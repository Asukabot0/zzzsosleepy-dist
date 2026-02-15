#!/usr/bin/env python3
import base64
import os
import random
import struct
import time
from typing import List, Tuple, Optional

from fpylll import IntegerMatrix, LLL, SVP, BKZ
from fpylll.fplll import bkz_param

HOST = os.environ.get("HOST", "challenges.1pc.tf")
PORT = int(os.environ.get("PORT", "28518"))

FRAME_LEN = 80
NUM_FRAMES = 120
ADC_BITS = 10

Q = 1 << 32
MASK = Q - 1


def _patch_fplll_strategies() -> None:
    """
    Some fpylll builds hardcode a non-existent strategies path at build time.
    Force SVP/BKZ to use the repo-bundled default.json if present.
    """
    strat_dir = os.path.join(os.path.dirname(__file__), "fplll_strategies")
    strat_file = os.path.join(strat_dir, "default.json")
    if not os.path.exists(strat_file):
        return

    strat_dir_b = strat_dir.encode()
    strat_file_b = strat_file.encode()

    # These attributes exist in current fpylll builds.
    BKZ.DEFAULT_STRATEGY_PATH = strat_dir_b
    BKZ.DEFAULT_STRATEGY = strat_file_b
    bkz_param.default_strategy_path = strat_dir_b
    bkz_param.default_strategy = strat_file_b


def adc_read(acc_u32: int) -> int:
    return acc_u32 >> (32 - ADC_BITS)


def get_signal() -> Tuple[bytes, List[int], object]:
    import socket

    print(f"[*] Connecting to {HOST}:{PORT} ...", flush=True)
    s = socket.create_connection((HOST, PORT), timeout=10)
    f = s.makefile("rb", buffering=0)

    b64_line = None
    while True:
        line = f.readline()
        if not line:
            break
        line = line.strip()
        if line == b"--- BEGIN SIGNAL ---":
            b64_line = f.readline().strip()
            _ = f.readline()
            break

    if not b64_line:
        raise RuntimeError("failed to read signal")

    raw = base64.b64decode(b64_line)
    if len(raw) != 8 + 2 * NUM_FRAMES:
        raise RuntimeError(f"unexpected signal length: {len(raw)}")

    seed = raw[:8]
    outs = list(struct.unpack("<" + "H" * NUM_FRAMES, raw[8:]))
    print(f"[*] Got signal: seed={seed.hex()} frames={len(outs)}", flush=True)
    return seed, outs, s


def gen_matrix(seed: bytes) -> Tuple[List[List[int]], List[List[int]]]:
    rng = random.Random(seed)
    A_unsigned: List[List[int]] = []
    A_signed: List[List[int]] = []

    for _ in range(NUM_FRAMES):
        row_u = [rng.getrandbits(32) for _ in range(FRAME_LEN)]
        row_s = [x if x < (1 << 31) else x - Q for x in row_u]
        A_unsigned.append(row_u)
        A_signed.append(row_s)

    return A_unsigned, A_signed


def build_B(A_signed: List[List[int]], outs: List[int]) -> List[int]:
    B: List[int] = []
    for j in range(len(outs)):
        s = 0
        row = A_signed[j]
        for x in row:
            s += x * 128
        bj = ((outs[j] << 22) + (1 << 21) - s) % Q
        if bj >= (1 << 31):
            bj -= Q
        B.append(bj)
    return B


def build_basis(A_signed: List[List[int]], B: List[int], *, W: int, WT: int) -> IntegerMatrix:
    n = FRAME_LEN
    m = len(B)
    dim = n + m + 1
    M = IntegerMatrix(dim, dim)

    # Top-left: W*I
    for i in range(n):
        M[i, i] = W
        for j in range(m):
            M[i, n + j] = A_signed[j][i]

    # Middle: -q*I
    for j in range(m):
        r = n + j
        M[r, r] = -Q

    # Last row: -B, WT
    last = dim - 1
    for j in range(m):
        M[last, n + j] = -B[j]
    M[last, last] = WT

    return M


def try_extract_key(
    vec: List[int],
    A_unsigned: List[List[int]],
    outs: List[int],
    *,
    W: int,
    WT: int,
) -> Optional[bytes]:
    if abs(vec[-1]) != WT:
        return None
    if vec[-1] == -WT:
        vec = [-x for x in vec]

    key = []
    for i in range(FRAME_LEN):
        if vec[i] % W != 0:
            return None
        kp = vec[i] // W
        k = kp + 128
        if not (0 <= k <= 255):
            return None
        key.append(k)

    # Verify on the equations used in the lattice (cheap) and then on all 120 frames (full).
    for j in range(len(outs)):
        acc = 0
        row = A_unsigned[j]
        for x, y in zip(row, key):
            acc = (acc + (x * y)) & MASK
        if adc_read(acc) != outs[j]:
            return None

    return bytes(key)


def verify_all(seed: bytes, outs_full: List[int], key: bytes) -> bool:
    A_unsigned, _ = gen_matrix(seed)
    for j in range(NUM_FRAMES):
        acc = 0
        for x, y in zip(A_unsigned[j], key):
            acc = (acc + (x * y)) & MASK
        if adc_read(acc) != outs_full[j]:
            return False
    return True


def solve(seed: bytes, outs_full: List[int]) -> bytes:
    # Centering/scaling parameters (from the intended solution writeup).
    W = 1 << 14
    WT = 1 << 21

    t0 = time.time()
    A_unsigned_full, A_signed_full = gen_matrix(seed)
    print(f"[*] Matrix generated in {time.time() - t0:.2f}s", flush=True)

    # Try increasing number of equations. More equations increases dimension but can help uniqueness.
    for frames in (80, 90, 100, 110, 120):
        outs = outs_full[:frames]
        A_unsigned = A_unsigned_full[:frames]
        A_signed = A_signed_full[:frames]

        print(f"[*] === frames={frames} dim={FRAME_LEN + frames + 1} ===", flush=True)
        t1 = time.time()
        B = build_B(A_signed, outs)
        M = build_basis(A_signed, B, W=W, WT=WT)
        print(f"[*] Built basis in {time.time() - t1:.2f}s", flush=True)

        # LLL pre-reduction.
        print("[*] LLL start", flush=True)
        t_lll = time.time()
        LLL.reduction(M, delta=0.99)
        print(f"[*] LLL done in {time.time() - t_lll:.2f}s", flush=True)

        # Check reduced basis vectors first (often hits immediately).
        dim = M.nrows
        print(f"[*] Scanning {dim} basis vectors ...", flush=True)
        for i in range(dim):
            vec = [int(M[i, j]) for j in range(dim)]
            key = try_extract_key(vec, A_unsigned, outs, W=W, WT=WT)
            if key is None:
                continue
            if verify_all(seed, outs_full, key):
                print("[+] Key verified on all frames", flush=True)
                return key

        # Avoid BKZ: many builds hit "infinite loop in babai". Use SVP enumeration instead.
        # Note: SVP may still rely on strategies; _patch_fplll_strategies() handles that.
        print("[*] Basis scan miss, trying SVP.shortest_vector(preprocess=False) ...", flush=True)
        try:
            v = SVP.shortest_vector(M, method="fast", pruning=False, preprocess=False)
        except Exception as e:
            print(f"[-] SVP failed: {e}", flush=True)
            continue

        key = try_extract_key([int(x) for x in v], A_unsigned, outs, W=W, WT=WT)
        if key is not None and verify_all(seed, outs_full, key):
            print("[+] Key verified on all frames", flush=True)
            return key

        print("[*] SVP miss", flush=True)

    raise RuntimeError("key not found")


def main() -> None:
    _patch_fplll_strategies()
    seed, outs, sock = get_signal()
    key = solve(seed, outs)
    sock.sendall(key.hex().encode() + b"\n")
    resp = sock.recv(8192)
    try:
        sock.settimeout(0.5)
        resp += sock.recv(8192)
    except Exception:
        pass
    print(resp.decode(errors="replace"), flush=True)


if __name__ == "__main__":
    main()

