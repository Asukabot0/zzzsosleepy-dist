#!/usr/bin/env python3
import base64
import os
import random
import struct
import time
from typing import List, Tuple, Optional

from fpylll import IntegerMatrix, LLL, BKZ
from fpylll.fplll import bkz_param

HOST = os.environ.get("HOST", "challenges.1pc.tf")
PORT = int(os.environ.get("PORT", "25857"))

FRAME_LEN = 80
NUM_FRAMES = 120
ADC_BITS = 10

Q = 1 << 32
MASK = Q - 1


def _patch_fplll_strategies() -> None:
    # Some fpylll wheels hardcode a CI path for default.json. Make SVP/BKZ load
    # strategies from the local checkout if present.
    strat_dir = os.path.join(os.path.dirname(__file__), "fplll_strategies")
    strat_file = os.path.join(strat_dir, "default.json")
    if not os.path.exists(strat_file):
        return

    strat_dir_b = strat_dir.encode()
    strat_file_b = strat_file.encode()
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

    for i in range(n):
        M[i, i] = W
        for j in range(m):
            M[i, n + j] = A_signed[j][i]

    for j in range(m):
        r = n + j
        M[r, r] = -Q

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
    n = FRAME_LEN
    m = len(outs)

    if abs(vec[-1]) != WT:
        return None
    if vec[-1] == -WT:
        vec = [-x for x in vec]

    kprime = []
    for i in range(n):
        if vec[i] % W != 0:
            return None
        kprime.append(vec[i] // W)

    key = []
    for kp in kprime:
        k = kp + 128
        if not (0 <= k <= 255):
            return None
        key.append(k)

    for j in range(m):
        acc = 0
        row = A_unsigned[j]
        for x, y in zip(row, key):
            acc = (acc + (x * y)) & MASK
        if adc_read(acc) != outs[j]:
            return None

    return bytes(key)


def _scan_basis(M: IntegerMatrix, A_unsigned_s, outs_s, *, W: int, WT: int, limit: int = 60) -> Optional[bytes]:
    dim = M.nrows
    for i in range(min(dim, limit)):
        vec = [int(M[i, j]) for j in range(dim)]
        key = try_extract_key(vec, A_unsigned_s, outs_s, W=W, WT=WT)
        if key is not None:
            return key
    return None


def solve(seed: bytes, outs: List[int]) -> bytes:
    W = 1 << 14
    WT = 1 << 21

    t0 = time.time()
    A_unsigned, A_signed = gen_matrix(seed)
    print(f"[*] Matrix generated in {time.time() - t0:.2f}s", flush=True)

    # Keep dimensions modest to avoid huge BKZ memory/time on Docker Desktop.
    for frames in (70, 78, 86, 94, 102):
        print(f"[*] === frames={frames} ===", flush=True)
        idx = list(range(frames))
        outs_s = [outs[i] for i in idx]
        A_unsigned_s = [A_unsigned[i] for i in idx]
        A_signed_s = [A_signed[i] for i in idx]

        t1 = time.time()
        B = build_B(A_signed_s, outs_s)
        M = build_basis(A_signed_s, B, W=W, WT=WT)
        print(f"[*] Built basis dim={M.nrows} in {time.time() - t1:.2f}s", flush=True)

        t_lll = time.time()
        print("[*] LLL start", flush=True)
        LLL.reduction(M, delta=0.99)
        print(f"[*] LLL done in {time.time() - t_lll:.2f}s, scanning basis ...", flush=True)
        key = _scan_basis(M, A_unsigned_s, outs_s, W=W, WT=WT, limit=M.nrows)
        if key is not None:
            return key
        print("[*] Scan miss after LLL", flush=True)

        # BKZ is usually enough; much cheaper than full SVP enumeration here.
        for bs in (12, 15, 18, 22, 26, 30):
            max_time = 30 if bs <= 18 else (90 if bs <= 26 else 120)
            max_loops = 2 if bs <= 18 else (5 if bs <= 26 else 6)
            print(f"[*] BKZ-{bs} start (max_time={max_time}s max_loops={max_loops})", flush=True)
            # Cap runtime: prevent runaway BKZ from getting SIGKILLed by the host.
            # Increase caps gradually with blocksize.
            t_bkz = time.time()
            BKZ.reduction(
                M,
                BKZ.Param(
                    block_size=bs,
                    max_loops=max_loops,
                    max_time=max_time,
                    rerandomization_density=3,
                ),
            )
            print(f"[*] BKZ-{bs} done in {time.time() - t_bkz:.2f}s, scanning basis ...", flush=True)
            key = _scan_basis(M, A_unsigned_s, outs_s, W=W, WT=WT, limit=M.nrows)
            if key is not None:
                return key
            print(f"[*] Scan miss after BKZ-{bs}", flush=True)

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
    print(resp.decode(errors="replace"))


if __name__ == "__main__":
    main()
