#!/usr/bin/env python3
"""
g6k-based solver for the zzzsosleepy challenge.

Rationale:
- fplll BKZ >= ~40 may hit the "infinite loop in babai" failure mode on some builds.
- g6k sieving/pump often finds the embedded short vector more reliably without deep BKZ.

Environment:
  - Requires: fpylll, cysignals, g6k
  - Uses HOST/PORT env vars (defaults match current challenge)
  - Verbose progress output (unbuffered recommended: python -u)
"""

import base64
import os
import random
import struct
import time
from typing import List, Tuple, Optional

from fpylll import IntegerMatrix, LLL, GSO

HOST = os.environ.get("HOST", "challenges.1pc.tf")
PORT = int(os.environ.get("PORT", "28518"))

FRAME_LEN = 80
NUM_FRAMES = 120
ADC_BITS = 10

Q = 1 << 32
MASK = Q - 1


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


def gen_random_signal() -> Tuple[bytes, List[int]]:
    key = os.urandom(FRAME_LEN)
    seed = os.urandom(8)
    rng = random.Random(seed)
    outs: List[int] = []
    for _ in range(NUM_FRAMES):
        row = [rng.getrandbits(32) for _ in range(FRAME_LEN)]
        acc = 0
        for x, y in zip(row, key):
            acc = (acc + (x * y)) & MASK
        outs.append(adc_read(acc))
    return seed, outs


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


def try_extract_key(vec: List[int], A_unsigned: List[List[int]], outs: List[int], *, W: int, WT: int) -> Optional[bytes]:
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
    # Import g6k lazily so the script still imports even if g6k isn't installed.
    from g6k import Siever
    from g6k.siever_params import SieverParams
    from g6k.algorithms.pump import pump

    frames = int(os.environ.get("FRAMES", "110"))
    W = int(os.environ.get("W", str(1 << 14)))
    WT = int(os.environ.get("WT", str(1 << 21)))

    # Sieving parameters / schedule
    blocksize = int(os.environ.get("BLOCKSIZE", "60"))
    pump_down = int(os.environ.get("PUMP_DOWN", "40"))
    dim4free = int(os.environ.get("DIM4FREE", "10"))

    t0 = time.time()
    A_unsigned_full, A_signed_full = gen_matrix(seed)
    print(f"[*] Matrix generated in {time.time() - t0:.2f}s", flush=True)

    outs = outs_full[:frames]
    A_unsigned = A_unsigned_full[:frames]
    A_signed = A_signed_full[:frames]

    B = build_B(A_signed, outs)
    Bmat = build_basis(A_signed, B, W=W, WT=WT)
    dim = Bmat.nrows
    print(f"[*] Lattice dim={dim} frames={frames} W={W} WT={WT}", flush=True)

    # LLL preprocess
    print("[*] LLL preprocess...", flush=True)
    t_lll = time.time()
    LLL.reduction(Bmat, delta=0.99)
    print(f"[*] LLL done in {time.time() - t_lll:.2f}s", flush=True)

    # Check basis vectors after LLL
    for i in range(min(dim, 200)):
        vec = [int(Bmat[i, j]) for j in range(dim)]
        key = try_extract_key(vec, A_unsigned, outs, W=W, WT=WT)
        if key is not None and verify_all(seed, outs_full, key):
            print("[+] Key found after LLL scan", flush=True)
            return key

    # Setup GSO + siever
    gso = GSO.Mat(Bmat)
    gso.update_gso()
    lll_obj = LLL.Reduction(gso)
    lll_obj()
    gso.update_gso()

    params = SieverParams()
    siever = Siever(gso, params=params)

    # Pump schedule: try a few kappas; each pump tends to be expensive but more reliable than BKZ here.
    # We keep it verbose but bounded; you can tune via env vars.
    kappas = list(range(0, max(1, dim - blocksize), max(1, blocksize // 4)))
    print(f"[*] Pump schedule: {len(kappas)} windows, blocksize={blocksize}, down={pump_down}, dim4free={dim4free}", flush=True)

    for it, kappa in enumerate(kappas, 1):
        print(f"[*] pump {it}/{len(kappas)} kappa={kappa}", flush=True)
        t_p = time.time()
        try:
            pump(
                siever,
                kappa=kappa,
                blocksize=blocksize,
                dim4free=dim4free,
                down_sieve=pump_down,
                verbose=True,
            )
        except Exception as e:
            print(f"[-] pump failed at kappa={kappa}: {e}", flush=True)
            continue
        print(f"[*] pump kappa={kappa} done in {time.time() - t_p:.2f}s", flush=True)

        # After each pump, the basis is modified in-place (through gso). Extract current integer basis.
        gso.update_gso()

        Bcur = gso.B
        # Scan a chunk of basis vectors
        for i in range(min(dim, 300)):
            vec = [int(Bcur[i, j]) for j in range(dim)]
            key = try_extract_key(vec, A_unsigned, outs, W=W, WT=WT)
            if key is not None and verify_all(seed, outs_full, key):
                print("[+] Key found after pump scan", flush=True)
                return key

    raise RuntimeError("key not found (g6k)")


def main() -> None:
    if os.environ.get("SELFTEST") == "1":
        n = int(os.environ.get("SELFTEST_N", "1"))
        for i in range(n):
            seed, outs = gen_random_signal()
            print(f"[*] SELFTEST {i+1}/{n} seed={seed.hex()}", flush=True)
            key = solve(seed, outs)
            if len(key) != FRAME_LEN:
                raise RuntimeError("SELFTEST produced wrong key length")
            print("[+] SELFTEST OK", flush=True)
        return

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
