#!/usr/bin/env python3
import base64
import struct
import random
import sys
from typing import List, Tuple, Optional

try:
    from fpylll import IntegerMatrix, LLL, BKZ, GSO
except ImportError:
    from fpylll import IntegerMatrix, LLL, BKZ, GSO

HOST = "challenges.1pc.tf"
PORT = 25857

FRAME_LEN = 80
NUM_FRAMES = 120
ADC_BITS = 10
Q = 1 << 32
MASK = Q - 1
W = 1 << 14
WT = 1 << 21

def u32(x: int) -> int: return x & MASK
def adc_read(acc_u32: int) -> int: return acc_u32 >> (32 - ADC_BITS)

def get_signal() -> Tuple[bytes, List[int], object]:
    import socket
    print("[*] Connecting to challenge...")
    s = socket.create_connection((HOST, PORT), timeout=10)
    f = s.makefile("rb", buffering=0)

    b64_line = None
    while True:
        line = f.readline()
        if not line: break
        line = line.strip()
        if line == b"--- BEGIN SIGNAL ---":
            b64_line = f.readline().strip()
            _ = f.readline()
            break

    if not b64_line: raise RuntimeError("failed to read signal")
    raw = base64.b64decode(b64_line)
    seed = raw[:8]
    outs = list(struct.unpack("<" + "H" * NUM_FRAMES, raw[8:]))
    print(f"[*] Received seed: {seed.hex()} and {len(outs)} outputs")
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
    for j in range(NUM_FRAMES):
        s = 0
        for x in A_signed[j]: s += x * 128
        bj = ((outs[j] << 22) + (1 << 21) - s) % Q
        if bj >= (1 << 31): bj -= Q
        B.append(bj)
    return B

def build_basis(A_signed: List[List[int]], B: List[int]) -> IntegerMatrix:
    n, m = FRAME_LEN, NUM_FRAMES
    dim = n + m + 1
    M = IntegerMatrix(dim, dim)
    for i in range(n):
        M[i, i] = W
        for j in range(m): M[i, n + j] = A_signed[j][i]
    for j in range(m):
        r = n + j
        M[r, r] = -Q
    last = dim - 1
    for j in range(m): M[last, n + j] = -B[j]
    M[last, last] = WT
    return M

def try_extract_key(vec: List[int], A_unsigned: List[List[int]], outs: List[int]) -> Optional[bytes]:
    if abs(vec[-1]) != WT: return None
    if vec[-1] == -WT: vec = [-x for x in vec]
    key = []
    for i in range(FRAME_LEN):
        if vec[i] % W != 0: return None
        kp = vec[i] // W
        k = kp + 128
        if not (0 <= k <= 255): return None
        key.append(k)
    for j in range(NUM_FRAMES):
        acc = 0
        for x, y in zip(A_unsigned[j], key): acc = (acc + x * y) & MASK
        if adc_read(acc) != outs[j]: return None
    return bytes(key)

def check_basis(M: IntegerMatrix, A_unsigned, outs) -> Optional[bytes]:
    dim = M.nrows
    # Check first 50 vectors, usually short vector is at start
    for i in range(min(50, dim)):
        vec = [int(M[i, j]) for j in range(dim)]
        key = try_extract_key(vec, A_unsigned, outs)
        if key: return key
    return None

def solve(seed: bytes, outs: List[int]) -> bytes:
    print("[*] Generating matrix...")
    A_unsigned, A_signed = gen_matrix(seed)
    B = build_B(A_signed, outs)
    print("[*] Building lattice basis...")
    M = build_basis(A_signed, B)

    # Use GSO with mpfr (explicitly) to improve stability of reductions
    print("[*] Initializing GSO (float_type='mpfr')...")
    G = GSO.Mat(M, float_type="mpfr")
    
    # Run LLL
    print("[*] Running LLL reduction (delta=0.99)...")
    lll_obj = LLL.Reduction(G, delta=0.99)
    lll_obj()
    
    print("[*] LLL done. Checking basis...")
    key = check_basis(M, A_unsigned, outs)
    if key:
        print("[+] Key found after LLL!")
        return key

    # Run BKZ-20
    print("[*] Running BKZ-20...")
    bkz_param = BKZ.Param(block_size=20, strategies=BKZ.DEFAULT_STRATEGY, flags=BKZ.AUTO_ABORT)
    bkz_obj = BKZ.Reduction(G, bkz_param)
    bkz_obj()
    
    print("[*] BKZ-20 done. Checking basis...")
    key = check_basis(M, A_unsigned, outs)
    if key:
        print("[+] Key found after BKZ-20!")
        return key

    # Run BKZ-30 (last resort)
    print("[*] Running BKZ-30...")
    bkz_param = BKZ.Param(block_size=30, strategies=BKZ.DEFAULT_STRATEGY, flags=BKZ.AUTO_ABORT)
    bkz_obj = BKZ.Reduction(G, bkz_param)
    bkz_obj()
    
    print("[*] BKZ-30 done. Checking basis...")
    key = check_basis(M, A_unsigned, outs)
    if key:
        print("[+] Key found after BKZ-30!")
        return key

    raise RuntimeError("key not found even after robust BKZ")

def main():
    seed, outs, sock = get_signal()
    try:
        key = solve(seed, outs)
        print(f"[+] Key: {key.hex()}")
        sock.sendall(key.hex().encode() + b"\n")
        resp = sock.recv(8192)
        try:
            sock.settimeout(1.0)
            resp += sock.recv(8192)
        except: pass
        print(resp.decode(errors="replace"))
    except Exception as e:
        print(f"[-] Solver failed: {e}")
        sock.close()

if __name__ == "__main__":
    main()
