#!/usr/bin/env python3
import base64
import struct
import random
import sys
from typing import List, Tuple, Optional

try:
    from fpylll import IntegerMatrix, LLL
except ImportError:
    from fpylll import IntegerMatrix, LLL

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
    for i in range(dim):
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

    deltas = [0.75, 0.90, 0.95, 0.98, 0.99]
    for d in deltas:
        print(f"[*] Running LLL reduction (delta={d})...")
        try:
            LLL.reduction(M, delta=d)
            print(f"[*] LLL delta={d} done. Checking basis...")
            key = check_basis(M, A_unsigned, outs)
            if key:
                print(f"[+] Key found with delta={d}!")
                return key
        except Exception as e:
            print(f"[-] Error during LLL delta={d}: {e}")
            break # Stop if crash/error 
    
    raise RuntimeError("key not found after trying all deltas")

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
