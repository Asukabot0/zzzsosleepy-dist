#!/usr/bin/env python3
import base64
import struct
import random
import sys
from typing import List, Tuple, Optional

try:
    from fpylll import IntegerMatrix, LLL, BKZ
except ImportError:
    # If not in path, hope it's installed
    from fpylll import IntegerMatrix, LLL, BKZ

HOST = "challenges.1pc.tf"
PORT = 25857

# Original constants
FRAME_LEN = 80
NUM_FRAMES_TOTAL = 120
ADC_BITS = 10
Q = 1 << 32
MASK = Q - 1

# Weights
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
    outs = list(struct.unpack("<" + "H" * NUM_FRAMES_TOTAL, raw[8:]))
    print(f"[*] Received seed: {seed.hex()} and {len(outs)} outputs")
    return seed, outs, s

def gen_matrix(seed: bytes, num_frames: int) -> Tuple[List[List[int]], List[List[int]]]:
    rng = random.Random(seed)
    A_unsigned: List[List[int]] = []
    A_signed: List[List[int]] = []
    
    # We must generate all frames to match the sequence, but we only store the ones we need?
    # Actually random state must be preserved.
    # The problem generates 120 frames. We can generate them all and slice.
    
    for _ in range(NUM_FRAMES_TOTAL):
        row_u = [rng.getrandbits(32) for _ in range(FRAME_LEN)]
        row_s = [x if x < (1 << 31) else x - Q for x in row_u]
        
        A_unsigned.append(row_u)
        A_signed.append(row_s)
        
    return A_unsigned[:num_frames], A_signed[:num_frames]

def build_B(A_signed: List[List[int]], outs: List[int], num_frames: int) -> List[int]:
    B: List[int] = []
    for j in range(num_frames):
        s = 0
        for x in A_signed[j]: s += x * 128
        bj = ((outs[j] << 22) + (1 << 21) - s) % Q
        if bj >= (1 << 31): bj -= Q
        B.append(bj)
    return B

def build_basis(A_signed: List[List[int]], B: List[int], num_frames: int) -> IntegerMatrix:
    n = FRAME_LEN
    m = num_frames
    dim = n + m + 1
    M = IntegerMatrix(dim, dim)
    
    # Fill diagonal for key weights
    for i in range(n):
        M[i, i] = W
    
    # Fill A transpose part
    for i in range(n):
        for j in range(m):
            M[i, n + j] = A_signed[j][i]
            
    # Fill -q I part
    for j in range(m):
        r = n + j
        M[r, r] = -Q
        
    # Fill last row
    last = dim - 1
    for j in range(m):
        M[last, n + j] = -B[j]
    M[last, last] = WT
    
    return M

def try_extract_key(vec: List[int], full_A_unsigned: List[List[int]], full_outs: List[int]) -> Optional[bytes]:
    # Check weight target
    if abs(vec[-1]) != WT: return None
    if vec[-1] == -WT: vec = [-x for x in vec]
    
    key = []
    for i in range(FRAME_LEN):
        if vec[i] % W != 0: return None
        kp = vec[i] // W
        k = kp + 128
        if not (0 <= k <= 255): return None
        key.append(k)
        
    # Verify against ALL outputs, not just the subset
    for j in range(len(full_outs)):
        acc = 0
        for x, y in zip(full_A_unsigned[j], key): acc = (acc + x * y) & MASK
        if adc_read(acc) != full_outs[j]: return None
        
    return bytes(key)

def check_basis(M: IntegerMatrix, full_A_unsigned, full_outs) -> Optional[bytes]:
    dim = M.nrows
    # Check first few vectors (usually sufficient if reduced)
    for i in range(min(dim, 50)): 
        vec = [int(M[i, j]) for j in range(dim)]
        key = try_extract_key(vec, full_A_unsigned, full_outs)
        if key: return key
    return None


def solve(seed: bytes, outs: List[int]) -> bytes:
    # Try multiple frame counts, starting from a safer number
    for target_frames in [100, 102, 104]:
        print(f"[*] Trying with {target_frames}/{NUM_FRAMES_TOTAL} frames...")
        
        full_A_unsigned, full_A_signed = gen_matrix(seed, NUM_FRAMES_TOTAL)
        
        # Slice for lattice
        A_use_s = full_A_signed[:target_frames]
        outs_use = outs[:target_frames]
        
        B = build_B(A_use_s, outs_use, target_frames)
        
        print("[*] Building lattice basis...")
        M = build_basis(A_use_s, B, target_frames)
        
        print(f"[*] Lattice Dimension: {M.nrows} x {M.ncols}")

        # LLL with safe delta
        print("[*] Running LLL reduction (delta=0.99)...")
        try:
            LLL.reduction(M, delta=0.99)
        except Exception as e:
            print(f"[-] LLL failed: {e}")
            continue

        print("[*] LLL done. Checking basis...")
        key = check_basis(M, full_A_unsigned, outs)
        if key: return key
        
        print("[*] Key not found by LLL. Running BKZ-15...")
        try:
            BKZ.reduction(M, BKZ.Param(block_size=15))
        except Exception as e:
            print(f"[-] BKZ-15 failed: {e}")
            continue
            
        key = check_basis(M, full_A_unsigned, outs)
        if key: return key

    raise RuntimeError("key not found / solver failed after all attempts")

def main():
    seed, outs, sock = get_signal()
    try:
        key = solve(seed, outs)
        if key:
            print(f"[+] Key: {key.hex()}")
            sock.sendall(key.hex().encode() + b"\n")
            resp = sock.recv(8192)
            try:
                sock.settimeout(1.0)
                resp += sock.recv(8192)
            except: pass
            print(resp.decode(errors="replace"))
        else:
            print("[-] Solver finished without finding key.")
    except Exception as e:
        print(f"[-] Solver failed: {e}")
    finally:
        sock.close()

if __name__ == "__main__":
    main()
