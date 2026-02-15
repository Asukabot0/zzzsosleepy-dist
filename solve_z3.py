#!/usr/bin/env python3
import base64
import struct
import random
from typing import List, Tuple, Optional

import z3

HOST = "challenges.1pc.tf"
PORT = 25857

FRAME_LEN = 80
NUM_FRAMES = 120
ADC_BITS = 10

Q = 1 << 32
MASK = Q - 1


def adc_read(acc_u32: int) -> int:
    return acc_u32 >> (32 - ADC_BITS)


def get_signal() -> Tuple[bytes, List[int], object]:
    import socket

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
            # consume end marker
            _ = f.readline()
            break

    if not b64_line:
        raise RuntimeError("failed to read signal")

    raw = base64.b64decode(b64_line)
    if len(raw) != 8 + 2 * NUM_FRAMES:
        raise RuntimeError(f"unexpected signal length: {len(raw)}")

    seed = raw[:8]
    outs = list(struct.unpack("<" + "H" * NUM_FRAMES, raw[8:]))
    return seed, outs, s


def gen_matrix(seed: bytes) -> List[List[int]]:
    rng = random.Random(seed)
    A: List[List[int]] = []
    for _ in range(NUM_FRAMES):
        A.append([rng.getrandbits(32) for _ in range(FRAME_LEN)])
    return A


def verify_key(A: List[List[int]], outs: List[int], key: bytes) -> bool:
    for j in range(NUM_FRAMES):
        acc = 0
        row = A[j]
        for x, y in zip(row, key):
            acc = (acc + (x * y)) & MASK
        if adc_read(acc) != outs[j]:
            return False
    return True


def solve_z3(seed: bytes, outs: List[int]) -> bytes:
    A = gen_matrix(seed)

    k = [z3.BitVec(f"k{i}", 8) for i in range(FRAME_LEN)]
    kz = [z3.ZeroExt(24, ki) for ki in k]  # -> 32-bit

    s = z3.SolverFor("QF_BV")
    s.set("timeout", 120_000)  # ms

    def add_equation(j: int) -> None:
        acc = z3.BitVecVal(0, 32)
        row = A[j]
        for i in range(FRAME_LEN):
            acc = acc + kz[i] * z3.BitVecVal(row[i], 32)
        s.add(z3.LShR(acc, 32 - ADC_BITS) == z3.BitVecVal(outs[j], 32))

    # Incrementally add constraints until the model is unique enough to verify.
    # This tends to solve much faster than throwing all 120 equations at once.
    checkpoints = [20, 30, 40, 60, 80, 120]
    added = 0
    for target in checkpoints:
        while added < target:
            add_equation(added)
            added += 1

        if s.check() != z3.sat:
            continue

        m = s.model()
        key = bytes(int(m[ki].as_long()) for ki in k)
        if verify_key(A, outs, key):
            return key

        # Not unique yet; keep adding equations.

    raise RuntimeError("no key found (z3)")


def main() -> None:
    seed, outs, sock = get_signal()
    key = solve_z3(seed, outs)

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

