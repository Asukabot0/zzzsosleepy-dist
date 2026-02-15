#!/usr/bin/env python3
import base64
import struct
import random
from typing import List, Tuple, Optional

from fpylll import IntegerMatrix, LLL, BKZ
from fpylll.fplll import bkz_param

HOST = "challenges.1pc.tf"
PORT = 25857

FRAME_LEN = 80
NUM_FRAMES = 120
ADC_BITS = 10

Q = 1 << 32
MASK = Q - 1

W = 1 << 14
WT = 1 << 21


def u32(x: int) -> int:
    return x & MASK


def adc_read(acc_u32: int) -> int:
    return acc_u32 >> (32 - ADC_BITS)


def get_signal() -> Tuple[bytes, List[int]]:
    # Avoid relying on pwntools for parsing; simple socket read is enough.
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

    # keep connection open: the service expects a hex guess line later.
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
        # B_j = out*2^22 + 2^21 - sum(A*128) (mod q), then choose signed rep
        bj = ((outs[j] << 22) + (1 << 21) - s) % Q
        if bj >= (1 << 31):
            bj -= Q
        B.append(bj)
    return B


def build_basis(A_signed: List[List[int]], B: List[int], *, W: int, WT: int) -> IntegerMatrix:
    n = FRAME_LEN
    m = len(B)
    dim = n + m + 1

    # We want column-basis C such that:
    # C * (k', y, t) = (W k', A k' - q y - t B, t WT)
    # fpylll uses row-basis, so we feed C^T.
    M = IntegerMatrix(dim, dim)

    # rows 0..n-1 correspond to k' columns
    for i in range(n):
        M[i, i] = W
        for j in range(m):
            M[i, n + j] = A_signed[j][i]
        M[i, dim - 1] = 0

    # rows n..n+m-1 correspond to y columns
    for j in range(m):
        r = n + j
        M[r, r] = -Q

    # last row corresponds to t column
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

    # normalize sign so last coord is +WT
    if vec[-1] == -WT:
        vec = [-x for x in vec]

    # k' must be exact multiples of W
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

    # Verify against observed outs
    for j in range(m):
        acc = 0
        row = A_unsigned[j]
        for x, y in zip(row, key):
            acc = (acc + (x * y)) & MASK
        if adc_read(acc) != outs[j]:
            return None

    return bytes(key)


def solve(seed: bytes, outs: List[int], *, tries: int = 4, frames: int = 96) -> bytes:
    A_unsigned, A_signed = gen_matrix(seed)

    if not (1 <= frames <= len(outs)):
        raise ValueError(f"frames must be in [1, {len(outs)}]")

    # Use a subset of equations to reduce lattice dimension and speed up LLL.
    # 80 unknowns => ~90-100 equations is typically enough.
    idx = list(range(frames))
    outs_s = [outs[i] for i in idx]
    A_unsigned_s = [A_unsigned[i] for i in idx]
    A_signed_s = [A_signed[i] for i in idx]

    B = build_B(A_signed_s, outs_s)

    # Parameter search: small changes in W can affect whether LLL exposes the
    # embedded uSVP vector in the reduced basis.
    candidate_W = [1 << 13, 1 << 14, 1 << 15, 1 << 16]
    WT_local = 1 << 21

    # LLL-only solving: uSVP embedding often places the target vector directly in the
    # reduced basis. We add a few randomized re-reductions to increase success rate.
    rng = random.Random(seed + b"LLL")

    def check_basis(M: IntegerMatrix, *, W_local: int, WT_local: int) -> Optional[bytes]:
        dim = M.nrows
        for i in range(dim):
            vec = [int(M[i, j]) for j in range(dim)]
            key = try_extract_key(vec, A_unsigned_s, outs_s, W=W_local, WT=WT_local)
            if key is not None:
                return key
        return None

    for W_local in candidate_W:
        M0 = build_basis(A_signed_s, B, W=W_local, WT=WT_local)
        dim = M0.nrows

        for attempt in range(tries):
            M = IntegerMatrix.from_matrix(M0)

            # Attempt 0: straight LLL on the constructed basis.
            # Later attempts: unimodular randomization + re-LLL.
            if attempt != 0:
                for i in range(dim - 1, 0, -1):
                    j = rng.randrange(i + 1)
                    if i != j:
                        M.swap_rows(i, j)

                for _ in range(dim // 3):
                    i = rng.randrange(dim)
                    j = rng.randrange(dim)
                    if i == j:
                        continue
                    c = rng.choice([-2, -1, 1, 2])
                    for col in range(dim):
                        M[i, col] = int(M[i, col]) + c * int(M[j, col])

            # 'fast' can hit a known fplll Babai infinite-loop bug on some builds.
            # 'heuristic' is slower but avoids that failure mode here.
            LLL.reduction(M, delta=0.99, float_type="d", method="heuristic")
            key = check_basis(M, W_local=W_local, WT_local=WT_local)
            if key is not None:
                return key

    raise RuntimeError("key not found (LLL random restarts)")


def main():
    # Some fpylll builds hardcode a CI path for the FPLLL strategies JSON, which
    # breaks SVP.shortest_vector() at runtime. Point it at the bundled file.
    import os

    strat_dir = os.path.join(os.path.dirname(__file__), "fplll_strategies")
    strat_dir_b = strat_dir.encode()
    strat_file_b = os.path.join(strat_dir, "default.json").encode()

    # These are used by various fpylll entrypoints (SVP/BKZ), depending on build.
    BKZ.DEFAULT_STRATEGY_PATH = strat_dir_b
    BKZ.DEFAULT_STRATEGY = strat_file_b
    bkz_param.default_strategy_path = strat_dir_b
    bkz_param.default_strategy = strat_file_b

    seed, outs, sock = get_signal()
    key = solve(seed, outs)

    # Send guess
    sock.sendall(key.hex().encode() + b"\n")
    resp = sock.recv(8192)
    # Best-effort: drain a bit more
    try:
        sock.settimeout(0.5)
        resp += sock.recv(8192)
    except Exception:
        pass
    print(resp.decode(errors="replace"))


if __name__ == "__main__":
    main()
