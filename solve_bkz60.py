#!/usr/bin/env python3
import base64
import multiprocessing as mp
import os
import queue
import random
import signal
import struct
import time
from typing import List, Tuple, Optional, Dict

from fpylll import IntegerMatrix, LLL, BKZ
from fpylll.fplll import bkz_param

HOST = os.environ.get("HOST", "challenges.1pc.tf")
PORT = int(os.environ.get("PORT", "28518"))

FRAME_LEN = 80
NUM_FRAMES = 120
ADC_BITS = 10

Q = 1 << 32
MASK = Q - 1


def _patch_fplll_strategies() -> None:
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


def _log2_norm2_prefix(vec: List[int], k: int = 80) -> float:
    # Approx log2 of squared norm, using only first k coordinates for speed.
    # Good enough to compare relative progress across reductions.
    s = 0
    kk = min(k, len(vec))
    for i in range(kk):
        x = int(vec[i])
        s += x * x
    if s <= 0:
        return float("-inf")
    return s.bit_length() - 1


def _report_basis(prefix: str, M: IntegerMatrix, WT: int, tag: str) -> None:
    # Print a tiny digest of the reduced basis so we can see if reduction helps.
    # We avoid expensive float norms; use log2 of partial norm^2.
    n = M.nrows
    top = min(6, n)
    vals = []
    for i in range(top):
        vec = [int(M[i, j]) for j in range(n)]
        ln = _log2_norm2_prefix(vec)
        last_ok = abs(int(vec[-1])) == WT
        vals.append(f"{ln}{'*' if last_ok else ''}")
    print(f"{prefix} {tag} basis[0:{top}] log2(norm2_prefix)={','.join(vals)}", flush=True)


def verify_all(seed: bytes, outs_full: List[int], key: bytes) -> bool:
    A_unsigned, _ = gen_matrix(seed)
    for j in range(NUM_FRAMES):
        acc = 0
        for x, y in zip(A_unsigned[j], key):
            acc = (acc + (x * y)) & MASK
        if adc_read(acc) != outs_full[j]:
            return False
    return True


def _worker_entry(
    result_q,
    stop_evt,
    seed: bytes,
    outs_full: List[int],
    A_unsigned_full: List[List[int]],
    A_signed_full: List[List[int]],
    *,
    frames: int,
    W: int,
    WT: int,
    r: int,
    bs_max: int,
    max_time: int,
    max_loops: int,
) -> None:
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass

    outs = outs_full[:frames]
    A_unsigned = A_unsigned_full[:frames]
    A_signed = A_signed_full[:frames]

    dim = FRAME_LEN + frames + 1
    prefix = f"[frames={frames} W={W} r={r} dim={dim}]"
    result_q.put(("stat", "start"))
    print(f"{prefix} start", flush=True)

    try:
        B = build_B(A_signed, outs)
        M = build_basis(A_signed, B, W=W, WT=WT)
    except Exception as e:
        result_q.put(("err", f"{prefix} build: {e}"))
        return

    # Randomize basis for restart r.
    rng = random.Random(seed + frames.to_bytes(2, "little") + W.to_bytes(4, "little") + r.to_bytes(4, "little"))
    if r != 0:
        for i in range(M.nrows - 1, 0, -1):
            j = rng.randrange(i + 1)
            if i != j:
                M.swap_rows(i, j)
        for _ in range(M.nrows // 3):
            i = rng.randrange(M.nrows)
            j = rng.randrange(M.nrows)
            if i == j:
                continue
            c = rng.choice([-2, -1, 1, 2])
            for col in range(M.ncols):
                M[i, col] = int(M[i, col]) + c * int(M[j, col])

    # LLL first.
    try:
        result_q.put(("stat", "lll"))
        t = time.time()
        LLL.reduction(M, delta=0.99)
        print(f"{prefix} LLL {time.time() - t:.2f}s", flush=True)
        _report_basis(prefix, M, WT, "after LLL")
    except Exception as e:
        result_q.put(("err", f"{prefix} LLL: {e}"))
        return

    # Scan after LLL.
    result_q.put(("stat", "scan"))
    for i in range(min(80, M.nrows)):
        vec = [int(M[i, j]) for j in range(M.nrows)]
        key = try_extract_key(vec, A_unsigned, outs, W=W, WT=WT)
        if key is not None and verify_all(seed, outs_full, key):
            result_q.put(("ok", key.hex()))
            return

    if stop_evt.is_set():
        result_q.put(("skip", f"{prefix} stopped"))
        return

    # BKZ ladder with caps. Babai may abort; treat as non-fatal and report.
    for bs in (20, 30, 40, 50, 60):
        if bs > bs_max:
            break

        result_q.put(("stat", "bkz"))
        print(f"{prefix} BKZ-{bs} start (caps: time={max_time}s loops={max_loops})", flush=True)
        try:
            t = time.time()
            BKZ.reduction(
                M,
                BKZ.Param(
                    block_size=bs,
                    max_time=max_time,
                    max_loops=max_loops,
                    rerandomization_density=3,
                ),
            )
            print(f"{prefix} BKZ-{bs} done {time.time() - t:.2f}s", flush=True)
            _report_basis(prefix, M, WT, f"after BKZ-{bs}")
        except Exception as e:
            # Known fplll failure mode; keep going with other restarts.
            result_q.put(("err", f"{prefix} BKZ-{bs}: {e}"))
            return

        # Scan some of the shortest-looking basis vectors.
        result_q.put(("stat", "scan"))
        for i in range(min(120, M.nrows)):
            vec = [int(M[i, j]) for j in range(M.nrows)]
            key = try_extract_key(vec, A_unsigned, outs, W=W, WT=WT)
            if key is not None and verify_all(seed, outs_full, key):
                result_q.put(("ok", key.hex()))
                return

        if stop_evt.is_set():
            result_q.put(("skip", f"{prefix} stopped"))
            return

    result_q.put(("miss", f"{prefix} no key"))


def solve(seed: bytes, outs_full: List[int]) -> bytes:
    WT = 1 << 21

    frames = int(os.environ.get("FRAMES", "110"))
    W = int(os.environ.get("W", str(1 << 14)))
    restarts = int(os.environ.get("RESTARTS", "50"))
    jobs = int(os.environ.get("JOBS", "0") or "0")
    if jobs <= 0:
        jobs = min(64, (os.cpu_count() or 8))
    jobs = max(1, jobs)

    bs_max = int(os.environ.get("BS_MAX", "60"))
    max_time = int(os.environ.get("BKZ_MAX_TIME", "120"))
    max_loops = int(os.environ.get("BKZ_MAX_LOOPS", "3"))

    t0 = time.time()
    A_unsigned_full, A_signed_full = gen_matrix(seed)
    print(f"[*] Matrix generated in {time.time() - t0:.2f}s", flush=True)
    print(
        f"[*] frames={frames} W={W} restarts={restarts} jobs={jobs} bs_max={bs_max} caps=(time={max_time}s loops={max_loops})",
        flush=True,
    )

    result_q: mp.Queue = mp.Queue()
    stop_evt = mp.Event()

    stages: Dict[str, int] = {"lll": 0, "bkz": 0, "scan": 0}
    running: List[Tuple[mp.Process, float, int]] = []
    next_r = 0
    done = 0

    def start_one(r: int) -> None:
        p = mp.Process(
            target=_worker_entry,
            args=(result_q, stop_evt, seed, outs_full, A_unsigned_full, A_signed_full),
            kwargs=dict(
                frames=frames,
                W=W,
                WT=WT,
                r=r,
                bs_max=bs_max,
                max_time=max_time,
                max_loops=max_loops,
            ),
            daemon=True,
        )
        p.start()
        running.append((p, time.time(), r))

    while len(running) < jobs and next_r < restarts:
        start_one(next_r)
        next_r += 1

    last = 0.0
    while done < restarts:
        now = time.time()
        if now - last >= 5:
            print(
                f"[*] progress: done={done}/{restarts} running={len(running)} next_r={next_r} lll={stages['lll']} bkz={stages['bkz']} scan={stages['scan']}",
                flush=True,
            )
            last = now

        try:
            msg = result_q.get(timeout=0.5)
        except queue.Empty:
            alive = []
            for p, ts, r in running:
                if p.is_alive():
                    alive.append((p, ts, r))
                else:
                    done += 1
            running = alive
            while len(running) < jobs and next_r < restarts and not stop_evt.is_set():
                start_one(next_r)
                next_r += 1
            if not running and next_r >= restarts:
                break
            continue

        kind = msg[0]
        if kind == "stat":
            stage = msg[1]
            if stage in stages:
                stages[stage] += 1
            continue
        if kind == "ok":
            key_hex = msg[1]
            print("[+] FOUND key", flush=True)
            stop_evt.set()
            for p, _, _ in running:
                if p.is_alive():
                    try:
                        p.terminate()
                    except Exception:
                        pass
            return bytes.fromhex(key_hex)
        if kind == "err":
            print(f"[-] {msg[1]}", flush=True)
        if kind == "miss":
            print(f"[*] {msg[1]}", flush=True)

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
