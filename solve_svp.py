#!/usr/bin/env python3
import base64
import os
import random
import queue
import signal
import struct
import time
import multiprocessing as mp
from typing import List, Tuple, Optional, Dict

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


def _worker_entry(
    result_q,
    stop_evt,
    seed: bytes,
    outs_full: List[int],
    A_unsigned_full: List[List[int]],
    A_signed_full: List[List[int]],
    WT: int,
    idx: int,
    frames: int,
    W: int,
    r: int,
) -> None:
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass

    outs = outs_full[:frames]
    A_unsigned = A_unsigned_full[:frames]
    A_signed = A_signed_full[:frames]

    dim = FRAME_LEN + frames + 1
    prefix = f"[w{idx:04d} frames={frames} W={W} r={r} dim={dim}]"
    print(f"{prefix} start", flush=True)
    result_q.put(("stat", idx, frames, W, r, "start"))

    try:
        B = build_B(A_signed, outs)
        M = build_basis(A_signed, B, W=W, WT=WT)
    except Exception as e:
        result_q.put(("err", idx, frames, W, r, f"build: {e}"))
        return

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

    try:
        result_q.put(("stat", idx, frames, W, r, "lll"))
        t_lll = time.time()
        LLL.reduction(M, delta=0.99)
        print(f"{prefix} LLL {time.time() - t_lll:.2f}s", flush=True)
    except Exception as e:
        result_q.put(("err", idx, frames, W, r, f"LLL: {e}"))
        return

    result_q.put(("stat", idx, frames, W, r, "scan"))
    for i in range(M.nrows):
        vec = [int(M[i, j]) for j in range(M.nrows)]
        key = try_extract_key(vec, A_unsigned, outs, W=W, WT=WT)
        if key is None:
            continue
        if verify_all(seed, outs_full, key):
            result_q.put(("ok", idx, frames, W, r, key.hex()))
            return

    if stop_evt.is_set():
        result_q.put(("skip", idx, frames, W, r, "stopped"))
        return

    print(f"{prefix} SVP start (no internal progress)", flush=True)
    result_q.put(("stat", idx, frames, W, r, "svp"))
    try:
        v = SVP.shortest_vector(M, method="fast", pruning=False, preprocess=False)
    except Exception as e:
        result_q.put(("err", idx, frames, W, r, f"SVP: {e}"))
        return

    key = try_extract_key([int(x) for x in v], A_unsigned, outs, W=W, WT=WT)
    if key is not None and verify_all(seed, outs_full, key):
        result_q.put(("ok", idx, frames, W, r, key.hex()))
        return

    result_q.put(("miss", idx, frames, W, r, "no key"))


def solve(seed: bytes, outs_full: List[int]) -> bytes:
    """
    Multi-core strategy:
    - Each LLL/SVP call is effectively single-threaded in most fpylll builds.
    - We parallelize across parameter sets (frames/weights) with multiprocessing.
    - First worker to find a key wins; others are terminated.
    """

    # Candidate parameter sets.
    # Expand aggressively so large machines can stay busy:
    # - frames sweep: [FRAMES_MIN, FRAMES_MAX] step FRAMES_STEP
    # - W sweep: [W_MIN_POW, W_MAX_POW] (inclusive) as powers of two
    # - restarts: repeat each (frames, W) with randomized unimodular perturbations
    frames_min = int(os.environ.get("FRAMES_MIN", "70"))
    frames_max = int(os.environ.get("FRAMES_MAX", "120"))
    frames_step = int(os.environ.get("FRAMES_STEP", "5"))
    if frames_step <= 0:
        raise ValueError("FRAMES_STEP must be > 0")

    w_min_pow = int(os.environ.get("W_MIN_POW", "10"))
    w_max_pow = int(os.environ.get("W_MAX_POW", "18"))
    if w_min_pow > w_max_pow:
        raise ValueError("W_MIN_POW must be <= W_MAX_POW")

    restarts = int(os.environ.get("RESTARTS", "30"))
    restarts = max(1, restarts)

    frames_list = list(range(frames_min, frames_max + 1, frames_step))
    W_list = [1 << p for p in range(w_min_pow, w_max_pow + 1)]
    WT = 1 << 21

    # Control concurrency. 384 threads does NOT mean 384 safe workers: memory spikes.
    jobs = int(os.environ.get("JOBS", "0") or "0")
    if jobs <= 0:
        jobs = min(32, (os.cpu_count() or 8))
    jobs = max(1, jobs)

    svp_timeout_s = int(os.environ.get("SVP_TIMEOUT", "600"))  # per-attempt cap (seconds)

    t0 = time.time()
    A_unsigned_full, A_signed_full = gen_matrix(seed)
    print(f"[*] Matrix generated in {time.time() - t0:.2f}s", flush=True)
    print(f"[*] jobs={jobs} svp_timeout={svp_timeout_s}s", flush=True)

    attempts: List[Tuple[int, int, int]] = []
    for frames in frames_list:
        for W in W_list:
            for r in range(restarts):
                attempts.append((frames, W, r))

    # Prioritize smaller dimensions first.
    attempts.sort(key=lambda x: (x[0], x[1], x[2]))
    total = len(attempts)

    # IPC
    # result_q carries both results and status messages.
    result_q: mp.Queue = mp.Queue()
    stop_evt = mp.Event()

    # Worker is defined at module level for spawn-based multiprocessing (macOS/Py3.13).

    procs: List[Tuple[mp.Process, float, int, int, int, int]] = []
    next_i = 0
    running = 0
    done = 0
    last_heartbeat = 0.0
    # Track latest stage per worker id for richer progress output.
    stages: Dict[int, str] = {}
    counts: Dict[str, int] = {"start": 0, "lll": 0, "scan": 0, "svp": 0}

    def start_one() -> None:
        nonlocal next_i, running
        if next_i >= total:
            return
        frames, W, r = attempts[next_i]
        idx = next_i
        p = mp.Process(
            target=_worker_entry,
            args=(
                result_q,
                stop_evt,
                seed,
                outs_full,
                A_unsigned_full,
                A_signed_full,
                WT,
                idx,
                frames,
                W,
                r,
            ),
            daemon=True,
        )
        p.start()
        procs.append((p, time.time(), idx, frames, W, r))
        running += 1
        next_i += 1

    # Fill initial slots.
    while running < jobs and next_i < total:
        start_one()

    # Main loop: monitor results + enforce per-worker timeout.
    while done < total:
        now = time.time()
        if now - last_heartbeat >= 5:
            stage_str = " ".join(f"{k}={counts.get(k,0)}" for k in ("lll", "scan", "svp"))
            print(
                f"[*] progress: done={done}/{total} running={running} queued={total - next_i} {stage_str}",
                flush=True,
            )
            last_heartbeat = now

        # Enforce SVP timeout by killing long-running workers.
        for i in range(len(procs)):
            p, start_ts, idx, frames, W, r = procs[i]
            if not p.is_alive():
                continue
            if now - start_ts > svp_timeout_s:
                print(f"[-] timeout: idx={idx} frames={frames} W={W} r={r} killing worker", flush=True)
                try:
                    p.terminate()
                except Exception:
                    pass

        # Drain results if available.
        try:
            msg = result_q.get(timeout=0.5)
        except queue.Empty:
            # Reap dead workers and start new ones.
            alive: List[Tuple[mp.Process, float, int, int, int, int]] = []
            for p, start_ts, idx, frames, W, r in procs:
                if p.is_alive():
                    alive.append((p, start_ts, idx, frames, W, r))
                else:
                    running -= 1
                    done += 1
            procs = alive
            while running < jobs and next_i < total and not stop_evt.is_set():
                start_one()
            if running == 0 and next_i >= total:
                break
            continue

        kind = msg[0]
        if kind == "stat":
            _, idx, frames, W, r, stage = msg
            prev = stages.get(idx)
            if prev is not None:
                counts[prev] = max(0, counts.get(prev, 0) - 1)
            stages[idx] = stage
            counts[stage] = counts.get(stage, 0) + 1
            continue

        _, idx, frames, W, r, payload = msg
        # One worker finished (or errored). Mark it done by reaping on next iteration.
        if kind == "ok":
            print(f"[+] FOUND key by idx={idx} frames={frames} W={W}", flush=True)
            key_hex = payload
            stop_evt.set()
            # Kill all workers.
            for p, _, _, _, _, _ in procs:
                if p.is_alive():
                    try:
                        p.terminate()
                    except Exception:
                        pass
            return bytes.fromhex(key_hex)
        elif kind in ("err", "miss", "skip"):
            print(f"[*] worker idx={idx} frames={frames} W={W} r={r} -> {kind}: {payload}", flush=True)

        # Reap + refill quickly.
        alive = []
        for p, start_ts, idx2, frames2, W2, r2 in procs:
            if p.is_alive():
                alive.append((p, start_ts, idx2, frames2, W2, r2))
            else:
                # Adjust stage counters for exited workers.
                st = stages.pop(idx2, None)
                if st is not None:
                    counts[st] = max(0, counts.get(st, 0) - 1)
                running -= 1
                done += 1
        procs = alive
        while running < jobs and next_i < total and not stop_evt.is_set():
            start_one()

    raise RuntimeError("key not found (all attempts exhausted)")


def selftest() -> None:
    """
    Local correctness check without network:
    - Generate random seed/key
    - Produce ADC-truncated outputs like chall.py
    - Ensure solver recovers the exact key
    """
    n = int(os.environ.get("SELFTEST_N", "3"))
    jobs = os.environ.get("JOBS", "")
    print(f"[*] SELFTEST enabled: n={n} JOBS={jobs or '(default)'}", flush=True)

    for t in range(n):
        seed = os.urandom(8)
        key = os.urandom(FRAME_LEN)

        rng = random.Random(seed)
        A = [[rng.getrandbits(32) for _ in range(FRAME_LEN)] for _ in range(NUM_FRAMES)]

        outs: List[int] = []
        for j in range(NUM_FRAMES):
            acc = 0
            row = A[j]
            for x, y in zip(row, key):
                acc = (acc + (x * y)) & MASK
            outs.append(adc_read(acc))

        print(f"[*] SELFTEST case {t+1}/{n}: seed={seed.hex()}", flush=True)
        got = solve(seed, outs)
        if got != key:
            raise RuntimeError(f"SELFTEST failed: got {got.hex()} expected {key.hex()}")

        print("[+] SELFTEST ok", flush=True)


def main() -> None:
    _patch_fplll_strategies()
    if os.environ.get("SELFTEST") == "1":
        selftest()
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
