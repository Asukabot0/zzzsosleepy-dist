#!/usr/bin/env python3
import base64
import os
import random
import queue
import signal
import struct
import time
import multiprocessing as mp
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
    """
    Multi-core strategy:
    - Each LLL/SVP call is effectively single-threaded in most fpylll builds.
    - We parallelize across parameter sets (frames/weights) with multiprocessing.
    - First worker to find a key wins; others are terminated.
    """

    # Candidate parameter sets. Keep this small-ish: each worker builds matrices and a lattice.
    frames_list = [70, 80, 90, 100, 110, 120]
    W_list = [1 << 13, 1 << 14, 1 << 15]
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

    attempts: List[Tuple[int, int]] = []
    for frames in frames_list:
        for W in W_list:
            attempts.append((frames, W))

    # Prioritize smaller dimensions first.
    attempts.sort(key=lambda x: (x[0], x[1]))
    total = len(attempts)

    # IPC
    result_q: mp.Queue = mp.Queue()
    stop_evt = mp.Event()

    def worker(idx: int, frames: int, W: int) -> None:
        try:
            # Allow parent to kill us cleanly.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        except Exception:
            pass

        outs = outs_full[:frames]
        A_unsigned = A_unsigned_full[:frames]
        A_signed = A_signed_full[:frames]

        dim = FRAME_LEN + frames + 1
        prefix = f"[w{idx:02d} frames={frames} W={W} dim={dim}]"
        print(f"{prefix} start", flush=True)

        try:
            B = build_B(A_signed, outs)
            M = build_basis(A_signed, B, W=W, WT=WT)
        except Exception as e:
            result_q.put(("err", idx, frames, W, f"build: {e}"))
            return

        try:
            t_lll = time.time()
            LLL.reduction(M, delta=0.99)
            print(f"{prefix} LLL {time.time() - t_lll:.2f}s", flush=True)
        except Exception as e:
            result_q.put(("err", idx, frames, W, f"LLL: {e}"))
            return

        # Scan basis (cheap).
        for i in range(M.nrows):
            vec = [int(M[i, j]) for j in range(M.nrows)]
            key = try_extract_key(vec, A_unsigned, outs, W=W, WT=WT)
            if key is None:
                continue
            if verify_all(seed, outs_full, key):
                result_q.put(("ok", idx, frames, W, key.hex()))
                return

        if stop_evt.is_set():
            result_q.put(("skip", idx, frames, W, "stopped"))
            return

        # SVP call: no incremental progress output inside fpylll, so we time-cap the attempt
        # by letting the parent terminate the process if it runs too long.
        print(f"{prefix} SVP start (no internal progress)", flush=True)
        try:
            v = SVP.shortest_vector(M, method="fast", pruning=False, preprocess=False)
        except Exception as e:
            result_q.put(("err", idx, frames, W, f"SVP: {e}"))
            return

        key = try_extract_key([int(x) for x in v], A_unsigned, outs, W=W, WT=WT)
        if key is not None and verify_all(seed, outs_full, key):
            result_q.put(("ok", idx, frames, W, key.hex()))
            return

        result_q.put(("miss", idx, frames, W, "no key"))

    procs: List[Tuple[mp.Process, float, int, int, int]] = []
    next_i = 0
    running = 0
    done = 0
    last_heartbeat = 0.0

    def start_one() -> None:
        nonlocal next_i, running
        if next_i >= total:
            return
        frames, W = attempts[next_i]
        idx = next_i
        p = mp.Process(target=worker, args=(idx, frames, W), daemon=True)
        p.start()
        procs.append((p, time.time(), idx, frames, W))
        running += 1
        next_i += 1

    # Fill initial slots.
    while running < jobs and next_i < total:
        start_one()

    # Main loop: monitor results + enforce per-worker timeout.
    while done < total:
        now = time.time()
        if now - last_heartbeat >= 5:
            print(f"[*] progress: done={done}/{total} running={running} queued={total - next_i}", flush=True)
            last_heartbeat = now

        # Enforce SVP timeout by killing long-running workers.
        for i in range(len(procs)):
            p, start_ts, idx, frames, W = procs[i]
            if not p.is_alive():
                continue
            if now - start_ts > svp_timeout_s:
                print(f"[-] timeout: idx={idx} frames={frames} W={W} killing worker", flush=True)
                try:
                    p.terminate()
                except Exception:
                    pass

        # Drain results if available.
        try:
            msg = result_q.get(timeout=0.5)
        except queue.Empty:
            # Reap dead workers and start new ones.
            alive: List[Tuple[mp.Process, float, int, int, int]] = []
            for p, start_ts, idx, frames, W in procs:
                if p.is_alive():
                    alive.append((p, start_ts, idx, frames, W))
                else:
                    running -= 1
                    done += 1
            procs = alive
            while running < jobs and next_i < total and not stop_evt.is_set():
                start_one()
            if running == 0 and next_i >= total:
                break
            continue

        kind, idx, frames, W, payload = msg
        # One worker finished (or errored). Mark it done by reaping on next iteration.
        if kind == "ok":
            print(f"[+] FOUND key by idx={idx} frames={frames} W={W}", flush=True)
            key_hex = payload
            stop_evt.set()
            # Kill all workers.
            for p, _, _, _, _ in procs:
                if p.is_alive():
                    try:
                        p.terminate()
                    except Exception:
                        pass
            return bytes.fromhex(key_hex)
        elif kind in ("err", "miss", "skip"):
            print(f"[*] worker idx={idx} frames={frames} W={W} -> {kind}: {payload}", flush=True)

        # Reap + refill quickly.
        alive = []
        for p, start_ts, idx2, frames2, W2 in procs:
            if p.is_alive():
                alive.append((p, start_ts, idx2, frames2, W2))
            else:
                running -= 1
                done += 1
        procs = alive
        while running < jobs and next_i < total and not stop_evt.is_set():
            start_one()

    raise RuntimeError("key not found (all attempts exhausted)")


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
