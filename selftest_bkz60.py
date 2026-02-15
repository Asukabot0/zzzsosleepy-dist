#!/usr/bin/env python3
"""
Local self-test for the BKZ-60 solver (no network):
- Generate random seed/key
- Generate outputs exactly like chall.py
- Run the BKZ-based solver to recover the key

Usage examples:
  python3 -u selftest_bkz60.py
  JOBS=16 RESTARTS=200 FRAMES=110 python3 -u selftest_bkz60.py
  SVP_TIMEOUT=... (not used here)
"""

import os
import random
from typing import List

import solve_bkz60


def gen_outs(seed: bytes, key: bytes) -> List[int]:
    rng = random.Random(seed)
    outs: List[int] = []
    for _ in range(solve_bkz60.NUM_FRAMES):
        row = [rng.getrandbits(32) for _ in range(solve_bkz60.FRAME_LEN)]
        acc = 0
        for x, y in zip(row, key):
            acc = (acc + (x * y)) & solve_bkz60.MASK
        outs.append(solve_bkz60.adc_read(acc))
    return outs


def main() -> None:
    n = int(os.environ.get("SELFTEST_N", "3"))
    for i in range(n):
        seed = os.urandom(8)
        key = os.urandom(solve_bkz60.FRAME_LEN)
        outs = gen_outs(seed, key)
        print(f"[*] case {i+1}/{n} seed={seed.hex()}", flush=True)
        got = solve_bkz60.solve(seed, outs)
        if got != key:
            raise RuntimeError(f"FAIL: got={got.hex()} expected={key.hex()}")
        print("[+] ok", flush=True)


if __name__ == "__main__":
    main()

