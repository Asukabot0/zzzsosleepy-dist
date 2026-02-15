#!/usr/bin/env python3
import subprocess
import time

while True:
    print("[Wrapper] Starting solve_final.py...")
    try:
        # Check if we should use a specific python path
        proc = subprocess.run(
            ["/opt/homebrew/Caskroom/miniconda/base/bin/python3", "solve_final.py"],
            capture_output=False
        )
        if proc.returncode == 0:
            print("[Wrapper] Success! Exiting.")
            break
        elif proc.returncode == 139:
            print("[Wrapper] Segmentation fault (139). Retrying...")
        else:
            print(f"[Wrapper] Exited with code {proc.returncode}. Retrying...")
    except Exception as e:
        print(f"[Wrapper] Error: {e}")
    
    time.sleep(1)
