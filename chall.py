#!/usr/bin/env python3
import os
import struct
import random
import sys
import base64

FRAME_LEN = 80     
NUM_FRAMES = 120   
ADC_BITS = 10      


DIM = FRAME_LEN
CNT = NUM_FRAMES
BITS = ADC_BITS
MASK = 0xFFFFFFFF

def _read_flag() -> str:
    flag = os.environ.get("FLAG") or os.environ.get("GZCTF_FLAG")
    if flag:
        return flag.strip()
    try:
        with open("flag.txt", "r") as f:
            return f.read().strip()
    except Exception:
        return "C2C{test_flag_local}"

class KeyGen:
    def __init__(self):
        self.key = os.urandom(FRAME_LEN)
        
    def get(self):
        return [b for b in self.key]

class Stream:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        
    def next(self):
        return [self.rng.getrandbits(32) for _ in range(FRAME_LEN)]

def _u32(x: int) -> int:
    return x & MASK

def _adc_read(acc_u32: int) -> int:
    return acc_u32 >> (32 - ADC_BITS)

def main():
    seed = os.urandom(8)
    st = Stream(seed)
    kg = KeyGen()
    k = kg.get()
    
    buf = bytearray()
    buf.extend(seed)
    
    for _ in range(CNT):
        v = st.next()
        acc = 0 
        for x, y in zip(v, k):
            acc = _u32(acc + x * y)
            
        out = _adc_read(acc)
        buf.extend(struct.pack('<H', out))
        
    print("--- BEGIN SIGNAL ---")
    print(base64.b64encode(buf).decode())
    print("--- END SIGNAL ---")
    
    sys.stdout.flush()
    try:
        line = sys.stdin.readline()
        if not line: return
        guess_hex = line.strip()
        if len(guess_hex) != DIM * 2:
            return
        guess = bytes.fromhex(guess_hex)
        
        if guess == kg.key:
            print("Access Granted")
            print(_read_flag())
        else:
            print("Access Denied")
    except:
        pass

if __name__ == "__main__":
    main()
