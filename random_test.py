import random
import os

seed = b'testseed'
rng = random.Random(seed)
v1 = rng.getrandbits(32)
v2 = rng.getrandbits(32)
print(f"{v1} {v2}")
