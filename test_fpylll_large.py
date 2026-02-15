from fpylll import IntegerMatrix, LLL
import sys
import random

print(sys.version)
try:
    print("Creating 201x201 matrix...")
    A = IntegerMatrix(201, 201)
    for i in range(201):
        for j in range(201):
            A[i, j] = random.randint(0, 2**32)
    print("Matrix created.")
    LLL.reduction(A, delta=0.99)
    print("LLL done.")
except Exception as e:
    print(e)
