from fpylll import IntegerMatrix, LLL
try:
    A = IntegerMatrix(10, 10)
    for i in range(10):
        A[i, i] = 1
        for j in range(i):
            A[i, j] = 2
    print("Matrix created")
    LLL.reduction(A)
    print("LLL done")
except Exception as e:
    print(e)
