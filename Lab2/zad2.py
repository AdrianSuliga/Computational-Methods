import numpy as np
from random import random

def zad2(A: np.array):
    n = len(A)
    for i in range(n):
        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            for k in range(i, n):
                A[j][k] -= factor * A[i][k]
            A[j][i] = factor

def testLU(A: np.array):
    A = A.astype(np.float64)
    B = A.copy()
    n = len(B)

    zad2(B)
    L = np.array([[B[i][j] if i > j else 0 for j in range(n)] for i in range(n)])
    U = np.array([[B[i][j] if i <= j else 0 for j in range(n)] for i in range(n)])
    for i in range(n): L[i][i] = 1

    I = np.subtract(A, np.matmul(L, U))
    print((abs(I) < 1e-10).all(), end = ' ')

for n in range(10, 201, 10):
    A = np.array([[random() for _ in range(n)] for _ in range(n)])
    A = A.astype(np.float64)
    testLU(A)
print()