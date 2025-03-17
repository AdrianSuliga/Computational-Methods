import numpy as np
from random import randint
import matplotlib.pyplot as plt
from time import time

def zad1(A: np.array):
    n = len(A)
    for i in range(n):
        for j in range(i + 1, n):
            # If there is 0 on a diagonal
            # than swap the rows and continue
            if A[i][i] < 10 ** -10:
                A[[i, j]] = A[[j, i]]
                continue

            factor = A[j][i] / A[i][i]
            
            for k in range(i, n + 1):
                A[j][k] -= factor * A[i][k]
    
    for i in range(n):
        if A[i][i] < 10 ** -10:
            return np.full(n, None)

    res_vec = np.zeros(n)
    
    for i in range(n - 1, -1, -1):
        for j in range(n - 1, i, -1):
            A[i][n] -= A[i][j] * res_vec[j]
        res_vec[i] = A[i][n] / A[i][i]

    return res_vec

xs = []
ys = []

for n in range(500, 1001):
    A = np.array([[randint(0, 100) for _ in range(n + 1)] for _ in range(n)])
    A = A.astype(np.float64)
    start = time()
    zad1(A)
    end = time()

    xs.append(n)
    ys.append(end - start)

xp = np.array(xs)
yp = np.array(ys)

plt.plot(xp, yp)
plt.show()