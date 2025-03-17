import numpy as np
from random import random
import matplotlib.pyplot as plt
from time import time

def zad1(A: np.array):
    n = len(A)
    for i in range(n):
        index_to_swap = i
        for j in range(i + 1, n):
            if abs(A[j][i]) > abs(A[index_to_swap][i]):
                index_to_swap = j

        A[i], A[index_to_swap] = A[index_to_swap], A[i]

        if abs(A[i][i]) < 1e-12:
            return np.full(n, None)

        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            
            for k in range(i, n + 1):
                A[j][k] -= factor * A[i][k]

    res_vec = np.zeros(n)
    
    for i in range(n - 1, -1, -1):
        for j in range(n - 1, i, -1):
            A[i][n] -= A[i][j] * res_vec[j]
        res_vec[i] = A[i][n] / A[i][i]

    return res_vec

xs = []
y_my_implementation = []
y_library = []

for n in range(500, 1001, 50):
    A = np.array([[random() for _ in range(n + 1)] for _ in range(n)])
    B = np.array([[A[i][j] for j in range(n)] for i in range(n)])
    X = np.array([A[i][n] for i in range(n)])

    A = A.astype(np.float64)
    B = B.astype(np.float64)
    X = X.astype(np.float64)

    start = time()
    zad1(A)
    end = time()
    y_my_implementation.append(end - start)

    start = time()
    np.linalg.solve(B, X)
    end = time()
    y_library.append(end - start)
    
    xs.append(n)

fig, axs = plt.subplots(2)

axs[0].plot(xs, y_my_implementation)
axs[0].set_title("Czas mojej implementacji")
axs[0].set_xlabel("Rozmiar macierzy")
axs[0].set_ylabel("Czas wykonania")

axs[1].plot(xs, y_library)
axs[1].set_title("Czas implementacji z biblioteki")
axs[1].set_xlabel("Rozmiar macierzy")
axs[1].set_ylabel("Czas wykonania")

plt.tight_layout()
plt.show()