from copy import deepcopy
import numpy as np
import time
import matplotlib.pyplot as plt

def power_method(matrix, max_iter, eps):
    x = np.random.rand(matrix.shape[1])

    for _ in range(max_iter):
        previous = deepcopy(x)
        x = matrix @ x
        x /= np.linalg.norm(x)
        if abs(np.linalg.norm(x - previous)) < eps: break

    return x, max(x, key = lambda xi: abs(xi))

def gen_matrix(n):
    return np.random.rand(n, n)

def test():
    sizes = np.linspace(50,10000,200,dtype=np.int64)
    times = []
    
    for size in sizes:
        matrix = gen_matrix(size)
        start_time = time.time()
        power_method(matrix, 10000, 1e-15)
        end_time = time.time()
        times.append(end_time - start_time)

    plt.plot(sizes, times)
    plt.show()

test()