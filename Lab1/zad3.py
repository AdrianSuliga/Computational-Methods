import numpy as np

def riemann_single_forward(s: np.float32, n:int):
    result = np.float32(0.0)
    for k in range(1, n + 1):
        result += 1 / (k ** s)
    return result

def riemann_double_forward(s: np.float64, n:int):
    result = np.float64(0.0)
    for k in range(1, n + 1):
        result += 1 / (k ** s)
    return result

def riemann_single_backward(s: np.float32, n:int):
    result = np.float32(0.0)
    for k in range(n + 1, 0, -1):
        result += 1 / (k ** s)
    return result

def riemann_double_backward(s: np.float64, n:int):
    result = np.float64(0.0)
    for k in range(n + 1, 0, -1):
        result += 1 / (k ** s)
    return result

def dirichlet_single_forward(s: np.float32, n:int):
    result = np.float32(0.0)
    for k in range(1, n + 1):
        result += ((-1) ** (k - 1)) / k ** s
    return result

def dirichlet_double_forward(s: np.float64, n:int):
    result = np.float64(0.0)
    for k in range(1, n + 1):
        result += ((-1) ** (k - 1)) / k ** s
    return result

def dirichlet_single_backward(s: np.float32, n:int):
    result = np.float32(0.0)
    for k in range(n + 1, 0, -1):
        result += ((-1) ** (k - 1)) / k ** s
    return result

def dirichlet_double_backward(s: np.float64, n:int):
    result = np.float64(0.0)
    for k in range(n + 1, 0, -1):
        result += ((-1) ** (k - 1)) / k ** s
    return result

ss = [2, 3.6667, 5, 7.2, 10]
ns = [50, 100, 200, 500, 1000]

for s in ss:
    for n in ns:
        print("\nFOR N = " + str(n) + ", S = " + str(s))
        print("=============RIEMANN=============")
        print("Single forward " + str(riemann_single_forward(np.float32(s), n)))
        print("Double forward " + str(riemann_double_forward(np.float32(s), n)))
        print("Single backward " + str(riemann_single_backward(np.float32(s), n)))
        print("Double backward " + str(riemann_double_backward(np.float32(s), n)))
        print("=============DIRICHLET=============")
        print("Single forward " + str(dirichlet_single_forward(np.float32(s), n)))
        print("Double forward " + str(dirichlet_double_forward(np.float32(s), n)))
        print("Single backward " + str(dirichlet_single_backward(np.float32(s), n)))
        print("Double backward " + str(dirichlet_double_backward(np.float32(s), n)))