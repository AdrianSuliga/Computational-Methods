import numpy as np
from random import randint
from time import time

def my_fft(n, vector):
    def ksi(i, k):
        return np.exp(-2j * np.pi * i * k / n)

    Fn = [[ksi(i, k) for k in range(n)] for i in range(n)]
    return np.dot(Fn, vector)

def my_fft2(n, vector):
    if n == 1:
        return [vector[0]]

    X_even = my_fft2(n // 2, vector[::2])
    X_odd = my_fft2(n // 2, vector[1::2])

    X = [0] * n
    for k in range(n // 2):
        twiddle = np.exp(-2j * np.pi * k / n) * X_odd[k]
        X[k] = X_even[k] + twiddle
        X[k + n // 2] = X_even[k] - twiddle

    return X

n = 1024
vector = [randint(1, 20) for _ in range(n)]

start = time()
result_dft = my_fft(n, vector)
end = time()
dft_time = end - start

start = time()
result_rec = my_fft2(n, vector)
end = time()
rec_time = end - start

start = time()
result_lib = np.fft.fft(vector)
end = time()
lib_time = end - start

print("Input vector:", vector)

print("CUSTOM DFT TIME:", dft_time)
#print("Result:", result_dft)

print("RECURSIVE FFT TIME:", rec_time)
#print("Result:", result_rec)

print("NUMPY FFT TIME:", lib_time)
#print("Result:", result_lib)
