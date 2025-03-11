import numpy as np
import time

v = np.float32(0.53125)
N = 10 ** 7

def kahan_algorithm(tab):
    sum = np.float32(0.0)
    err = np.float32(0.0)
    for i in  range(N):
        y = np.float32(tab[i] - err)
        temp = sum + y
        err = (temp - sum) - y
        sum = temp
    return sum

def rec_algorithm(tab):
    def rec(tab, left, right):
        if left == right:
            return tab[left]
        
        pivot = (left + right) // 2
        return rec(tab, left, pivot) + rec(tab, pivot + 1, right)

    return rec(tab, 0, N - 1)

def zad2_1():
    tab = [v for _ in range(N)]
    result = kahan_algorithm(tab)
    print("Wynik: " + str(result))
    print("Błąd bezwzględny: " + str(abs(N * v - result)))
    print("Błąd względny: " + str(abs(N * v - result) / (N * v)))

def zad2_2():
    # err variable allows us to keep track of error we made
    # while summing previous numbers. Algorithm itself is also
    # O(n) so pretty cool.
    pass

def zad2_3():
    tab = [v for _ in range(N)]

    print("Measuring time...")

    start = time.time()
    kahan_algorithm(tab)
    end = time.time()
    print("Kahan algorithm time: " + str(end - start))

    start = time.time()
    rec_algorithm(tab)
    end = time.time()
    print("Recursive algorithm time: " + str(end - start))

zad2_1()
zad2_3()