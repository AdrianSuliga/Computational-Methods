import numpy as np
import matplotlib.pyplot as plt
import time

N = 10**7
v = np.float32(0.53125)

def zad1_1():
    tab = [v for _ in range(N)]
    result = np.float32(0.0)

    for i in range(N):
        result += tab[i]

    return result

def zad1_2():
    result = zad1_1()

    print("Błąd bezwzględny: " + str(abs(N * v - result)))
    print("Błąd względny: " + str(abs((N * v - result) / (N * v))))

def zad1_3():
    xs = []
    ys = []
    tab = [v for _ in range(N)]
    result = np.float32(0.0)
    raport_step = 25000

    for i in range(N):
        result += tab[i]
        if i % raport_step == 0:
            xs.append(i)
            ys.append(abs(((i + 1) * v - result) / ((i + 1) * v)))

    xpoints = np.array(xs)
    ypoints = np.array(ys)

    plt.plot(xpoints, ypoints)
    plt.show()

def zad1_4(val = v):
    tab = [val for _ in range(N)]

    def rec(tab, left, right):
        if left == right:
            return tab[left]
        
        pivot = (left + right) // 2
        return rec(tab, left, pivot) + rec(tab, pivot + 1, right)

    return rec(tab, 0, N - 1)

def zad1_5(val = v):
    result = zad1_4(val)

    # Errors are significantly smaller because algorithm sums
    # numbers that are similar in size. It reduces error that 
    # arises during exponent shifts.
    print("Błąd bezwzględny: " + str(abs(N * val - result)))
    print("Błąd względny: " + str(abs((N * val - result) / (N * val))))

def zad1_6():
    print("Measuring time...")
    start = time.time()
    zad1_1()
    end = time.time()
    print("Normal algorithm time: " + str(end - start))

    start = time.time()
    zad1_4()
    end = time.time()
    print("Recursive algorithm time: " + str(end - start))

def zad1_7():
    new_v = np.float32(0.31244121241124122)
    print(zad1_4(new_v))
    zad1_5(new_v)

#print(zad1_1())
#zad1_2()
#zad1_3()
#print(zad1_4())
#zad1_5()
#zad1_6()
zad1_7()