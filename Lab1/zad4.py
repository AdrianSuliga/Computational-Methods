import numpy as np
import matplotlib.pyplot as plt

def recursive(r:np.float32, x0:np.float32, N:int):
    for _ in range(N):
        x0 = r * x0 * (1 - x0)

    return x0

def recursive64(r:np.float64, x0:np.float64, N:int):
    for _ in range(N):
        x0 = r * x0 * (1 - x0)
    
    return x0

def zad4a(x0:np.float32, N:int, rs):
    ys = []

    for r in rs:
        x0 = np.random.random()
        ys.append(recursive64(r, x0, N))

    return ys

rs = np.linspace(0, 4, 100000)
N = 1000

plt.plot(rs, zad4a(0.3, N, rs), ls='', marker=',')
#plt.plot(rs, zad4a(0.5, N, rs))
#plt.plot(rs, zad4a(0.7, N, rs))
plt.show()

def zad4b32(x32:np.float32, N:int, rs):
    y32 = []

    for r in rs:
        y32.append(recursive(r, x32, N))

    return y32

def zad4b64(x64:np.float64, N:int, rs):
    y64 = []

    for r in rs:
        y64.append(recursive64(r, x64, N))

    return y64

#rs = np.linspace(3.75, 3.8, 10000)
#N = 1000

#plt.plot(rs, zad4b64(np.float64(0.3), N, rs))
#plt.plot(rs, zad4b64(np.float64(0.5), N, rs))
#plt.plot(rs, zad4b64(np.float64(0.7), N, rs))

#plt.plot(rs, zad4b32(np.float32(0.3), N, rs))
#plt.plot(rs, zad4b32(np.float32(0.5), N, rs))
#plt.plot(rs, zad4b32(np.float32(0.7), N, rs))

#plt.show()

def zad4c(xs, N:int):
    r = np.float32(4.0)
    ys = []    

    for x in xs:
        x0, cnt = x, 0
        for i in range(N):
            x0 = r * x0 * (1 - x0)
            cnt += 1
            if x0 < 10 ** (-6):
                break
        ys.append(cnt)

    return ys

#N = 10 ** 5
#xs = np.linspace(np.float32(0.0), np.float32(1.0), 100)
#plt.plot(xs, zad4c(xs, N))
#plt.show()