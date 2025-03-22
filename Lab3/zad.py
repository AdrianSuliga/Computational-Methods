import mpmath

def f1(x): return mpmath.cos(x) * mpmath.cosh(x) - 1
def df1(x): return mpmath.cos(x) * mpmath.sinh(x) - mpmath.cosh(x) * mpmath.sin(x)

def f2(x): return mpmath.mpf(1/x) - mpmath.tan(x)
def df2(x): return -(mpmath.sec(x) ** 2) - mpmath.mpf(1 / (x ** 2))

def f3(x): return 2 ** (-x) + mpmath.exp(x) + 2 * mpmath.cos(x) - 6
def df3(x): return -2 * mpmath.sin(x) + mpmath.exp(x) - mpmath.ln(2) / mpmath.mpf(2 ** x)

def zad1_bisection(f, a, b, precision, error):
    c, cnt = mpmath.mpf((a + b) / 2), 1
    error = mpmath.mpf(10 ** (-error))
    mpmath.mp.dps = precision

    while abs(f(c)) >= error:
        if mpmath.sign(f(c)) == mpmath.sign(f(a)): a = c
        else: b = c
        c = mpmath.mpf(a + (b - a) / 2) 
        cnt += 1

    return (c, cnt)

def zad2_newton(f, df, x0, precision, error, max_iter):
    error = mpmath.mpf(10 ** (-error))
    mpmath.mp.dps = precision

    for i in range(max_iter):
        x1 = x0 - f(x0) / df(x0)
        if abs(x1 - x0) < error: return (i, x1)
        x0 = x1
    
    return (max_iter, None)

def zad2_secants(f, x0, x1, precision, error, max_iter):
    error = mpmath.mpf(10 ** (-error))
    mpmath.mp.dps = precision

    for i in range(max_iter):
        pass

print(zad2_newton(f1, df1, 13, 30, 30, 1000))

print(zad2_newton(f2, df2, 3, 30, 30, 1000))
print(zad2_newton(f3, df3, 1, 30, 30, 1000))