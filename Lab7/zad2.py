import scipy.linalg as linalg
import numpy as np

def inverse_power_method(A, sigma, eps, max_iter):
    n = A.shape[0]
    I = np.eye(n)
    B = A - sigma * I
    
    P, L, U = linalg.lu(B)
    
    x = np.random.rand(n)
    x /= np.linalg.norm(x)
    
    for i in range(max_iter):
        y = linalg.solve(U, linalg.solve(L, P @ x))
        
        x_new = y / np.linalg.norm(y)
        
        if np.linalg.norm(x_new - x) < eps: break
        x = x_new
    
    lambda_approx = x.T @ A @ x / (x.T @ x)

    return lambda_approx, x

A = np.array([
    [4, 1, 3],
    [2, 3, 2],
    [9, 3, 4]
])

sigma = 6.1
lambda_approx, eigenvector = inverse_power_method(A, sigma, 1e-6, 1000000)

print(f"Approximated eigenvalue: {lambda_approx}")
print(f"Corresponding eigenvector: {eigenvector}")

