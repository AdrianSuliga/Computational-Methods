from random import randint
import numpy as np

def get_matrix(n: int) -> np.array:
    M = np.zeros(shape = (n, n))

    for i in range(n):
        for j in range(n):
            M[i, j] = randint(0, 100)
    
    return M

def gram_shmidt(M: np.array) -> tuple[np.array, np.array]:
    n = M.shape[0]
    Q = []
    M = M.T # żeby móc odnosić się do kolumn jak do wierszy

    Q.append(M[0] / np.linalg.norm(M[0]))

    for k in range(1, n):
        weird_sum = np.dot(Q[0], M[k]) * Q[0]
        for i in range(1, k):
            weird_sum += np.dot(Q[i], M[k]) * Q[i]
        Q.append(M[k] - weird_sum)
        Q[k] /= np.linalg.norm(Q[k])

    Q = np.array(Q)

    R = np.zeros(shape = (n, n))
    for i in range(n):
        for j in range(i, n):
            R[i, j] = np.dot(Q[i], M[j])

    Q = Q.T
    M = M.T

    # Wyniki różnią się znakiem od np.linalg.qr
    return Q, R

def test_implementation(M: np.array) -> None:
    Q, R = gram_shmidt(M)
    Q_correct, R_correct = np.linalg.qr(M)
    n = M.shape[0]

    max_q_difference = -float('inf')
    max_r_difference = -float('inf')

    for i in range(n):
        for j in range(n):
            max_q_difference = max(max_q_difference, abs(Q[i, j]) - abs(Q_correct[i, j])) # biorę abs bo moja implementacja zwraca z przeciwnym znakiem
            max_r_difference = max(max_r_difference, abs(R[i, j]) - abs(R_correct[i, j])) 

    print("Max difference in Q", max_q_difference, "for matrix of size", n)
    print("Max difference in R", max_r_difference, "for matrix of size", n)

test_implementation(get_matrix(5))
test_implementation(get_matrix(10))
test_implementation(get_matrix(15))
test_implementation(get_matrix(20))
test_implementation(get_matrix(100))