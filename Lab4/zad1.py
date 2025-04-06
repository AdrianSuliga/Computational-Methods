from random import randint, random
from math import exp
import matplotlib.pyplot as plt 
from PIL import Image
import io

def distance(p1, p2): 
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def manhatan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def path_length(points):
    size = len(points)
    sum = 0

    for i in range(size - 1):
        sum += distance(points[i], points[i + 1])
    
    sum += distance(points[0], points[size - 1])

    return sum

def schedule_temp(x, T0):
    return T0 * exp(-x / T0)

def schedule_prob(E, T):
    return exp(E / T)

def schedule_neighbour(state: list):
    result = state[:]
    size = len(state)
    i = j = 0

    while i == j:
        i, j = randint(0, size - 1), randint(0, size - 1)

    result[i], result[j] = result[j], result[i]
    return result

def gif_maker():
    pass

def simulated_annealing(points, max_iter, initial_temp):
    xs, ys = [], []
    for i in range(1, max_iter + 1):
        T = schedule_temp(i, initial_temp)
        if T < 1e-12: break
        candidate = schedule_neighbour(points)
        E = path_length(points) - path_length(candidate)
        if E > 0:
            points = candidate[:]
        else:
            prob = schedule_prob(E, T)
            if random() < prob:
                points = candidate[:]
        xs.append(i)
        ys.append(path_length(points))

        print(f"Done in {i * 100 / max_iter}%")

    plt.plot(xs, ys)
    plt.title("Długość znalezionej ścieżki")
    plt.tight_layout()
    plt.show()

    return path_length(points)

n = 20
points = [(randint(0, 100), randint(0, 100)) for _ in range(n)]

print(simulated_annealing(points, 10000, 1000))
