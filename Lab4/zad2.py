import matplotlib.pyplot as plt
from random import random, randint
from math import exp
from time import time

def random_image(n:int, delta:float):
    M = [[1 if random() > delta else 0 for _ in range(n)] for _ in range(n)]

    plt.imshow(M, cmap = 'gray', interpolation = 'nearest')
    plt.axis('off')
    plt.show()

    return M

def point_energy_8_neighbours(M, i, j):
    result, n = 0, len(M)
    points = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]

    for x_offset, y_offset in points:
        if -1 < i + x_offset < n and -1 < j + y_offset < n:
            result += 1 if M[i][j] != M[i + x_offset][j + y_offset] else 0

    return result

def point_energy_8_neighbours_v2(M, i, j):
    result, n = 0, len(M)
    points = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]

    for x_offset, y_offset in points:
        if -1 < i + x_offset < n and -1 < j + y_offset < n:
            result += 1 if M[i][j] == M[i + x_offset][j + y_offset] else 0

    return result

def point_energy_4_neighbours_plus(M, i, j):
    result, n = 0, len(M)
    points = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    for x_offset, y_offset in points:
        if -1 < i + x_offset < n and -1 < j + y_offset < n:
            result += 1 if M[i][j] != M[i + x_offset][j + y_offset] else 0

    return result

def point_energy_4_neighbours_cross(M, i, j):
    result, n = 0, len(M)
    points = [(-1, -1), (1, 1), (-1, 1), (1, -1)]

    for x_offset, y_offset in points:
        if -1 < i + x_offset < n and -1 < j + y_offset < n:
            result += 1 if M[i][j] != M[i + x_offset][j + y_offset] else 0

    return result

def point_energy_8_neighbours_cross(M, i, j): 
    result, n = 0, len(M)

    points = [(-1, -1), (1, 1), (1, -1), (-1, 1), (-2, -2), (2, 2), (2, -2), (-2, 2)]

    for x_offset, y_offset in points:
        if -1 < i + x_offset < n and -1 < j + y_offset < n:
            result += 1 if M[i + x_offset][j + y_offset] == M[i][j] else 0

    return result

def calculate_energy(M, point_energy_function):
    sum, n = 0, len(M)

    for i in range(n):
        for j in range(n):
            sum += point_energy_function(M, i, j)

    return sum

def temp_fun(T0, a):
    return T0 * (1 - a)

def schedule_prob(E, T):
    if E < 0: return 1
    return exp(-E / T)

def schedule_neighbour(points):
    n = len(points)

    while True:
        x1, y1 = randint(0, n - 1), randint(0, n - 1)
        x2, y2 = randint(0, n - 1), randint(0, n - 1)
        if points[x1][y1] != points[x2][y2]: break

    return ((x1, y1), (x2, y2))

def calculate_point_energy_difference(points, P1, P2, point_energy_function, offsets):
    x1, y1 = P1
    x2, y2 = P2

    n = len(points)
    start_energy, end_energy = 0, 0

    for x_offset, y_offset in offsets:
        if -1 < x1 + x_offset < n and -1 < y1 + y_offset < n:
            start_energy += point_energy_function(points, x1 + x_offset, y1 + y_offset)
        if -1 < x2 + x_offset < n and -1 < y2 + y_offset < n:
            start_energy += point_energy_function(points, x2 + x_offset, y2 + y_offset)

    start_energy += point_energy_function(points, x1, y1)
    start_energy += point_energy_function(points, x2, y2)

    points[x1][y1], points[x2][y2] = points[x2][y2], points[x1][y1]

    for x_offset, y_offset in offsets:
        if -1 < x1 + x_offset < n and -1 < y1 + y_offset < n:
            end_energy += point_energy_function(points, x1 + x_offset, y1 + y_offset)
        if -1 < x2 + x_offset < n and -1 < y2 + y_offset < n:
            end_energy += point_energy_function(points, x2 + x_offset, y2 + y_offset)

    end_energy += point_energy_function(points, x1, y1)
    end_energy += point_energy_function(points, x2, y2)

    points[x1][y1], points[x2][y2] = points[x2][y2], points[x1][y1]

    return end_energy - start_energy

def simulated_annealing(points, max_iter, init_temp, point_energy_function, offsets, a):
    xs, ys = [], []
    T = init_temp

    all_energy = calculate_energy(points, point_energy_function)

    for i in range(1, max_iter + 1):
        T = temp_fun(T, a)
        if T < 1e-12: break

        P1, P2 = schedule_neighbour(points)

        dE = calculate_point_energy_difference(points, P1, P2, point_energy_function, offsets)

        x1, y1 = P1
        x2, y2 = P2

        if schedule_prob(dE, T) > random():
            all_energy += dE
            points[x1][y1], points[x2][y2] = points[x2][y2], points[x1][y1]

        xs.append(i)
        ys.append(all_energy)

    plt.imshow(points, cmap = 'gray', interpolation = 'nearest')
    plt.axis('off')
    plt.grid(True)
    plt.show()

    plt.plot(xs, ys)
    plt.grid(True)
    plt.show()

offsets = [(-1, -1), (1, 1), (1, -1), (-1, 1), (-2, -2), (2, 2), (2, -2), (-2, 2)]
simulated_annealing(random_image(100, 0.1), 1000000, 100, point_energy_8_neighbours_cross, offsets, 1e-5)
simulated_annealing(random_image(100, 0.2), 1000000, 100, point_energy_8_neighbours_cross, offsets, 1e-5)
simulated_annealing(random_image(100, 0.4), 1000000, 100, point_energy_8_neighbours_cross, offsets, 1e-5)