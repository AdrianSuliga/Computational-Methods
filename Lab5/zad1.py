import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw_sphere(n: int) -> None:
    s = np.linspace(0, 2 * np.pi, n)
    t = np.linspace(0, np.pi, n) 
    S, T = np.meshgrid(s, t)

    X = np.cos(S) * np.sin(T)
    Y = np.sin(S) * np.sin(T)
    Z = np.cos(T)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')

    ax.plot_surface(X, Y, Z)

    ax.set_title("Sfera jednostkowa")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_aspect('equal')

    plt.show()

    return X, Y, Z

def draw(P:tuple):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')

    ax.plot_surface(P[0].reshape(100, 100), P[1].reshape(100, 100), P[2].reshape(100, 100))

    ax.set_title("Elipsoida")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_aspect('equal')

    plt.show()

def elipsoid(T: tuple) -> None:
    X, Y, Z = T
    A1 = np.random.randint(5, size = (3, 3))
    A2 = np.random.randint(5, size = (3, 3))
    A3 = np.random.randint(5, size = (3, 3))

    P = A1 @ [X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]
    P = A2 @ P
    P = A3 @ P

elipsoid(draw_sphere(100))