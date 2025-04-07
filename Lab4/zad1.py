from random import randint, random
from math import exp, log
import matplotlib.pyplot as plt 
import os, tempfile
from PIL import Image

# CLOUD GENERATORS
def random_cloud(n):
    return [(randint(0, 100), randint(0, 100)) for _ in range(n)]

def normal_distribution(n):
    return []

def nine_groups(n):
    return[]

# LENGTH CALCULATOR
def path_length(points, distance):
    size = len(points)
    sum = 0

    for i in range(size - 1):
        sum += distance(points[i], points[i + 1])

    return sum

# DISTANCE FUNCTIONS
def euclidean_distance(p1, p2): 
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def manhatan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

# TEMPERATURE
def euler_temp(x, T0):
    return T0 * exp(-x / T0)

def exponential_temp(x, T0):
    return T0 * ((0.9) ** x)

def linear_temp(x, T0):
    return T0 - 0.01 * x

def log_temp(x, T0):
    return T0 / log(1 + x)

def schedule_prob(E, T):
    return exp(E / T)

# SWAPPING
def arbitrary_swap(state: list):
    result = state[:]
    size = len(state)
    i = j = 0

    while i == j:
        i, j = randint(0, size - 1), randint(0, size - 1)

    result[i], result[j] = result[j], result[i]
    return result

def consecutive_swap(state: list):
    result = state[:]
    size = len(state)
    i = randint(0, size - 1)

    result[i], result[i + 1] = result[i + 1], result[i]
    return result

# SIMULATED ANNEALING
# Generowanie GIFa pozostawiam jako opcjonalne, ponieważ zabiera dużo czasu.
def simulated_annealing(points, max_iter, initial_temp, 
                        temp_fun, neighbour_fun, distance_fun, make_gif = False):
    xs, ys = [], []
    frame_paths = []  
    temp_dir = None

    # W celu nieprzepełnienia RAMu tworzę tymczasowy folder, gdzie
    # będę zapisywał wygenerowane klatki GIFa
    if make_gif:
        fig, ax = plt.subplots()
        line, = ax.plot([], [], "ro-")
        title = ax.set_title("Wizualizacja działania")
        plt.tight_layout()
        temp_dir = tempfile.mkdtemp()

    # Algorytm wyżarzania
    for i in range(1, max_iter + 1):
        T = temp_fun(i, initial_temp)
        if T < 1e-12: break
        candidate = neighbour_fun(points)
        
        E = path_length(points, distance_fun) - \
            path_length(candidate, distance_fun)
        if E > 0:
            points = candidate[:]
        else:
            prob = schedule_prob(E, T)
            if random() < prob:
                points = candidate[:]

        # W każdej iteracji zapisujemy dane do późniejszego 
        # wykresu (numer iteracji, długość znalezionej ścieżki)
        xs.append(i)
        ys.append(path_length(points, distance_fun))

        if make_gif and (i - 1) % 10 == 0:
            # Zapis klatki to pliku
            x_coords = [x for x, _ in points]
            y_coords = [y for _, y in points]
            line.set_data(x_coords, y_coords)

            ax.set_xlim(-5, 105)
            ax.set_ylim(-5, 105)
            title.set_text(f"Progres = {i * 100 / max_iter}%")
            print(f"Generowanie klatek... {i * 100 / max_iter}%")
            
            frame_path = os.path.join(temp_dir, f"frame_{i:05d}.png")
            fig.savefig(frame_path, dpi=80)
            frame_paths.append(frame_path)

    if make_gif:
        print("Wygenerowano dane\nTworzenie pliku GIF, może to chwilę potrwać...")
        try:
            with Image.open(frame_paths[0]) as first_frame:
                other_frames = [Image.open(f) for f in frame_paths[1:]]
                first_frame.save(
                    'animation.gif',
                    format = 'GIF',
                    append_images = other_frames,
                    save_all = True,
                    duration = 10,
                    loop = 0
                )
                for frame in other_frames:
                    frame.close()
        except Exception as e:
            print(f"Error creating GIF: {e}")
            
        for frame_path in frame_paths:
            try:
                os.remove(frame_path)
            except:
                pass
        try:
            os.rmdir(temp_dir)
        except:
            pass
        plt.close(fig)

    # Przedstawiamy na wykresie uzyskane rozwiązanie
    fig, axs = plt.subplots(2)
    axs[0].plot(xs, ys)
    axs[0].set_title(f"Wynik algorytmu dla {max_iter} wykonań")
    axs[0].set_xlabel("Ilość iteracji")
    axs[0].set_ylabel("Długość ścieżki")

    x_coords = [x for x, _ in points]
    y_coords = [y for _, y in points]
    axs[1].plot(x_coords, y_coords, "ro-")
    axs[1].set_title(f"Znaleziona ścieżka o długości " 
                    f"{round(path_length(points, distance_fun), 2)}")

    plt.tight_layout()
    plt.show()

init_temp = 1000
max_iter = 10000
#simulated_annealing(random_cloud(10), max_iter, init_temp, euler_temp, arbitrary_swap, euclidean_distance, True)
#simulated_annealing(random_cloud(20), 2 * max_iter, 2 * init_temp, euler_temp, arbitrary_swap, euclidean_distance, True)
#simulated_annealing(random_cloud(50), 5 * max_iter, 5 * init_temp, euler_temp, arbitrary_swap, euclidean_distance, True)