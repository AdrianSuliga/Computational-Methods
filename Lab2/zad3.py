import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def readGraph(path: str):
    file = open(path, "r")
    E = []

    for line in file:
        u, v, w = map(int, line.split(','))
        E.append((u, v, w))
     
    return E

def list_form(E):
    n = graph_size_from_list(E)
    G = [[] for _ in range(n)]

    for edge in E:
        G[edge[0]].append((edge[1], edge[2]))
        G[edge[1]].append((edge[0], edge[2]))

    return G

def graph_size_from_list(E):
    size = 0

    for u, v, _ in E:
        size = max(size, u, v)
    
    return size + 1

def drawGraph(Edges, Weights, Currents):
    G = nx.DiGraph()

    for u, v, _ in Edges:
        w = Weights[Currents[u][v]]
        if u < v and w < 0: G.add_edge(v, u, weight = abs(w))
        elif u > v and w < 0: G.add_edge(u, v, weight = abs(w))
        elif u < v and w >= 0: G.add_edge(u, v, weight = w)
        elif u > v and w >= 0: G.add_edge(v, u, weight = w)
        
    pos = nx.spring_layout(G, seed = 89)
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", edgecolors="black")
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=15)
    nx.draw_networkx_labels(G, pos, font_size=12, font_color="black")

    edge_labels = {(u, v): f"{w:.2f}" for u, v, w in G.edges(data="weight")}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.tight_layout()
    plt.show()

def findCycleFrom(G:list, u:int, path:list, cycles:list, size:int):
    for v, _ in G[u]:
        if v not in path:
            findCycleFrom(G, v, [v] + path, cycles, size)
        elif v == path[-1] and len(path) > 2:
            min_index = path.index(min(path))
            path = path[min_index:] + path[:min_index]
            inv = [path[0]] + path[1:][::-1]
            if path not in cycles and inv not in cycles and len(cycles) < size:
                cycles.append(path)

def findCycles(G, size):
    cycles = []
    n = len(G)

    for u in range(n):
        findCycleFrom(G, u, [u], cycles, size)
        if len(cycles) == size: break

    if len(cycles) < size:
        print("Nie można znaleźć wystarczająco dużo cykli!")
    
    return cycles

def findPath(G, start, end):
    def dfsVisit(G, u, end, visited, path):
        visited[u] = True
        for v, _ in G[u]:
            if v == end:
                return path + [v]
            if not visited[v]:
                return dfsVisit(G, v, end, visited, path + [v])

    n = len(G)
    visited = [False for _ in range(n)]
    visited[start] = True
    path = [start]

    return dfsVisit(G, start, end, visited, path)

def edgesToCurrents(V, Edges):
    Currents = [[-1 for _ in range(V)] for _ in range(V)]
    it = 0
    
    for u, v, _ in Edges:
        Currents[u][v] = it
        Currents[v][u] = it
        it += 1

    return Currents

def zad3Kirchoff(Edges, s, t, SEM):
    G = list_form(Edges)
    E = len(Edges)
    V = len(G)

    Eqs = [[0 for _ in range(E)] for _ in range(E)]
    Currents = edgesToCurrents(V, Edges)

    # Wektor wyników
    A = [0 for _ in range(E)]
    A[0] = SEM

    eq_row = 0

    # Ścieżka z SEM
    SEM_path = findPath(G, s, t)
    for i in range(len(SEM_path) - 1):
        Eqs[eq_row][Currents[SEM_path[i]][SEM_path[i + 1]]] = Edges[Currents[SEM_path[i]][SEM_path[i + 1]]][2] * (1 if SEM_path[i] > SEM_path[i + 1] else -1)
    eq_row += 1

    # 1. prawo Kirchoffa
    for u in range(V):
        if u == s or u == t: continue
        for v, _ in G[u]:
            if u < v: Eqs[eq_row][Currents[u][v]] = -1
            if u > v: Eqs[eq_row][Currents[u][v]] = 1
        eq_row += 1

    # 2. prawo Kirchoffa
    Cycles = findCycles(G, E - V + 1)

    for cycle in Cycles:
        cycle = cycle + [cycle[0]]
        cycle_len = len(cycle)

        for i in range(cycle_len - 1):
            Eqs[eq_row][Currents[cycle[i]][cycle[i + 1]]] = Edges[Currents[cycle[i]][cycle[i + 1]]][2] * (1 if cycle[i] > cycle[i + 1] else -1)
        eq_row += 1

    Weights = np.linalg.solve(Eqs, A)
    print("Wyliczone natężenia:")
    print(Weights)

    drawGraph(Edges, Weights, Currents)

zad3Kirchoff(readGraph("input/graph1.txt"), 1, 3, 200)
zad3Kirchoff(readGraph("input/graph2.txt"), 0, 1, 50)

