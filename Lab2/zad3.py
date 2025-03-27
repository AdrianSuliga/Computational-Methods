import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from random import randint
from collections import deque

def readGraph(path: str):
    file = open(path, "r")
    E = []

    for line in file:
        u, v, w = map(int, line.split(','))
        E.append((u, v, w))
     
    return E

def drawGraph(Edges, Weights, Currents, seed, sem_nodes):
    G = nx.DiGraph()

    for u, v, _ in Edges:
        w = Weights[Currents[u][v]]
        if u < v and w < 0: G.add_edge(v, u, weight = abs(w))
        elif u > v and w < 0: G.add_edge(u, v, weight = abs(w))
        elif u < v and w >= 0: G.add_edge(u, v, weight = w)
        elif u > v and w >= 0: G.add_edge(v, u, weight = w)
        
    pos = nx.spring_layout(G, k=1.0, seed=seed, scale=5)

    node_colors = ["lightgreen" if node in sem_nodes else "lightblue" for node in G.nodes]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, edgecolors="black")
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=10)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color="darkblue")

    edge_labels = {(u, v): f"{w:.2f} A" for u, v, w in G.edges(data="weight")}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.show()

def draw2DGraph(Edges, Weights, Currents, sem_nodes):
    G = nx.DiGraph()

    for u, v, _ in Edges:
        w = Weights[Currents[u][v]]
        if u < v and w < 0: G.add_edge(v, u, weight = abs(w))
        elif u > v and w < 0: G.add_edge(u, v, weight = abs(w))
        elif u < v and w >= 0: G.add_edge(u, v, weight = w)
        elif u > v and w >= 0: G.add_edge(v, u, weight = w)
        
    n = int(np.sqrt(len(set(G.nodes)))) 
    pos = {i: (i % n, -i // n) for i in G.nodes}  

    node_colors = ["lightgreen" if node in sem_nodes else "lightblue" for node in G.nodes]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, edgecolors="black")
    nx.draw_networkx_edges(G, pos, edge_color="black", arrowstyle="->", arrowsize=10)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color="darkblue")

    edge_labels = {(u, v): f"{w:.2f} A" for u, v, w in G.edges(data="weight")}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.show()

def listForm(E):
    n = graphSizeFromList(E)
    G = [[] for _ in range(n)]

    for edge in E:
        G[edge[0]].append((edge[1], edge[2]))
        G[edge[1]].append((edge[0], edge[2]))

    return G

def graphSizeFromList(E):
    size = 0

    for u, v, _ in E:
        size = max(size, u, v)
    
    return size + 1

def parentToCycle(Parent, start, end):
    result = [end]
    while end != start:
        end = Parent[end]
        result.append(end)
    return result

def removeEdgeFromListForm(G, u, v):
    G[u] = [n for n in G[u] if n[0] != v]
    G[v] = [n for n in G[v] if n[0] != u]

def findCycleFrom(G, start, end):
    # Usuń połączenie pomiędzy start i end aby nie
    # powodowało zbyt wczesnego wykrycia cyklu
    removeEdgeFromListForm(G, start, end)

    V = len(G)
    Q = deque()
    Q.append(start)
    parent = [None for _ in range(V)]
    visited = [False for _ in range(V)]
    visited[start] = True

    while Q:
        vertex = Q.pop()
        for v, _ in G[vertex]:
            if v == end:
                parent[end] = vertex
                parent[start] = end
                return parent
            if not visited[v]:
                Q.append(v)
                visited[v] = True
                parent[v] = vertex
    
    # Nie udało się znaleźć cyklu
    return None

def findCycles(G, size):
    cycle_count = 0
    cycles = []

    for u in range(len(G)):
        for v, _ in G[u]:
            result = findCycleFrom(G, u, v)
            if result != None:
                cycle_count += 1
                cycles.append(parentToCycle(result, u, v))
                removeEdgeFromListForm(G, u, v)
            if cycle_count == size:
                return cycles
    
    # Nie udało się znaleźć odpowiedniej ilości cykli,
    # więc nie można rozwiązać równania
    return None

def findPath(G, start, end):
    def dfsVisit(G, u, end, visited, path):
        visited[u] = True
        if u == end: return path

        for v, _ in G[u]:
            if not visited[v]:
                result = dfsVisit(G, v, end, visited, path + [v])
                if result is not None: return result
        
        return None

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

def zad3Kirchoff(Edges, s, t, SEM, seed, NetGraph = False):
    G = listForm(Edges)
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
        Eqs[eq_row][Currents[SEM_path[i]][SEM_path[i + 1]]] = \
        Edges[Currents[SEM_path[i]][SEM_path[i + 1]]][2] * \
        (1 if SEM_path[i] > SEM_path[i + 1] else -1)
    eq_row += 1

    # 1. prawo Kirchoffa
    for u in range(V):
        if u == s or u == t: continue
        for v, _ in G[u]:
            if u < v: Eqs[eq_row][Currents[u][v]] = -1
            if u > v: Eqs[eq_row][Currents[u][v]] = 1
        eq_row += 1

    # 2. prawo Kirchoffa
    Cycles = findCycles(listForm(Edges), E - V + 1)
    
    if Cycles is None:
        print("Nie można rozwiązać układu, za mało równań")
        return

    for cycle in Cycles:
        cycle = cycle + [cycle[0]]
        cycle_len = len(cycle)

        for i in range(cycle_len - 1):
            Eqs[eq_row][Currents[cycle[i]][cycle[i + 1]]] = \
            Edges[Currents[cycle[i]][cycle[i + 1]]][2] * \
            (1 if cycle[i] > cycle[i + 1] else -1)
        eq_row += 1

    Weights = np.linalg.solve(Eqs, A)

    print("Wyliczone natężenia:")
    print(Weights)

    if NetGraph: draw2DGraph(Edges, Weights, Currents, [s,t])
    else: drawGraph(Edges, Weights, Currents, seed, [s,t])

zad3Kirchoff(readGraph("input/graph1.txt"), s = 0, t = 1, SEM = 200, seed = 25)
zad3Kirchoff(readGraph("input/graph2.txt"), s = 2, t = 1, SEM = 100, seed = 19)
zad3Kirchoff(readGraph("input/graph_erdos_renyi.txt"), s = 0, t = 1, SEM = 200, seed = 31)
zad3Kirchoff(readGraph("input/graph_cubical.txt"), s = 0, t = 1, SEM = 100, seed = 2)
zad3Kirchoff(readGraph("input/graph_bridge.txt"), s = 0, t = 12, SEM = 200, seed = 173)
zad3Kirchoff(readGraph("input/graph_2d_net.txt"), s = 0, t = 1, SEM = 1000, seed = 10, NetGraph = True)
zad3Kirchoff(readGraph("input/graph_small_world.txt"), s = 2, t = 3, SEM = 500, seed = 1, NetGraph = True)