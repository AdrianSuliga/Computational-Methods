import networkx as nx
import matplotlib.pyplot as plt

def readGraph(path: str):
    file = open(path, "r")
    E = []

    for line in file:
        u, v, w = map(int, line.split(','))
        E.append((u, v, w))
     
    return E

def drawGraph(E):
    # Does not work for now
    G = nx.from_edgelist(E)
    pos = nx.spring_layout(G, seed = 3)

    nx.draw_networkx_nodes(G, pos, node_size = 400)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=3)

    nx.draw_networkx_labels(G, pos, font_size=16, font_family="sans-serif")
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)

    plt.title("Przykładowy graf")
    plt.show()

def visited(node, path):
    return node in path

# Function to rotate a cycle such that 
# it starts with the smallest node
def rotate_to_smallest(path):
    min_index = path.index(min(path))
    return path[min_index:] + path[:min_index]

# Function to invert the cycle order
def invert(path):
    return rotate_to_smallest(path[::-1])

# Function to check if a cycle is new
def is_new(path, cycles):
    return path not in cycles

def findOneNewCycle(path, graph, cycles):
    start_node = path[0]
    
    # Visit each edge and node of each edge
    for node1, node2, _ in graph:
        if start_node in (node1, node2):
            next_node = node2 if node1 == start_node else node1
            
            if not visited(next_node, path):
                # Neighbor node not on path yet
                sub = [next_node] + path
                # Explore extended path
                findOneNewCycle(sub, graph, cycles)
            elif len(path) > 2 and next_node == path[-1]:
                # Cycle found
                p = rotate_to_smallest(path)
                inv = invert(p)

                if is_new(p, cycles) and is_new(inv, cycles):
                    cycles.append(p)

def findAllCycles(E):
    cycles = []

    for u, v, _ in E:
        findOneNewCycle([u], E, cycles)
        findOneNewCycle([v], E, cycles)

    return cycles

E = readGraph("input/graph.txt")
print(E)
print(findAllCycles(E))

