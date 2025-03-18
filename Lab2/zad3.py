import networkx as nx
import matplotlib.pyplot as plt

def readGraph(path: str):
    file = open(path, "r")
    G = nx.Graph()

    for line in file:
        u, v, w = map(int, line.split(','))
        G.add_edge(u, v, weight = w) 

    return G

def drawGraph(G):
    pos = nx.spring_layout(G, seed = 3)

    nx.draw_networkx_nodes(G, pos, node_size = 400)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=3)

    nx.draw_networkx_labels(G, pos, font_size=16, font_family="sans-serif")
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)

    plt.title("Przykładowy graf")
    plt.show()

G = readGraph("input/graph.txt")