import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt

# Create a simple temporal graph
def create_temporal_graph():
    # Number of nodes and time steps
    num_nodes = 5
    num_timesteps = 3

    # Create a DGL graph
    g = dgl.DGLGraph()

    # Add nodes for each time step
    for _ in range(num_timesteps):
        g.add_nodes(num_nodes)

    # Add temporal edges between nodes of consecutive time steps
    for t in range(num_timesteps - 1):
        src = torch.arange(num_nodes) + t * num_nodes
        dst = torch.arange(num_nodes) + (t + 1) * num_nodes
        g.add_edges(src, dst)
        g.add_edges(dst, src)  # If the edges are bidirectional

    return g

# Visualize the graph
def visualize_graph(g):
    nx_g = g.to_networkx()
    pos = {i: (i // 5, -i % 5) for i in range(g.number_of_nodes())}
    nx.draw(nx_g, pos, with_labels=True, node_size=500, node_color='lightblue')
    plt.show()

g = create_temporal_graph()
visualize_graph(g)
