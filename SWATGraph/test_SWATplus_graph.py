import networkx as nx
import pickle
import pandas as pd
import numpy as np

def load_graph(graph_path, temporal_data_path):
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    with open(temporal_data_path, 'rb') as f:
        temporal_data = pickle.load(f)
    return G, temporal_data

def print_graph_info(G, temporal_data):
    print("Graph Information:")
    print("===================")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")

    # Print node and edge attributes
    print("\nSample of node attributes:")
    for i, (node, attr) in enumerate(G.nodes(data=True)):
        if i >= 10:  # limit to 10 samples
            break
        print(f"Node {node}: {attr}")
        if attr.get('node_role') == 'weather_station':
            print(f"Temporal data for node {node}: {temporal_data.get(node)}")

    print("\nSample of edge attributes for each role:")
    for i, (source, target, attr) in enumerate(G.edges(data=True)):
        if i >= 10:
            break
        print(f"Edge ({source}, {target}): {attr}")

    # Unique node roles
    node_roles = nx.get_node_attributes(G, 'node_role').values()
    unique_node_roles = set(node_roles)
    print(f"\nUnique node roles: {unique_node_roles}")

    # unique edge roles
    edge_roles = nx.get_edge_attributes(G, 'edge_role').values()
    unique_edge_roles = set(edge_roles)
    print(f"\nUnique edge roles: {unique_edge_roles}")

    # Unique spatial keys
    spatial_keys = nx.get_node_attributes(G, 'spatial_key').values()
    unique_spatial_keys = len(set(spatial_keys))
    print(f"\nUnique spatial keys: {unique_spatial_keys}")

    # Print temporal data keys and shapes
    print("\nTemporal data keys and shapes:")
    for key, value in temporal_data.items():
        print(f"Temporal_Key: {key}, Shape: {value.shape} (time_steps, features)")

    # Check for missing or NaN values in node attributes
    print("\nChecking for missing or NaN values in node attributes:")
    for node, attr in G.nodes(data=True):
        for key, value in attr.items():
            if isinstance(value, (list, np.ndarray)):
                if np.isnan(value).any():
                    print(f"Node {node} has NaN value(s) in attribute {key}")
            elif pd.isna(value):
                print(f"Node {node} has NaN value for attribute {key}")

    # Check how many nodes do not have the 'node_role' attribute
    nodes_without_role = [node for node, attr in G.nodes(data=True) if 'node_role' not in attr]
    print(f"\nNumber of nodes without 'node_role' attribute: {len(nodes_without_role)}")
    if nodes_without_role:
        print(f"Nodes without 'node_role' attribute: {nodes_without_role}")

    # Check for the number of features each node role has
    print("\nNumber of features for each node role:")
    role_feature_counts = {role: [] for role in unique_node_roles}
    for node, attr in G.nodes(data=True):
        if 'node_role' in attr and 'feature' in attr:
            role_feature_counts[attr['node_role']].append(len(attr['feature']))
    for role, counts in role_feature_counts.items():
        if counts:
            print(f"Node role '{role}' has {len(counts)} nodes with an average of {sum(counts)/len(counts):.2f} features each.")
        else:
            print(f"Node role '{role}' has no features.")

    # Check for nodes without any attributes
    nodes_without_attributes = [node for node, attr in G.nodes(data=True) if not attr]
    print(f"\nNumber of nodes without any attributes: {len(nodes_without_attributes)}")
    if nodes_without_attributes:
        print(f"Nodes without any attributes: {nodes_without_attributes}")

    # Check for edges without any attributes 
    edges_without_attributes = [(u, v) for u, v, attr in G.edges(data=True) if not attr]
    print(f"\nNumber of edges without any attributes: {len(edges_without_attributes)}")
    if edges_without_attributes:
        print(f"Edges without any attributes: {edges_without_attributes}")

if __name__ == "__main__":
    graph_path = '/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/04115000/Graphs/SWAT_plus_streams.gpickle'
    temporal_data_path = '/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/04115000/Graphs/temporal_data.pkl'

    G, temporal_data = load_graph(graph_path, temporal_data_path)
    print_graph_info(G, temporal_data)
