import pickle   
import networkx as nx
import os
import pandas as pd

def check_hetero_data_for_none(hetero_data):
    # Check node features
    print("Checking node features:")
    for node_type, x in hetero_data.x_dict.items():
        if x is None:
            print(f"Node type {node_type} has None as features.")
        else:
            print(f"Node type {node_type} features: {x.shape}")

    # Check edge indices
    print("\nChecking edge indices:")
    for edge_type, edge_index in hetero_data.edge_index_dict.items():
        if edge_index is None:
            print(f"Edge type {edge_type} has None as edge indices.")
        else:
            print(f"Edge type {edge_type} edge indices: {edge_index.shape}")

    # Optionally, raise an error or handle None values if found
    for node_type, x in hetero_data.x_dict.items():
        if x is None:
            raise ValueError(f"Node type {node_type} has None as features.")

    for edge_type, edge_index in hetero_data.edge_index_dict.items():
        if edge_index is None:
            raise ValueError(f"Edge type {edge_type} has None as edge indices.")

    print("\nNo None values found in the graph.")
    return hetero_data
def test_graph_consistency(swat_output_base_path, name):
    ## load the graph and print
    with open(f'{swat_output_base_path}/{name}/Graphs/SWAT_plus_streams.gpickle', 'rb') as f:
        G = pickle.load(f)

        ## Print unique edge types
        edge_types = set()
        for _, _, data in G.edges(data=True):
            if edge_type := data.get('edge_role'):
                edge_types.add(edge_type)
        print(f"Unique Edge Types: {edge_types}")

        ## Number of edges for each edge type and check feature consistency
        for edge_type in edge_types:
            feature_lengths = set()
            for u, v, data in G.edges(data=True):
                if data.get('edge_role') == edge_type:
                    features = data.get('feature', None)
                    if features is not None:
                        feature_lengths.add(len(features))
            
            if len(feature_lengths) > 1:
                print(f"Inconsistent number of features found for edges with role {edge_type}: {feature_lengths}")
            else:
                print(f"All edges with role {edge_type} have {feature_lengths.pop() if feature_lengths else 0} features")

        ## Print node types
        node_types = set()
        for _, data in G.nodes(data=True):
            if node_type := data.get('node_role'):
                node_types.add(node_type)
        print(f"Unique Node Types: {node_types}")

        ## Number of nodes for each node type and check feature consistency
        for node_type in node_types:
            feature_lengths = set()
            for node, data in G.nodes(data=True):
                if data.get('node_role') == node_type:
                    features = data.get('feature', None)
                    if features is not None:
                        feature_lengths.add(len(features))
            
            if len(feature_lengths) > 1:
                print(f"Inconsistent number of features found for nodes with role {node_type}: {feature_lengths}")
            else:
                print(f"All nodes with role {node_type} have {feature_lengths.pop() if feature_lengths else 0} features")
