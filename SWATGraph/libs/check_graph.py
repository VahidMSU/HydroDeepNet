
import logging
import networkx as nx

def check_graph(graph):
    print("############### CHECKING GRAPH DATA ###########")
    node_roles = set(nx.get_node_attributes(graph, 'node_role').values())
    edge_roles = set(nx.get_edge_attributes(graph, 'edge_role').values())
    
    logging.info(f"Node types in the graph: {node_roles}")
    logging.info(f"Edge types in the graph: {edge_roles}")
    
    # Check if 'channel' node type exists in the graph
    if 'channel' not in node_roles:
        logging.warning("'channel' node type not found in the graph nodes.")
    else:
        logging.info("'channel' node type found in the graph nodes.")
    
    # Check edges to ensure 'channel' is a destination node in at least one edge type
    channel_as_dst = any(dst for _, dst, attr in graph.edges(data=True) if graph.nodes[dst].get('node_role') == 'channel')
    if not channel_as_dst:
        logging.warning("'channel' node type not found as a destination in any edge type.")
    else:
        logging.info("'channel' node type is a destination in at least one edge type.")

    # Check edges to ensure 'channel' is a source node in at least one edge type
    channel_as_src = any(src for src, _, attr in graph.edges(data=True) if graph.nodes[src].get('node_role') == 'channel')
    if not channel_as_src:
        logging.warning("'channel' node type not found as a source in any edge type.")
    else:
        logging.info("'channel' node type is a source in at least one edge type.")

