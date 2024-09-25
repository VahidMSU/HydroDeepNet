import os
from torch_geometric.data import HeteroData
import logging
from load_relation_tables import generate_relation_table
from libs.load_climate  import load_climate_datas
import torch
from torch_geometric.utils import add_self_loops
import pandas as pd
from libs.utils import check_hetero_data_for_none
def graph_consistency_check(node_tracker, processed_hru_ids, G):
    for hru_id in node_tracker['hru']:
        if hru_id not in processed_hru_ids:
            logging.error(f"HRU {hru_id} does not have a connection to any weather station.")

    # Logging node roles and the number of nodes and edges
    logging.info(f"Unique node types: {G.node_types}")
    logging.info(f"Created graph with {G.num_nodes} nodes and {G.num_edges} edges across {G.edge_types} edge types.")
def add_self_loops_to_graph(G):
    for edge_type in G.edge_types:
        src_type, rel_type, dst_type = edge_type
        
        # Skip adding self-loops for channel-channel edges
        if src_type == dst_type and rel_type != 'hydrograph':
            G[edge_type].edge_index = add_self_loops(G[edge_type].edge_index)[0]
    return G


def add_gw_cell_node(G, node_tracker, row):
    if pd.notna(row['cell_id']):
        cell_id = row['cell_id']
        if cell_id not in node_tracker['gw_cell']:
            if 'x' in G['gw_cell']:
                G['gw_cell'].x = torch.cat([G['gw_cell'].x, torch.tensor([[row['cell_area'], row['overlap_area_m2']]], dtype=torch.float)], dim=0)
            else:
                G['gw_cell'].x = torch.tensor([[row['cell_area'], row['overlap_area_m2']]], dtype=torch.float)
            node_tracker['gw_cell'][cell_id] = len(node_tracker['gw_cell'])
    return G, node_tracker

def add_edge(G, src, dest, src_type, dest_type, edge_type):
    if (src_type, edge_type, dest_type) not in G.edge_types:
        G[(src_type, edge_type, dest_type)].edge_index = torch.tensor([[src], [dest]], dtype=torch.long)
    else:
        G[(src_type, edge_type, dest_type)].edge_index = torch.cat(
            [G[(src_type, edge_type, dest_type)].edge_index, torch.tensor([[src], [dest]], dtype=torch.long)], dim=1
        )
    return G

def add_HRU_node(G, node_tracker, row):
    if pd.notna(row['hru_id']) and row['hru_elev'] > 0:
        hru_id = row['hru_id']
        if hru_id not in node_tracker['hru']:
            if 'x' in G['hru']:
                G['hru'].x = torch.cat([G['hru'].x, torch.tensor([[row['hru_elev'], row['hru_area']]], dtype=torch.float)], dim=0)
            else:
                G['hru'].x = torch.tensor([[row['hru_elev'], row['hru_area']]], dtype=torch.float)
            node_tracker['hru'][hru_id] = len(node_tracker['hru'])
    return G, node_tracker, hru_id

def add_channel_node(G, node_tracker, row):
    if pd.notna(row['channel_id']):
        channel_id = row['channel_id']
        if channel_id not in node_tracker['channel']:
            if 'x' in G['channel']:
                G['channel'].x = torch.cat([G['channel'].x, torch.tensor([[row['channel_length_m'], row['drop']]], dtype=torch.float)], dim=0)
            else:
                G['channel'].x = torch.tensor([[row['channel_length_m'], row['drop']]], dtype=torch.float)
            node_tracker['channel'][channel_id] = len(node_tracker['channel'])
    return G, node_tracker  

def check_dslinkno(G, node_tracker, row):
    if pd.notna(row['dslinkno']):
        dslinkno = row['dslinkno']
        if dslinkno not in node_tracker['channel']:
            if 'x' in G['channel']:
                G['channel'].x = torch.cat([G['channel'].x, torch.tensor([[0.0, 0.0]], dtype=torch.float)], dim=0)
            else:
                G['channel'].x = torch.tensor([[0.0, 0.0]], dtype=torch.float)
            node_tracker['channel'][dslinkno] = len(node_tracker['channel'])
    return G, node_tracker
def add_channel_edge(G, node_tracker, row, channel_edges_tracker):
    if pd.notna(row['channel_id']) and pd.notna(row['dslinkno']):
        src_channel = row['channel_id']
        dest_channel = row['dslinkno']
        channel_edge = (src_channel, dest_channel)
        # Only add the edge if it doesn't already exist
        if channel_edge not in channel_edges_tracker and (src_channel in node_tracker['channel'] and dest_channel in node_tracker['channel']):
            G = add_edge(G, node_tracker['channel'][src_channel], node_tracker['channel'][dest_channel], 'channel', 'channel', 'hydrograph')
            channel_edges_tracker.add(channel_edge)
    return G, channel_edges_tracker


def add_weather_station_hru_edge(G, node_tracker, row, hru_id, processed_hru_ids, unique_wst_ids, temporal_data, txtintout_path):
    # Add Weather Station node with temporal features
    if pd.notna(row['wst_id']) and row['wst_id'] in unique_wst_ids:
        wst_id = row['wst_id']
        logging.info(f"Processing weather station {wst_id}")  # Log the wst_id being processed
        if wst_id not in node_tracker['weather_station']:
            temporal_array = load_climate_datas(txtintout_path, row['wst'], start_year=2000, end_year=2020)

            if 'x' not in G['weather_station']:
                G['weather_station'].x = torch.zeros((len(unique_wst_ids), 7671, 6), dtype=torch.float)

            G['weather_station'].x[len(node_tracker['weather_station'])] = torch.tensor(temporal_array, dtype=torch.float)
            
            # Track the index of this weather station node
            node_tracker['weather_station'][wst_id] = len(node_tracker['weather_station'])

            temporal_data[wst_id] = temporal_array
            unique_wst_ids = unique_wst_ids[unique_wst_ids != wst_id]

    # Adding climate edges: connecting HRU to the appropriate weather station
    if hru_id not in processed_hru_ids:
        if pd.notna(row['wst_id']) and pd.notna(row['hru_id']):
            if row['wst_id'] in node_tracker['weather_station'] and row['hru_id'] in node_tracker['hru']:
                G = add_edge(G, node_tracker['weather_station'][row['wst_id']], node_tracker['hru'][row['hru_id']], 'weather_station', 'hru', 'climate')
            else:
                logging.error(f"Weather station {row['wst_id']} or HRU {row['hru_id']} not found in node tracker")
        processed_hru_ids.add(hru_id)  # Mark this HRU as processed
    return G, node_tracker, processed_hru_ids, unique_wst_ids, temporal_data




def add_gw_channel_edge(G, node_tracker, row, processed_gw_channel_edges):
    if pd.notna(row['cell_id']) and pd.notna(row['channel_id']):
        gw_channel_edge = (row['cell_id'], row['channel_id'])
        channel_gw_edge = (row['channel_id'], row['cell_id'])  # Reverse edge

        if gw_channel_edge not in processed_gw_channel_edges:
            if row['cell_id'] in node_tracker['gw_cell'] and row['channel_id'] in node_tracker['channel']:
                # Add the gw_cell to channel edge
                G = add_edge(G, node_tracker['gw_cell'][row['cell_id']], node_tracker['channel'][row['channel_id']], 'gw_cell', 'channel', 'gw_sw')
                # Add the channel to gw_cell edge (reverse direction)
                G = add_edge(G, node_tracker['channel'][row['channel_id']], node_tracker['gw_cell'][row['cell_id']], 'channel', 'gw_cell', 'sw_gw')
                # Track these edges as processed
                processed_gw_channel_edges.add(gw_channel_edge)
                processed_gw_channel_edges.add(channel_gw_edge)  # Track the reverse edge as well
            else:
                logging.error(f"Groundwater Cell {row['cell_id']} or Channel {row['channel_id']} not found in node tracker")
    return G, processed_gw_channel_edges



def add_sw_gw_edge(G, node_tracker, row):
    if pd.notna(row['cell_id']) and pd.notna(row['hru_id']):
        if row['hru_id'] in node_tracker['hru'] and row['cell_id'] in node_tracker['gw_cell']:
            G = add_edge(G, node_tracker['hru'][row['hru_id']], node_tracker['gw_cell'][row['cell_id']], 'hru', 'gw_cell', 'sw_gw')
        else:
            logging.error(f"HRU {row['hru_id']} or Groundwater Cell {row['cell_id']} not found in node tracker")
    return G

# Ensure each HRU is connected to only one channel
def add_hru_channel_edge(G, node_tracker, row, processed_hru_channel_edges):
    if pd.notna(row['hru_id']) and pd.notna(row['channel_id']):
        hru_channel_edge = (row['hru_id'], row['channel_id'])
        channel_hru_edge = (row['channel_id'], row['hru_id'])  # Reverse edge

        if hru_channel_edge not in processed_hru_channel_edges:
            if row['hru_id'] in node_tracker['hru'] and row['channel_id'] in node_tracker['channel']:
                # Add the hru to channel edge
                G = add_edge(G, node_tracker['hru'][row['hru_id']], node_tracker['channel'][row['channel_id']], 'hru', 'channel', 'sw')
                # Add the channel to hru edge (reverse direction)
                G = add_edge(G, node_tracker['channel'][row['channel_id']], node_tracker['hru'][row['hru_id']], 'channel', 'hru', 'sw_reverse')
                # Track these edges as processed
                processed_hru_channel_edges.add(hru_channel_edge)
                processed_hru_channel_edges.add(channel_hru_edge)  # Track the reverse edge as well
            else:
                logging.error(f"HRU {row['hru_id']} or Channel {row['channel_id']} not found in node tracker")

    return G, processed_hru_channel_edges


class SWATGraphProcessor:
    def __init__(self, swat_output_base_path):
        self.swat_output_base_path = swat_output_base_path
        self.temporal_data = {}

    def create_graph(self):
        G = HeteroData()  # Create an empty HeteroData object

        unique_wst_ids = self.swat_streams['wst_id'].unique()
        logging.info(f"Number of unique wst_ids: {len(unique_wst_ids)}")

        # A dictionary to keep track of existing nodes with their indices
        node_tracker = {
            'hru': {},
            'gw_cell': {},
            'channel': {},
            'weather_station': {}
        }

        processed_hru_ids = set()  # To ensure each HRU is only processed once
        channel_edges_tracker = set()  # To track unique channel-channel edges
        hru_channel_edges_tracker = set()  # To track unique HRU-channel edges
        processed_hru_channel_edges = set()  # To track HRU-channel edges
        processed_gw_channel_edges = set()  # To track GW cell-channel edges

        adding_wst = False
        for _, row in self.swat_streams.iterrows():
            # Add HRU node
            G, node_tracker, hru_id = add_HRU_node(G, node_tracker, row)

            # Add Groundwater Cell node
            G, node_tracker = add_gw_cell_node(G, node_tracker, row)

            # Add Channel node
            G, node_tracker = add_channel_node(G, node_tracker, row)

            # Ensure downstream channels (dslinkno) are added as nodes
            G, node_tracker = check_dslinkno(G, node_tracker, row)
            if adding_wst:
                G, node_tracker, processed_hru_ids, unique_wst_ids, self.temporal_data = add_weather_station_hru_edge(G, node_tracker, row, hru_id, processed_hru_ids, unique_wst_ids, self.temporal_data, self.txtintout_path)
            # Adding hydrograph edges, ensuring only unique edges are added
            G, channel_edges_tracker = add_channel_edge(G, node_tracker, row, channel_edges_tracker)

            # Adding surface water to groundwater edges
            
            G = add_sw_gw_edge(G, node_tracker, row)


            
            G, processed_hru_channel_edges = add_hru_channel_edge(G, node_tracker, row, processed_hru_channel_edges)


            G, processed_gw_channel_edges = add_gw_channel_edge(G, node_tracker, row, processed_gw_channel_edges)

        # Ensure all HRU nodes have a connection to a weather station

        graph_consistency_check(node_tracker, processed_hru_ids, G)

        return G

    def process(self, name):

        os.makedirs(f'{self.swat_output_base_path}/{name}/Graphs', exist_ok=True)
        self.txtintout_path = f'{self.swat_output_base_path}/{name}/SWAT_gwflow_MODEL/Scenarios/Default/TxtInOut'
        self.swat_streams = pd.read_csv(f'{self.swat_output_base_path}/{name}/Graphs/cell_hru_riv_wst.csv')
        logging.info(f"swat_streams columns: {self.swat_streams.columns}")

        self.swat_streams['wst_id'] = pd.factorize(self.swat_streams['wst'])[0]
        self.swat_streams = self.swat_streams[self.swat_streams['hru_elev'] > 0]
        
        logging.info(f"Length of the SWAT streams before merging with graph con: {len(self.swat_streams)}")
        logging.info(f"Number of unique wst_id: {len(self.swat_streams['wst_id'].unique())}")
        logging.info(f"Number of unique linkno: {len(self.swat_streams['linkno'].unique())}")
        
        G = self.create_graph()
        G = add_self_loops_to_graph(G)
        
        print(G)    
        # Now G will include self-loops for all the nodes of the same type
        check_hetero_data_for_none(G)
        return G


if __name__ == "__main__":
    
    swat_output_base_path = "/data/MyDataBase/SWATplus_by_VPUID/0000/huc12"
    processor = SWATGraphProcessor(swat_output_base_path)
    names = os.listdir(swat_output_base_path)
    names.remove("log.txt")

    name = "04115000"
    print(f"Processing {name}")
    logger_path = os.path.join(swat_output_base_path, name, "Graphs/log.txt")
    generate_relation_table(name, logger_path)
    G = processor.process(name)
    ### save 