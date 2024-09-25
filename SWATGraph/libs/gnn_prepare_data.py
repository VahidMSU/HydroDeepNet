import numpy as np
import pandas as pd
from datetime import datetime
import logging
import torch
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset
from torch_geometric.nn import HeteroConv
from torch_geometric.nn import GCNConv
from torch_geometric.data import HeteroData

class TemporalGraphDataset(Dataset):
    def __init__(self, temporal_data, graph, target_channel_id, streamflow_data_path):
        self.data, self.targets = prepare_data(temporal_data, graph, target_channel_id, streamflow_data_path)
        self.graph = graph
        
        self.hetero_data = HeteroData()
        
        node_counts = {}
        
        for node, attr in graph.nodes(data=True):
            if 'node_role' not in attr:
                continue
            node_type = attr['node_role']
            if node_type not in self.hetero_data:
                self.hetero_data[node_type].x = []
                node_counts[node_type] = 0
            self.hetero_data[node_type].x.append([attr.get('hru_elev', 0), attr.get('hru_area', 0), 
                                                  attr.get('hru_lat', 0), attr.get('hru_lon', 0), 
                                                  attr.get("channel_length_m", 0), attr.get("drop", 0),
                                                  attr.get('cell_area', 0), attr.get('overlap_area_m2', 0)])
            node_counts[node_type] += 1
        
        for node_type in self.hetero_data.node_types:
            self.hetero_data[node_type].x = torch.tensor(self.hetero_data[node_type].x, dtype=torch.float32)
            self.hetero_data[node_type].num_nodes = node_counts[node_type]  # Explicitly set the number of nodes
            logging.info(f"Node type: {node_type}, Features shape: {self.hetero_data[node_type].x.shape}")
        
        for edge in graph.edges(data=True):
            src, dst, attr = edge
            edge_type = attr['edge_role']
            if edge_type not in self.hetero_data:
                self.hetero_data[edge_type].edge_index = [[], []]
            self.hetero_data[edge_type].edge_index[0].append(src)
            self.hetero_data[edge_type].edge_index[1].append(dst)
        
        for edge_type in self.hetero_data.edge_types:
            self.hetero_data[edge_type].edge_index = torch.tensor(self.hetero_data[edge_type].edge_index, dtype=torch.long)
            self.hetero_data[edge_type].num_nodes = max(max(self.hetero_data[edge_type].edge_index[0]), max(self.hetero_data[edge_type].edge_index[1])) + 1  # Set the number of nodes
            logging.info(f"Edge type: {edge_type}, Edge index shape: {self.hetero_data[edge_type].edge_index.shape}")
        
        logging.info(f"Data shape: {self.data.shape}")
        logging.info(f"Targets shape: {self.targets.shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32), self.hetero_data
def hetero_data_collate(batch):
    inputs, targets, hetero_data_list = zip(*batch)
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    hetero_data_batch = Batch.from_data_list(hetero_data_list)
    return inputs, targets, hetero_data_batch



def prepare_data(temporal_data, graph, target_channel_id, streamflow_data_path):
    data = []
    targets = []
    start_year = 2000
    end_year = 2002
    logging.info(f"Preparing data for target channel ID: {target_channel_id}")

    if target_channel_id not in temporal_data:
        logging.error(f"Target channel ID {target_channel_id} not found in temporal data.")
        return np.array(data), np.array(targets)

    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    
    start_idx = (start_date - datetime(2000, 1, 1)).days
    end_idx = (end_date - datetime(2000, 1, 1)).days + 1
    num_days = (end_date - start_date).days + 1
    logging.info(f"Start index: {start_idx}, End index: {end_idx}, Number of days: {num_days}")
    
    streamflow_df = pd.read_csv(streamflow_data_path)
    logging.info(f"Streamflow data shape: {streamflow_df.shape}")

    streamflow_df['date'] = pd.to_datetime(streamflow_df['date'])
    streamflow_df = streamflow_df[(streamflow_df['date'] >= start_date) & (streamflow_df['date'] <= end_date)]
    logging.info(f"Streamflow data shape after filtering: {streamflow_df.shape}")

    targets_array = streamflow_df['streamflow'].values

    if len(targets_array) < num_days:
        logging.error("The length of the target streamflow data is less than the number of days in the range.")
        return np.array(data), np.array(targets)
    
    for node in graph.nodes:
        if graph.nodes[node].get('node_role') == 'channel' and node == target_channel_id:
            for i in range(start_idx, min(end_idx, len(temporal_data[node]))):
                data.append(temporal_data[node][i : i + 1])
                targets.append(targets_array[i])

    if not data:
        logging.warning(f"No data prepared for target channel ID {target_channel_id}.")

    logging.info(f"Prepared {len(data)} data samples and {len(targets)} target samples.")

    return np.array(data), np.array(targets)