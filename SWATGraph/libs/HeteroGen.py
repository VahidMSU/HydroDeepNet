import os
import torch
import numpy as np
from torch_geometric.data import HeteroData
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
from libs.load_climate import load_climate_datas
import pandas as pd
def test_hetero_data(data):
    try:
        # Check if all node types have features
        for node_type in data.node_types:
            if 'x' not in data[node_type]:
                raise ValueError(f"Node type '{node_type}' does not have features (x).")

        # Check if all edge types have edge_index
        for edge_type in data.edge_types:
            if 'edge_index' not in data[edge_type]:
                raise ValueError(f"Edge type '{edge_type}' does not have 'edge_index'.")

        # Verify that edge indices reference valid nodes
        for edge_type in data.edge_types:
            src_type, _, dst_type = edge_type
            src_nodes = data[edge_type].edge_index[0].unique()
            dst_nodes = data[edge_type].edge_index[1].unique()
            if src_nodes.max() >= data[src_type].num_nodes:
                raise ValueError(f"Edge type '{edge_type}' has source nodes out of range.")
            if dst_nodes.max() >= data[dst_type].num_nodes:
                raise ValueError(f"Edge type '{edge_type}' has destination nodes out of range.")
        # Check there is no NaN in the data
        for node_type in data.node_types:
            if torch.isnan(data[node_type].x).any():
                raise ValueError(f"Node type '{node_type}' has NaN values in features.")
            else:
                print(f"Node type '{node_type}' has no NaN values in features.")
        ####### 
        print("All tests passed. The HeteroData is suitable for a HeteroGNN.")
    except ValueError as e:
        print(f"Test failed: {e}")


def generate_hetero_data(name, base_path):
    df = pd.read_csv(f'{base_path}/{name}/Graphs/cell_hru_riv_wst.csv')
    df = df[df.hru_elev > 0].reset_index(drop=True)

    df['wst_lat'] = [int(x.split('n')[0][1:]) for x in df.wst.values]
    df['wst_lon'] = [int(x.split('n')[1][:-1]) for x in df.wst.values]
    df['wst_id'] = pd.factorize(df['wst'])[0]
    

    df = df.dropna(subset=['wst_id', 'hru_id', 'cell_id', 'linkno', 'dslinkno'])

    df['wst_id'] = df['wst_id'].astype(int)
    df['hru_id'] = df['hru_id'].astype(int)
    df['cell_id'] = df['cell_id'].astype(int)
    df['linkno'] = df['linkno'].astype(int)
    df['dslinkno'] = df['dslinkno'].astype(int)

    # Create mappings to ensure indices are continuous
    wst_mapping = {index: i for i, index in enumerate(df['wst_id'].unique())}
    hru_mapping = {index: i for i, index in enumerate(df['hru_id'].unique())}
    cell_mapping = {index: i for i, index in enumerate(df['cell_id'].unique())}
    link_mapping = {index: i for i, index in enumerate(df['linkno'].unique())}
    dslink_mapping = {index: i for i, index in enumerate(df['dslinkno'].unique())}

    # Map the indices
    df['wst_id'] = df['wst_id'].map(wst_mapping)
    df['hru_id'] = df['hru_id'].map(hru_mapping)
    df['cell_id'] = df['cell_id'].map(cell_mapping)
    df['linkno'] = df['linkno'].map(link_mapping)
    df['dslinkno'] = df['dslinkno'].map(dslink_mapping)

    data = HeteroData()

    # Adding time series data for weather stations
    weather_station_features = df[['wst_id', 'wst_lat', 'wst_lon']].drop_duplicates().set_index('wst_id')
    txtintout_path = f'{base_path}/{name}/SWAT_gwflow_MODEL/Scenarios/Default/TxtInOut'
    weather_station_time_series = []
    
    for wst in df['wst'].unique():
        ts_data = load_climate_datas(txtintout_path, wst, start_year=2000, end_year=2020)
        weather_station_time_series.append(ts_data)

    # Stack the time series data
    weather_station_time_series = np.stack(weather_station_time_series)

    data['weather_station'].x = torch.tensor(weather_station_features.values, dtype=torch.float)
    data['weather_station'].time_series = torch.tensor(weather_station_time_series, dtype=torch.float)

    hru_features = df[['hru_id', 'hru_elev', 'hru_area', 'hru_lat', 'hru_lon']].drop_duplicates().set_index('hru_id')
    data['hru'].x = torch.tensor(hru_features.values, dtype=torch.float)

    gw_cell_features = df[['cell_id', 'cell_area', 'overlap_area_m2']].drop_duplicates().set_index('cell_id')
    data['gw_cell'].x = torch.tensor(gw_cell_features.values, dtype=torch.float)

    channel_features = df[['linkno', 'channel_length_m', 'drop']].drop_duplicates().set_index('linkno')
    data['channel'].x = torch.tensor(channel_features.values, dtype=torch.float)

    # Ensure consistent time series length between weather stations and channels
    min_length = 7671
    data['weather_station'].time_series = data['weather_station'].time_series[:, :min_length, :]

    # Assuming channels also have time series data
    channel_time_series = np.random.rand(data['channel'].x.shape[0], min_length, 1)  # Dummy data
    data['channel'].time_series = torch.tensor(channel_time_series, dtype=torch.float)

    # Filter out invalid edge indices
    valid_edges = df.dropna(subset=['wst_id', 'hru_id', 'cell_id', 'linkno', 'dslinkno'])

    # Create edges using unique values only
    weather_station_hru_edges = valid_edges[['wst_id', 'hru_id']].drop_duplicates().values.T
    hru_gw_cell_edges = valid_edges[['hru_id', 'cell_id']].drop_duplicates().values.T
    hru_channel_edges = valid_edges[['hru_id', 'linkno']].drop_duplicates().values.T
    channel_gw_cell_edges = valid_edges[['linkno', 'cell_id']].drop_duplicates().values.T
    channel_channel_edges = valid_edges[['linkno', 'dslinkno']].drop_duplicates().values.T
    weather_station_hru_edges_flipped = np.flip(weather_station_hru_edges, axis=0).copy()
    channel_gw_cell_edges_flipped = np.flip(channel_gw_cell_edges, axis=0).copy()


    # Add assertions to check the indices
    assert max(weather_station_hru_edges[0]) < len(data['weather_station']['x'])
    assert max(weather_station_hru_edges[1]) < len(data['hru']['x'])
    assert max(hru_gw_cell_edges[0]) < len(data['hru']['x'])
    assert max(hru_gw_cell_edges[1]) < len(data['gw_cell']['x'])
    assert max(hru_channel_edges[0]) < len(data['hru']['x'])
    assert max(hru_channel_edges[1]) < len(data['channel']['x'])
    assert max(channel_gw_cell_edges[0]) < len(data['channel']['x'])
    assert max(channel_gw_cell_edges[1]) < len(data['gw_cell']['x'])
    assert max(channel_channel_edges[0]) < len(data['channel']['x'])
    assert max(channel_channel_edges[1]) < len(data['channel']['x'])

    data['weather_station', 'climate', 'hru'].edge_index = torch.tensor(weather_station_hru_edges, dtype=torch.long)
    data['hru', 'sw_gw', 'gw_cell'].edge_index = torch.tensor(hru_gw_cell_edges, dtype=torch.long)
    data['hru', 'sw', 'channel'].edge_index = torch.tensor(hru_channel_edges, dtype=torch.long)
    data['channel', 'gw_sw', 'gw_cell'].edge_index = torch.tensor(channel_gw_cell_edges, dtype=torch.long)
    data['channel', 'hydrograph', 'channel'].edge_index = torch.tensor(channel_channel_edges, dtype=torch.long)
    data['hru', 'climate', 'weather_station'].edge_index = torch.tensor(weather_station_hru_edges_flipped, dtype=torch.long)
    data['gw_cell', 'gw_sw', 'channel'].edge_index = torch.tensor(channel_gw_cell_edges_flipped, dtype=torch.long)
    
    print(data)
    return data


if __name__ == "__main__":
    name = "04164300"
    swat_output_base_path = "/data/MyDataBase/SWATplus_by_VPUID/0000/huc12"

    data = generate_hetero_data(name, swat_output_base_path)
    test_hetero_data(data)

