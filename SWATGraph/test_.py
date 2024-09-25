import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.data import HeteroData
from torch_geometric_temporal.nn.recurrent import TGCN

# Generate random temporal heterogeneous graph data
def generate_random_temporal_hetero_data(num_nodes, num_edges, num_timesteps, num_features):
    data = HeteroData()
    
    # Randomly generate node features
    data['weather_station'].x = torch.randn(num_nodes['weather_station'], num_features)
    data['hru'].x = torch.randn(num_nodes['hru'], num_features)
    data['gw_cell'].x = torch.randn(num_nodes['gw_cell'], num_features)
    data['channel'].x = torch.randn(num_nodes['channel'], num_features)

    # Randomly generate time series data
    data['weather_station'].time_series = torch.randn(num_nodes['weather_station'], num_timesteps, num_features)
    data['channel'].time_series = torch.randn(num_nodes['channel'], num_timesteps, num_features)

    # Randomly generate edge indices
    data['weather_station', 'climate', 'hru'].edge_index = torch.randint(0, num_nodes['weather_station'], (2, num_edges['weather_station_hru']))
    data['hru', 'sw_gw', 'gw_cell'].edge_index = torch.randint(0, num_nodes['hru'], (2, num_edges['hru_gw_cell']))
    data['hru', 'sw', 'channel'].edge_index = torch.randint(0, num_nodes['hru'], (2, num_edges['hru_channel']))
    data['channel', 'gw_sw', 'gw_cell'].edge_index = torch.randint(0, num_nodes['channel'], (2, num_edges['channel_gw_cell']))
    data['channel', 'hydrograph', 'channel'].edge_index = torch.randint(0, num_nodes['channel'], (2, num_edges['channel_channel']))
    data['hru', 'climate', 'weather_station'].edge_index = torch.randint(0, num_nodes['hru'], (2, num_edges['hru_weather_station']))
    data['gw_cell', 'gw_sw', 'channel'].edge_index = torch.randint(0, num_nodes['gw_cell'], (2, num_edges['gw_cell_channel']))

    return data

class TemporalHeteroGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(TemporalHeteroGNN, self).__init__()
        self.tgcn = TGCN(in_channels, hidden_channels, out_channels)
        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        
        self.conv1 = HeteroConv({
            ('weather_station', 'climate', 'hru'): SAGEConv(hidden_channels + in_channels, hidden_channels),
            ('hru', 'sw_gw', 'gw_cell'): SAGEConv(hidden_channels + in_channels, hidden_channels),
            ('hru', 'sw', 'channel'): SAGEConv(hidden_channels + in_channels, hidden_channels),
            ('channel', 'gw_sw', 'gw_cell'): SAGEConv(hidden_channels + in_channels, hidden_channels),
            ('channel', 'hydrograph', 'channel'): SAGEConv(hidden_channels + in_channels, hidden_channels),
            ('hru', 'climate', 'weather_station'): SAGEConv(hidden_channels + in_channels, hidden_channels),
            ('gw_cell', 'gw_sw', 'channel'): SAGEConv(hidden_channels + in_channels, hidden_channels)
        }, aggr='sum')

        self.conv2 = HeteroConv({
            ('weather_station', 'climate', 'hru'): SAGEConv(hidden_channels, out_channels),
            ('hru', 'sw_gw', 'gw_cell'): SAGEConv(hidden_channels, out_channels),
            ('hru', 'sw', 'channel'): SAGEConv(hidden_channels, out_channels),
            ('channel', 'gw_sw', 'gw_cell'): SAGEConv(hidden_channels, out_channels),
            ('channel', 'hydrograph', 'channel'): SAGEConv(hidden_channels, out_channels),
            ('hru', 'climate', 'weather_station'): SAGEConv(hidden_channels, out_channels),
            ('gw_cell', 'gw_sw', 'channel'): SAGEConv(hidden_channels, out_channels)
        }, aggr='sum')

    def forward(self, x_dict, time_series_dict, edge_index_dict):
        h_dict = {}
        for node_type, ts in time_series_dict.items():
            h = self.tgcn(ts, edge_index_dict)
            h_dict[node_type] = h

        for node_type, x in x_dict.items():
            if node_type in h_dict:
                h_dict[node_type] = torch.cat([x, h_dict[node_type]], dim=-1)
            else:
                h_dict[node_type] = x

        for node_type, h in h_dict.items():
            expected_dim = self.hidden_channels + self.in_channels
            if h.shape[1] != expected_dim:
                padding_size = expected_dim - h.shape[1]
                h_dict[node_type] = torch.cat([h, torch.zeros(h.shape[0], padding_size, device=h.device)], dim=1)

        h_dict = self.conv1(h_dict, edge_index_dict)
        h_dict = {key: F.relu(h) for key, h in h_dict.items()}
        h_dict = self.conv2(h_dict, edge_index_dict)

        return h_dict

if __name__ == "__main__":
    num_nodes = {
        'weather_station': 40,
        'hru': 1551,
        'gw_cell': 4735,
        'channel': 1059
    }
    
    num_edges = {
        'weather_station_hru': 1551,
        'hru_gw_cell': 4775,
        'hru_channel': 3002,
        'channel_gw_cell': 1084,
        'channel_channel': 137,
        'hru_weather_station': 1551,
        'gw_cell_channel': 1084
    }
    
    num_timesteps = 7671
    num_features = 6

    data = generate_random_temporal_hetero_data(num_nodes, num_edges, num_timesteps, num_features)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    time_series_dict = {
        'weather_station': data['weather_station'].time_series,
        'channel': data['channel'].time_series
    }

    print("Node feature shapes:")
    for key, value in data.x_dict.items():
        print(f"{key}: {value.shape}")
    print("Time series shapes:")
    for key, value in time_series_dict.items():
        print(f"{key}: {value.shape}")
    print("Edge index shapes:")
    for key, value in data.edge_index_dict.items():
        print(f"{key}: {value.shape}")

    streamflow = np.random.rand(num_timesteps)
    print(f"streamflow shape: {streamflow.shape}")

    out_channels = 1
    model = TemporalHeteroGNN(in_channels=num_features, hidden_channels=32, out_channels=out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(data.x_dict, time_series_dict, data.edge_index_dict)

        target = torch.tensor(streamflow, dtype=torch.float).unsqueeze(1).to(device)
        
        print(f"target shape: {target.shape}")
        print(f"target: {target}")
        
        loss = criterion(out['channel'], target)
        loss.backward()

        optimizer.step()
        
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
