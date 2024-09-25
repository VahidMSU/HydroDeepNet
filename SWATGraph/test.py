import torch
from libs.HeteroGen import generate_hetero_data, test_hetero_data
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from torch.nn import ModuleList, LSTM, ModuleDict

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') 

class HetGNN(torch.nn.Module):
    def __init__(self, in_channels_dict, hidden_channels, out_channels, lstm_hidden_dim, lstm_num_layers):
        super(HetGNN, self).__init__()
        logging.info(f"Input channels dictionary: {in_channels_dict}")
        # LSTM for channel time series
        self.lstm = LSTM(input_size=6, hidden_size=lstm_hidden_dim, num_layers=lstm_num_layers, batch_first=True)

        # Linear layers to ensure feature dimensions match before HeteroConv
        self.linear_proj = ModuleDict()
        for node_type, in_channels in in_channels_dict.items():
            if node_type == 'channel':
                self.linear_proj[node_type] = Linear(in_channels + lstm_hidden_dim, hidden_channels)
            else:
                self.linear_proj[node_type] = Linear(in_channels, hidden_channels)

        # Define a dictionary to store the convolution layers for each edge type
        self.convs = ModuleList()
        for _ in in_channels_dict.keys():
            conv = SAGEConv(hidden_channels, hidden_channels)
            self.convs.append(conv)

        self.hetero_conv = HeteroConv(dict(zip(in_channels_dict.keys(), self.convs)), aggr='sum')

        # Linear layer for output
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # Apply LSTM to channel time series data
        logging.info(f"Channel time series shape: {x_dict['channel_time_series'].shape}")
        lstm_out, _ = self.lstm(x_dict['channel_time_series'])
        lstm_out = lstm_out[:, -1, :]  # Get the last output of the LSTM

        # Concatenate LSTM output with other features
        x_dict['channel'] = torch.cat([x_dict['channel'], lstm_out], dim=1)
        logging.info(f"Concatenated channel features shape: {x_dict['channel'].shape}") 
        # Project node features to consistent dimensions
        for node_type, x in x_dict.items():
            if node_type != 'channel_time_series':  # Skip non-node type keys
                x_dict[node_type] = self.linear_proj[node_type](x)

        # Apply hetero convolution
        x_dict = self.hetero_conv(x_dict, edge_index_dict)

        # Apply linear layer to each node type's features
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin(x).relu()

        return x_dict

class HetGNNTrainer:
    def __init__(self, name, base_path, hidden_channels, out_channels, lstm_hidden_dim, lstm_num_layers, lr=0.01, num_epochs=100):
        self.name = name
        self.base_path = base_path
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.lr = lr
        self.num_epochs = num_epochs
        
        self.data = self.load_data()
        self.model = self.build_model()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Define masks for training and validation
        self.train_mask, self.val_mask = self.split_data()

    def load_data(self):
        data = generate_hetero_data(self.name, self.base_path)
        test_hetero_data(data)
        return data

    def build_model(self):
        in_channels_dict = {
            'weather_station': 2,
            'hru': 4,
            'gw_cell': 2,
            'channel': 2  # This will be adjusted after LSTM
        }

        edge_types = [
            ('weather_station', 'climate', 'hru'),
            ('hru', 'sw_gw', 'gw_cell'),
            ('hru', 'sw', 'channel'),
            ('channel', 'gw_sw', 'gw_cell'),
            ('channel', 'hydrograph', 'channel'),
            ('hru', 'climate', 'weather_station'),
            ('gw_cell', 'gw_sw', 'channel')

        ]

        return HetGNN(in_channels_dict, self.hidden_channels, self.out_channels, self.lstm_hidden_dim, self.lstm_num_layers)

    def prepare_data(self):
        x_dict = {
            'weather_station': self.data['weather_station'].x,
            'hru': self.data['hru'].x,
            'gw_cell': self.data['gw_cell'].x,
            'channel': self.data['channel'].x,
            'channel_time_series': self.data['channel'].time_series  # Add time series for channel
        }

        edge_index_dict = {
            ('weather_station', 'climate', 'hru'): self.data['weather_station', 'climate', 'hru'].edge_index,
            ('hru', 'sw_gw', 'gw_cell'): self.data['hru', 'sw_gw', 'gw_cell'].edge_index,
            ('hru', 'sw', 'channel'): self.data['hru', 'sw', 'channel'].edge_index,
            ('channel', 'gw_sw', 'gw_cell'): self.data['channel', 'gw_sw', 'gw_cell'].edge_index,
            ('channel', 'hydrograph', 'channel'): self.data['channel', 'hydrograph', 'channel'].edge_index,
            ('hru', 'climate', 'weather_station'): self.data['hru', 'climate', 'weather_station'].edge_index,
            ('gw_cell', 'gw_sw', 'channel'): self.data['gw_cell', 'gw_sw', 'channel'].edge_index
        }
        return x_dict, edge_index_dict

    def split_data(self):
        target_node_type = 'hru'
        num_nodes = self.data[target_node_type].num_nodes
        train_mask = torch.rand(num_nodes) < 0.8
        val_mask = ~train_mask
        return train_mask, val_mask

    def train(self):
        target_node_type = 'hru'
        target_feature_index = 0  # Index of the target feature in the node feature matrix
        targets = self.data[target_node_type].x[:, target_feature_index]

        x_dict, edge_index_dict = self.prepare_data()

        for epoch in range(self.num_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            
            # Forward pass
            out = self.model(x_dict, edge_index_dict)[target_node_type]
            
            # Compute loss on the training set
            loss = self.criterion(out[self.train_mask, target_feature_index], targets[self.train_mask])
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_out = self.model(x_dict, edge_index_dict)[target_node_type]
                val_loss = self.criterion(val_out[self.val_mask, target_feature_index], targets[self.val_mask])
            
            print(f'Epoch: {epoch + 1}, Loss: {loss.item()}, Val Loss: {val_loss.item()}')

        print("Training complete!")

# Example usage
name = "04115000"
swat_output_base_path = "/data/MyDataBase/SWATplus_by_VPUID/0000/huc12"
hidden_channels = 16
out_channels = 8
lstm_hidden_dim = 32
lstm_num_layers = 2

trainer = HetGNNTrainer(name, swat_output_base_path, hidden_channels, out_channels, lstm_hidden_dim, lstm_num_layers)
trainer.train()
