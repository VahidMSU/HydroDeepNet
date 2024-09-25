import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool
from torch_geometric.utils import from_networkx
from libs.HeteroData_generation import SWATGraphProcessor, generate_relation_table

class GNNModel(nn.Module):
    def __init__(self, in_channels_dict, train_input_dim, out_channels, num_layers=2, hidden_channels=64):
        super(GNNModel, self).__init__()
        
        # Define GNN layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('hru', 'sw_gw', 'gw_cell'): SAGEConv(in_channels_dict['hru'], hidden_channels),
                ('channel', 'hydrograph', 'channel'): SAGEConv(in_channels_dict['channel'], hidden_channels),
                ('hru', 'sw', 'channel'): SAGEConv(in_channels_dict['hru'], hidden_channels),
                ('gw_cell', 'gw_sw', 'channel'): SAGEConv(in_channels_dict['gw_cell'], hidden_channels),
                ('hru', 'self_loop', 'hru'): SAGEConv(in_channels_dict['hru'], hidden_channels),
            }, aggr='mean')
            self.convs.append(conv)
        
        # Define the MLP layers
        self.fc1 = nn.Linear(hidden_channels + train_input_dim, 128)
        self.fc2 = nn.Linear(128, out_channels)
    def forward(self, data, train_data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict

        # Debugging: Print the contents of x_dict and edge_index_dict
        print("x_dict keys and shapes:")
        for key, value in x_dict.items():
            if value is not None:
                print(f"{key}: {value.shape}")
            else:
                print(f"{key}: None")

        print("edge_index_dict keys and shapes:")
        for key, value in edge_index_dict.items():
            if value is not None:
                print(f"{key}: {value.shape}")
            else:
                print(f"{key}: None")

        # Forward pass through GNN layers
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # Pooling operation (for graph-level prediction)
        if 'channel' in x_dict:
            x = global_mean_pool(x_dict['channel'], data.batch)  # assuming 'channel' is a key node type for prediction
        else:
            x = global_mean_pool(next(iter(x_dict.values())), data.batch)

        # Concatenate the pooled graph features with the SWAT parameters (train_data)
        x = torch.cat([x, train_data], dim=1)

        # Pass through MLP layers
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def main():
    name = "04115000"
    print(f"Processing {name}")

    # Define paths
    swat_output_base_path = "/data/MyDataBase/SWATplus_by_VPUID/0000/huc12"
    logger_path = os.path.join(swat_output_base_path, name, "Graphs/log.txt")
    scores_path = os.path.join(swat_output_base_path, name, "local_best_solution_SWAT_gwflow_MODEL.txt")
    
    # Generate relation table
    generate_relation_table(name, logger_path)
    
    # Process SWAT data to create graph
    processor = SWATGraphProcessor(swat_output_base_path)
    data = processor.process(name)
    
    # Load performance scores
    scores = pd.read_csv(scores_path, sep=",")
    target = torch.tensor(scores['best_score'].values, dtype=torch.float32).unsqueeze(1)  # Make sure target has the correct shape
    train = torch.tensor(scores.drop(columns=['best_score']).values, dtype=torch.float32)
    print(scores.head())
    print(f"scores columns: {scores.columns}")

    # Define model input dimensions
    in_channels_dict = {
        'hru': 2,
        'channel': 2,
        'gw_cell': 2,
    }
    train_input_dim = train.shape[1]

    # Initialize model, optimizer, and loss function
    model = GNNModel(in_channels_dict, train_input_dim, out_channels=1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Training loop
    epochs = 100
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data, train)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch}, Loss: {loss.item()}')

if __name__ == "__main__":
    main()
