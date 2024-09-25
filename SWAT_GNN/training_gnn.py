import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv, global_mean_pool
from torch_geometric.data import DataLoader, Dataset
import torch.optim as optim
import os
import logging
from libs.HeteroData_generation import SWATGraphProcessor, generate_relation_table

class GraphLevelGNN(nn.Module):
    def __init__(self, in_channels_dict, gnn_hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super(GraphLevelGNN, self).__init__()
        
        # Define GNN layers with HeteroConv
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('hru', 'sw_gw', 'gw_cell'): SAGEConv((-1, -1), gnn_hidden_dim),
                ('channel', 'sw', 'hru'): SAGEConv((-1, -1), gnn_hidden_dim),
                ('gw_cell', 'gw_sw', 'channel'): SAGEConv((-1, -1), gnn_hidden_dim),
                ('hru', 'sw', 'channel'): SAGEConv((-1, -1), gnn_hidden_dim),
                ('channel', 'hydrograph', 'channel'): SAGEConv((-1, -1), gnn_hidden_dim),
            }, aggr='sum')
            self.convs.append(conv)

        # Define the fully connected layers
        self.fc1 = nn.Linear(gnn_hidden_dim, gnn_hidden_dim // 2)
        self.fc2 = nn.Linear(gnn_hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict

        # Debugging: Check the contents of x_dict and edge_index_dict
        print("x_dict:", {k: (v.shape if v is not None else None) for k, v in x_dict.items()})
        print("edge_index_dict:", {k: (v.shape if v is not None else None) for k, v in edge_index_dict.items()})

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # Pooling operation to get graph-level representation
        x = global_mean_pool(x_dict['channel'], data.batch)  # assuming 'channel' nodes are the key for aggregation

        # Final prediction
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc2(x)

        return out


# Custom dataset for graph-level predictions
class GraphLevelDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data  # Assuming this is a single HeteroData object
        self.targets = targets  # The target performance metrics for each graph

    def __len__(self):
        return 1  # There's only one graph in the dataset

    def __getitem__(self, idx):
        # Ignore idx since we have only one graph
        return self.data, self.targets[0]  # Return the single graph and its target

def train_model(data, targets, in_channels_dict, epochs=100, batch_size=1, learning_rate=0.001):
    # Define the dataset and data loaders
    dataset = GraphLevelDataset(data, targets)  # data is your HeteroData graph object
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)  # No need to shuffle a single graph

    model = GraphLevelGNN(in_channels_dict, gnn_hidden_dim=64, output_dim=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_data, target in train_loader:
            batch_data = batch_data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(batch_data)
            loss = criterion(output.squeeze(), target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')


if __name__ == "__main__":
    swat_output_base_path = "/data/MyDataBase/SWATplus_by_VPUID/0000/huc12"
    processor = SWATGraphProcessor(swat_output_base_path)
    names = os.listdir(swat_output_base_path)
    names.remove("log.txt")

    name = "04115000"
    print(f"Processing {name}")
    logger_path = os.path.join(swat_output_base_path, name, "Graphs/log.txt")
    generate_relation_table(name, logger_path)
    data = processor.process(name)

    # Define the in_channels_dict based on your graph's node features
    in_channels_dict = {
        'hru': 2,
        'channel': 2,
        'gw_cell': 2,
    }

    # Example target values (e.g., NSE, MPE, etc.) - replace with actual targets
    targets = torch.tensor([1.2, 3.5, 4.6])  # Replace with actual target values

    # Train the model
    train_model(data, targets, in_channels_dict)
