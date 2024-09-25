import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.datasets import MovieLens
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.loader import DataLoader
import os

# Set CUDA launch blocking to help with debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# Load the MovieLens dataset
dataset = MovieLens('/tmp/MovieLens')
data = dataset[0]
print(data)

import time
time.sleep(5)

class MovieLensHetGNN(nn.Module):
    """
    A Heterogeneous Graph Neural Network for the MovieLens dataset.
    
    This model uses two layers of HeteroConv with SAGEConv to process
    user and movie node features along with the edges between them.
    
    Args:
        hidden_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, hidden_channels, out_channels):
        super(MovieLensHetGNN, self).__init__()
        self.conv1 = HeteroConv({
            ('user', 'rates', 'movie'): SAGEConv((-1, -1), hidden_channels),
            ('movie', 'rev_rates', 'user'): SAGEConv((-1, -1), hidden_channels)
        }, aggr='sum')

        self.conv2 = HeteroConv({
            ('user', 'rates', 'movie'): SAGEConv(hidden_channels, out_channels),
            ('movie', 'rev_rates', 'user'): SAGEConv(hidden_channels, out_channels)
        }, aggr='sum')

    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass of the model.
        
        Args:
            x_dict (dict): Dictionary of node features.
            edge_index_dict (dict): Dictionary of edge indices.
        
        Returns:
            dict: Dictionary of output node features.
        """
        h_dict = self.conv1(x_dict, edge_index_dict)
        h_dict = {key: F.relu(h) for key, h in h_dict.items()}
        h_dict = self.conv2(h_dict, edge_index_dict)
        return h_dict

if __name__ == "__main__":

    
    # Set the device to GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Example data creation (replace with your actual data loading process)
    data = HeteroData()
    data['user'].x = torch.randn((100, 16)).to(device)  # Example user node features
    data['movie'].x = torch.randn((100, 16)).to(device)  # Example movie node features
    data['user', 'rates', 'movie'].edge_index = torch.randint(0, 100, (2, 500)).to(device)  # Example edges
    data['movie', 'rev_rates', 'user'].edge_index = torch.randint(0, 100, (2, 500)).to(device)  # Example edges

    # Initialize the model and move it to the appropriate device
    model = MovieLensHetGNN(hidden_channels=32, out_channels=16).to(device)

    # Perform a forward pass with the example data
    out = model(data.x_dict, data.edge_index_dict)
    print(out)

    # Define the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Dummy target data for the sake of demonstration
    target = torch.randn(data['movie'].num_nodes, 1).to(device)

    # Training loop
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        loss = criterion(out['movie'], target)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
