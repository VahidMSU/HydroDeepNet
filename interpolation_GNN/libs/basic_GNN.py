import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, GraphNorm
import torch_geometric.utils as pyg_utils
import torch.nn as nn


class GNNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_attr_dim, dropout=0.5):
        super(GNNModel, self).__init__()
        
        # Define the HeteroConv layer for different edge types
        self.conv1 = HeteroConv({
            ('centroid', 'connected_to', 'centroid'): SAGEConv(in_channels, hidden_channels)
        }, aggr='mean')

        self.conv2 = HeteroConv({
            ('centroid', 'connected_to', 'centroid'): SAGEConv(hidden_channels, hidden_channels)
        }, aggr='mean')
        
        # Add Graph Normalization
        self.gn1 = GraphNorm(hidden_channels)
        self.gn2 = GraphNorm(hidden_channels)
        
        # Dropout layer to prevent overfitting (optional, can uncomment to use)
        #self.dropout = torch.nn.Dropout(dropout)
        
        # Final linear layer to predict for each node
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        # Get the edge attributes for the specific edge type
        edge_attr = edge_attr_dict[('centroid', 'connected_to', 'centroid')] if edge_attr_dict else None
        
        # Add self-loops to the edge_index and edge_attr for 'centroid' nodes
        edge_index_dict[('centroid', 'connected_to', 'centroid')], edge_attr = pyg_utils.add_self_loops(
            edge_index_dict[('centroid', 'connected_to', 'centroid')],
            edge_attr=edge_attr
        )
        
        # Update the edge_attr_dict with the new edge attributes (with self-loops)
        if edge_attr_dict is not None:
            edge_attr_dict[('centroid', 'connected_to', 'centroid')] = edge_attr

        # First layer HeteroConv
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict['centroid'] = F.relu(x_dict['centroid'])  # Apply non-linearity (ReLU) after SAGEConv
        x_dict['centroid'] = F.relu(self.gn1(x_dict['centroid']))  # Apply GraphNorm
        
        # Second layer HeteroConv
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict['centroid'] = F.relu(x_dict['centroid'])  # Apply non-linearity (ReLU) after SAGEConv
        x_dict['centroid'] = F.relu(self.gn2(x_dict['centroid']))  # Apply GraphNorm again
        
        # Return node-level predictions
        return self.lin(x_dict['centroid'])  # We are interested in predictions for 'centroid' nodes




