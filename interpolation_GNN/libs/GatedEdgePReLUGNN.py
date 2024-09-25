import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, GraphNorm
import torch_geometric.utils as pyg_utils
import torch.nn as nn



class SAGEConvWithEdgeAttr(nn.Module):
    def __init__(self, in_channels, out_channels, edge_attr_dim):
        super(SAGEConvWithEdgeAttr, self).__init__()
        self.sage_conv = SAGEConv(in_channels, out_channels)
        self.edge_transform = nn.Linear(edge_attr_dim, out_channels)
        self.gate = nn.Linear(out_channels * 2, out_channels)
        self.norm = nn.BatchNorm1d(out_channels)
        self.residual = nn.Identity()

    def forward(self, x, edge_index, edge_attr):
        # Apply transformation to edge attributes
        edge_attr_transformed = self.edge_transform(edge_attr)

        # Perform the SAGE convolution
        out = self.sage_conv(x, edge_index)

        # Add the edge attribute influence to the destination nodes
        row, col = edge_index
        combined_features = torch.cat([out[col], edge_attr_transformed], dim=-1)
        
        # Compute gating values
        gate_values = torch.sigmoid(self.gate(combined_features))
        
        # Weight the node features by the gate values
        edge_contributions = gate_values * edge_attr_transformed
        out = out + torch.zeros_like(out).scatter_(0, col.unsqueeze(-1).expand_as(edge_contributions), edge_contributions)
        
        # Apply normalization and residual connection
        out = self.norm(out)
        out = self.residual(out) + out

        return F.relu(out)


class GatedEdgePReLUGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_attr_dim, dropout=0.5):
        super(GatedEdgePReLUGNN, self).__init__()
        
        # Define the HeteroConv layer for different edge types using SAGEConvWithEdgeAttr
        self.conv1 = HeteroConv({
            ('centroid', 'connected_to', 'centroid'): SAGEConvWithEdgeAttr(in_channels, hidden_channels, edge_attr_dim)
        }, aggr='mean')

        self.conv2 = HeteroConv({
            ('centroid', 'connected_to', 'centroid'): SAGEConvWithEdgeAttr(hidden_channels, hidden_channels, edge_attr_dim)
        }, aggr='mean')
        
        # Add Graph Normalization
        self.gn1 = GraphNorm(hidden_channels)
        self.gn2 = GraphNorm(hidden_channels)
        
        # Dropout layer to prevent overfitting (optional, can uncomment to use)
        # self.dropout = torch.nn.Dropout(dropout)
        
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

        # First layer HeteroConv with edge attributes
        x_dict = self.conv1(x_dict, edge_index_dict, edge_attr_dict)
        x_dict['centroid'] = F.relu(x_dict['centroid'])  # Apply non-linearity (ReLU) after SAGEConvWithEdgeAttr
        x_dict['centroid'] = F.relu(self.gn1(x_dict['centroid']))  # Apply GraphNorm
        
        # Second layer HeteroConv with edge attributes
        x_dict = self.conv2(x_dict, edge_index_dict, edge_attr_dict)
        x_dict['centroid'] = F.relu(x_dict['centroid'])  # Apply non-linearity (ReLU) after SAGEConvWithEdgeAttr
        x_dict['centroid'] = F.relu(self.gn2(x_dict['centroid']))  # Apply GraphNorm again
        
        # Return node-level predictions
        return self.lin(x_dict['centroid'])  # We are interested in predictions for 'centroid' nodes
