
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv
from torch.nn import BatchNorm1d, LayerNorm




class SAGEConvWithEdgeAttrAndAttention(nn.Module):
    def __init__(self, in_channels, out_channels, edge_attr_dim):
        super(SAGEConvWithEdgeAttrAndAttention, self).__init__()
        self.sage_conv = SAGEConv(in_channels, out_channels)

        # Attention mechanism for edge attributes
        self.attention = nn.Linear(out_channels + out_channels, 1)

        # Transformation of edge attributes
        self.edge_transform = nn.Linear(edge_attr_dim, out_channels)
        
        # Normalization layer
        self.norm = nn.BatchNorm1d(out_channels)
        self.residual = nn.Identity()

    def forward(self, x, edge_index, edge_attr):
        # Perform the SAGE convolution
        out = self.sage_conv(x, edge_index)

        # Transform edge attributes
        edge_attr_transformed = self.edge_transform(edge_attr)

        # Calculate attention scores
        row, col = edge_index
        combined_features = torch.cat([out[col], edge_attr_transformed], dim=-1)

        # Ensure the combined feature size is correct
        assert combined_features.shape[-1] == self.attention.in_features, \
            f"Expected input dimension for attention: {self.attention.in_features}, but got {combined_features.shape[-1]}"

        attention_weights = torch.sigmoid(self.attention(combined_features))

        # Apply attention to edge attributes
        edge_contributions = attention_weights * edge_attr_transformed
        out = out + torch.zeros_like(out).scatter_(0, col.unsqueeze(-1).expand_as(edge_contributions), edge_contributions)

        # Apply normalization and residual connection
        out = self.norm(out)
        out = self.residual(out) + out

        return F.relu(out)

class AttentionEdgePReLUGNN(nn.Module):
    def __init__(self, in_channels_dict, out_channels, aggregation, edge_attr_dict, reduced_dim=32, use_batchnorm=True):
        super(AttentionEdgePReLUGNN, self).__init__()
        
        # Choose normalization type
        self.use_batchnorm = use_batchnorm

        # Dimensionality reduction for node features
        self.node_reduce_dim = nn.ModuleDict({
            node_type: nn.Linear(in_channels, reduced_dim) for node_type, in_channels in in_channels_dict.items()
        })
        
        # Normalization layers for reduced node features
        self.node_norm = nn.ModuleDict({
            node_type: BatchNorm1d(reduced_dim) if self.use_batchnorm else LayerNorm(reduced_dim) 
            for node_type in in_channels_dict.keys()
        })
        
        # Dimensionality reduction and normalization for edge attributes
        edge_attr_dim = edge_attr_dict[('pfas_sites', 'dis_edge', 'gw_wells')].shape[1]
        
        self.edge_reduce_dim = nn.ModuleDict({
            str(('pfas_sites', 'dis_edge', 'gw_wells')): nn.Linear(edge_attr_dim, reduced_dim),
            str(('pfas_sites', 'dis_edge', 'sw_stations')): nn.Linear(edge_attr_dim, reduced_dim),

            str(('sw_stations', 'dis_edge', 'pfas_sites')): nn.Linear(edge_attr_dim, reduced_dim),
            str(('sw_stations', 'dis_edge', 'gw_wells')): nn.Linear(edge_attr_dim, reduced_dim),

            str(('gw_wells', 'dis_edge', 'gw_wells')): nn.Linear(edge_attr_dim, reduced_dim),

            str(('gw_wells', 'dis_edge', 'pfas_sites')): nn.Linear(edge_attr_dim, reduced_dim),
            str(('gw_wells', 'dis_edge', 'sw_stations')): nn.Linear(edge_attr_dim, reduced_dim),
            str(('gw_wells', 'dis_edge', 'gw_wells')): nn.Linear(edge_attr_dim, reduced_dim),
        })
        
        # Normalization layers for edge attributes
        self.edge_norm = nn.ModuleDict({
            edge_type_str: BatchNorm1d(reduced_dim) if self.use_batchnorm else LayerNorm(reduced_dim)
            for edge_type_str in self.edge_reduce_dim.keys()
        })
        
        # First convolutional layer with edge attributes and attention
        self.conv1 = HeteroConv({
            ('pfas_sites', 'dis_edge', 'gw_wells'): SAGEConvWithEdgeAttrAndAttention(reduced_dim, out_channels, reduced_dim),
            ('pfas_sites', 'dis_edge', 'sw_stations'): SAGEConvWithEdgeAttrAndAttention(reduced_dim, out_channels, reduced_dim),
            ('sw_stations', 'dis_edge', 'pfas_sites'): SAGEConvWithEdgeAttrAndAttention(reduced_dim, out_channels, reduced_dim),
            ('sw_stations', 'dis_edge', 'gw_wells'): SAGEConvWithEdgeAttrAndAttention(reduced_dim, out_channels, reduced_dim),
            ('gw_wells', 'dis_edge', 'sw_stations'): SAGEConvWithEdgeAttrAndAttention(reduced_dim, out_channels, reduced_dim),
            ('gw_wells', 'dis_edge', 'gw_wells'): SAGEConvWithEdgeAttrAndAttention(reduced_dim, out_channels, reduced_dim),
            ('gw_wells', 'dis_edge', 'pfas_sites'): SAGEConvWithEdgeAttrAndAttention(reduced_dim, out_channels, reduced_dim),
            ('gw_wells', 'self_loop', 'gw_wells'): SAGEConv(reduced_dim, out_channels),
            ('sw_stations', 'self_loop', 'sw_stations'): SAGEConv(reduced_dim, out_channels),
            ('pfas_sites', 'self_loop', 'pfas_sites'): SAGEConv(reduced_dim, out_channels), 
        }, aggr=aggregation)

        # Second convolutional layer with edge attributes and attention (same structure as the first)
        self.conv2 = HeteroConv({
            ('pfas_sites', 'dis_edge', 'gw_wells'): SAGEConvWithEdgeAttrAndAttention(out_channels, out_channels, reduced_dim),
            ('pfas_sites', 'dis_edge', 'sw_stations'): SAGEConvWithEdgeAttrAndAttention(out_channels, out_channels, reduced_dim),
            ('sw_stations', 'dis_edge', 'pfas_sites'): SAGEConvWithEdgeAttrAndAttention(out_channels, out_channels, reduced_dim),
            ('sw_stations', 'dis_edge', 'gw_wells'): SAGEConvWithEdgeAttrAndAttention(out_channels, out_channels, reduced_dim),
            ('gw_wells', 'dis_edge', 'pfas_sites'): SAGEConvWithEdgeAttrAndAttention(out_channels, out_channels, reduced_dim),
            ('gw_wells', 'dis_edge', 'sw_stations'): SAGEConvWithEdgeAttrAndAttention(out_channels, out_channels, reduced_dim),
            ('gw_wells', 'dis_edge', 'gw_wells'): SAGEConvWithEdgeAttrAndAttention(out_channels, out_channels, reduced_dim),
            ('gw_wells', 'self_loop', 'gw_wells'): SAGEConv(out_channels, out_channels),
            ('sw_stations', 'self_loop', 'sw_stations'): SAGEConv(out_channels, out_channels),
            ('pfas_sites', 'self_loop', 'pfas_sites'): SAGEConv(out_channels, out_channels),
        }, aggr=aggregation)
        
        # Linear layers and PReLU activation
        self.linear = nn.Linear(out_channels, 1)
        self.prelu = nn.PReLU()

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # Apply dimensionality reduction for node features (without normalization)
        x_dict = {node_type: self.node_reduce_dim[node_type](x) for node_type, x in x_dict.items()}
        
        # Apply normalization after dimensionality reduction for node features
        x_dict = {node_type: F.relu(self.node_norm[node_type](x)) for node_type, x in x_dict.items()}
        
        # Dimensionality reduction and normalization for edge attributes
        edge_attr_dict = {
            edge_type: F.relu(self.edge_norm[str(edge_type)](self.edge_reduce_dim[str(edge_type)](edge_attr))) 
            for edge_type, edge_attr in edge_attr_dict.items()
        }
        
        # First convolutional layer
        x = self.conv1(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)

        # Apply activation after first layer
        x = {key: F.relu(x[key]) for key in x.keys()}

        # Second convolutional layer
        x = self.conv2(x, edge_index_dict, edge_attr_dict=edge_attr_dict)

        # Apply activation after second layer
        x = {key: F.relu(x[key]) for key in x.keys()}

        # Apply linear layer and PReLU activation
        x['gw_wells'] = self.linear(x['gw_wells'])
        x['gw_wells'] = self.prelu(x['gw_wells'])

        x['sw_stations'] = self.linear(x['sw_stations'])
        x['sw_stations'] = self.prelu(x['sw_stations'])

        return x



