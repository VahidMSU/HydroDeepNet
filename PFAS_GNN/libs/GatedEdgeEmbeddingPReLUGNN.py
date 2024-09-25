import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv
from torch.nn import BatchNorm1d, LayerNorm




class SAGEConvWithEdgeAttrAndEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, edge_attr_dim, edge_embedding_dim=16):
        super(SAGEConvWithEdgeAttrAndEmbedding, self).__init__()
        self.sage_conv = SAGEConv(in_channels, out_channels)
        
        # Edge embedding layer
        self.edge_embedding = nn.Linear(edge_attr_dim, edge_embedding_dim)
        
        # Transform edge embeddings to match the output dimension
        self.edge_transform_emb = nn.Linear(edge_embedding_dim, out_channels)
        
        # Direct edge attribute transformation
        self.edge_transform_attr = nn.Linear(edge_attr_dim, out_channels)
        
        self.gate = nn.Linear(out_channels * 3, out_channels)  # Update to reflect combined feature size
        self.norm = nn.BatchNorm1d(out_channels)
        self.residual = nn.Identity()

    def forward(self, x, edge_index, edge_attr):
        # Compute edge embeddings
        edge_embeddings = F.relu(self.edge_embedding(edge_attr))
        
        # Transform edge embeddings to match the output dimension
        edge_attr_transformed_emb = self.edge_transform_emb(edge_embeddings)

        # Directly transform edge attributes
        edge_attr_transformed_attr = self.edge_transform_attr(edge_attr)
        
        # Combine both transformed edge features
        edge_attr_transformed = edge_attr_transformed_emb + edge_attr_transformed_attr

        # Perform the SAGE convolution
        out = self.sage_conv(x, edge_index)

        # Add the edge attribute influence to the destination nodes
        row, col = edge_index
        combined_features = torch.cat([out[col], edge_attr_transformed, edge_attr_transformed_attr], dim=-1)
        
        # Compute gating values
        gate_values = torch.sigmoid(self.gate(combined_features))
        
        # Weight the node features by the gate values
        edge_contributions = gate_values * edge_attr_transformed
        out = out + torch.zeros_like(out).scatter_(0, col.unsqueeze(-1).expand_as(edge_contributions), edge_contributions)
        
        # Apply normalization and residual connection
        out = self.norm(out)
        out = self.residual(out) + out

        return F.relu(out)

class GatedEdgeEmbeddingPReLUGNN(nn.Module):
    def __init__(self, in_channels_dict, out_channels, aggregation, edge_attr_dict, edge_embedding_dim=16, reduced_dim=32, use_batchnorm=True):
        super(GatedEdgeEmbeddingPReLUGNN, self).__init__()
        
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
        gw_edge_attr_dim_pg = edge_attr_dict[('pfas_sites', 'dis_edge', 'gw_wells')].shape[1]
        
        self.edge_reduce_dim = nn.ModuleDict({
            str(('pfas_sites', 'dis_edge', 'gw_wells')): nn.Linear(gw_edge_attr_dim_pg, reduced_dim),
            str(('pfas_sites', 'dis_edge', 'sw_stations')): nn.Linear(gw_edge_attr_dim_pg, reduced_dim),

            str(('sw_stations', 'dis_edge', 'pfas_sites')): nn.Linear(gw_edge_attr_dim_pg, reduced_dim),
            str(('sw_stations', 'dis_edge', 'gw_wells')): nn.Linear(gw_edge_attr_dim_pg, reduced_dim),

            str(('gw_wells', 'dis_edge', 'pfas_sites')): nn.Linear(gw_edge_attr_dim_pg, reduced_dim),
            str(('gw_wells', 'dis_edge', 'sw_stations')): nn.Linear(gw_edge_attr_dim_pg, reduced_dim),
            str(('gw_wells', 'dis_edge', 'gw_wells')): nn.Linear(gw_edge_attr_dim_pg, reduced_dim),
        })
        
        # Normalization layers for edge attributes
        self.edge_norm = nn.ModuleDict({
            edge_type_str: BatchNorm1d(reduced_dim) if self.use_batchnorm else LayerNorm(reduced_dim)
            for edge_type_str in self.edge_reduce_dim.keys()
        })
        
        # First convolutional layer with edge attributes and embeddings
        self.conv1 = HeteroConv({
            ('pfas_sites', 'dis_edge', 'gw_wells'): SAGEConvWithEdgeAttrAndEmbedding(reduced_dim, out_channels, reduced_dim, edge_embedding_dim),
            ('pfas_sites', 'dis_edge', 'sw_stations'): SAGEConvWithEdgeAttrAndEmbedding(reduced_dim, out_channels, reduced_dim, edge_embedding_dim),
            ('sw_stations', 'dis_edge', 'pfas_sites'): SAGEConvWithEdgeAttrAndEmbedding(reduced_dim, out_channels, reduced_dim, edge_embedding_dim),
            ('sw_stations', 'dis_edge', 'gw_wells'): SAGEConvWithEdgeAttrAndEmbedding(reduced_dim, out_channels, reduced_dim, edge_embedding_dim),
            ('gw_wells', 'dis_edge', 'sw_stations'): SAGEConvWithEdgeAttrAndEmbedding(reduced_dim, out_channels, reduced_dim, edge_embedding_dim),
            ('gw_wells', 'dis_edge', 'gw_wells'): SAGEConvWithEdgeAttrAndEmbedding(reduced_dim, out_channels, reduced_dim, edge_embedding_dim),
            ('gw_wells', 'dis_edge', 'pfas_sites'): SAGEConvWithEdgeAttrAndEmbedding(reduced_dim, out_channels, reduced_dim, edge_embedding_dim),
            ('gw_wells', 'self_loop', 'gw_wells'): SAGEConv(reduced_dim, out_channels),
            ('sw_stations', 'self_loop', 'sw_stations'): SAGEConv(reduced_dim, out_channels),
            ('pfas_sites', 'self_loop', 'pfas_sites'): SAGEConv(reduced_dim, out_channels), 
        }, aggr=aggregation)

        # Second convolutional layer with edge attributes and embeddings (same structure as the first)
        self.conv2 = HeteroConv({
            ('pfas_sites', 'dis_edge', 'gw_wells'): SAGEConvWithEdgeAttrAndEmbedding(out_channels, out_channels, reduced_dim, edge_embedding_dim),
            ('pfas_sites', 'dis_edge', 'sw_stations'): SAGEConvWithEdgeAttrAndEmbedding(out_channels, out_channels, reduced_dim, edge_embedding_dim),
            ('sw_stations', 'dis_edge', 'pfas_sites'): SAGEConvWithEdgeAttrAndEmbedding(out_channels, out_channels, reduced_dim, edge_embedding_dim),
            ('sw_stations', 'dis_edge', 'gw_wells'): SAGEConvWithEdgeAttrAndEmbedding(out_channels, out_channels, reduced_dim, edge_embedding_dim),
            ('gw_wells', 'dis_edge', 'pfas_sites'): SAGEConvWithEdgeAttrAndEmbedding(out_channels, out_channels, reduced_dim, edge_embedding_dim),
            ('gw_wells', 'dis_edge', 'sw_stations'): SAGEConvWithEdgeAttrAndEmbedding(out_channels, out_channels, reduced_dim, edge_embedding_dim),
            ('gw_wells', 'dis_edge', 'gw_wells'): SAGEConvWithEdgeAttrAndEmbedding(out_channels, out_channels, reduced_dim, edge_embedding_dim),
            ('gw_wells', 'self_loop', 'gw_wells'): SAGEConv(out_channels, out_channels),
            ('sw_stations', 'self_loop', 'sw_stations'): SAGEConv(out_channels, out_channels),
            ('pfas_sites', 'self_loop', 'pfas_sites'): SAGEConv(out_channels, out_channels),
        }, aggr=aggregation)
        
        # Linear layers and PReLU activation
        self.linear = nn.Linear(out_channels, 1)
        self.prelu = nn.PReLU()
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # Filter out 'rivs' node type
        x_dict = {node_type: self.node_reduce_dim[node_type](x) for node_type, x in x_dict.items() if node_type != 'rivs'}
        
        # Apply normalization after dimensionality reduction for node features
        x_dict = {node_type: F.relu(self.node_norm[node_type](x)) for node_type, x in x_dict.items() if node_type != 'rivs'}
        
        # Dimensionality reduction and normalization for edge attributes, filtering out 'rivs'
        edge_attr_dict = {
            edge_type: F.relu(self.edge_norm[str(edge_type)](self.edge_reduce_dim[str(edge_type)](edge_attr))) 
            for edge_type, edge_attr in edge_attr_dict.items() if edge_type != ('rivs', 'dis_edge', 'rivs')
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

