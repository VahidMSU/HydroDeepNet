import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv



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


class GatedEdgePReLUGNN(nn.Module):
    def __init__(self, in_channels_dict, out_channels, aggregation, edge_attr_dict):
        super(GatedEdgePReLUGNN, self).__init__()
        
        # Automatically determine the dimension of edge attributes
        edge_attr_dim = edge_attr_dict[('pfas_sites', 'dis_edge', 'gw_wells')].shape[1]
        
        # First convolutional layer with edge attributes
        self.conv1 = HeteroConv({
            ('pfas_sites', 'dis_edge', 'gw_wells'): SAGEConvWithEdgeAttr(in_channels_dict['pfas_sites'], out_channels, edge_attr_dim),
            ('gw_wells', 'dis_edge', 'pfas_sites'): SAGEConvWithEdgeAttr(in_channels_dict['gw_wells'], out_channels, edge_attr_dim),
            ('pfas_sites', 'dis_edge', 'sw_stations'): SAGEConvWithEdgeAttr(in_channels_dict['pfas_sites'], out_channels, edge_attr_dim),
            ('sw_stations', 'dis_edge', 'pfas_sites'): SAGEConvWithEdgeAttr(in_channels_dict['sw_stations'], out_channels, edge_attr_dim),
            ('sw_stations', 'dis_edge', 'gw_wells'): SAGEConvWithEdgeAttr(in_channels_dict['sw_stations'], out_channels, edge_attr_dim),
            ('gw_wells', 'dis_edge', 'sw_stations'): SAGEConvWithEdgeAttr(in_channels_dict['gw_wells'], out_channels, edge_attr_dim),
            ('gw_wells', 'dis_edge', 'gw_wells'): SAGEConvWithEdgeAttr(in_channels_dict['gw_wells'], out_channels, edge_attr_dim),
            ('sw_stations', 'self_loop', 'sw_stations'): SAGEConv(in_channels_dict['sw_stations'], out_channels),
            ('pfas_sites', 'self_loop', 'pfas_sites'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),          
            ('gw_wells', 'self_loop', 'gw_wells'): SAGEConv(in_channels_dict['gw_wells'], out_channels),
        }, aggr=aggregation)

        # Second convolutional layer with edge attributes
        self.conv2 = HeteroConv({
            ('pfas_sites', 'dis_edge', 'gw_wells'): SAGEConvWithEdgeAttr(out_channels, out_channels, edge_attr_dim),
            ('gw_wells', 'dis_edge', 'pfas_sites'): SAGEConvWithEdgeAttr(out_channels, out_channels, edge_attr_dim),
            ('pfas_sites', 'dis_edge', 'sw_stations'): SAGEConvWithEdgeAttr(out_channels, out_channels, edge_attr_dim),
            ('sw_stations', 'dis_edge', 'pfas_sites'): SAGEConvWithEdgeAttr(out_channels, out_channels, edge_attr_dim),
            ('sw_stations', 'dis_edge', 'gw_wells'): SAGEConvWithEdgeAttr(out_channels, out_channels, edge_attr_dim),
            ('gw_wells', 'dis_edge', 'sw_stations'): SAGEConvWithEdgeAttr(out_channels, out_channels, edge_attr_dim),
            ('gw_wells', 'dis_edge', 'gw_wells'): SAGEConvWithEdgeAttr(out_channels, out_channels, edge_attr_dim),
            ('gw_wells', 'self_loop', 'gw_wells'): SAGEConv(out_channels, out_channels),
            ('sw_stations', 'self_loop', 'sw_stations'): SAGEConv(out_channels, out_channels),
            ('pfas_sites', 'self_loop', 'pfas_sites'): SAGEConv(out_channels, out_channels),
        }, aggr=aggregation)

        # Dynamic PReLU activations for each node type
        self.prelu_gw_wells = nn.PReLU(num_parameters=out_channels)
        self.prelu_sw_stations = nn.PReLU(num_parameters=out_channels)

        # Linear layers to reduce dimensionality to 1
        self.linear_gw_wells = nn.Linear(out_channels, 1)
        self.linear_sw_stations = nn.Linear(out_channels, 1)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # First convolutional layer
        x = self.conv1(x_dict, edge_index_dict, edge_attr_dict={
            ('pfas_sites', 'dis_edge', 'gw_wells'): edge_attr_dict[('pfas_sites', 'dis_edge', 'gw_wells')],
            ('gw_wells', 'dis_edge', 'pfas_sites'): edge_attr_dict[('gw_wells', 'dis_edge', 'pfas_sites')],
            ('pfas_sites', 'dis_edge', 'sw_stations'): edge_attr_dict[('pfas_sites', 'dis_edge', 'sw_stations')],
            ('sw_stations', 'dis_edge', 'gw_wells'): edge_attr_dict[('sw_stations', 'dis_edge', 'gw_wells')],
            ('gw_wells', 'dis_edge', 'sw_stations'): edge_attr_dict[('gw_wells', 'dis_edge', 'sw_stations')],
            ('sw_stations', 'dis_edge', 'pfas_sites'): edge_attr_dict[('sw_stations', 'dis_edge', 'pfas_sites')],
            ('gw_wells', 'dis_edge', 'gw_wells'): edge_attr_dict[('gw_wells', 'dis_edge', 'gw_wells')],
        })

        # Apply activation after first layer
        x = {key: F.relu(x[key]) for key in x.keys()}

        # Second convolutional layer
        x = self.conv2(x, edge_index_dict, edge_attr_dict={
            ('pfas_sites', 'dis_edge', 'gw_wells'): edge_attr_dict[('pfas_sites', 'dis_edge', 'gw_wells')],
            ('gw_wells', 'dis_edge', 'pfas_sites'): edge_attr_dict[('gw_wells', 'dis_edge', 'pfas_sites')],
            ('pfas_sites', 'dis_edge', 'sw_stations'): edge_attr_dict[('pfas_sites', 'dis_edge', 'sw_stations')],
            ('sw_stations', 'dis_edge', 'pfas_sites'): edge_attr_dict[('sw_stations', 'dis_edge', 'pfas_sites')],
            ('sw_stations', 'dis_edge', 'gw_wells'): edge_attr_dict[('sw_stations', 'dis_edge', 'gw_wells')],
            ('gw_wells', 'dis_edge', 'sw_stations'): edge_attr_dict[('gw_wells', 'dis_edge', 'sw_stations')],
            ('gw_wells', 'dis_edge', 'gw_wells'): edge_attr_dict[('gw_wells', 'dis_edge', 'gw_wells')],
        })

        # Apply activation after second layer
        x = {key: F.relu(x[key]) for key in x.keys()}

        # Apply dynamic PReLU activation
        x['gw_wells'] = self.prelu_gw_wells(x['gw_wells'])
        x['sw_stations'] = self.prelu_sw_stations(x['sw_stations'])

        # Apply linear layer to reduce dimensionality to 1
        x['gw_wells'] = self.linear_gw_wells(x['gw_wells'])
        x['sw_stations'] = self.linear_sw_stations(x['sw_stations'])

        return x
