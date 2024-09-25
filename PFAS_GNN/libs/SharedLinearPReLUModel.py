import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv


class SharedLinearPReLUModel(nn.Module):
    def __init__(self, in_channels_dict, out_channels, aggregation, edge_attr_dict=None):
        super(SharedLinearPReLUModel, self).__init__()
        
        # First HeteroConv layer with all relevant edges
        self.conv1 = HeteroConv({
            ('pfas_sites', 'dis_edge', 'gw_wells'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('gw_wells', 'dis_edge', 'pfas_sites'): SAGEConv(in_channels_dict['gw_wells'], out_channels),
            ('pfas_sites', 'dis_edge', 'sw_stations'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('sw_stations', 'dis_edge', 'pfas_sites'): SAGEConv(in_channels_dict['sw_stations'], out_channels),
            ('sw_stations', 'dis_edge', 'gw_wells'): SAGEConv(in_channels_dict['sw_stations'], out_channels),
            ('gw_wells', 'dis_edge', 'sw_stations'): SAGEConv(in_channels_dict['gw_wells'], out_channels),
            ('gw_wells', 'dis_edge', 'gw_wells'): SAGEConv(in_channels_dict['gw_wells'], out_channels),
            ('gw_wells', 'self_loop', 'gw_wells'): SAGEConv(in_channels_dict['gw_wells'], out_channels),
            ('sw_stations', 'self_loop', 'sw_stations'): SAGEConv(in_channels_dict['sw_stations'], out_channels),
            ('pfas_sites', 'self_loop', 'pfas_sites'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
        }, aggr=aggregation)
        
        # Second HeteroConv layer with all relevant edges
        self.conv2 = HeteroConv({
            ('pfas_sites', 'dis_edge', 'gw_wells'): SAGEConv(out_channels, out_channels),
            ('gw_wells', 'dis_edge', 'pfas_sites'): SAGEConv(out_channels, out_channels),
            ('pfas_sites', 'dis_edge', 'sw_stations'): SAGEConv(out_channels, out_channels),
            ('sw_stations', 'dis_edge', 'pfas_sites'): SAGEConv(out_channels, out_channels),
            ('sw_stations', 'dis_edge', 'gw_wells'): SAGEConv(out_channels, out_channels),
            ('gw_wells', 'dis_edge', 'sw_stations'): SAGEConv(out_channels, out_channels),
            ('gw_wells', 'dis_edge', 'gw_wells'): SAGEConv(out_channels, out_channels),
            ('gw_wells', 'self_loop', 'gw_wells'): SAGEConv(out_channels, out_channels),
            ('sw_stations', 'self_loop', 'sw_stations'): SAGEConv(out_channels, out_channels),
            ('pfas_sites', 'self_loop', 'pfas_sites'): SAGEConv(out_channels, out_channels),
        }, aggr=aggregation)

        # Linear layer and PReLU activation
        self.linear = nn.Linear(out_channels, 1)
        self.prelu = nn.PReLU()

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        # First convolution layer
        x = self.conv1(x_dict, edge_index_dict)
        x = {key: F.relu(x[key]) for key in x.keys()}
        
        # Second convolution layer
        x = self.conv2(x, edge_index_dict)
        x = {key: F.relu(x[key]) for key in x.keys()}
        
        # Apply linear layers and PReLU activations
        x['gw_wells'] = self.prelu(self.linear(x['gw_wells']))
        x['sw_stations'] = self.prelu(self.linear(x['sw_stations']))
        
        return x
