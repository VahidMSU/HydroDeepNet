import torch
from torch_geometric.nn import HeteroConv, SAGEConv
import torch.nn as nn
from torch.nn import functional as F


class DeepGraphSAGEModel(nn.Module):
    def __init__(self, in_channels_dict, out_channels, aggregation, edge_attr_dict=None):
        super(DeepGraphSAGEModel, self).__init__()
        
        # First HeteroConv layer with GraphSAGE (SAGEConv) for each edge type
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
        
        # Second HeteroConv layer with GraphSAGE (SAGEConv) for each edge type
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
        
        # Third HeteroConv layer with GraphSAGE (SAGEConv) for each edge type
        self.conv3 = HeteroConv({
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
        
        # Separate linear layers for gw_wells and sw_stations
        self.gw_wells_linear = nn.Linear(out_channels, 1)
        self.sw_stations_linear = nn.Linear(out_channels, 1)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        # First convolution layer
        x = self.conv1(x_dict, edge_index_dict)
        x = {key: F.relu(x[key]) for key in x.keys()}
        
        # Second convolution layer
        x = self.conv2(x, edge_index_dict)
        x = {key: F.relu(x[key]) for key in x.keys()}

        # Third convolution layer
        x = self.conv3(x, edge_index_dict)
        x = {key: F.relu(x[key]) for key in x.keys()}
        
        # Apply the separate linear layers for gw_wells and sw_stations
        x['gw_wells'] = self.gw_wells_linear(x['gw_wells'])
        x['sw_stations'] = self.sw_stations_linear(x['sw_stations'])
        
        return x
