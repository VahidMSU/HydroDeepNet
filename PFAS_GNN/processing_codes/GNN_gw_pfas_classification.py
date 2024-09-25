import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.data import HeteroData
import pandas as pd
import geopandas as gpd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from GNN_gw_pfas import load_pfas_sites, load_pfas_gw, classify_pfas_gw, plot_distribution, split_sampled_pfas_gw, plot_site_samples, create_hetdata, split_data
from GNN_gw_pfas import calculate_distances_gpu
import itertools


def create_edges_and_distances(gw, sites, device, threshold):
    assert len(gw) > 0, logging.ERROR("There are no gw wells")
    assert len(sites) > 0, logging.ERROR("There are no sites")
    assert gw.crs.to_epsg() == 26990, logging.ERROR("gw crs is not 26990")
    assert sites.crs.to_epsg() == 26990, logging.ERROR("sites crs is not 26990")
    assert len(gw['WSSN'].unique()) == len(gw), logging.ERROR("There are duplicate WSSN values in gw")

    gw_coords = np.array([(geom.x, geom.y) for geom in gw.geometry])
    sites_coords = np.array([(geom.x, geom.y) for geom in sites.geometry])

    sites_gw_dists = calculate_distances_gpu(gw_coords, sites_coords, device)

    edges = []
    distances = []
    dem_differences = []
    swl_differences = []
    num_gw = len(gw_coords)
    num_sites = len(sites_coords)

    for i, j in itertools.product(range(num_sites), range(num_gw)):
        distance = sites_gw_dists[i, j].item()
        if distance <= threshold:
            edges.append([i, j])
            distances.append(distance)
            if "DEM_250m" in sites.columns and "DEM_250m" in gw.columns:
                dem_difference = sites['DEM_250m'].iloc[i] - gw['DEM_250m'].iloc[j]
                dem_differences.append(dem_difference)

            if "kriging_output_SWL_250m" in sites.columns and "kriging_output_SWL_250m" in gw.columns:
                fit_to_meter = 0.3048
                site_head = sites['DEM_250m'].iloc[i] - (sites['kriging_output_SWL_250m'].iloc[i]*fit_to_meter)
                gw_head = gw['DEM_250m'].iloc[j] - (gw['kriging_output_SWL_250m'].iloc[j]*fit_to_meter)
                swl_difference = site_head - gw_head
                swl_differences.append(swl_difference)

    logging.info(f'stage:create_edges_and_distances ==##== Number of edges: {len(edges)}')
    logging.info(f'stage:create_edges_and_distances ==##== Number of distances: {len(distances)}')

    return edges, distances, dem_differences, swl_differences

def create_hetdata(pfas_gw, pfas_sites, device, pfas_gw_columns, pfas_sites_columns, gw_features, distance_threshold):
    logging.info(f"stage:create_hetdata ==##== distance_threshold: {distance_threshold}")
    pfas_gw_columns.append('gw_node_index')
    pfas_sites_columns.append('site_node_index')
    data = HeteroData()

    #NOTE: pfas_gw columns include all PFAS components at this state 
    ### include PFAS components in pfas_gw_columns
    ### this can be used for node classification based on PFAS composition
    ### pfas_components = [col for col in pfas_gw.columns if col.endswith('Result')]
    
    data['gw_wells'].x = torch.tensor(pfas_gw[pfas_gw_columns+gw_features].values, dtype=torch.float)
    data['pfas_sites'].x = torch.tensor(pfas_sites[pfas_sites_columns+gw_features].values, dtype=torch.float)
    assert pfas_gw.crs.to_epsg() == 26990, "pfas_gw crs is not 26990"
    assert pfas_sites.crs.to_epsg() == 26990, "pfas_sites crs is not 26990"

    edges, distances, dem_differences, swl_differences = create_edges_and_distances(pfas_gw, pfas_sites, device, threshold=distance_threshold)
    data['pfas_sites', 'distance', 'gw_wells'].edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    if len(dem_differences) > 0 and len(swl_differences) == 0:
        logging.info("#### DEM difference is included ##########")
        data['pfas_sites', 'distance', 'gw_wells'].edge_attr = torch.tensor(np.vstack((distances, dem_differences)).T, dtype=torch.float)
    elif len(swl_differences) > 0 and len(dem_differences) == 0:
        logging.info("#### SWL difference is included ##########")
        data['pfas_sites', 'distance', 'gw_wells'].edge_attr = torch.tensor(np.vstack((distances, swl_differences)).T, dtype=torch.float)
    elif len(dem_differences) > 0 and len(swl_differences) > 0:
        logging.info("#### DEM difference and SWL difference is included ##########")
        data['pfas_sites', 'distance', 'gw_wells'].edge_attr = torch.tensor(np.vstack((distances, dem_differences, swl_differences)).T, dtype=torch.float)
    else:
        data['pfas_sites', 'distance', 'gw_wells'].edge_attr = torch.tensor(np.vstack((distances)).T, dtype=torch.float)

    return data




def load_dataset(args):
    data_dir = args["data_dir"]
    pfas_gw_columns = args["pfas_gw_columns"]
    pfas_sites_columns = args["pfas_sites_columns"]
    gw_features = args["gw_features"]
    distance_threshold = args["distance_threshold"]

    pfas_sites = load_pfas_sites(data_dir)
    pfas_gw = load_pfas_gw(data_dir, gw_features)
    pfas_gw = classify_pfas_gw(pfas_gw)

    plot_distribution(pfas_gw)

    train_gw, val_gw, test_gw = split_sampled_pfas_gw(pfas_gw)
    plot_site_samples(train_gw, val_gw, test_gw, pfas_sites)
    data = create_hetdata(pfas_gw, pfas_sites, device, pfas_gw_columns, pfas_sites_columns, gw_features, distance_threshold)

    data = split_data(data, train_gw, val_gw, test_gw).to(device)
    ## assert geometry is in pfas_gw
    assert 'geometry' in pfas_gw.columns, "geometry column is missing in pfas_gw"
    return data, pfas_gw




class GAE(nn.Module):
    def __init__(self, in_channels_dict, out_channels):
        super(GAE, self).__init__()
        self.encoder = HeteroConv({
            ('pfas_sites', 'distance', 'gw_wells'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('gw_wells', 'distance', 'pfas_sites'): SAGEConv(in_channels_dict['gw_wells'], out_channels)
        }, aggr='mean')
        self.decoder = nn.ModuleDict({
            'pfas_sites-gw_wells': nn.Sequential(
                nn.Linear(out_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, 1),
                nn.Sigmoid()
            )
        })

    def encode(self, x_dict, edge_index_dict):
        x = self.encoder(x_dict, edge_index_dict)
        return x

    def decode(self, z, edge_index):
        edge_type = 'pfas_sites-gw_wells'
        src, dst = edge_index
        z_src = z['site_node_index'][src]
        z_dst = z['gw_node_index'][dst]
        z = torch.cat([z_src, z_dst], dim=-1)
        return self.decoder[edge_type](z).view(-1)

    def forward(self, x_dict, edge_index_dict):
        z = self.encode(x_dict, edge_index_dict)
        return z

    def recon_loss(self, z, edge_index):
        return F.binary_cross_entropy(self.decode(z, edge_index), torch.ones(edge_index.size(1), device=z.device))


# Training the GAE
def train_gae(model, data, optimizer, epochs=200):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        z = model(data.x_dict, data.edge_index_dict)
        loss = model.recon_loss(z, data['pfas_sites', 'distance', 'gw_wells'].edge_index)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = {
        "data_dir": "/data/MyDataBase/HuronRiverPFAS/",
        "pfas_gw_columns": ['sum_PFAS'],
        "pfas_sites_columns": ['Industry'],
        "gw_features":[
                       "kriging_output_SWL_250m",
                       "DEM_250m",
                    #    "kriging_output_H_COND_1_250m",
                    #    "kriging_output_H_COND_2_250m",
                    #    "kriging_output_AQ_THK_1_250m",
                    #    "kriging_output_AQ_THK_2_250m",
                        ],

        "distance_threshold": 10000
    }


    # Set up the model
    in_channels_dict = {
        'pfas_sites': len(args['gw_features']) + 2,
        'gw_wells': len(args['gw_features']) + 2
    }
    model = GAE(in_channels_dict, out_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



    data, _ = load_dataset(args)

    print("======================================")
    print(f"############################## OUT FINAL IMPORTED DATA: {data} ##############################")
    print("======================================")





    # Train the model
    train_gae(model, data, optimizer)

    # After training, use z for node classification or clustering
    z = model.encode(data.x_dict, data.edge_index_dict)

    # Example: KMeans for clustering
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=3)
    labels = kmeans.fit_predict(z['gw_wells'].cpu().detach().numpy())

    # Visualize or evaluate the clustering results based on labels
