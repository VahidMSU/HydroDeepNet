import numpy as np
import torch
from torch_geometric.data import HeteroData
import itertools
import torch
import pandas as pd

import numpy as np
import torch
import itertools
from joblib import Parallel, delayed
import numpy as np
import torch
import itertools
from joblib import Parallel, delayed
from tqdm import tqdm  # Import tqdm for progress bars
import numpy as np
import torch
from shapely.geometry import Point
from rtree import index
import itertools

def calculate_distances_gpu(des_coords, src_coords, device, logger, dtype=torch.float):
    src_tensor = torch.tensor(src_coords, dtype=dtype, device=device)
    des_tensor = torch.tensor(des_coords, dtype=dtype, device=device)
    
    src_des_dists = torch.cdist(src_tensor, des_tensor, p=2)
    
    logger.info(f"stage:calculate_distances_gpu ==##== Shape of src_des_dists: {src_des_dists.shape}")
    logger.info(f"stage:calculate_distances_gpu ==##== Distances shape: {src_des_dists.shape}")
    
    return src_des_dists

def create_edges_and_distances(des_node, src_node, device, threshold, logger):
    des_node_coords = np.array([(geom.x, geom.y) for geom in des_node.geometry])
    src_node_coords = np.array([(geom.x, geom.y) for geom in src_node.geometry])

    # Build a spatial index for the destination nodes
    src_index = index.Index()
    for i, coord in enumerate(src_node_coords):
        src_index.insert(i, (coord[0], coord[1], coord[0], coord[1]))

    potential_pairs = []
    for i, des_coord in enumerate(des_node_coords):
        point = Point(des_coord)
        # Query spatial index to find potential nearby source nodes within the threshold
        nearby_src_nodes = list(src_index.intersection((point.x - threshold, point.y - threshold, point.x + threshold, point.y + threshold)))
        for j in nearby_src_nodes:
            potential_pairs.append((j, i))

    logger.info(f"stage:create_edges_and_distances ==##== Potential pairs to process: {len(potential_pairs)}")

    src_node_des_node_dists = calculate_distances_gpu(des_node_coords, src_node_coords, device, logger)

    edges = []
    distances = []
    dem_differences = []
    swl_differences = []

    for j, i in potential_pairs:
        distance = src_node_des_node_dists[j, i].item()
        if distance <= threshold:
            edges.append([j, i])
            distances.append(distance)
            if "DEM_250m" in src_node.columns and "DEM_250m" in des_node.columns:
                dem_difference = src_node['DEM_250m'].iloc[j] - des_node['DEM_250m'].iloc[i]
                dem_differences.append(dem_difference)

            if "kriging_output_SWL_250m" in src_node.columns and "kriging_output_SWL_250m" in des_node.columns:
                fit_to_meter = 0.3048
                site_head = src_node['DEM_250m'].iloc[j] - (src_node['kriging_output_SWL_250m'].iloc[j] * fit_to_meter)
                des_node_head = des_node['DEM_250m'].iloc[i] - (des_node['kriging_output_SWL_250m'].iloc[i] * fit_to_meter)
                swl_differences.append(site_head - des_node_head)

    logger.info(f'stage:create_edges_and_distances ==##== Number of edges: {len(edges)}')
    logger.info(f'stage:create_edges_and_distances ==##== Number of distances: {len(distances)}')

    logger.info(
        f'stage:create_edges_and_distances ==##== Number of des_node wells not connected to at least one src_node {len(des_node_coords) - len({edge[1] for edge in edges})}'
    )

    return edges, distances, dem_differences, swl_differences


def create_hetdata(pfas_gw, pfas_sw, unsampled_gw, pfas_sites, rivs, device, pfas_gw_columns, pfas_sites_columns,pfas_sw_station_columns, gw_features, distance_threshold, logger, gw_gw_distance_threshold):
    
    logger.info(f"stage:create_hetdata ==##== distance_threshold: {distance_threshold}")
    logger.info(f"stage:create_hetdata ==##== Number of pfas_gw: {len(pfas_gw)}")
    logger.info(f"stage:create_hetdata ==##== Number of unsampled gw: {len(unsampled_gw)}")
    logger.info(f"stage:create_hetdata ==##== Number of pfas_sites: {len(pfas_sw)}")


    pfas_gw_columns.append('gw_node_index')
    pfas_sites_columns.append('site_node_index')
    pfas_sw_station_columns.append('sw_node_index')
    rivs_columns = ['riv_node_index', 'ChannelR']
    data = HeteroData()

    ### normalize gw_features data in pfas_gw, pfas_sw and pfas_sites based on the entire dataset (pfas_gw, pfas_sw and pfas_sites)
    for feature in gw_features:
        all_data_mean = max (np.mean(pfas_gw[feature].values), np.mean(pfas_sw[feature].values), np.mean(pfas_sites[feature].values))
        all_data_std = max (np.std(pfas_gw[feature].values), np.std(pfas_sw[feature].values), np.std(pfas_sites[feature].values))
        pfas_gw[feature] = (pfas_gw[feature] - all_data_mean) / all_data_std
        pfas_sw[feature] = (pfas_sw[feature] - all_data_mean) / all_data_std
        pfas_sites[feature] = (pfas_sites[feature] - all_data_mean) / all_data_std

    ## add node features
    data['sw_stations'].x = torch.tensor(pfas_sw[pfas_sw_station_columns + gw_features].values, dtype=torch.float)
    data['gw_wells'].x = torch.tensor(pfas_gw[pfas_gw_columns + gw_features].values, dtype=torch.float)
    data['pfas_sites'].x = torch.tensor(pfas_sites[pfas_sites_columns + gw_features].values, dtype=torch.float)
    data['rivs'].x = torch.tensor(rivs[rivs_columns].values, dtype=torch.float)
    

    pfas_gw = pfas_gw.drop_duplicates(subset='WSSN')  ### NOTE: need to be verified later
    #pfas_sw = pfas_sw.drop_duplicates(subset='SiteCode')  ### NOTE: need to be verified later

    assert pfas_gw.crs.to_epsg() == 26990, "pfas_gw crs is not 26990"
    assert pfas_sites.crs.to_epsg() == 26990, "pfas_sites crs is not 26990"
    assert len(pfas_gw['WSSN'].unique()) == len(pfas_gw), f"There are duplicate WSSN values in pfas_gw {len(pfas_gw['WSSN'].unique())} - {len(pfas_gw)}"
    assert len(pfas_sw['SiteCode'].unique()) == len(pfas_sw), f"There are duplicate SiteCode values in pfas_sw {len(pfas_sw['SiteCode'].unique())} - {len(pfas_sw)}"

    
    data = create_node_edge_main(data, pfas_gw, pfas_sw, pfas_sites, rivs, device, distance_threshold, logger, gw_gw_distance_threshold)

    data = add_self_loops(data, device)

    assert_HeteroData_creation(data, logger)


    return data

def create_rivs_edges_and_nodes(rivs, data):

    # Step 1: Define node features (assuming AreaC, strmOrder, and Len2 are node features)
    data['rivs'].x = torch.tensor(rivs[['AreaC', 'strmOrder']].values, dtype=torch.float)

    # Step 2: Create a mapping from Channel (HydroSequence) to riv_node_index
    channel_to_index = pd.Series(rivs.index, index=rivs['Channel']).to_dict()

    # Step 3: Create edges based on Channel (source) and ChannelR (destination)
    riv_riv_edges = rivs[['Channel', 'ChannelR']].dropna().values
    riv_riv_edges = np.array([
        [channel_to_index[src], channel_to_index[dst]] 
        for src, dst in riv_riv_edges 
        if dst in channel_to_index  # Ensure that ChannelR is in the mapping
    ])

    # Step 4: Convert to torch tensor and ensure it’s contiguous
    riv_riv_edge_index = torch.tensor(riv_riv_edges.T, dtype=torch.long).contiguous()

    edge_attrs = [
        rivs.loc[src, ['AreaC', 'strmOrder', 'Len2']].values
        for src, dst in riv_riv_edges
    ]
    
    riv_riv_edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    # Step 6: Assign edge_index and edge_attr to the HeteroData object
    data['rivs', 'dis_edge', 'rivs'].edge_index = riv_riv_edge_index
    data['rivs', 'dis_edge', 'rivs'].edge_attr = riv_riv_edge_attr

    return data


def create_node_edge_main(data, pfas_gw, pfas_sw, pfas_sites,rivs, device, distance_threshold, logger, gw_gw_distance_threshold):

    """" Create edges and distances between nodes """

    # Step 1: Create a mapping from Channel (HydroSequence) to riv_node_index
    data = create_rivs_edges_and_nodes(rivs, data)

    gw_site_edges, gw_distances, gw_site_dem_differences, gw_site_swl_differences = create_edges_and_distances(pfas_gw, pfas_sites, device, threshold=distance_threshold, logger=logger)
    gw_edge_index = torch.tensor(gw_site_edges, dtype=torch.long).t().contiguous()
    data['pfas_sites', 'dis_edge', 'gw_wells'].edge_index = gw_edge_index
    data['gw_wells', 'dis_edge', 'pfas_sites'].edge_index = gw_edge_index[[1, 0], :]  # Reverse the edges for the reverse direction

    sw_site_edges, sw_distances, sw_site_dem_differences, sw_site_swl_differences = create_edges_and_distances(pfas_sw, pfas_sites, device, threshold=distance_threshold, logger=logger)
    sw_edge_index = torch.tensor(sw_site_edges, dtype=torch.long).t().contiguous()
    data['pfas_sites', 'dis_edge', 'sw_stations'].edge_index = sw_edge_index
    data['sw_stations', 'dis_edge', 'pfas_sites'].edge_index = sw_edge_index[[1, 0], :]  # Reverse the edges for the reverse direction

    gw_sw_edges, gw_sw_distances, gw_sw_dem_differences, gw_sw_swl_differences = create_edges_and_distances(pfas_gw, pfas_sw, device, threshold=distance_threshold, logger=logger)
    gw_sw_edge_index = torch.tensor(gw_sw_edges, dtype=torch.long).t().contiguous()
    data['sw_stations', 'dis_edge', 'gw_wells'].edge_index = gw_sw_edge_index
    data['gw_wells', 'dis_edge', 'sw_stations'].edge_index = gw_sw_edge_index[[1, 0], :]  # Reverse the edges for the reverse direction

    gw_gw_edges, gw_gw_distances, gw_gw_dem_differences, gw_gw_swl_differences = create_edges_and_distances(pfas_gw, pfas_gw, device, threshold=gw_gw_distance_threshold, logger=logger)
    gw_gw_edge_index = torch.tensor(gw_gw_edges, dtype=torch.long).t().contiguous()
    data['gw_wells', 'dis_edge', 'gw_wells'].edge_index = gw_gw_edge_index

    if len(gw_site_dem_differences) > 0 and len(gw_site_swl_differences) == 0:
        logger.info("#### DEM difference is included ##########")
        gw_site_edge_attr = torch.tensor(np.vstack((gw_distances, gw_site_dem_differences)).T, dtype=torch.float)
        sw_site_edge_attr = torch.tensor(np.vstack((sw_distances, sw_site_dem_differences)).T, dtype=torch.float)
        gw_sw_edge_attr = torch.tensor(np.vstack((gw_sw_distances, gw_sw_dem_differences)).T, dtype=torch.float)
        gw_gw_edge_attr = torch.tensor(np.vstack((gw_gw_distances, gw_gw_dem_differences)).T, dtype=torch.float)    
    elif len(gw_site_swl_differences) > 0 and len(gw_site_dem_differences) == 0:
        logger.info("#### SWL difference is included ##########")
        gw_site_edge_attr = torch.tensor(np.vstack((gw_distances, gw_site_swl_differences)).T, dtype=torch.float)
        sw_site_edge_attr = torch.tensor(np.vstack((sw_distances, sw_site_swl_differences)).T, dtype=torch.float)
        gw_sw_edge_attr = torch.tensor(np.vstack((gw_sw_distances, gw_sw_swl_differences)).T, dtype=torch.float)
        gw_gw_edge_attr = torch.tensor(np.vstack((gw_gw_distances, gw_gw_swl_differences)).T, dtype=torch.float)

    elif len(gw_site_dem_differences) > 0 and len(gw_site_swl_differences) > 0:
        logger.info("#### DEM difference and SWL difference is included ##########")
        gw_site_edge_attr = torch.tensor(np.vstack((gw_distances, gw_site_dem_differences, gw_site_swl_differences)).T, dtype=torch.float)
        sw_site_edge_attr = torch.tensor(np.vstack((sw_distances, sw_site_dem_differences, sw_site_swl_differences)).T, dtype=torch.float)
        gw_sw_edge_attr = torch.tensor(np.vstack((gw_sw_distances, gw_sw_dem_differences, gw_sw_swl_differences)).T, dtype=torch.float)
        gw_gw_edge_attr = torch.tensor(np.vstack((gw_gw_distances, gw_gw_dem_differences, gw_gw_swl_differences)).T, dtype=torch.float)
    else:
        gw_site_edge_attr = torch.tensor(np.vstack((gw_distances)).T, dtype=torch.float)
        sw_site_edge_attr = torch.tensor(np.vstack((sw_distances)).T, dtype=torch.float)
        gw_sw_edge_attr = torch.tensor(np.vstack((gw_sw_distances)).T, dtype=torch.float)
        gw_gw_edge_attr = torch.tensor(np.vstack((gw_gw_distances)).T, dtype=torch.float)

    data['pfas_sites', 'dis_edge', 'gw_wells'].edge_attr = gw_site_edge_attr
    data['gw_wells', 'dis_edge', 'pfas_sites'].edge_attr = gw_site_edge_attr

    data['pfas_sites', 'dis_edge', 'sw_stations'].edge_attr = sw_site_edge_attr
    data['sw_stations', 'dis_edge', 'pfas_sites'].edge_attr = sw_site_edge_attr

    data['sw_stations', 'dis_edge', 'gw_wells'].edge_attr = gw_sw_edge_attr
    data['gw_wells', 'dis_edge', 'sw_stations'].edge_attr = gw_sw_edge_attr

    data['gw_wells', 'dis_edge', 'gw_wells'].edge_attr = gw_gw_edge_attr
    data['gw_wells', 'dis_edge', 'gw_wells'].edge_attr = gw_gw_edge_attr

    return data

def assert_HeteroData_creation(data, logger):
    # Ensure 'pfas_sites' is a destination node type
    assert ('pfas_sites', 'dis_edge', 'gw_wells') in data.edge_index_dict.keys(), "There is no edge from 'pfas_sites' to 'gw_wells'"
    assert ('gw_wells', 'dis_edge', 'pfas_sites') in data.edge_index_dict.keys(), "There is no edge from 'gw_wells' to 'pfas_sites'"

    assert ('pfas_sites', 'self_loop', 'pfas_sites') in data.edge_index_dict.keys(), "There is no self-loop edge for 'pfas_sites'"
    assert ('gw_wells', 'self_loop', 'gw_wells') in data.edge_index_dict.keys(), "There is no self-loop edge for 'gw_wells'"

    assert ('pfas_sites', 'dis_edge', 'sw_stations') in data.edge_index_dict.keys(), "There is no edge from 'pfas_sites' to 'sw_stations'"
    assert ('sw_stations', 'dis_edge', 'pfas_sites') in data.edge_index_dict.keys(), "There is no edge from 'sw_stations' to 'pfas_sites'"

    assert ('pfas_sites', 'self_loop', 'pfas_sites') in data.edge_index_dict.keys(), "There is no self-loop edge for 'pfas_sites'"
    assert ('sw_stations', 'self_loop', 'sw_stations') in data.edge_index_dict.keys(), "There is no self-loop edge for 'sw_stations'"
    
    assert ('sw_stations', 'dis_edge', 'gw_wells') in data.edge_index_dict.keys(), "There is no edge from 'sw_stations' to 'gw_wells'"
    assert ('gw_wells', 'dis_edge', 'sw_stations') in data.edge_index_dict.keys(), "There is no edge from 'gw_wells' to 'sw_stations'"
    assert ('sw_stations', 'self_loop', 'sw_stations') in data.edge_index_dict.keys(), "There is no self-loop edge for 'sw_stations'"

    logger.info(f"Edges from 'pfas_sites' to 'sw_stations': {data['pfas_sites', 'dis_edge', 'sw_stations'].edge_index.shape[1]}")
    logger.info(f"Edges from 'sw_stations' to 'pfas_sites': {data['sw_stations', 'dis_edge', 'pfas_sites'].edge_index.shape[1]}")

    logger.info(f"Edges from 'pfas_sites' to 'gw_wells': {data['pfas_sites', 'dis_edge', 'gw_wells'].edge_index.shape[1]}")
    logger.info(f"Edges from 'gw_wells' to 'pfas_sites': {data['gw_wells', 'dis_edge', 'pfas_sites'].edge_index.shape[1]}")

    logger.info(f"Edges from 'sw_stations' to 'gw_wells': {data['sw_stations', 'dis_edge', 'gw_wells'].edge_index.shape[1]}")
    logger.info(f"Edges from 'gw_wells' to 'sw_stations': {data['gw_wells', 'dis_edge', 'sw_stations'].edge_index.shape[1]}")

    logger.info(f"Self-loop edges for 'pfas_sites': {data['pfas_sites', 'self_loop', 'pfas_sites'].edge_index.shape[1]}")
    logger.info(f"Self-loop edges for 'gw_wells': {data['gw_wells', 'self_loop', 'gw_wells'].edge_index.shape[1]}")
    logger.info(f"Self-loop edges for 'sw_stations': {data['sw_stations', 'self_loop', 'sw_stations'].edge_index.shape[1]}")


def add_self_loops(data, device):
    """Add self-loops to the HeteroData."""
    num_pfas_sites = data['pfas_sites'].x.size(0)
    num_gw_wells = data['gw_wells'].x.size(0)
    num_sw_stations = data['sw_stations'].x.size(0)

    pfas_sites_self_loops = torch.arange(num_pfas_sites, dtype=torch.long, device=device).unsqueeze(0).repeat(2, 1)
    gw_wells_self_loops = torch.arange(num_gw_wells, dtype=torch.long, device=device).unsqueeze(0).repeat(2, 1)
    sw_stations_self_loops = torch.arange(num_sw_stations, dtype=torch.long, device=device).unsqueeze(0).repeat(2, 1)

    data['pfas_sites', 'self_loop', 'pfas_sites'].edge_index = pfas_sites_self_loops
    data['gw_wells', 'self_loop', 'gw_wells'].edge_index = gw_wells_self_loops
    data['sw_stations', 'self_loop', 'sw_stations'].edge_index = sw_stations_self_loops

    return data


def load_rivers(path, device, logger):
    import geopandas as gpd
    import pandas as pd
    import matplotlib.pyplot as plt
    # Load the shapefile containing river data
    riv = gpd.read_file(path)
    ##
    logger.info(f"stage:create_river_nodes_edges ==##== Number of rows: {riv.shape[0]}")
    # Assign node indices based on the unique 'Channel' values
    riv['riv_node_index'] = pd.factorize(riv['Channel'])[0]
    riv['sampled'] = 0
    riv.plot()
    plt.tight_layout()  
    plt.savefig('figs/sw_rivers.png',dpi=300)
    return riv
import numpy as np
import torch
import pandas as pd

def create_river_nodes_edges(riv, device, logger):
    # Create edges based on the 'Channel' (Channel) and 'ChannelR' relationships
    riv_edges = riv[['riv_node_index', 'ChannelR']].dropna().values
    
    # Convert ChannelR to riv_node_index to get the destination node index
    # First, map ChannelR to riv_node_index based on the Channel mapping
    riv_node_index = pd.Series(riv['riv_node_index'].values, index=riv['Channel']).to_dict()
    riv_edges = np.array([[src, riv_node_index.get(dst, -1)] for src, dst in riv_edges if dst in riv_node_index])

    # Remove any edges with a -1 index (which indicates no corresponding downstream node was found)
    riv_edges = riv_edges[riv_edges[:, 1] != -1]

    # Convert edges to PyTorch tensors
    riv_edge_index = torch.tensor(riv_edges.T, dtype=torch.long, device=device)
    
    # Define edge attributes using relevant columns 
    assert "Drop" in riv.columns, "Drop column is not in the riv dataframe"
    assert "Length" in riv.columns, "Length column is not in the riv dataframe"
    assert "Subbasin" in riv.columns, "Subbasin column is not in the riv dataframe"
    
    # Extract the relevant columns and convert to a NumPy array
    riv_edge_attr_values = riv[['Drop', 'Length', 'Subbasin']].values
    
    # Convert the NumPy array to a PyTorch tensor
    riv_edge_attr = torch.tensor(riv_edge_attr_values, dtype=torch.float, device=device)

    ### print number of edges and their features
    logger.info(f"Number of nodes: {riv['riv_node_index'].nunique()}")
    logger.info(f"Number of edges: {riv_edge_index.shape[1]}")   
    logger.info(f"Edge attributes: {riv_edge_attr.shape[1]}")

    return riv['riv_node_index'], riv_edge_index, riv_edge_attr


if __name__ == "__main__":
    from libs.utils import setup_logging
    path = "/data/MyDataBase/CIWRE-BAE/SWAT_input/huc8/4100013/SWAT_MODEL/Watershed/Shapes/SWAT_plus_streams_modified.shp"
    logger = setup_logging()
    sw = load_rivers(path, device='cpu', logger=logger)
    riv_node_index, riv_edge_index, riv_edge_attr = create_river_nodes_edges(sw, device='cpu', logger=logger)
    
    print("End of the script")