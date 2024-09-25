import os
import torch
import pandas as pd

# Initialize a list to hold the information for each HeteroData file
hetero_data_info = []

files = os.listdir('Hetero_data')
for file in files:
    path = 'Hetero_data/' + file
    data = torch.load(path)
    
    # Extract information from the filename
    gw_site_sw_distance_threshold = file.split('_')[-2]
    gw_gw_distance_threshold = file.split('_')[-1].split('.')[0]
    gw_features = file.split('_')[:-2]

    # Extract node counts
    sw_node_count = data['sw_stations'].x.size(0)
    gw_node_count = data['gw_wells'].x.size(0)
    pfas_node_count = data['pfas_sites'].x.size(0)

    # Extract edge counts for each edge type
    gw_sw_edges = data[('gw_wells', 'dis_edge', 'sw_stations')].edge_index.size(1) 
    sw_gw_edges = data[('sw_stations', 'dis_edge', 'gw_wells')].edge_index.size(1) 
    gw_pfas_edges = data[('gw_wells', 'dis_edge', 'pfas_sites')].edge_index.size(1) 
    pfas_gw_edges = data[('pfas_sites', 'dis_edge', 'gw_wells')].edge_index.size(1)
    sw_pfas_edges = data[('sw_stations', 'dis_edge', 'pfas_sites')].edge_index.size(1) 
    pfas_sw_edges = data[('pfas_sites', 'dis_edge', 'sw_stations')].edge_index.size(1) 
    gw_gw_edges = data[('gw_wells', 'dis_edge', 'gw_wells')].edge_index.size(1)

    # Store the extracted information in a dictionary
    hetero_data_info.append({
        "File": file,
        "GW Site/SW Distance Threshold": gw_site_sw_distance_threshold,
        "GW-GW Distance Threshold": gw_gw_distance_threshold,
        "GW Features": ', '.join(gw_features),
        "SW Nodes": sw_node_count,
        "GW Nodes": gw_node_count,
        "PFAS Nodes": pfas_node_count,
        "GW-SW Edges": gw_sw_edges,
        "SW-GW Edges": sw_gw_edges,
        "GW-PFAS Edges": gw_pfas_edges,
        "PFAS-GW Edges": pfas_gw_edges,
        "SW-PFAS Edges": sw_pfas_edges,
        "PFAS-SW Edges": pfas_sw_edges,
        "GW-GW Edges": gw_gw_edges
    })

# Convert the list of dictionaries to a DataFrame for easier visualization
df = pd.DataFrame(hetero_data_info)

# Display the DataFrame
pd.set_option('display.max_colwidth', None)  # To ensure that all content in columns is fully visible
print(df)

# Optionally save to a CSV file
df.drop(columns=['File']).to_csv('results/HeteroData_info.csv', index=False)
