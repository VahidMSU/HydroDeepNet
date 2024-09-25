import os
import pandas as pd
import networkx as nx
import logging
from libs.relation_table import generate_relation_table
from libs.load_climate import load_climate_datas
import pickle
import numpy as np
import geopandas as gpd
# Configure logging
logging.basicConfig(level=logging.INFO)

class SWATGraphProcessor:
    def __init__(self, swat_output_base_path, num_days=565):
        self.swat_output_base_path = swat_output_base_path
        self.num_days = num_days
        self.temporal_data = {}

    def create_graph(self):
        G = nx.DiGraph()
        unique_wst_ids = self.swat_streams['wst_id'].unique()
        logging.info(f"Number of unique wst_ids: {len(unique_wst_ids)}")

        for _, row in self.swat_streams.iterrows():
            ############# defining hru node ################
            if pd.notna(row['hru_id']) and row['hru_elev'] > 0:
                node_attr = {
                    'node_role': 'hru',
                    'spatial_key': row['hru_id'],
                }
                node_features = [row['hru_elev'],
                                 row['hru_area'],
                                 row['hru_lat'],
                                 row['hru_lon']]
                
                G.add_node(row['hru_id'], **node_attr, feature=node_features)

            ############# defining cell node ################
            if pd.notna(row['cell_id']):
                node_attr = {
                    'node_role': 'gw_cell', 
                    'spatial_key': row['cell_id'], 
                }
                node_features = [row['cell_area'],
                                 row['overlap_area_m2']]
                G.add_node(row['cell_id'], **node_attr, feature=node_features)

            ############# defining channel node ################
            if pd.notna(row['channel_id']):
                node_attr = {
                    'node_role': 'channel', 
                    'spatial_key': row['channel_id'],
                }
                node_features = [row['channel_length_m'],
                                 row['drop']]
                G.add_node(row['channel_id'], **node_attr, feature=node_features)

            # Ensure downstream channels (dslinkno) are added as nodes
            if pd.notna(row['dslinkno']) and not G.has_node(row['dslinkno']):
                node_attr = {
                    'node_role': 'channel', 
                    'spatial_key': row['dslinkno'],
                }
                G.add_node(row['dslinkno'], **node_attr)

            ############# defining weather station node ################
            if pd.notna(row['wst_id']) and row['wst_id'] in unique_wst_ids:
                node_attr = {
                    'spatial_key': row['wst_id'],
                    'node_role': 'weather_station'
                }
                # Assuming temporal data are features
                temporal_array = load_climate_datas(self.txtintout_path, row['wst'], start_year=2000, end_year=2020)
                print(f"temporal_array shape: {temporal_array.shape}")
                logging.info(f"temporal_array shape for node {row['wst_id']}: {temporal_array.shape}")
                logging.info(f"Number of features in temporal_array for node {row['wst_id']}: {temporal_array.shape[1]}")
     
                node_features = temporal_array   ### the size of temporal data is (7671, 6)
                G.add_node(row['wst_id'], **node_attr, feature=node_features)
                self.temporal_data[row['wst_id']] = temporal_array
                unique_wst_ids = unique_wst_ids[unique_wst_ids != row['wst_id']]

            ############# defining edge ################
            def add_edge(G, u, v, role):
                if not G.has_edge(u, v):
                    edge_role = {'edge_role': role}
                    G.add_edge(u, v, **edge_role)

            if pd.notna(row['wst_id']) and pd.notna(row['hru_id']):
                add_edge(G, row['wst_id'], row['hru_id'], 'climate')

            if pd.notna(row['channel_id']) and pd.notna(row['dslinkno']):
                add_edge(G, row['channel_id'], row['dslinkno'], 'hydrograph')

            if pd.notna(row['cell_id']) and pd.notna(row['hru_id']):
                add_edge(G,  row['hru_id'],row['cell_id'], 'sw_gw')

            if pd.notna(row['cell_id']) and pd.notna(row['channel_id']):
                add_edge(G, row['cell_id'], row['channel_id'], 'gw_sw')

            if pd.notna(row['hru_id']) and pd.notna(row['channel_id']):
                add_edge(G, row['hru_id'], row['channel_id'], 'sw')

        logging.info(f"Unique roles: {set(nx.get_node_attributes(G, 'node_role').values())}")
        logging.info(f"Created graph with {len(G.nodes())} nodes and {len(G.edges())} edges.")

        return G

    def process(self, name):

        os.makedirs(f'{self.swat_output_base_path}/{name}/Graphs', exist_ok=True)
        ### path to the SWAT txtintout folder
        self.txtintout_path = f'{self.swat_output_base_path}/{name}/SWAT_gwflow_MODEL/Scenarios/Default/TxtInOut'
        ## read the SWATGWFLOW connectivity file
        self.swat_streams = pd.read_csv(f'{self.swat_output_base_path}/{name}/Graphs/cell_hru_riv_wst.csv')
        logging.info(f"swat_streams columns: {self.swat_streams.columns}")

        self.swat_streams['wst_id'] = pd.factorize(self.swat_streams['wst'])[0]
        ### remove rows with hru_elev < 0
        self.swat_streams = self.swat_streams[self.swat_streams['hru_elev'] > 0]
        
        logging.info(f"Length of the SWAT streams before merging with graph con: {len(self.swat_streams)}")
        logging.info(f"Number of unique wst_id: {len(self.swat_streams['wst_id'].unique())}")
        logging.info(f"Number of unique linkno: {len(self.swat_streams['linkno'].unique())}")
        
        G = self.create_graph()
        ### Ensure all nodes have the 'node_role' and 'spatial_key' attributes
        for node, attr in G.nodes(data=True):
            if 'node_role' not in attr:
                logging.warning(f"Node {node} missing 'node_role'. Adding default value.")
                attr['node_role'] = 'unknown'
            if 'spatial_key' not in attr:
                logging.warning(f"Node {node} missing 'spatial_key'. Adding default value.")
                attr['spatial_key'] = node
        
        ### Ensure all nodes have features
        for node, attr in G.nodes(data=True):
            if 'feature' not in attr:
                logging.warning(f"Node {node} missing 'feature'. Adding default features.")
                attr['feature'] = np.random.rand(5).tolist()  # Example default feature

        logging.info(f"Unique roles: {set(nx.get_node_attributes(G, 'node_role').values())}")
        logging.info(f"Created graph with {len(G.nodes())} nodes and {len(G.edges())} edges.")

        ### print what we are saving
        print(G)
        ### Save the NetworkX graph and temporal data
        with open(f'{self.swat_output_base_path}/{name}/Graphs/SWAT_plus_streams.gpickle', 'wb') as f:
            pickle.dump(G, f)
        with open(f'{self.swat_output_base_path}/{name}/Graphs/temporal_data.pkl', 'wb') as f:
            pickle.dump(self.temporal_data, f)
        logging.info(f'Saved graph data to {self.swat_output_base_path}/{name}/Graphs/SWAT_plus_streams.gpickle')
        logging.info(f'Saved temporal data to {self.swat_output_base_path}/{name}/Graphs/temporal_data.pkl')

if __name__ == "__main__":
    
    swat_output_base_path = "/data/MyDataBase/SWATplus_by_VPUID/0000/huc12"
    processor = SWATGraphProcessor(swat_output_base_path, num_days=10)
    names = os.listdir(swat_output_base_path)
    
    names.remove("log.txt")

    for name in names:
        #if name != "04115000":
        #    continue
        print(f"Processing {name}")
        generate_relation_table(name)
        processor.process(name)
