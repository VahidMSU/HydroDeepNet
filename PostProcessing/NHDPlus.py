import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import os
def generate_data(layer='WBDHU8', name='huc8'):
    all_df = []
    VPUID_path = f"/data/MyDataBase/CIWRE-BAE/NHDPlusData/SWATPlus_NHDPlus/"
    VPUIDs = os.listdir(VPUID_path)
    ## keep only those starting with 04
    #VPUIDs = [VPUID for VPUID in VPUIDs if VPUID.startswith("04")]
    VPUIDs = ["0405", "0406", "0407", "0408", "0409", "0410"]#, "0411", "0412"]
    for VPUID in VPUIDs:
        path = f"/data/MyDataBase/CIWRE-BAE/NHDPlusData/SWATPlus_NHDPlus/{VPUID}/unzipped_NHDPlusVPU/NHDPLUS_H_{VPUID}_HU4_GDB.gdb"
        try:
            df = gpd.read_file(path, driver='FileGDB', layer=layer).to_crs('EPSG:4326')
        except:
            continue

        all_df.append(df)

    all_df = gpd.GeoDataFrame(pd.concat(all_df, ignore_index=True))
    ## save the model bounds
    os.makedirs("model_bounds", exist_ok=True)
    all_df.to_pickle(f"model_bounds/{name}_model_bounds.pkl")
    #plot
    fig, ax = plt.subplots(figsize=(15, 10))
    all_df.boundary.plot(ax=ax, color='black', linewidth=1)
    all_df.plot(ax=ax, alpha=1,  edgecolor='black', linewidth=3, facecolor='none')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    ### average huc8 area in sqkm
    avg_area = all_df.to_crs("EPSG:26990").area.quantile(0.10)/10**6
    plt.title(f'# {len(all_df)} model bounds with average area {avg_area:.2f} (km\u00b2)')
    plt.savefig(f'model_bounds/{name}_model_bounds.png', dpi=300)



import os
import geopandas as gpd
import networkx as nx

def generate_huc12_list(HUC8, VPUID):
    path = f"/data/MyDataBase/CIWRE-BAE/NHDPlusData/SWATPlus_NHDPlus/{VPUID}/unzipped_NHDPlusVPU/"
    gdb = os.listdir(path)
    gdb = [g for g in gdb if g.endswith('.gdb')]
    path = os.path.join(path, gdb[0])

    huc12 = gpd.read_file(path, driver='FileGDB', layer='WBDHU12').to_crs('EPSG:4326')
    huc8 = gpd.read_file(path, driver='FileGDB', layer='WBDHU8').to_crs('EPSG:4326')

    # Intersect HUC8 with HUC12
    huc12 = gpd.overlay(huc12, huc8, how='intersection')
    huc12_grouped = huc12.groupby('HUC8')

    # Helper function to find clusters within HUC12s
    def find_clusters(huc12_group):
        clusters = []
        # Create a graph where nodes are HUC12 geometries and edges represent touching geometries
        G = nx.Graph()
        for idx, geom in enumerate(huc12_group.geometry):
            G.add_node(idx, geometry=geom)

        # Add edges for geometries that touch each other
        for i, geom1 in enumerate(huc12_group.geometry):
            for j, geom2 in enumerate(huc12_group.geometry):
                if i != j and geom1.touches(geom2):
                    G.add_edge(i, j)

        # Find connected components (clusters) in the graph
        for component in nx.connected_components(G):
            cluster = huc12_group.iloc[list(component)]
            clusters.append(cluster['HUC12'].values)

        return clusters

    # Process each HUC8 group to find clusters
    huc12_clusters = {}
    for huc8, group in huc12_grouped:
        clusters = find_clusters(group)
        for i, cluster in enumerate(clusters, start=1):
            huc12_clusters[f"{huc8}_cluster_{i}"] = cluster
    ## now loop and plot
    for huc8 in huc12_clusters.keys():
        fig, ax = plt.subplots(figsize=(15, 10))
        huc12[huc12['HUC12'].isin(huc12_clusters[huc8])].boundary.plot(ax=ax, color='black', linewidth=1)
        huc12[huc12['HUC12'].isin(huc12_clusters[huc8])].plot(ax=ax, alpha=1,  edgecolor='black', linewidth=3, facecolor='none')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'{huc8} with {len(huc12_clusters[huc8])} HUC12s')
        plt.savefig(f'model_bounds/{huc8}.png', dpi=300)

    return huc12_clusters

# Example usage:
# huc12_clusters = generate_huc12_list('some_HUC8', 'some_VPUID')


if __name__ == '__main__':

    #generate_data()
    VPUID = "0405"
    HUC8 = "04050001"
    huc12_dict = generate_huc12_list(HUC8, VPUID)

    print(huc12_dict)