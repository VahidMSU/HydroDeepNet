import h5py
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.cluster import KMeans




def models_variables(path):

    with h5py.File(path, 'r') as f:
        variables = list(f['hru_wb_30m/2001/1'].keys())
    return variables


def landuse_lookup(lookup_table):
    lookup = pd.read_csv(lookup_table)
    return lookup

def get_model_path(path, name):
    """Get the centroid of a model's geometry"""
    model_path = os.path.join(path, name, f"SWAT_gwflow_MODEL/gwflow_gis/Subbasin_shape.shp")
    
    gdf = gpd.read_file(model_path).to_crs("EPSG:4326") 
    unioned_geo = gdf['geometry'].union_all()
    unioned_geo_centroid = unioned_geo.centroid

    return unioned_geo_centroid

def plot_clusters_on_map(df, output_path='./Michigan/figs/michigan_clusters_map.png'):
    """
    Plot clusters on a Michigan map using contextily for basemap
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame with columns 'name', 'x', 'y', 'cluster'
    output_path : str
        Path to save the output map
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        import contextily as ctx
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        import geopandas as gpd
        from shapely.geometry import Point
        
        # Convert to GeoDataFrame
        geometry = [Point(x, y) for x, y in zip(df['x'], df['y'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        
        # Create a colormap for the clusters
        n_clusters = len(df['cluster'].unique())
        colors = plt.cm.tab10(range(n_clusters))
        cmap = ListedColormap(colors)
        
        # Convert to Web Mercator for contextily
        gdf_web = gdf.to_crs(epsg=3857)
        
        # Michigan Lower Peninsula bounding box (approximate)
        michigan_bounds = {
            'north': 45.8, 
            'south': 41.5, 
            'west': -87.5, 
            'east': -82.3
        }
        
        # Convert bounds to Web Mercator
        bounds_points = [
            Point(michigan_bounds['west'], michigan_bounds['south']),
            Point(michigan_bounds['east'], michigan_bounds['north'])
        ]
        bounds_gdf = gpd.GeoDataFrame(geometry=bounds_points, crs="EPSG:4326").to_crs(epsg=3857)
        x_min, y_min, x_max, y_max = bounds_gdf.total_bounds
        
        # Create plot
        fig, ax = plt.subplots(figsize=(15, 15))
        
        # Plot points with clusters
        gdf_web.plot(column='cluster', ax=ax, markersize=100, cmap=cmap, 
                   edgecolor='black', legend=True, categorical=True,
                   legend_kwds={'loc': 'upper left', 'title': 'Cluster'})
        
        # Add basemap
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom=8)
        
        # Set plot extent to Michigan Lower Peninsula
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Add title and labels
        plt.title("Watershed Clusters in Michigan's Lower Peninsula", fontsize=16)
        
        # Remove axis labels as they're web mercator coordinates
        ax.set_axis_off()
        
        # Add scale bar and north arrow
        # Scale bar
        scale_bar_length = 50000  # 50 km in meters
        x_offset = (x_max - x_min) * 0.05
        y_offset = (y_max - y_min) * 0.05
        ax.plot([x_min + x_offset, x_min + x_offset + scale_bar_length], 
                [y_min + y_offset, y_min + y_offset], 'k-', lw=2)
        ax.text(x_min + x_offset + scale_bar_length/2, y_min + y_offset*2, 
                '50 km', ha='center', fontsize=12)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Map with clusters saved to {output_path}")
        return output_path
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install contextily with: pip install contextily")
        return None

def get_model_cluster_mapping(base_path):
    """Get mapping from model name to cluster based on geographical location"""
    NAMES = os.listdir(base_path)
    NAMES = [name for name in NAMES if os.path.isdir(os.path.join(base_path, name))]
    
    names_coord = []
    for name in NAMES:
        try:
            model_path = os.path.join(base_path, name, f"SWAT_gwflow_MODEL/gwflow_gis/Subbasin_shape.shp")
            if os.path.exists(model_path):
                gdf = gpd.read_file(model_path).to_crs("EPSG:4326") 
                unioned_geo = gdf['geometry'].union_all()
                unioned_geo_centroid = unioned_geo.centroid
                names_coord.append((name, unioned_geo_centroid.x, unioned_geo_centroid.y))
            else:
                print(f"Warning: Shapefile not found for {name}")
        except Exception as e:
            print(f"Error processing shapefile for {name}: {e}")
    
    df = pd.DataFrame(names_coord, columns=['name', 'x', 'y'])
    
    # Cluster the watersheds
    kmeans = KMeans(n_clusters=5, random_state=0).fit(df[['x', 'y']])
    df['cluster'] = kmeans.labels_
    
    # Ensure output directory exists
    os.makedirs('./Michigan/figs', exist_ok=True)
    
    # Plot the clusters on standard scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    for cluster in df['cluster'].unique():
        ax.scatter(df[df['cluster'] == cluster]['x'], df[df['cluster'] == cluster]['y'], label=f'Cluster {cluster}')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Watershed Clusters', fontsize=14)
    # Move legend to the left side
    ax.legend(loc='upper left', fontsize=10, frameon=True, framealpha=0.9)
    plt.tight_layout()
    plt.savefig('./Michigan/figs/clusters.png', dpi=300)
    
    # Also create a proper map with Michigan context
    try:
        plot_clusters_on_map(df, './Michigan/figs/michigan_clusters_map.png')
    except Exception as e:
        print(f"Could not create Michigan map: {e}")
    
    # Return mapping from name to cluster
    return {row['name']: row['cluster'] for _, row in df.iterrows()}