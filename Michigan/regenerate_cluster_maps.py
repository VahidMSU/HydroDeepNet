#!/usr/bin/env python3
"""
Script to regenerate watershed cluster maps.
This will force a recalculation of the clusters and regenerate the maps.
"""

import os
import sys
import time

# Add the utilities module to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Import the necessary function
    from utils.geolocate import get_model_cluster_mapping, plot_clusters_on_map
    import pandas as pd
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def main():
    print("Starting cluster map regeneration...")
    
    # Define the base path to SWAT model directories - use the default path
    base_path = "/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12"
    
    # Force recalculation of clusters
    print(f"Forcing recalculation of clusters from {base_path}")
    name_to_cluster = get_model_cluster_mapping(base_path, force_recalculate=True)
    
    # Additional step: directly try to create the map from the cached CSV
    cache_dir = "./Michigan/cache"
    cluster_points_file = os.path.join(cache_dir, "cluster_points.csv")
    
    if os.path.exists(cluster_points_file):
        try:
            print(f"Reading cluster points from {cluster_points_file}")
            df = pd.read_csv(cluster_points_file)
            
            # Ensure output directory exists
            os.makedirs('./Michigan/figs', exist_ok=True)
            
            # Try to create the map directly
            print("Attempting to create Michigan map directly...")
            output_path = plot_clusters_on_map(df, './Michigan/figs/michigan_clusters_map.png')
            if output_path:
                print(f"Successfully created Michigan map at {output_path}")
            else:
                print("Failed to create Michigan map directly")
        except Exception as e:
            print(f"Error creating map directly: {e}")
    
    print("Cluster map regeneration completed")

if __name__ == "__main__":
    main()