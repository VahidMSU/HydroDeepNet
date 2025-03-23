"""
Module for creating error zones for the MODFLOW model.
This helps identify areas with different hydrogeological properties.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from MODGenX.logger_singleton import get_logger
import rasterio
from MODGenX.utils import load_raster, match_raster_dimensions

logger = get_logger()

def create_error_zones_and_save(model_path, load_raster_args, ML=False):
    """
    Create error zones based on parameter uncertainty and save as shapefiles.
    
    Parameters:
    -----------
    model_path : str
        Path to the MODFLOW model directory
    load_raster_args : dict
        Dictionary containing parameters for raster loading with path_handler
    ML : bool, optional
        Whether machine learning predictions were used
    """
    logger.info("Creating error zones for model parameters")
    
    # Ensure path_handler is provided
    assert 'path_handler' in load_raster_args, "path_handler is required in load_raster_args"
    path_handler = load_raster_args['path_handler']
    ref_raster_path = path_handler.get_ref_raster_path()
    fit_to_meter = path_handler.config.fit_to_meter
    
    # Get raster paths using path_handler
    raster_paths = path_handler.get_raster_paths(ML)
    
    # Create error paths dictionary from raster_paths
    error_paths = {}
    for key, path in raster_paths.items():
        if key.endswith('_er'):
            error_paths[key] = path
    
    # Create zones directory
    zones_dir = os.path.join(model_path, "error_zones")
    os.makedirs(zones_dir, exist_ok=True)
    
    try:
        # Load reference raster to get dimensions
        with rasterio.open(ref_raster_path) as src:
            ref_shape = src.shape
            ref_transform = src.transform
            ref_crs = src.crs
            
            # Get the geotransform to convert indices to coordinates
            xmin, xres, xskew, ymax, yskew, yres = src.transform.to_gdal()
        
        # Process each error raster
        summary = []
        processed_count = 0
        
        for param_name, error_path in error_paths.items():
            if os.path.exists(error_path):
                logger.info(f"Processing error zones for {param_name}")
                
                # Load error raster
                error_data = load_raster(error_path, load_raster_args)
                error_data = match_raster_dimensions(np.zeros(ref_shape), error_data)
                
                # Scale values by fit_to_meter for parameters in feet
                if param_name not in ["recharge_data_er"]:
                    error_data = error_data * fit_to_meter
                
                # Create zones based on percentiles
                # Zone 1: 0-25% (low uncertainty)
                # Zone 2: 25-50% (medium-low uncertainty)
                # Zone 3: 50-75% (medium-high uncertainty)
                # Zone 4: 75-100% (high uncertainty)
                valid_data = error_data[error_data > 0]
                if len(valid_data) == 0:
                    logger.warning(f"No valid data for {param_name}, skipping")
                    continue
                    
                # Calculate percentiles
                p25 = np.percentile(valid_data, 25)
                p50 = np.percentile(valid_data, 50)
                p75 = np.percentile(valid_data, 75)
                
                # Create zones
                zones = np.zeros_like(error_data, dtype=int)
                zones[(error_data > 0) & (error_data <= p25)] = 1
                zones[(error_data > p25) & (error_data <= p50)] = 2
                zones[(error_data > p50) & (error_data <= p75)] = 3
                zones[error_data > p75] = 4
                
                # Create a GeoDataFrame with zones
                rows, cols = np.where(zones > 0)
                zone_values = zones[rows, cols]
                
                # Convert indices to coordinates
                x_coords = xmin + cols * xres
                y_coords = ymax + rows * yres
                
                # Create geometries and attributes
                geometries = [Point(x, y) for x, y in zip(x_coords, y_coords)]
                gdf = gpd.GeoDataFrame({
                    'zone': zone_values,
                    'error': error_data[rows, cols],
                    'geometry': geometries
                }, crs=ref_crs)
                
                # Save to file
                output_file = os.path.join(zones_dir, f"{param_name}_zones.geojson")
                gdf.to_file(output_file, driver='GeoJSON')
                logger.info(f"Saved error zones for {param_name} to {output_file}")
                
                # Create visualization
                plt.figure(figsize=(10, 8))
                plt.imshow(zones, cmap='viridis')
                plt.colorbar(label='Zone')
                plt.title(f"Error Zones for {param_name}")
                plt.savefig(os.path.join(zones_dir, f"{param_name}_zones.png"))
                plt.close()
                
                # Add stats to summary
                for zone in range(1, 5):
                    zone_data = gdf[gdf['zone'] == zone]
                    if len(zone_data) > 0:
                        summary.append({
                            'parameter': param_name,
                            'zone': zone,
                            'count': len(zone_data),
                            'min_error': zone_data['error'].min(),
                            'max_error': zone_data['error'].max(),
                            'mean_error': zone_data['error'].mean()
                        })
                
                processed_count += 1
        
        # Save summary to file if we processed any data
        if summary:
            summary_df = pd.DataFrame(summary)
            summary_df.to_csv(os.path.join(zones_dir, "error_zones_summary.csv"), index=False)
            logger.info(f"Saved error zones summary to {os.path.join(zones_dir, 'error_zones_summary.csv')}")
            
        logger.info(f"Created error zones for {processed_count} parameters")
        return True
    
    except Exception as e:
        logger.error(f"Error creating error zones: {str(e)}")
        return False


