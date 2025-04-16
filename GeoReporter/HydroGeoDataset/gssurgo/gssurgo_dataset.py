import h5py
import numpy as np
import os
import logging
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from typing import Tuple, Optional, List, Dict, Any
import fiona
from shapely.geometry import box, mapping
import matplotlib.pyplot as plt
try:
    from config import AgentConfig
except ImportError:
    from GeoReporter.config import AgentConfig
# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def get_coordinates(lat=None, lon=None, lat_range=None, lon_range=None) -> Tuple[Optional[Tuple], Optional[Tuple]]:
    """
    Get array indices from geographic coordinates.
    
    Args:
        lat: Target latitude for point query
        lon: Target longitude for point query
        lat_range: (min_lat, max_lat) for area query
        lon_range: (min_lon, max_lon) for area query
        
    Returns:
        Tuple containing:
            - For point query: (row_index, col_index), None
            - For area query: None, (min_row, max_row, min_col, max_col)
    """
    try:
        with h5py.File(AgentConfig.HydroGeoDataset_ML_250_path, 'r') as f:
            lat_ = f["geospatial/lat_250m"][:]
            lon_ = f["geospatial/lon_250m"][:]
            lat_ = np.where(lat_ == -999, np.nan, lat_)
            lon_ = np.where(lon_ == -999, np.nan, lon_)

            if lat is not None and lon is not None:
                valid_mask = ~np.isnan(lat_) & ~np.isnan(lon_)
                coordinates = np.column_stack((lat_[valid_mask], lon_[valid_mask]))
                from scipy.spatial import cKDTree
                _, idx = cKDTree(coordinates).query([lat, lon])
                valid_indices = np.where(valid_mask)
                return (valid_indices[0][idx], valid_indices[1][idx]), None

            if lat_range and lon_range:
                min_lat, max_lat = lat_range
                min_lon, max_lon = lon_range
                mask = (lat_ >= min_lat) & (lat_ <= max_lat) & (lon_ >= min_lon) & (lon_ <= max_lon)
                if np.any(mask):
                    rows, cols = np.where(mask)
                    return None, (np.min(rows), np.max(rows), np.min(cols), np.max(cols))
                else:
                    logger.warning("No data points found within the specified geographic range")
    except Exception as e:
        logger.error(f"Error determining coordinates: {e}")
        
    return None, None


def extract_gssurgo_data(bounding_box, output_dir=None):
    """
    Extract GSSURGO data for a specified bounding box.
    
    Args:
        bounding_box: List [min_lon, min_lat, max_lon, max_lat]
        output_dir: Directory to save output files (optional)
    
    Returns:
        Dictionary containing extracted GSSURGO parameters
    """
    path = AgentConfig.HydroGeoDataset_ML_250_path
    results = {}
    
    print("Opening HDF5 file...")
    with h5py.File(path, "r") as f:
        # Print available datasets in gssurgo
        gssurgo = f['gssurgo']
        print(f"Available GSSURGO parameters: {list(gssurgo.keys())}")
        
        # Extract coordinates
        lat = f['geospatial/lat_250m'][:]
        lon = f['geospatial/lon_250m'][:]
        print(f"Coordinate dimensions: {lat.shape}, {lon.shape}")
        
        # Find indices corresponding to the bounding box
        print("Finding coordinates within bounding box...")
        lat_range = (bounding_box[1], bounding_box[3])
        lon_range = (bounding_box[0], bounding_box[2])
        point_idx, range_idx = get_coordinates(lat_range=lat_range, lon_range=lon_range)
        
        if range_idx is None:
            print("No coordinates found within bounding box!")
            return {}
        
        min_row, max_row, min_col, max_col = range_idx
        print(f"Bounding box indices: rows [{min_row}, {max_row}], cols [{min_col}, {max_col}]")
        
        # Extract each GSSURGO parameter within the bounding box
        for param in gssurgo.keys():
            try:
                data = gssurgo[param][min_row:max_row+1, min_col:max_col+1]
                # Replace fill values with NaN
                data = np.where(data == -999, np.nan, data)
                results[param] = data
                print(f"Extracted {param} with shape {data.shape}")
                
                # Save visualization if output_dir is provided
                if output_dir is not None:
                    os.makedirs(output_dir, exist_ok=True)
                    if data.size > 0 and not np.all(np.isnan(data)):
                        plt.figure(figsize=(10, 8))
                        plt.imshow(data, cmap='viridis')
                        plt.colorbar(label=param)
                        plt.title(f"GSSURGO {param}")
                        plt.savefig(os.path.join(output_dir, f"gssurgo_{param}.png"))
                        plt.close()
            except Exception as e:
                print(f"Error extracting {param}: {e}")
    
    return results

# Example usage when run directly
if __name__ == "__main__":
    bounding_box= [-85.444332, 43.158148, -84.239256, 44.164683]
    # Extract GSSURGO data for the specified bounding box
    results = extract_gssurgo_data(bounding_box, output_dir="gssurgo_output")
    print(results.keys())