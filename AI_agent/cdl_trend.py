import h5py
import numpy as np
from scipy.spatial import cKDTree
import pandas as pd
import os
import time

try:
    from config import AgentConfig
except ImportError:
    from AI_agent.config import AgentConfig
    
def cdl_trends(config):
    start_time = time.time()
    print("Starting CDL data extraction...")
    
    lat_range = (config['bounding_box'][1], config['bounding_box'][3])
    lon_range = (config['bounding_box'][0], config['bounding_box'][2])
    
    # Make sure years is always a list of integers
    if isinstance(config['start_year'], str):
        config['start_year'] = int(config['start_year'])
    if isinstance(config['end_year'], str):
        config['end_year'] = int(config['end_year'])
        
    years = list(range(config['start_year'], config['end_year']+1))
    print(f"Querying years: {years}")

    extracted_data = {}

    # First check which years are available
    available_years = []
    try:
        with h5py.File(AgentConfig.HydroGeoDataset_ML_250_path, 'r') as f:
            for year in years:
                if f"CDL/{year}" in f:
                    available_years.append(year)
                else:
                    print(f"Year {year} not found in dataset")

        if not available_years:
            print(f"Warning: None of the requested years {years} found in dataset")
            # Try to find any available years in the dataset as fallback
            with h5py.File(AgentConfig.HydroGeoDataset_ML_250_path, 'r') as f:
                if 'CDL' in f:
                    all_years = [int(y) for y in list(f['CDL'].keys()) if y.isdigit()]
                    print(f"Available years in dataset: {all_years}")
                    if all_years:
                        closest_years = sorted(all_years, key=lambda y: min(abs(y - yr) for yr in years))
                        use_years = closest_years[:3]  # Use up to 3 closest years
                        print(f"Using fallback years: {use_years}")
                        years = use_years
    except Exception as e:
        print(f"Error checking available years: {e}")

    # Process each year
    for year in years:
        year_start = time.time()
        try:
            data = read_h5_file(
                lat_range=lat_range,
                lon_range=lon_range,
                address=f"CDL/{year}",
            )
            if data:
                # Store data using integer keys to ensure consistency
                extracted_data[int(year)] = data
                print(f"CDL data extraction for year {year} took {time.time() - year_start:.2f} seconds")
        except Exception as e:
            print(f"Error extracting CDL data for year {year}: {e}")

    total_time = time.time() - start_time
    print(f"Total CDL data extraction took {total_time:.2f} seconds")
    
    if not extracted_data:
        print("Warning: No data could be extracted for any year")
    else:
        print(f"Successfully extracted data for years: {list(extracted_data.keys())}")

    return extracted_data


def CDL_lookup(code):
    df = pd.read_csv(AgentConfig.CDL_CODES_path)
    return df[df['CODE'] == code].NAME.values[0]

def get_coordinates(lat=None, lon=None, lat_range=None, lon_range=None):
    with h5py.File(AgentConfig.HydroGeoDataset_ML_250_path, 'r') as f:
        lat_ = f["geospatial/lat_250m"][:]
        lon_ = f["geospatial/lon_250m"][:]
        lat_ = np.where(lat_ == -999, np.nan, lat_)
        lon_ = np.where(lon_ == -999, np.nan, lon_)

        if lat is not None and lon is not None:
            valid_mask = ~np.isnan(lat_) & ~np.isnan(lon_)
            coordinates = np.column_stack((lat_[valid_mask], lon_[valid_mask]))
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
    
    return None, None


def read_h5_file(address, lat=None, lon=None, lat_range=None, lon_range=None, path = AgentConfig.HydroGeoDataset_ML_250_path):
    
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None

    try:
        point_idx, range_idx = get_coordinates(lat, lon, lat_range, lon_range)
        with h5py.File(path, 'r') as f:
            if point_idx:
                data = f[address][point_idx[0], point_idx[1]]
                return {"value": CDL_lookup(data) if "CDL" in address else data}
            
            if range_idx:
                min_row, max_row, min_col, max_col = range_idx
                data = f[address][min_row:max_row+1, min_col:max_col+1]
                data = np.where(data == -999, np.nan, data)
                return process_data(data, address)
            
            return {"value": f[address][:]}
    except Exception as e:
        print(f"Error reading data: {e}")
        return None

def process_data(data, address):
    
    if "CDL" in address:
        unique, counts = np.unique(data, return_counts=True)
        cell_area_ha = 6.25
        result = {CDL_lookup(key): float(value * cell_area_ha) for key, value in zip(unique, counts)}
        result["Total Area"] = float(np.nansum(list(result.values())))
        result["unit"] = "hectares"
        return result

    return {
        "number of cells": int(data.size),
        "median": float(np.nanmedian(data).round(2)),
        "max": float(np.nanmax(data).round(2)),
        "min": float(np.nanmin(data).round(2)),
        "mean": float(np.nanmean(data).round(2)),
        "std": float(np.nanstd(data).round(2)),
    }


if __name__ == "__main__":
    config = {
        "RESOLUTION": 250,
        "aggregation": "annual",
        "start_year": 2010,
        "end_year": 2012,
        "bounding_box": [-85.5, 42.5, -85.0, 43.0]
    }
    cdl_trends(config)
    print("Done.")