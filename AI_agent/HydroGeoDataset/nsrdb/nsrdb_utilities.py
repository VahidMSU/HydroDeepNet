"""
NSRDB (National Solar Radiation Database) data processing and visualization utilities.

This module provides functions for analyzing, visualizing, and exporting
NSRDB data extracted from HDF5 datasets, similar to the PRISM utilities.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import os
import logging
import geopandas as gpd
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime, timedelta
import seaborn as sns
import calendar
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata
import rasterio
from rasterio.transform import from_origin
from utils.plot_utils import safe_figure, save_figure
# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Set matplotlib style for consistent visuals
plt.style.use('seaborn-v0_8-whitegrid')

# NSRDB variable information for consistent processing
NSRDB_VARIABLES = {
    'ghi': {
        'description': 'Global Horizontal Irradiance',
        'units': 'W/m²',
        'color_map': 'YlOrRd',
        'line_color': '#ff7f0e',
        'aggregation': 'mean',
        'scale_factor': 1.0,  # Corrected: no scaling needed (factor = 1.0)
        'display_name': 'Solar Irradiance'
    },
    'wind_speed': {
        'description': 'Wind Speed',
        'units': 'm/s',
        'color_map': 'Blues',
        'line_color': '#1f77b4',
        'aggregation': 'mean',
        'scale_factor': 0.1,  # Correct: divide by 10 (factor = 10.0)
        'display_name': 'Wind Speed'
    },
    'relative_humidity': {
        'description': 'Relative Humidity',
        'units': '%',
        'color_map': 'BuPu',
        'line_color': '#9467bd',
        'aggregation': 'mean',
        'scale_factor': 0.01,  # Correct: divide by 100 (factor = 100.0)
        'display_name': 'Humidity'
    }
}

def get_coordinates_from_bbox(coor_path: str, bbox: List[float]) -> gpd.GeoDataFrame:
    """
    Get coordinates within a bounding box from the NSRDB coordinates file.
    
    Args:
        coor_path: Path to the NSRDB coordinates index shapefile
        bbox: Bounding box as [lon_min, lat_min, lon_max, lat_max]
        
    Returns:
        GeoDataFrame of coordinates within the bounding box
    """
    try:
        file = gpd.read_file(coor_path)
        logger.info(f"Loaded {len(file)} coordinates from {coor_path}")
        
        # Extract coordinates within the bounding box
        index = file.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        
        logger.info(f"Found {len(index)} coordinates within the bounding box")
        return index
    except Exception as e:
        logger.error(f"Error getting coordinates from bbox: {e}")
        return gpd.GeoDataFrame()

def extract_nsrdb_data(year: int, coordinates_index: gpd.GeoDataFrame,
                      nsrdb_path_template: str = "/data/SWATGenXApp/GenXAppData/NSRDB/nsrdb_{}_full_filtered.h5") -> Dict[str, np.ndarray]:
    """
    Extract NSRDB data for a given year and set of coordinates.
    
    Args:
        year: Year to extract data for
        coordinates_index: GeoDataFrame with NSRDB_inde column for data lookup
        nsrdb_path_template: Path template for NSRDB HDF5 files with {} for year
        
    Returns:
        Dictionary with arrays for each variable
    """
    try:
        path = nsrdb_path_template.format(year)
        if not os.path.exists(path):
            logger.error(f"NSRDB file not found: {path}")
            return {}
            
        nsrdb_indices = coordinates_index.NSRDB_inde.values
        
        with h5py.File(path, 'r') as f:
            # Get all variable data
            variables = {}
            for var_name in NSRDB_VARIABLES.keys():
                if var_name in f:
                    variables[var_name] = f[var_name][:, nsrdb_indices]
                    logger.info(f"Extracted {var_name} data with shape {variables[var_name].shape}")
                else:
                    logger.warning(f"Variable {var_name} not found in {path}")
                    
            # Also get time index if available
            if 'time_index' in f:
                variables['time_index'] = f['time_index'][:]
                
        return variables
    except Exception as e:
        logger.error(f"Error extracting NSRDB data: {e}")
        return {}

def extract_nsrdb_multiyear(years: List[int], coordinates_index: gpd.GeoDataFrame,
                           nsrdb_path_template: str = "/data/SWATGenXApp/GenXAppData/NSRDB/nsrdb_{}_full_filtered.h5") -> Dict[str, np.ndarray]:
    """
    Extract NSRDB data for multiple years and combine them.
    
    Args:
        years: List of years to extract data for
        coordinates_index: GeoDataFrame with NSRDB_inde column for data lookup
        nsrdb_path_template: Path template for NSRDB HDF5 files with {} for year
        
    Returns:
        Dictionary with combined arrays for each variable
    """
    try:
        all_data = {}
        
        for year in years:
            year_data = extract_nsrdb_data(year, coordinates_index, nsrdb_path_template)
            
            if not year_data:
                logger.warning(f"No data extracted for year {year}")
                continue
                
            # Initialize or append data
            for var_name, var_data in year_data.items():
                if var_name == 'time_index':
                    # Special handling for time_index
                    continue
                    
                if var_name not in all_data:
                    all_data[var_name] = var_data
                else:
                    # Append along the time dimension (first axis)
                    all_data[var_name] = np.concatenate([all_data[var_name], var_data], axis=0)
                    
            logger.info(f"Extracted data for year {year}")
                    
        # Report final data shapes
        for var_name, var_data in all_data.items():
            logger.info(f"Final {var_name} data shape: {var_data.shape}")
            
        return all_data
    except Exception as e:
        logger.error(f"Error in multi-year extraction: {e}")
        return {}

def create_interpolated_grid(data: Dict[str, np.ndarray], coordinates_index: gpd.GeoDataFrame, 
                           bbox: List[float], resolution: float = 0.01) -> Dict[str, np.ndarray]:
    """
    Create interpolated grids for each variable and time step.
    
    Args:
        data: Dictionary with variable arrays
        coordinates_index: GeoDataFrame with coordinate information
        bbox: Bounding box as [lon_min, lat_min, lon_max, lat_max]
        resolution: Grid resolution in degrees
        
    Returns:
        Dictionary with interpolated grid arrays
    """
    try:
        # Create grid
        x = np.arange(bbox[0], bbox[2] + resolution, resolution)
        y = np.arange(bbox[1], bbox[3] + resolution, resolution)
        grid_x, grid_y = np.meshgrid(x, y)
        
        # Get coordinates as points
        points = np.column_stack((coordinates_index.longitude, coordinates_index.latitude))
        
        # Initialize output
        grid_data = {}
        
        # For demonstration, interpolate first time step of each variable
        for var_name, var_data in data.items():
            if var_name == 'time_index':
                continue
                
            # Get scale factor from our defined dictionary
            scale_factor = NSRDB_VARIABLES[var_name]['scale_factor']
            
            # For first time step only (for efficiency)
            time_step = 0
            values = var_data[time_step, :] * scale_factor
            
            # Interpolate
            grid = griddata(points, values, (grid_x, grid_y), method='linear')
            grid_data[var_name] = grid
            
        return grid_data
    except Exception as e:
        logger.error(f"Error creating interpolated grid: {e}")
        return {}

def save_as_raster(data: np.ndarray, output_path: str, bbox: List[float], crs: str) -> bool:
    """
    Save a numpy array as a GeoTIFF raster.
    
    Args:
        data: 2D numpy array of data
        output_path: Path to save the raster
        bbox: Bounding box as [lon_min, lat_min, lon_max, lat_max]
        crs: Coordinate reference system
        
    Returns:
        Boolean indicating success
    """
    try:
        height, width = data.shape
        
        # Calculate resolution
        lon_res = (bbox[2] - bbox[0]) / (width - 1) if width > 1 else 0.01
        lat_res = (bbox[3] - bbox[1]) / (height - 1) if height > 1 else 0.01
        
        # Create transform from upper left corner
        transform = from_origin(bbox[0], bbox[3], lon_res, lat_res)
        
        # Create metadata
        meta = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'dtype': data.dtype,
            'crs': crs,
            'transform': transform,
            'nodata': np.nan
        }
        
        # Write raster
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(data, 1)
            
        logger.info(f"Saved raster to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving raster: {e}")
        return False

def aggregate_nsrdb_daily(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Aggregate 30-minute NSRDB data to daily values.
    
    Args:
        data: Dictionary with variable arrays
        
    Returns:
        Dictionary with daily aggregated arrays
    """
    try:
        daily_data = {}
        
        for var_name, var_data in data.items():
            if var_name == 'time_index':
                continue
                
            # Reshape to (days, timesteps_per_day, locations)
            # Assuming 48 half-hour intervals per day
            original_shape = var_data.shape
            days = original_shape[0] // 48
            
            # Skip if we don't have complete days
            if days * 48 != original_shape[0]:
                logger.warning(f"Skipping {var_name} - incomplete days")
                continue
                
            # Reshape and aggregate
            reshaped = var_data[:days*48].reshape(days, 48, -1)
            
            # Apply appropriate aggregation and scaling based on variable
            if var_name == 'ghi':
                # For GHI, calculate the daily average power (W/m²)
                # instead of converting to energy units
                scale = NSRDB_VARIABLES[var_name]['scale_factor']
                daily = reshaped * scale
                daily = daily.mean(axis=1)  # Take mean over 48 half-hour periods
            else:
                # For other variables, take the mean with appropriate scaling
                scale = NSRDB_VARIABLES[var_name]['scale_factor']
                daily = reshaped * scale
                daily = daily.mean(axis=1)
                
            daily_data[var_name] = daily
            logger.info(f"Aggregated {var_name} to daily values with shape {daily.shape}")
            
        return daily_data
    except Exception as e:
        logger.error(f"Error aggregating to daily values: {e}")
        return {}

def create_nsrdb_timeseries(data: Dict[str, np.ndarray], start_year: int, 
                          aggregation: str = 'daily',
                          output_path: Optional[str] = None) -> plt.Figure:
    """
    Create time series plots for NSRDB variables.
    
    Args:
        data: Dictionary with variable arrays
        start_year: Starting year for labeling
        aggregation: Time aggregation ('daily', 'monthly', etc.)
        output_path: Optional path to save the figure
        
    Returns:
        Matplotlib Figure object
    """
    try:
        with safe_figure(figsize=(12, 10)) as fig:
            # Create subplots
            gs = gridspec.GridSpec(len(data), 1)
            
            # Calculate time axis
            if aggregation == 'daily':
                days = data[list(data.keys())[0]].shape[0]
                dates = pd.date_range(start=f"{start_year}-01-01", periods=days)
                x = np.arange(days)
            else:
                # Default to using index numbers
                x = np.arange(data[list(data.keys())[0]].shape[0])
            
            # Plot each variable
            for i, (var_name, var_data) in enumerate(data.items()):
                if var_name == 'time_index':
                    continue
                    
                # Calculate spatial mean
                spatial_mean = np.nanmean(var_data, axis=1)
                
                # Create subplot
                ax = fig.add_subplot(gs[i])
                
                # Plot data
                color = NSRDB_VARIABLES[var_name]['line_color']
                ax.plot(x, spatial_mean, color=color, linewidth=1.5)
                
                # Add labels
                ax.set_ylabel(f"{NSRDB_VARIABLES[var_name]['description']} ({NSRDB_VARIABLES[var_name]['units']})")
                
                # Add title to top subplot
                if i == 0:
                    ax.set_title(f"NSRDB Climate Variables - {aggregation.capitalize()} Mean")
                
                # Only add x-label to bottom subplot
                if i == len(data) - 1:
                    if aggregation == 'daily':
                        ax.set_xlabel("Day of Year")
                        
                        # Add some date tick labels
                        tick_positions = np.linspace(0, len(x) - 1, min(12, len(x))).astype(int)
                        tick_labels = [dates[pos].strftime("%Y-%m-%d") for pos in tick_positions]
                        ax.set_xticks(tick_positions)
                        ax.set_xticklabels(tick_labels, rotation=45)
                    else:
                        ax.set_xlabel("Time Step")
                
                # Add grid
                ax.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            # Save if path provided
            if output_path:
                save_figure(fig, output_path)
                
            return fig
    except Exception as e:
        logger.error(f"Error creating time series: {e}")
        plt.close('all')
        return None

def create_nsrdb_map(grid_data: Dict[str, np.ndarray], bbox: List[float], 
                    output_path: Optional[str] = None) -> plt.Figure:
    """
    Create spatial maps for NSRDB variables.
    
    Args:
        grid_data: Dictionary with gridded data arrays
        bbox: Bounding box as [lon_min, lat_min, lon_max, lat_max]
        output_path: Optional path to save the figure
        
    Returns:
        Matplotlib Figure object
    """
    try:
        with safe_figure(figsize=(18, 6)) as fig:  # Wider figure for horizontal layout
            # Create subplots based on number of variables
            n_vars = len(grid_data)
            
            # Use horizontal layout (one row) for all variables
            n_rows = 1
            n_cols = n_vars
            
            # Plot each variable
            for i, (var_name, var_data) in enumerate(grid_data.items()):
                # Create subplot
                ax = fig.add_subplot(n_rows, n_cols, i + 1)
                
                # Create masked array to handle NaN values
                masked_data = np.ma.masked_invalid(var_data)
                
                # Get colormap
                cmap = plt.cm.get_cmap(NSRDB_VARIABLES[var_name]['color_map'])
                
                # Plot data
                im = ax.imshow(masked_data, cmap=cmap, origin='lower', 
                              extent=[bbox[0], bbox[2], bbox[1], bbox[3]])
                
                # Add colorbar
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(f"{NSRDB_VARIABLES[var_name]['description']} ({NSRDB_VARIABLES[var_name]['units']})")
                
                # Add title
                ax.set_title(NSRDB_VARIABLES[var_name]['display_name'])
                
                # Add lat/lon labels
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
            
            # Add overall title
            plt.suptitle("NSRDB Spatial Distribution", fontsize=16, y=1.05)
            
            plt.tight_layout()
            
            # Save if path provided
            if output_path:
                save_figure(fig, output_path)
                
            return fig
    except Exception as e:
        logger.error(f"Error creating spatial maps: {e}")
        plt.close('all')
        return None

def extract_for_swat(nsrdb_data: Dict[str, np.ndarray], coordinates_index: gpd.GeoDataFrame, 
                    output_dir: str, start_year: int) -> bool:
    """
    Extract and format NSRDB data for SWAT+ model, similar to NSRDB_SWATplus_extraction.py.
    
    Args:
        nsrdb_data: Dictionary with NSRDB data arrays
        coordinates_index: GeoDataFrame with coordinate information
        output_dir: Directory to save output files
        start_year: Starting year for output files
        
    Returns:
        Boolean indicating success
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Variable mapping
        swat_dict = {
            'ghi': 'slr',
            'wind_speed': 'wnd',
            'relative_humidity': 'hmd',
        }
        
        # Create daily aggregated data
        daily_data = aggregate_nsrdb_daily(nsrdb_data)
        
        if not daily_data:
            logger.error("Failed to aggregate daily data")
            return False
        
        # Create date range
        days = daily_data[list(daily_data.keys())[0]].shape[0]
        date_range = pd.date_range(start=f"{start_year}-01-01", periods=days)
        
        # Process each location
        for i, idx_row in coordinates_index.iterrows():
            nsrdb_idx = idx_row.NSRDB_inde
            lat = idx_row.latitude
            lon = idx_row.longitude
            
            # Find position of this index in the array
            idx_pos = np.where(coordinates_index.NSRDB_inde.values == nsrdb_idx)[0]
            if len(idx_pos) == 0:
                logger.warning(f"Index {nsrdb_idx} not found in data")
                continue
            
            idx_pos = idx_pos[0]
            
            # Create file for each variable
            for var_name, swat_ext in swat_dict.items():
                if var_name not in daily_data:
                    continue
                    
                # Create output filename based on row/col if available or coordinates
                if hasattr(idx_row, 'row') and hasattr(idx_row, 'col'):
                    output_file = os.path.join(output_dir, f"r{idx_row.row}_c{idx_row.col}.{swat_ext}")
                else:
                    # Use rounded coordinates if row/col not available
                    output_file = os.path.join(output_dir, f"lat{lat:.2f}_lon{lon:.2f}.{swat_ext}")
                
                with open(output_file, 'w') as f:
                    # Write header
                    f.write(f"NSRDB Index: {nsrdb_idx}\n")
                    f.write("nbyr nstep lat lon elev\n")
                    
                    # Calculate years
                    years = len(date_range) // 365  # Approximate
                    
                    # Write position data (elev set to 0 if not available)
                    elev = idx_row.get('elev', 0)
                    f.write(f"{years}\t0\t{lat:.4f}\t{lon:.4f}\t{elev:.1f}\n")
                    
                    # Write data for each day
                    for day_idx, date in enumerate(date_range):
                        # Get value for this location
                        value = daily_data[var_name][day_idx, idx_pos]
                        
                        # Write year, day of year, value
                        f.write(f"{date.year}\t{date.dayofyear}\t{value:.4f}\n")
            
        # Create .cli files for each variable
        for var_name, swat_ext in swat_dict.items():
            if var_name not in daily_data:
                continue
                
            cli_file = os.path.join(output_dir, f"{swat_ext}.cli")
            with open(cli_file, 'w') as f:
                f.write(f"NSRDB Climate Data - {NSRDB_VARIABLES[var_name]['description']}\n")
                f.write(f"{var_name} file\n")
                
                # Add entries for each location
                for i, idx_row in coordinates_index.iterrows():
                    if hasattr(idx_row, 'row') and hasattr(idx_row, 'col'):
                        f.write(f"r{idx_row.row}_c{idx_row.col}.{swat_ext}\n")
                    else:
                        lat = idx_row.latitude
                        lon = idx_row.longitude
                        f.write(f"lat{lat:.2f}_lon{lon:.2f}.{swat_ext}\n")
        
        logger.info(f"Extracted SWAT+ formatted files to {output_dir}")
        return True
    except Exception as e:
        logger.error(f"Error extracting for SWAT+: {e}")
        return False

def calculate_statistics(data: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """
    Calculate basic statistics for NSRDB variables.
    
    Args:
        data: Dictionary with variable arrays
        
    Returns:
        Dictionary with statistics for each variable
    """
    stats = {}
    
    for var_name, var_data in data.items():
        if var_name == 'time_index':
            continue
            
        # Calculate spatial mean for each time step
        spatial_mean = np.nanmean(var_data, axis=1)
        
        # Calculate statistics
        stats[var_name] = {
            'mean': np.nanmean(spatial_mean),
            'min': np.nanmin(spatial_mean),
            'max': np.nanmax(spatial_mean),
            'std': np.nanstd(spatial_mean),
            'temporal_variability': np.nanstd(spatial_mean) / np.nanmean(spatial_mean) if np.nanmean(spatial_mean) != 0 else 0,
            'spatial_variability': np.nanmean(np.nanstd(var_data, axis=1)) / np.nanmean(var_data) if np.nanmean(var_data) != 0 else 0
        }
        
    return stats

def calculate_monthly_averages(data: Dict[str, np.ndarray], start_year: int) -> Dict[str, np.ndarray]:
    """
    Calculate monthly averages for NSRDB variables.
    
    Args:
        data: Dictionary with daily aggregated variable arrays
        start_year: Starting year for the data
        
    Returns:
        Dictionary with monthly averages for each variable
    """
    try:
        monthly_avgs = {}
        
        # Create date range
        days = data[list(data.keys())[0]].shape[0]
        date_range = pd.date_range(start=f"{start_year}-01-01", periods=days)
        
        # Create DataFrame with dates
        df = pd.DataFrame({'date': date_range})
        df['year'] = df.date.dt.year
        df['month'] = df.date.dt.month
        
        # Calculate monthly averages for each variable
        for var_name, var_data in data.items():
            if var_name == 'time_index':
                continue
                
            # Reshape var_data to 2D if needed (time x locations)
            if var_data.ndim > 2:
                var_data_2d = var_data.reshape(var_data.shape[0], -1)
            else:
                var_data_2d = var_data
                
            # Calculate spatial mean for each time step
            spatial_mean = np.nanmean(var_data_2d, axis=1)
            
            # Add to DataFrame
            df[var_name] = spatial_mean
            
            # Group by year and month
            monthly = df.groupby(['year', 'month'])[var_name].agg(['mean', 'min', 'max', 'std']).reset_index()
            
            monthly_avgs[var_name] = monthly
            
        return monthly_avgs
    except Exception as e:
        logger.error(f"Error calculating monthly averages: {e}")
        return {}

def export_data_to_csv(data: Dict[str, np.ndarray], 
                     coordinates_index: gpd.GeoDataFrame,
                     output_path: str, 
                     start_year: int) -> bool:
    """
    Export NSRDB data to CSV format.
    
    Args:
        data: Dictionary with variable arrays
        coordinates_index: GeoDataFrame with coordinate information
        output_path: Path to save the CSV file
        start_year: Starting year for the data
        
    Returns:
        Boolean indicating success
    """
    try:
        # Create daily aggregated data
        daily_data = aggregate_nsrdb_daily(data)
        
        if not daily_data:
            logger.error("Failed to aggregate daily data for CSV export")
            return False
            
        # Create date range
        days = daily_data[list(daily_data.keys())[0]].shape[0]
        date_range = pd.date_range(start=f"{start_year}-01-01", periods=days)
        
        # Create DataFrame with dates
        df = pd.DataFrame({'date': date_range})
        df['year'] = df.date.dt.year
        df['month'] = df.date.dt.month
        df['day'] = df.date.dt.day
        df['doy'] = df.date.dt.dayofyear
        
        # Calculate spatial mean for each variable
        for var_name, var_data in daily_data.items():
            spatial_mean = np.nanmean(var_data, axis=1)
            df[var_name] = spatial_mean
            
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Exported data to {output_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error exporting to CSV: {e}")
        return False

if __name__ == "__main__":
    print("NSRDB utilities module loaded. Import to use functions.")
