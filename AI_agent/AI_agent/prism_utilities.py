"""
PRISM climate data processing and visualization utilities.

This module provides functions for analyzing, visualizing, and exporting
PRISM (Parameter-elevation Regressions on Independent Slopes Model) climate data
extracted from HDF5 datasets.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import h5py
import os
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime, timedelta
import seaborn as sns
import calendar
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from AI_agent.plot_utils import safe_figure, save_figure

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

# PRISM variable information for consistent processing
PRISM_VARIABLES = {
    'ppt': {
        'description': 'Precipitation',
        'units': 'mm',
        'color_map': 'Blues',
        'aggregation': 'sum',
        'scale_factor': 1.0,
        'display_name': 'Precipitation'
    },
    'tmax': {
        'description': 'Maximum Temperature',
        'units': '°C',
        'color_map': 'hot_r',
        'aggregation': 'mean',
        'scale_factor': 1.0,
        'display_name': 'Max Temperature'
    },
    'tmin': {
        'description': 'Minimum Temperature',
        'units': '°C',
        'color_map': 'cool',
        'aggregation': 'mean',
        'scale_factor': 1.0,
        'display_name': 'Min Temperature'
    },
    'tmean': {
        'description': 'Mean Temperature',
        'units': '°C',
        'color_map': 'RdYlBu_r',
        'aggregation': 'mean',
        'scale_factor': 1.0,
        'display_name': 'Mean Temperature'
    }
}

def get_rowcol_range_by_latlon(database_path: str, desired_min_lat: float, desired_max_lat: float, 
                          desired_min_lon: float, desired_max_lon: float, buffer: float = 0.01) -> Tuple[int, int, int, int]:
    """
    Get row and column indices from the HDF5 file for a given lat/lon bounding box.
    
    Args:
        database_path: Path to the HDF5 file
        desired_min_lat: Minimum latitude
        desired_max_lat: Maximum latitude
        desired_min_lon: Minimum longitude
        desired_max_lon: Maximum longitude
        buffer: Buffer distance in degrees to expand the search area
        
    Returns:
        Tuple of (min_row, max_row, min_col, max_col) indices
    """
    try:
        with h5py.File(database_path, 'r') as f:
            lat_ = f["geospatial/lat_250m"][:]
            lon_ = f["geospatial/lon_250m"][:]
            
            # Clean up invalid values
            lat_ = np.where(lat_ == -999, np.nan, lat_)
            lon_ = np.where(lon_ == -999, np.nan, lon_)

            # Add buffer to improve matching
            lat_mask = (lat_ >= (desired_min_lat - buffer)) & (lat_ <= (desired_max_lat + buffer))
            lon_mask = (lon_ >= (desired_min_lon - buffer)) & (lon_ <= (desired_max_lon + buffer))
            combined_mask = lat_mask & lon_mask

            if not np.any(combined_mask):
                logger.warning("No exact matches found, expanding search area...")
                buffer = buffer * 5  # Expand buffer
                lat_mask = (lat_ >= (desired_min_lat - buffer)) & (lat_ <= (desired_max_lat + buffer))
                lon_mask = (lon_ >= (desired_min_lon - buffer)) & (lon_ <= (desired_max_lon + buffer))
                combined_mask = lat_mask & lon_mask

            if np.any(combined_mask):
                row_indices, col_indices = np.where(combined_mask)
                return (
                    np.min(row_indices),
                    np.max(row_indices),
                    np.min(col_indices),
                    np.max(col_indices)
                )
            
            logger.error("Could not find valid points for the given coordinates")
            return None, None, None, None
            
    except Exception as e:
        logger.error(f"Error getting row/col range: {e}")
        return None, None, None, None

def get_mask(database_path: str, resolution: int = 250) -> np.ndarray:
    """
    Get the base mask for the given resolution.
    
    Args:
        database_path: Path to the HDF5 file
        resolution: Grid resolution in meters
        
    Returns:
        Numpy array containing the mask
    """
    try:
        with h5py.File(database_path, 'r') as f:
            DEM_ = f[f"geospatial/BaseRaster_{resolution}m"][:]
            return np.where(DEM_ == -999, 0, 1)
    except Exception as e:
        logger.error(f"Error getting mask: {e}")
        return np.array([])

def extract_prism_data(prism_path: str, base_path: str, start_year: int, end_year: int,
                      bounding_box: Optional[Tuple[float, float, float, float]] = None,
                      aggregation: str = 'annual', resolution: int = 250) -> Dict[str, np.ndarray]:
    """
    Extract PRISM climate data for a given period and region.
    
    Args:
        prism_path: Path to the PRISM HDF5 file
        base_path: Path to the base HDF5 file with geographic data
        start_year: First year to extract
        end_year: Last year to extract
        bounding_box: Optional [min_lon, min_lat, max_lon, max_lat] for spatial subset
        aggregation: Temporal aggregation ('daily', 'monthly', 'seasonal', 'annual')
        resolution: Grid resolution in meters
        
    Returns:
        Dictionary with arrays for 'ppt', 'tmax', 'tmin', 'tmean'
    """
    if not os.path.exists(prism_path):
        logger.error(f"PRISM data file not found: {prism_path}")
        return {}
    
    if not os.path.exists(base_path):
        logger.error(f"Base data file not found: {base_path}")
        return {}
    
    # Get base mask
    base_mask = get_mask(base_path, resolution)
    if base_mask.size == 0:
        logger.error("Could not retrieve base mask")
        return {}
    
    # Calculate bounding box indices if specified
    bounding_indices = None
    if bounding_box:
        min_lon, min_lat, max_lon, max_lat = bounding_box
        min_x, max_x, min_y, max_y = get_rowcol_range_by_latlon(
            base_path, min_lat, max_lat, min_lon, max_lon
        )
        if min_x is not None:
            logger.info(f"Spatial subset indices: {min_x}, {max_x}, {min_y}, {max_y}")
            bounding_indices = (min_x, max_x, min_y, max_y)
        else:
            logger.warning(f"Could not determine indices for bounding box {bounding_box}")
    
    # Available years in the requested range
    available_years = list(range(start_year, end_year + 1))
    
    try:
        with h5py.File(prism_path, 'r') as f:
            # Check available variables
            available_vars = list(f.keys())
            logger.info(f"Available variables: {available_vars}")
            
            # Lists to store data for each variable
            ppt_data = []
            tmax_data = []
            tmin_data = []
            
            # Check each year for data availability
            for year in available_years:
                if f"ppt/{year}" not in f or f"tmax/{year}" not in f or f"tmin/{year}" not in f:
                    logger.warning(f"Skipping year {year} - not all variables available")
                    continue
                
                try:
                    # Load data for this year
                    if bounding_indices:
                        min_x, max_x, min_y, max_y = bounding_indices
                        mask = base_mask[min_x:max_x+1, min_y:max_y+1]
                        
                        # Extract data with bounding box
                        year_ppt = f[f'ppt/{year}/data'][:, min_x:max_x+1, min_y:max_y+1]
                        year_tmax = f[f'tmax/{year}/data'][:, min_x:max_x+1, min_y:max_y+1]
                        year_tmin = f[f'tmin/{year}/data'][:, min_x:max_x+1, min_y:max_y+1]
                    else:
                        mask = base_mask
                        
                        # Extract full data
                        year_ppt = f[f'ppt/{year}/data'][:]
                        year_tmax = f[f'tmax/{year}/data'][:]
                        year_tmin = f[f'tmin/{year}/data'][:]
                    
                    # Standardize time dimension if needed
                    min_days = min(year_ppt.shape[0], year_tmax.shape[0], year_tmin.shape[0])
                    year_ppt = year_ppt[:min_days]
                    year_tmax = year_tmax[:min_days]
                    year_tmin = year_tmin[:min_days]
                    
                    # Apply mask
                    time_mask = np.broadcast_to(mask, year_ppt.shape)
                    year_ppt = np.where(time_mask != 1, np.nan, year_ppt)
                    year_tmax = np.where(time_mask != 1, np.nan, year_tmax)
                    year_tmin = np.where(time_mask != 1, np.nan, year_tmin)
                    
                    # Append to lists
                    ppt_data.append(year_ppt)
                    tmax_data.append(year_tmax)
                    tmin_data.append(year_tmin)
                    
                    logger.info(f"Successfully extracted data for year {year}")
                except Exception as e:
                    logger.error(f"Error extracting data for year {year}: {e}")
            
            # Concatenate all years
            if not ppt_data:
                logger.warning("No data could be extracted")
                return {}
                
            ppt = np.concatenate(ppt_data, axis=0)
            tmax = np.concatenate(tmax_data, axis=0)
            tmin = np.concatenate(tmin_data, axis=0)
            
            # Calculate mean temperature
            tmean = (tmax + tmin) / 2
            
            # Clean up any invalid values
            ppt = np.where(np.isnan(ppt) | (ppt < 0), np.nan, ppt)
            tmax = np.where(np.isnan(tmax) | (tmax < -50) | (tmax > 60), np.nan, tmax)
            tmin = np.where(np.isnan(tmin) | (tmin < -60) | (tmin > 50), np.nan, tmin)
            tmean = np.where(np.isnan(tmean) | (tmean < -50) | (tmean > 50), np.nan, tmean)
            
            # Perform temporal aggregation if requested
            if aggregation != 'daily':
                ppt, tmax, tmin, tmean = aggregate_climate_data(
                    ppt, tmax, tmin, tmean, 
                    start_year, 
                    aggregation
                )
            
            logger.info(f"Final data shapes - PPT: {ppt.shape}, TMAX: {tmax.shape}, TMIN: {tmin.shape}, TMEAN: {tmean.shape}")
            
            # Return the data
            return {
                'ppt': ppt,
                'tmax': tmax,
                'tmin': tmin,
                'tmean': tmean
            }
    
    except Exception as e:
        logger.error(f"Error extracting PRISM data: {e}")
        return {}

def aggregate_climate_data(ppt: np.ndarray, tmax: np.ndarray, tmin: np.ndarray, tmean: np.ndarray, 
                          start_year: int, aggregation: str = 'monthly') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate climate data temporally.
    
    Args:
        ppt: Precipitation data array
        tmax: Maximum temperature data array
        tmin: Minimum temperature data array
        tmean: Mean temperature data array
        start_year: First year in the data
        aggregation: Aggregation period ('monthly', 'seasonal', 'annual')
        
    Returns:
        Tuple of (aggregated_ppt, aggregated_tmax, aggregated_tmin, aggregated_tmean)
    """
    total_days = ppt.shape[0]
    start_date = f"{start_year}-01-01"
    dates = pd.date_range(start=start_date, periods=total_days)
    
    if aggregation == 'monthly':
        logger.info("Performing monthly aggregation...")
        # Create monthly aggregates for each variable
        ppt_monthly = [ppt[(dates.year == year) & (dates.month == month), :, :].sum(axis=0)
                     for year in np.unique(dates.year)
                     for month in range(1, 13) if np.sum((dates.year == year) & (dates.month == month)) > 0]
        
        tmax_monthly = [tmax[(dates.year == year) & (dates.month == month), :, :].mean(axis=0)
                       for year in np.unique(dates.year)
                       for month in range(1, 13) if np.sum((dates.year == year) & (dates.month == month)) > 0]
        
        tmin_monthly = [tmin[(dates.year == year) & (dates.month == month), :, :].mean(axis=0)
                       for year in np.unique(dates.year)
                       for month in range(1, 13) if np.sum((dates.year == year) & (dates.month == month)) > 0]
                       
        tmean_monthly = [tmean[(dates.year == year) & (dates.month == month), :, :].mean(axis=0)
                        for year in np.unique(dates.year)
                        for month in range(1, 13) if np.sum((dates.year == year) & (dates.month == month)) > 0]
        
        return (np.array(ppt_monthly), np.array(tmax_monthly), 
                np.array(tmin_monthly), np.array(tmean_monthly))
        
    elif aggregation == 'seasonal':
        logger.info("Performing seasonal aggregation...")
        # Define seasons (DJF, MAM, JJA, SON)
        seasons = {
            'DJF': [12, 1, 2],
            'MAM': [3, 4, 5],
            'JJA': [6, 7, 8],
            'SON': [9, 10, 11]
        }
        
        # Create seasonal aggregates for each variable
        ppt_seasonal = [ppt[np.isin(dates.month, months) & (dates.year == year), :, :].sum(axis=0)
                      for year in np.unique(dates.year)
                      for _, months in seasons.items()]
        
        tmax_seasonal = [tmax[np.isin(dates.month, months) & (dates.year == year), :, :].mean(axis=0)
                        for year in np.unique(dates.year)
                        for _, months in seasons.items()]
        
        tmin_seasonal = [tmin[np.isin(dates.month, months) & (dates.year == year), :, :].mean(axis=0)
                        for year in np.unique(dates.year)
                        for _, months in seasons.items()]
                        
        tmean_seasonal = [tmean[np.isin(dates.month, months) & (dates.year == year), :, :].mean(axis=0)
                         for year in np.unique(dates.year)
                         for _, months in seasons.items()]
        
        return (np.array(ppt_seasonal), np.array(tmax_seasonal), 
                np.array(tmin_seasonal), np.array(tmean_seasonal))
        
    elif aggregation == 'annual':
        logger.info("Performing annual aggregation...")
        # Create annual aggregates for each variable
        ppt_annual = [ppt[dates.year == year, :, :].sum(axis=0) 
                    for year in np.unique(dates.year)]
        
        tmax_annual = [tmax[dates.year == year, :, :].mean(axis=0) 
                      for year in np.unique(dates.year)]
        
        tmin_annual = [tmin[dates.year == year, :, :].mean(axis=0) 
                      for year in np.unique(dates.year)]
                      
        tmean_annual = [tmean[dates.year == year, :, :].mean(axis=0) 
                      for year in np.unique(dates.year)]
        
        return (np.array(ppt_annual), np.array(tmax_annual), 
                np.array(tmin_annual), np.array(tmean_annual))
    
    else:
        logger.warning(f"Unknown aggregation: aggregation, returning original data")
        return ppt, tmax, tmin, tmean

def get_prism_spatial_means(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Calculate spatial means for PRISM variables over time.
    
    Args:
        data: Dictionary with arrays for climate variables
        
    Returns:
        Dictionary with 1D arrays of spatial means for each variable
    """
    means = {}
    
    for var_name, var_data in data.items():
        if var_data.size > 0:
            # Average across spatial dimensions to get time series
            means[var_name] = np.nanmean(var_data, axis=(1, 2))
            logger.info(f"Computed spatial means for {var_name}, shape: {means[var_name].shape}")
        else:
            means[var_name] = np.array([])
            logger.warning(f"Empty data for {var_name}")
    
    return means

def create_period_labels(start_year: int, end_year: int, aggregation: str = 'annual') -> List[str]:
    """
    Create time period labels based on aggregation type.
    
    Args:
        start_year: First year in the data
        end_year: Last year in the data
        aggregation: Aggregation period ('monthly', 'seasonal', 'annual')
        
    Returns:
        List of period labels
    """
    if aggregation == 'annual':
        return [str(year) for year in range(start_year, end_year + 1)]
    
    elif aggregation == 'monthly':
        labels = []
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                labels.append(f"{year}-{month:02d}")
        return labels
    
    elif aggregation == 'seasonal':
        seasons = ['DJF', 'MAM', 'JJA', 'SON']
        labels = []
        for year in range(start_year, end_year + 1):
            for season in seasons:
                labels.append(f"{year}-{season}")
        return labels
    
    else:
        logger.warning(f"Unknown aggregation: {aggregation}")
        return []

def plot_climate_timeseries(data: Dict[str, np.ndarray], 
                           start_year: int, 
                           end_year: int,
                           aggregation: str = 'monthly',
                           output_path: Optional[str] = None,
                           title: Optional[str] = None) -> bool:
    """
    Create a time series plot of climate data.
    
    Args:
        data: Dictionary with climate variable arrays
        start_year: Starting year
        end_year: Ending year
        aggregation: Temporal aggregation type
        output_path: Path to save the plot
        title: Optional title for the plot
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if not data:
            logger.warning("No data to plot")
            return False

        # Using the safe_figure context manager
        with safe_figure(figsize=(12, 8)) as fig:
            # Set up the figure
            gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
            
            # Temperature subplot
            ax1 = fig.add_subplot(gs[0])
            
            # Precipitation subplot
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            
            # Create period labels
            period_labels = create_period_labels(start_year, end_year, aggregation)
            x = np.arange(len(period_labels))
            
            # Plot temperature variables
            temp_vars = ['tmean', 'tmax', 'tmin']
            for var_name in temp_vars:
                if var_name in data and len(data[var_name]) > 0:
                    ax1.plot(x[:len(data[var_name])], data[var_name], 
                           label=PRISM_VARIABLES[var_name]['description'], 
                           color=PRISM_VARIABLES[var_name]['color'],
                           linewidth=2)
            
            # Plot precipitation
            if 'ppt' in data and len(data['ppt']) > 0:
                ax2.bar(x[:len(data['ppt'])], data['ppt'], 
                       label='Precipitation', color='blue', alpha=0.7)
            
            # Set labels and title
            ax1.set_ylabel('Temperature (°C)')
            ax1.set_title(title or f"Climate Data Time Series ({start_year}-{end_year})")
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.legend()
            
            ax2.set_ylabel('Precipitation (mm)')
            ax2.set_xlabel('Time Period')
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # Set x-axis ticks based on aggregation
            if aggregation == 'annual':
                # For annual data, show all years
                tick_indices = range(0, len(period_labels), max(1, len(period_labels) // 10))
                ax2.set_xticks(tick_indices)
                ax2.set_xticklabels([period_labels[i] for i in tick_indices], rotation=45)
            elif aggregation == 'monthly':
                # For monthly data, show January of each year
                years = range(start_year, end_year + 1)
                tick_indices = [period_labels.index(f"{year}-01") for year in years if f"{year}-01" in period_labels]
                ax2.set_xticks(tick_indices)
                ax2.set_xticklabels([f"{period_labels[i]}" for i in tick_indices], rotation=45)
            else:
                # Default tick behavior
                tick_indices = range(0, len(period_labels), max(1, len(period_labels) // 10))
                ax2.set_xticks(tick_indices)
                ax2.set_xticklabels([period_labels[i] for i in tick_indices], rotation=45)
            
            plt.tight_layout()
            
            # Save figure if output path provided
            if output_path:
                return save_figure(fig, output_path)
            
            return True
            
    except Exception as e:
        logger.error(f"Error plotting climate time series: {e}", exc_info=True)
        plt.close('all')  # Emergency cleanup
        return False

def create_climate_spatial_plot(data: Dict[str, np.ndarray], time_index: int = 0,
                               output_path: Optional[str] = None, title: Optional[str] = None,
                               figsize: Tuple[int, int] = (12, 10)) -> Optional[plt.Figure]:
    """
    Create a spatial plot of PRISM climate variables for a specific time point.
    
    Args:
        data: Dictionary with arrays for climate variables
        time_index: Index of time slice to visualize
        output_path: Path to save the figure (optional)
        title: Custom title (optional)
        figsize: Figure size as tuple (width, height)
        
    Returns:
        Matplotlib Figure object or None if error occurs
    """
    try:
        # Check if we have data
        if not data:
            logger.warning("No data to plot")
            return None
        
        # Get valid variables
        valid_vars = []
        for var_name, var_data in data.items():
            if var_name in PRISM_VARIABLES and var_data.size > 0:
                if time_index < var_data.shape[0]:
                    valid_vars.append(var_name)
                else:
                    logger.warning(f"Time index {time_index} out of range for {var_name}")
        
        if not valid_vars:
            logger.warning("No valid variables to plot")
            return None
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
        
        # Plot variables
        plot_positions = {
            'ppt': 0,
            'tmax': 1,
            'tmin': 2,
            'tmean': 3
        }
        
        for var_name in valid_vars:
            var_info = PRISM_VARIABLES[var_name]
            pos = plot_positions.get(var_name, 0)
            
            # Get data slice
            img = data[var_name][time_index]
            
            # Create masked array to handle NaN values
            masked_img = np.ma.masked_invalid(img)
            
            # Set up colormap
            cmap = plt.get_cmap(var_info.get('color_map', 'viridis'))
            
            # Determine min/max values for colorbar, excluding outliers
            valid_data = masked_img.compressed()
            if len(valid_data) > 0:
                vmin = np.percentile(valid_data, 5)
                vmax = np.percentile(valid_data, 95)
            else:
                vmin, vmax = 0, 1
                
            # Create subplot
            ax = plt.subplot(gs[pos])
            
            # Create image
            im = ax.imshow(masked_img, cmap=cmap, interpolation='nearest', 
                          vmin=vmin, vmax=vmax)
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax, pad=0.01)
            cbar.set_label(f"{var_info['description']} ({var_info['units']})")
            
            # Set title for subplot
            display_name = var_info.get('display_name', var_name)
            ax.set_title(f"{display_name}", fontsize=12, fontweight='bold')
            
            # Remove axes ticks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(True)
        
        # Set overall title
        if title:
            plt.suptitle(title, fontsize=14, fontweight='bold')
        else:
            plt.suptitle(f"PRISM Climate Data - Time Index: {time_index}", 
                        fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved climate spatial plot to {output_path}")
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating climate spatial plot: {e}", exc_info=True)
        return None

def create_climate_seasonal_plot(data: Dict[str, np.ndarray], start_year: int, end_year: int,
                                output_path: Optional[str] = None, figsize: Tuple[int, int] = (14, 10)) -> Optional[plt.Figure]:
    """
    Create a seasonal analysis plot showing monthly patterns of climate variables.
    
    Args:
        data: Dictionary with arrays for climate variables (should be monthly or daily)
        start_year: Starting year of the data
        end_year: Ending year of the data
        output_path: Path to save the figure (optional)
        figsize: Figure size as tuple (width, height)
        
    Returns:
        Matplotlib Figure object or None if error occurs
    """
    try:
        # Check if we have data
        if not data:
            logger.warning("No data to plot")
            return None
            
        # Get spatial means for each variable
        means = get_prism_spatial_means(data)
        
        if not means:
            logger.warning("Could not calculate spatial means")
            return None
            
        # Create monthly averages dataframe
        total_months = (end_year - start_year + 1) * 12
        dates = pd.date_range(start=f"{start_year}-01-01", periods=total_months, freq='M')
        
        # Ensure we don't exceed the length of our data
        min_len = min(len(d) for d in means.values() if len(d) > 0)
        dates = dates[:min_len]
        
        # Create dataframe
        df = pd.DataFrame({'date': dates})
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        
        # Add climate variables to dataframe
        for var_name, var_data in means.items():
            if len(var_data) >= len(dates):
                df[var_name] = var_data[:len(dates)]
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Plot variables
        plot_positions = {
            'ppt': 0,
            'tmax': 1,
            'tmin': 2,
            'tmean': 3
        }
        
        month_names = [calendar.month_abbr[m] for m in range(1, 13)]
        
        for var_name in means.keys():
            if var_name in plot_positions and var_name in df.columns:
                var_info = PRISM_VARIABLES[var_name]
                pos = plot_positions[var_name]
                ax = axes[pos]
                
                # Calculate monthly statistics
                monthly_stats = df.groupby('month')[var_name].agg(['mean', 'std', 'min', 'max']).reset_index()
                
                # Create line plot with error bars
                ax.errorbar(
                    monthly_stats['month'],
                    monthly_stats['mean'],
                    yerr=monthly_stats['std'],
                    marker='o',
                    linestyle='-',
                    capsize=4,
                    color=plt.cm.get_cmap(var_info['color_map'])(0.6)
                )
                
                # Add min/max range
                ax.fill_between(
                    monthly_stats['month'],
                    monthly_stats['min'],
                    monthly_stats['max'],
                    alpha=0.2,
                    color=plt.cm.get_cmap(var_info['color_map'])(0.6)
                )
                
                # Format x-axis with month names
                ax.set_xticks(range(1, 13))
                ax.set_xticklabels(month_names)
                
                # Set labels
                ax.set_xlabel('Month', fontsize=10)
                ax.set_ylabel(f"{var_info['description']} ({var_info['units']})", fontsize=10)
                ax.set_title(f"{var_info['display_name']} Seasonal Pattern", fontsize=12, fontweight='bold')
                
                # Add grid
                ax.grid(True, linestyle='--', alpha=0.6)
        
        # Set overall title
        plt.suptitle(f"PRISM Seasonal Analysis ({start_year}-{end_year})", 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved seasonal plot to {output_path}")
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating seasonal plot: {e}", exc_info=True)
        return None

def export_climate_data_to_csv(data: Dict[str, np.ndarray], start_year: int, end_year: int,
                              aggregation: str, output_path: str) -> bool:
    """
    Export processed PRISM climate data to CSV format.
    
    Args:
        data: Dictionary with arrays for climate variables
        start_year: First year of the data
        end_year: Last year of the data
        aggregation: Data aggregation level ('daily', 'monthly', 'seasonal', 'annual')
        output_path: Path to save the CSV file
        
    Returns:
        Boolean indicating success
    """
    try:
        # Calculate spatial means for each variable
        means = get_prism_spatial_means(data)
        
        if not means:
            logger.warning("No data to export")
            return False
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create period labels
        period_labels = create_period_labels(start_year, end_year, aggregation)
        
        # Find minimum length to ensure all variables have same length
        min_len = min(len(d) for var_name, d in means.items() if var_name in PRISM_VARIABLES and len(d) > 0)
        period_labels = period_labels[:min_len]
        
        # Create DataFrame
        df = pd.DataFrame({'Period': period_labels})
        
        # Add year and month columns if monthly aggregation
        if aggregation == 'monthly':
            df['Year'] = [int(p.split('-')[0]) for p in period_labels]
            df['Month'] = [int(p.split('-')[1]) for p in period_labels]
        elif aggregation == 'seasonal':
            df['Year'] = [int(p.split('-')[0]) for p in period_labels]
            df['Season'] = [p.split('-')[1] for p in period_labels]
        elif aggregation == 'annual':
            df['Year'] = [int(p) for p in period_labels]
        
        # Add climate variables to dataframe
        for var_name, var_data in means.items():
            if var_name in PRISM_VARIABLES and len(var_data) > 0:
                # Trim to minimum length and add to dataframe
                df[var_name] = var_data[:min_len]
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Exported climate data to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting climate data: {e}", exc_info=True)
        return False

def calculate_climate_trends(data: Dict[str, np.ndarray], start_year: int, end_year: int) -> Dict[str, Dict]:
    """
    Calculate trends in climate variables over time.
    
    Args:
        data: Dictionary with arrays for climate variables
        start_year: First year of the data
        end_year: Last year of the data
        
    Returns:
        Dictionary with trend statistics for each variable
    """
    try:
        from scipy.stats import linregress
        
        # Calculate spatial means for each variable
        means = get_prism_spatial_means(data)
        
        if not means:
            logger.warning("No data to calculate trends")
            return {}
            
        # Create time axis (assume annual data)
        years = np.arange(start_year, end_year + 1)
        
        # Calculate trends for each variable
        trends = {}
        
        for var_name, var_data in means.items():
            if var_name in PRISM_VARIABLES and len(var_data) > 0:
                # Use only data that matches our year range
                y_data = var_data[:len(years)]
                
                if len(y_data) < 3:  # Need at least 3 points for a trend
                    continue
                    
                # Calculate linear regression
                slope, intercept, r_value, p_value, std_err = linregress(years, y_data)
                
                # Calculate trend values
                trend_start = intercept + slope * start_year
                trend_end = intercept + slope * end_year
                total_change = trend_end - trend_start
                pct_change = (total_change / trend_start * 100) if trend_start != 0 else float('inf')
                
                # Store trend information
                trends[var_name] = {
                    'slope': slope,  # Change per year
                    'intercept': intercept,
                    'r_squared': r_value**2,
                    'p_value': p_value,
                    'std_err': std_err,
                    'total_change': total_change,
                    'percent_change': pct_change,
                    'significant': p_value < 0.05
                }
                
        return trends
        
    except Exception as e:
        logger.error(f"Error calculating climate trends: {e}", exc_info=True)
        return {}

if __name__ == "__main__":
    print("PRISM utilities module loaded. Import to use functions.")
