"""
Groundwater data processing and visualization utilities.

This module provides functions for analyzing, visualizing, and exporting
groundwater properties data extracted from HDF5 HydroGeoDataset.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter
import pandas as pd
import h5py
import os
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
from pathlib import Path
import seaborn as sns
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

# Unit conversion constants
M_TO_FT = 3.28084  # meters to feet
M2_TO_FT2 = 10.7639  # square meters to square feet

# Groundwater properties information
GROUNDWATER_VARIABLES = {
    'AQ_THK_1': {
        'description': 'Upper Aquifer Thickness',
        'units': 'ft',
        'units_si': 'm',
        'color_map': 'viridis',
        'log_scale': False,
        'display_name': 'Upper Aquifer Thickness',
        'valid_range': (0, 200),
        'conversion_factor': M_TO_FT
    },
    'AQ_THK_2': {
        'description': 'Lower Aquifer Thickness',
        'units': 'ft',
        'units_si': 'm',
        'color_map': 'plasma',
        'log_scale': False,
        'display_name': 'Lower Aquifer Thickness',
        'valid_range': (0, 300),
        'conversion_factor': M_TO_FT
    },
    'H_COND_1': {
        'description': 'Upper Aquifer Horizontal Hydraulic Conductivity',
        'units': 'ft/day',
        'units_si': 'm/day',
        'color_map': 'YlGnBu',
        'log_scale': True,
        'display_name': 'Upper K',
        'valid_range': (0.001, 1000),
        'conversion_factor': M_TO_FT
    },
    'H_COND_2': {
        'description': 'Lower Aquifer Horizontal Hydraulic Conductivity',
        'units': 'ft/day',
        'units_si': 'm/day',
        'color_map': 'YlGnBu',
        'log_scale': True,
        'display_name': 'Lower K',
        'valid_range': (0.001, 1000),
        'conversion_factor': M_TO_FT
    },
    'V_COND_1': {
        'description': 'Upper Aquifer Vertical Hydraulic Conductivity',
        'units': 'ft/day',
        'units_si': 'm/day',
        'color_map': 'GnBu',
        'log_scale': True,
        'display_name': 'Upper Vertical K',
        'valid_range': (0.0001, 100),
        'conversion_factor': M_TO_FT
    },
    'V_COND_2': {
        'description': 'Lower Aquifer Vertical Hydraulic Conductivity',
        'units': 'ft/day',
        'units_si': 'm/day',
        'color_map': 'GnBu',
        'log_scale': True,
        'display_name': 'Lower Vertical K',
        'valid_range': (0.0001, 100),
        'conversion_factor': M_TO_FT
    },
    'TRANSMSV_1': {
        'description': 'Upper Aquifer Transmissivity',
        'units': 'ft²/day',
        'units_si': 'm²/day',
        'color_map': 'cool',
        'log_scale': True,
        'display_name': 'Upper Transmissivity',
        'valid_range': (0.1, 10000),
        'conversion_factor': M2_TO_FT2
    },
    'TRANSMSV_2': {
        'description': 'Lower Aquifer Transmissivity',
        'units': 'ft²/day',
        'units_si': 'm²/day',
        'color_map': 'cool',
        'log_scale': True,
        'display_name': 'Lower Transmissivity',
        'valid_range': (0.1, 10000),
        'conversion_factor': M2_TO_FT2
    },
    'SWL': {
        'description': 'Static Water Level',
        'units': 'ft below surface',
        'units_si': 'm below surface',
        'color_map': 'Blues_r',
        'log_scale': False,
        'display_name': 'Water Table Depth',
        'valid_range': (0, 100),
        'conversion_factor': M_TO_FT
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

def extract_groundwater_data(database_path: str, bounding_box: Optional[Tuple[float, float, float, float]] = None,
                            resolution: int = 250) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Extract groundwater properties data for a given region.
    
    Args:
        database_path: Path to the HDF5 file with groundwater data
        bounding_box: Optional [min_lon, min_lat, max_lon, max_lat] for spatial subset
        resolution: Grid resolution in meters
        
    Returns:
        Dictionary mapping variable names to tuples of (data_array, error_array)
    """
    if not os.path.exists(database_path):
        logger.error(f"Data file not found: {database_path}")
        return {}
    
    # Calculate bounding box indices if specified
    bounding_indices = None
    if bounding_box:
        min_lon, min_lat, max_lon, max_lat = bounding_box
        min_row, max_row, min_col, max_col = get_rowcol_range_by_latlon(
            database_path, min_lat, max_lat, min_lon, max_lon
        )
        if min_row is not None:
            logger.info(f"Spatial subset indices: {min_row}, {max_row}, {min_col}, {max_col}")
            bounding_indices = (min_row, max_row, min_col, max_col)
        else:
            logger.warning(f"Could not determine indices for bounding box {bounding_box}")
    
    # Extract groundwater data
    extracted_data = {}
    
    try:
        with h5py.File(database_path, 'r') as f:
            # List available EBK datasets
            if 'EBK' in f:
                available_datasets = list(f['EBK'].keys())
                logger.info(f"Available EBK datasets: {available_datasets}")
                
                # Process each groundwater variable
                for var_name in GROUNDWATER_VARIABLES.keys():
                    # Look for output dataset and error dataset
                    output_key = f"kriging_output_{var_name}_250m"
                    error_key = f"kriging_stderr_{var_name}_250m"
                    
                    if output_key in f['EBK'] and error_key in f['EBK']:
                        try:
                            # Extract data with bounding box if available
                            if bounding_indices:
                                min_row, max_row, min_col, max_col = bounding_indices
                                output_data = f[f'EBK/{output_key}'][min_row:max_row+1, min_col:max_col+1]
                                error_data = f[f'EBK/{error_key}'][min_row:max_row+1, min_col:max_col+1]
                            else:
                                output_data = f[f'EBK/{output_key}'][:]
                                error_data = f[f'EBK/{error_key}'][:]
                            
                            # Clean up any invalid values
                            output_data = np.where((output_data == -999) | np.isnan(output_data), np.nan, output_data)
                            error_data = np.where((error_data == -999) | np.isnan(error_data), np.nan, error_data)
                            
                            # Store data
                            extracted_data[var_name] = (output_data, error_data)
                            logger.info(f"Successfully extracted {var_name} data, shape: {output_data.shape}")
                            
                        except Exception as e:
                            logger.error(f"Error extracting {var_name} data: {e}")
                    else:
                        logger.warning(f"Dataset not found for {var_name}")
            else:
                logger.error("EBK group not found in the HDF5 file")
                
        return extracted_data
    
    except Exception as e:
        logger.error(f"Error extracting groundwater data: {e}")
        return {}

def convert_units(value: float, var_name: str, to_us: bool = True) -> float:
    """
    Convert between SI and US units for groundwater variables.
    
    Args:
        value: Value to convert
        var_name: Variable name for unit lookup
        to_us: Convert from SI to US (True) or US to SI (False)
        
    Returns:
        Converted value
    """
    if var_name not in GROUNDWATER_VARIABLES:
        return value
    
    factor = GROUNDWATER_VARIABLES[var_name].get('conversion_factor', 1.0)
    
    if to_us:
        return value * factor
    else:
        return value / factor

def get_groundwater_spatial_stats(data: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                                use_us_units: bool = True) -> Dict[str, Dict[str, float]]:
    """
    Calculate spatial statistics for groundwater properties.
    
    Args:
        data: Dictionary with extracted groundwater data
        use_us_units: Whether to convert values to US units
        
    Returns:
        Dictionary with statistics for each variable
    """
    stats = {}
    
    for var_name, (var_data, error_data) in data.items():
        if var_data.size > 0:
            # Calculate statistics on the variable data
            mean_val = float(np.nanmean(var_data))
            median_val = float(np.nanmedian(var_data))
            min_val = float(np.nanmin(var_data))
            max_val = float(np.nanmax(var_data))
            std_val = float(np.nanstd(var_data))
            error_mean_val = float(np.nanmean(error_data))
            error_max_val = float(np.nanmax(error_data))
            
            # Convert to US units if requested
            if use_us_units and var_name in GROUNDWATER_VARIABLES:
                factor = GROUNDWATER_VARIABLES[var_name].get('conversion_factor', 1.0)
                mean_val *= factor
                median_val *= factor
                min_val *= factor
                max_val *= factor
                std_val *= factor
                error_mean_val *= factor
                error_max_val *= factor
            
            var_stats = {
                'mean': mean_val,
                'median': median_val,
                'min': min_val,
                'max': max_val,
                'std': std_val,
                'cv': float(np.nanstd(var_data) / np.nanmean(var_data) * 100),  # Coefficient of variation (unchanged)
                'valid_cells': int(np.sum(~np.isnan(var_data))),
                'total_cells': int(var_data.size),
                'coverage': float(np.sum(~np.isnan(var_data)) / var_data.size * 100),
                'error_mean': error_mean_val,
                'error_max': error_max_val
            }
            
            stats[var_name] = var_stats
            logger.info(f"Computed statistics for {var_name}")
        
    return stats

def create_groundwater_maps(data: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                           output_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (15, 12),
                           use_us_units: bool = True) -> Optional[plt.Figure]:
    """
    Create spatial maps of groundwater properties.
    
    Args:
        data: Dictionary with extracted groundwater data
        output_path: Path to save the figure (optional)
        figsize: Figure dimensions (width, height) in inches
        use_us_units: Whether to display units in US system
        
    Returns:
        Matplotlib Figure object or None if error occurs
    """
    try:
        # Only process variables with data
        valid_vars = [k for k, v in data.items() if v[0].size > 0 and np.sum(~np.isnan(v[0])) > 0]
        
        if not valid_vars:
            logger.warning("No valid groundwater data to visualize")
            return None
            
        # Determine layout based on number of variables
        n_vars = len(valid_vars)
        n_cols = min(3, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols  # Ceiling division
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])  # Make sure axes is array-like
        axes = axes.flatten()
        
        # Plot each variable
        for i, var_name in enumerate(valid_vars):
            var_data, _ = data[var_name]
            ax = axes[i]
            var_info = GROUNDWATER_VARIABLES[var_name]
            
            # Create masked array to handle NaN values
            masked_data = np.ma.masked_invalid(var_data)
            
            # Determine min/max values for colorbar
            valid_data = masked_data.compressed()
            if len(valid_data) > 0:
                # Use percentiles to avoid extreme outliers
                vmin = np.nanpercentile(valid_data, 1)
                vmax = np.nanpercentile(valid_data, 99)
                
                # For log scale plots, ensure positive values
                if var_info['log_scale']:
                    vmin = max(vmin, var_info['valid_range'][0])
                    vmax = min(vmax, var_info['valid_range'][1])
            else:
                vmin, vmax = var_info['valid_range']
            
            # Create colormap instance
            cmap = plt.get_cmap(var_info['color_map'])
            
            # Plot the data with appropriate normalization
            if var_info['log_scale']:
                norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
            else:
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                
            im = ax.imshow(masked_data, cmap=cmap, norm=norm)
            
            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label(f"{var_info['units' if use_us_units else 'units_si']}")
            
            # Format colorbar for log scale
            if var_info['log_scale']:
                cbar.formatter = ScalarFormatter()
                cbar.formatter.set_scientific(True)
                cbar.formatter.set_powerlimits((0, 0))
                cbar.update_ticks()
            
            # Add title and labels
            ax.set_title(f"{var_info['display_name']}")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(True)
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        # Add overall title
        plt.suptitle("Groundwater Properties", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Groundwater maps saved to {output_path}")
            
        return fig
    
    except Exception as e:
        logger.error(f"Error creating groundwater maps: {e}", exc_info=True)
        return None

def create_groundwater_error_maps(data: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                                output_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (15, 12),
                                use_us_units: bool = True) -> Optional[plt.Figure]:
    """
    Create maps of standard errors for groundwater properties.
    
    Args:
        data: Dictionary with extracted groundwater data
        output_path: Path to save the figure (optional)
        figsize: Figure dimensions (width, height) in inches
        use_us_units: Whether to display units in US system
        
    Returns:
        Matplotlib Figure object or None if error occurs
    """
    try:
        # Only process variables with data
        valid_vars = [k for k, v in data.items() if v[1].size > 0 and np.sum(~np.isnan(v[1])) > 0]
        
        if not valid_vars:
            logger.warning("No valid groundwater error data to visualize")
            return None
            
        # Determine layout based on number of variables
        n_vars = len(valid_vars)
        n_cols = min(3, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols  # Ceiling division
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])  # Make sure axes is array-like
        axes = axes.flatten()
        
        # Plot each variable
        for i, var_name in enumerate(valid_vars):
            _, error_data = data[var_name]
            ax = axes[i]
            var_info = GROUNDWATER_VARIABLES[var_name]
            
            # Create masked array to handle NaN values
            masked_data = np.ma.masked_invalid(error_data)
            
            # Determine min/max values for colorbar
            valid_data = masked_data.compressed()
            if len(valid_data) > 0:
                # Use percentiles to avoid extreme outliers
                vmin = np.nanpercentile(valid_data, 1)
                vmax = np.nanpercentile(valid_data, 99)
            else:
                vmin, vmax = 0, 1
            
            # Create image with custom colormap for errors
            im = ax.imshow(masked_data, cmap='Reds', vmin=vmin, vmax=vmax)
            
            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label(f"Std. Error ({var_info['units' if use_us_units else 'units_si']})")
            
            # Add title and labels
            ax.set_title(f"{var_info['display_name']} Error")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(True)
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        # Add overall title
        plt.suptitle("Groundwater Properties - Estimation Errors", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Groundwater error maps saved to {output_path}")
            
        return fig
    
    except Exception as e:
        logger.error(f"Error creating groundwater error maps: {e}", exc_info=True)
        return None

def create_groundwater_histograms(data: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                               output_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (15, 12),
                               use_us_units: bool = True) -> Optional[plt.Figure]:
    """
    Create histograms of groundwater properties distribution.
    
    Args:
        data: Dictionary with extracted groundwater data
        output_path: Path to save the figure (optional)
        figsize: Figure dimensions (width, height) in inches
        use_us_units: Whether to display units in US system
        
    Returns:
        Matplotlib Figure object or None if error occurs
    """
    try:
        # Only process variables with data
        valid_vars = [k for k, v in data.items() if v[0].size > 0 and np.sum(~np.isnan(v[0])) > 0]
        
        if not valid_vars:
            logger.warning("No valid groundwater data to visualize")
            return None
            
        # Determine layout based on number of variables
        n_vars = len(valid_vars)
        n_cols = min(3, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols  # Ceiling division
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])  # Make sure axes is array-like
        axes = axes.flatten()
        
        # Plot histogram for each variable
        for i, var_name in enumerate(valid_vars):
            var_data, _ = data[var_name]
            ax = axes[i]
            var_info = GROUNDWATER_VARIABLES[var_name]
            
            # Get valid data
            valid_data = var_data[~np.isnan(var_data)]
            
            if len(valid_data) > 0:
                # Determine histogram bins
                if var_info['log_scale']:
                    # Use log scale bins
                    min_val = max(np.nanmin(valid_data), var_info['valid_range'][0])
                    max_val = min(np.nanmax(valid_data), var_info['valid_range'][1])
                    bins = np.logspace(np.log10(min_val), np.log10(max_val), 30)
                    ax.set_xscale('log')
                else:
                    # Use linear bins
                    bins = 30
                
                # Create histogram
                ax.hist(valid_data, bins=bins, color=plt.cm.get_cmap(var_info['color_map'])(0.6),
                       edgecolor='black', alpha=0.7)
                
                # Add statistics
                mean_val = np.nanmean(valid_data)
                median_val = np.nanmedian(valid_data)
                
                # Add lines for mean and median
                ax.axvline(x=mean_val, color='red', linestyle='-', label=f'Mean: {mean_val:.2f}')
                ax.axvline(x=median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
                
                # Add legend
                ax.legend(loc='upper right', fontsize='x-small')
                
                # Add labels
                ax.set_xlabel(f"{var_info['units' if use_us_units else 'units_si']}")
                ax.set_ylabel("Frequency")
                ax.set_title(f"{var_info['display_name']}")
                
                # For better histogram appearance
                ax.grid(True, alpha=0.3, linestyle='--')
            else:
                ax.text(0.5, 0.5, "No valid data", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{var_info['display_name']}")
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        # Add overall title
        plt.suptitle("Distribution of Groundwater Properties", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Groundwater histograms saved to {output_path}")
            
        return fig
    
    except Exception as e:
        logger.error(f"Error creating groundwater histograms: {e}", exc_info=True)
        return None

def create_groundwater_correlation_matrix(data: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                                        output_path: Optional[str] = None,
                                        figsize: Tuple[int, int] = (10, 8)) -> Optional[plt.Figure]:
    """
    Create correlation matrix of groundwater properties.
    
    Args:
        data: Dictionary with extracted groundwater data
        output_path: Path to save the figure (optional)
        figsize: Figure dimensions (width, height) in inches
        
    Returns:
        Matplotlib Figure object or None if error occurs
    """
    try:
        # Only process variables with data
        valid_vars = [k for k, v in data.items() if v[0].size > 0 and np.sum(~np.isnan(v[0])) > 0]
        
        if len(valid_vars) < 2:
            logger.warning("Need at least two variables for correlation analysis")
            return None
            
        # Create a dataframe with flattened data
        df_dict = {}
        
        # Number of cells to sample (for very large datasets)
        max_sample = 10000
        
        # Extract data for each variable
        for var_name in valid_vars:
            var_data, _ = data[var_name]
            flat_data = var_data.flatten()
            # If dataset is very large, take a random sample
            if len(flat_data) > max_sample:
                np.random.seed(42)  # For reproducibility
                indices = np.random.choice(len(flat_data), max_sample, replace=False)
                flat_data = flat_data[indices]
            df_dict[var_name] = flat_data
        
        # Create DataFrame with matched index (to handle NaN values)
        df = pd.DataFrame(df_dict)
        
        # Calculate correlation matrix
        corr_matrix = df.corr(method='pearson')
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot correlation matrix
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                   annot=True, fmt='.2f', square=True, linewidths=.5,
                   cbar_kws={"shrink": .5})
        
        # Add labels
        ax.set_title("Correlation Matrix of Groundwater Properties", fontsize=14, fontweight='bold')
        
        # Replace variable names with display names in axis labels
        display_names = [GROUNDWATER_VARIABLES[var]['display_name'] for var in corr_matrix.columns]
        ax.set_xticklabels(display_names, rotation=45, ha='right')
        ax.set_yticklabels(display_names, rotation=0)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Correlation matrix saved to {output_path}")
            
        return fig
    
    except Exception as e:
        logger.error(f"Error creating correlation matrix: {e}", exc_info=True)
        return None

def export_groundwater_data_to_csv(data: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                                 stats: Dict[str, Dict[str, float]],
                                 output_path: str,
                                 use_us_units: bool = True) -> bool:
    """
    Export groundwater data statistics to CSV format.
    
    Args:
        data: Dictionary with extracted groundwater data
        stats: Dictionary with calculated statistics
        output_path: Path to save the CSV file
        use_us_units: Whether to display units in US system
        
    Returns:
        Boolean indicating success
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Prepare data for CSV output
        rows = []
        
        for var_name, var_stats in stats.items():
            if var_name in GROUNDWATER_VARIABLES:
                # Add basic information
                row = {
                    'Variable': var_name,
                    'Description': GROUNDWATER_VARIABLES[var_name]['description'],
                    'Units': GROUNDWATER_VARIABLES[var_name]['units' if use_us_units else 'units_si']
                }
                
                # Add all statistics
                row.update(var_stats)
                
                # Format some values for better readability
                if 'cv' in row:
                    row['cv'] = f"{row['cv']:.2f}%"
                if 'coverage' in row:
                    row['coverage'] = f"{row['coverage']:.2f}%"
                
                rows.append(row)
        
        # Create and save DataFrame
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Data exported to {output_path}")
        
        # Export spatial distribution percentiles for each variable
        percentile_path = output_path.replace('.csv', '_percentiles.csv')
        percentile_rows = []
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        
        for var_name, (var_data, _) in data.items():
            if var_name in GROUNDWATER_VARIABLES:
                # Calculate percentiles
                valid_data = var_data[~np.isnan(var_data)]
                if len(valid_data) > 0:
                    perc_values = np.percentile(valid_data, percentiles)
                    
                    # Create row
                    row = {
                        'Variable': var_name,
                        'Description': GROUNDWATER_VARIABLES[var_name]['description'],
                        'Units': GROUNDWATER_VARIABLES[var_name]['units' if use_us_units else 'units_si']
                    }
                    
                    # Add percentile values
                    for p, val in zip(percentiles, perc_values):
                        row[f'p{p}'] = val
                    
                    percentile_rows.append(row)
        
        # Save percentiles
        if percentile_rows:
            df_perc = pd.DataFrame(percentile_rows)
            df_perc.to_csv(percentile_path, index=False)
            logger.info(f"Percentile data exported to {percentile_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error exporting data: {e}", exc_info=True)
        return False

def compare_groundwater_variables(data: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                               var1: str, var2: str,
                               output_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (10, 8),
                               use_us_units: bool = True) -> Optional[plt.Figure]:
    """
    Create a scatter plot comparing two groundwater variables.
    
    Args:
        data: Dictionary with extracted groundwater data
        var1: First variable to compare
        var2: Second variable to compare
        output_path: Path to save the figure (optional)
        figsize: Figure dimensions (width, height) in inches
        use_us_units: Whether to display units in US system
        
    Returns:
        Matplotlib Figure object or None if error occurs
    """
    try:
        if var1 not in data or var2 not in data:
            logger.warning(f"One or both variables not available: {var1}, {var2}")
            return None
            
        # Get data arrays
        data1, _ = data[var1]
        data2, _ = data[var2]
        
        # Reshape to 1D arrays
        flat1 = data1.flatten()
        flat2 = data2.flatten()
        
        # Create mask for valid data points in both arrays
        valid_mask = ~np.isnan(flat1) & ~np.isnan(flat2)
        x = flat1[valid_mask]
        y = flat2[valid_mask]
        
        if len(x) == 0:
            logger.warning("No valid overlapping data points")
            return None
            
        # Get variable information
        var1_info = GROUNDWATER_VARIABLES.get(var1, {'description': var1, 'units': '', 'log_scale': False})
        var2_info = GROUNDWATER_VARIABLES.get(var2, {'description': var2, 'units': '', 'log_scale': False})
        
        # Convert to US units if requested
        if use_us_units:
            x = convert_units(x, var1)
            y = convert_units(y, var2)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create scatter plot
        scatter = ax.scatter(x, y, alpha=0.5, s=10)
        
        # Set scales if needed
        if var1_info['log_scale']:
            ax.set_xscale('log')
        if var2_info['log_scale']:
            ax.set_yscale('log')
            
        # Add regression line
        from scipy import stats
        if var1_info['log_scale'] and var2_info['log_scale']:
            # Log-log regression
            log_x = np.log10(x)
            log_y = np.log10(y)
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
            
            # Plot in original scale
            x_line = np.logspace(np.log10(min(x)), np.log10(max(x)), 100)
            y_line = 10**(slope * np.log10(x_line) + intercept)
            
            correlation_text = f"Log-Log Regression\nR² = {r_value**2:.3f}\ny = {10**intercept:.3e} · x^{slope:.3f}"
            
        else:
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            x_line = np.linspace(min(x), max(x), 100)
            y_line = slope * x_line + intercept
            
            correlation_text = f"Linear Regression\nR² = {r_value**2:.3f}\ny = {slope:.3f}x + {intercept:.3f}"
            
        # Plot regression line
        ax.plot(x_line, y_line, 'r-', linewidth=2)
        
        # Add correlation text
        ax.text(
            0.05, 0.95, correlation_text,
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8}
        )
        
        # Add labels and title
        ax.set_xlabel(f"{var1_info['description']} ({var1_info['units' if use_us_units else 'units_si']})", fontsize=12)
        ax.set_ylabel(f"{var2_info['description']} ({var2_info['units' if use_us_units else 'units_si']})", fontsize=12)
        ax.set_title(f"Relationship between {var1_info['description']} and {var2_info['description']}", 
                   fontsize=14, fontweight='bold')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {output_path}")
            
        return fig
        
    except Exception as e:
        logger.error(f"Error creating comparison plot: {e}", exc_info=True)
        return None