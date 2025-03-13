"""
SNODAS data processing and visualization utilities.

This module provides functions for analyzing, visualizing, and exporting
SNODAS (Snow Data Assimilation System) data extracted from HDF5 datasets.
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
try:
    from AI_agent.plot_utils import safe_figure, save_figure
except ImportError:
    from plot_utils import safe_figure, save_figure

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

# SNODAS variable information for consistent processing
SNODAS_VARIABLES = {
    'snow_water_equivalent': {
        'description': 'Snow Water Equivalent',
        'units': 'mm',
        'color_map': 'Blues',
        'line_color': '#1f77b4',
        'aggregation': 'mean',
        'scale_factor': 1.0,
        'display_name': 'SWE'
    },
    'snow_layer_thickness': {
        'description': 'Snow Layer Thickness',
        'units': 'mm',
        'color_map': 'cool',
        'line_color': '#9467bd',
        'aggregation': 'mean',
        'scale_factor': 1.0,
        'display_name': 'Snow Depth'
    },
    'snow_accumulation': {
        'description': 'Snow Accumulation',
        'units': 'mm',
        'color_map': 'winter',
        'line_color': '#8c564b',
        'aggregation': 'sum',
        'scale_factor': 0.01,
        'display_name': 'Accumulation'
    },
    'snowpack_sublimation_rate': {
        'description': 'Snowpack Sublimation Rate',
        'units': 'mm',
        'color_map': 'BuPu',
        'line_color': '#e377c2',
        'aggregation': 'sum',
        'scale_factor': 0.01,
        'display_name': 'Sublimation'
    },
    'melt_rate': {
        'description': 'Snowmelt Rate',
        'units': 'mm',
        'color_map': 'YlGnBu',
        'line_color': '#17becf',
        'aggregation': 'sum',
        'scale_factor': 0.01,
        'display_name': 'Melt'
    }
}

def get_snodas_spatial_means(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Calculate spatial means for SNODAS variables over time.
    
    Args:
        data: Dictionary with arrays for snow variables
        
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

def plot_snow_timeseries(data: Dict[str, np.ndarray], 
                        start_year: int, 
                        end_year: int,
                        aggregation: str = 'monthly',
                        output_path: Optional[str] = None,
                        title: Optional[str] = None) -> bool:
    """
    Create a time series plot of SNODAS data.
    
    Args:
        data: Dictionary with snow variable arrays
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
            
            # SWE subplot
            ax1 = fig.add_subplot(gs[0])
            
            # Snow Depth subplot
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            
            # Create period labels
            period_labels = create_period_labels(start_year, end_year, aggregation)
            x = np.arange(len(period_labels))
            
            # Plot SWE
            if 'snow_water_equivalent' in data and len(data['snow_water_equivalent']) > 0:
                var_info = SNODAS_VARIABLES['snow_water_equivalent']
                ax1.plot(x[:len(data['snow_water_equivalent'])], data['snow_water_equivalent'], 
                       label=var_info['description'], 
                       color=var_info['line_color'],
                       linewidth=2)
                
                # Add melt rate on same axis with different scale if available
                if 'melt_rate' in data and len(data['melt_rate']) > 0:
                    melt_info = SNODAS_VARIABLES['melt_rate']
                    ax1_melt = ax1.twinx()
                    ax1_melt.plot(x[:len(data['melt_rate'])], data['melt_rate'],
                                 label=melt_info['description'],
                                 color='r', linestyle='--', alpha=0.7)
                    ax1_melt.set_ylabel(f"{melt_info['description']} ({melt_info['units']})", color='r')
                    ax1_melt.tick_params(axis='y', labelcolor='r')
            
            # Plot snow depth
            if 'snow_layer_thickness' in data and len(data['snow_layer_thickness']) > 0:
                var_info = SNODAS_VARIABLES['snow_layer_thickness']
                ax2.plot(x[:len(data['snow_layer_thickness'])], data['snow_layer_thickness'], 
                       label=var_info['description'], 
                       color=var_info['line_color'],
                       linewidth=2)
                
                # Add accumulation on same axis with different scale if available
                if 'snow_accumulation' in data and len(data['snow_accumulation']) > 0:
                    accum_info = SNODAS_VARIABLES['snow_accumulation']
                    ax2_accum = ax2.twinx()
                    ax2_accum.plot(x[:len(data['snow_accumulation'])], data['snow_accumulation'],
                                  label=accum_info['description'],
                                  color='g', linestyle='--', alpha=0.7)
                    ax2_accum.set_ylabel(f"{accum_info['description']} ({accum_info['units']})", color='g')
                    ax2_accum.tick_params(axis='y', labelcolor='g')
            
            # Set labels and title
            ax1.set_ylabel('Snow Water Equivalent (mm)')
            ax1.set_title(title or f"SNODAS Data Time Series ({start_year}-{end_year})")
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.legend(loc='upper left')
            
            ax2.set_ylabel('Snow Depth (mm)')
            ax2.set_xlabel('Time Period')
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.legend(loc='upper left')
            
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
        logger.error(f"Error plotting snow time series: {e}", exc_info=True)
        plt.close('all')  # Emergency cleanup
        return False

def create_snow_spatial_plot(data: Dict[str, np.ndarray], time_index: int = 0,
                          output_path: Optional[str] = None, title: Optional[str] = None,
                          figsize: Tuple[int, int] = (12, 10)) -> Optional[plt.Figure]:
    """
    Create a spatial plot of SNODAS variables for a specific time point.
    
    Args:
        data: Dictionary with arrays for snow variables
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
            if var_name in SNODAS_VARIABLES and var_data.size > 0:
                if time_index < var_data.shape[0]:
                    valid_vars.append(var_name)
                else:
                    logger.warning(f"Time index {time_index} out of range for {var_name}")
        
        if not valid_vars:
            logger.warning("No valid variables to plot")
            return None
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Determine grid layout based on number of variables
        n_vars = len(valid_vars)
        n_cols = min(2, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        # Plot variables
        for i, var_name in enumerate(valid_vars):
            var_info = SNODAS_VARIABLES[var_name]
            
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
            ax = fig.add_subplot(n_rows, n_cols, i+1)
            
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
            plt.suptitle(f"SNODAS Data - Time Index: {time_index}", 
                        fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved snow spatial plot to {output_path}")
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating snow spatial plot: {e}", exc_info=True)
        return None

def create_snow_seasonal_plot(data: Dict[str, np.ndarray], start_year: int, end_year: int,
                            output_path: Optional[str] = None, figsize: Tuple[int, int] = (14, 10)) -> Optional[plt.Figure]:
    """
    Create a seasonal analysis plot showing monthly patterns of snow variables.
    
    Args:
        data: Dictionary with arrays for snow variables (should be monthly or daily)
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
        means = get_snodas_spatial_means(data)
        
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
        
        # Add snow variables to dataframe
        for var_name, var_data in means.items():
            if len(var_data) >= len(dates):
                df[var_name] = var_data[:len(dates)]
        
        # Create figure with subplots based on available variables
        fig = plt.figure(figsize=figsize)
        
        # Plot variables 
        plot_vars = [v for v in ['snow_water_equivalent', 'snow_layer_thickness', 
                                'melt_rate', 'snow_accumulation']
                    if v in means and len(means[v]) > 0]
        
        n_vars = len(plot_vars)
        if n_vars == 0:
            logger.warning("No variables to plot")
            return None
            
        n_cols = min(2, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        month_names = [calendar.month_abbr[m] for m in range(1, 13)]
        
        for i, var_name in enumerate(plot_vars):
            if var_name in df.columns:
                var_info = SNODAS_VARIABLES[var_name]
                ax = fig.add_subplot(n_rows, n_cols, i+1)
                
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
                    color=var_info['line_color']
                )
                
                # Add min/max range
                ax.fill_between(
                    monthly_stats['month'],
                    monthly_stats['min'],
                    monthly_stats['max'],
                    alpha=0.2,
                    color=var_info['line_color']
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
        plt.suptitle(f"SNODAS Seasonal Analysis ({start_year}-{end_year})", 
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

def export_snow_data_to_csv(data: Dict[str, np.ndarray], start_year: int, end_year: int,
                           aggregation: str, output_path: str) -> bool:
    """
    Export processed SNODAS data to CSV format.
    
    Args:
        data: Dictionary with arrays for snow variables
        start_year: First year of the data
        end_year: Last year of the data
        aggregation: Data aggregation level ('daily', 'monthly', 'seasonal', 'annual')
        output_path: Path to save the CSV file
        
    Returns:
        Boolean indicating success
    """
    try:
        # Calculate spatial means for each variable
        means = get_snodas_spatial_means(data)
        
        if not means:
            logger.warning("No data to export")
            return False
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create period labels
        period_labels = create_period_labels(start_year, end_year, aggregation)
        
        # Find minimum length to ensure all variables have same length
        min_len = min(len(d) for var_name, d in means.items() if var_name in SNODAS_VARIABLES and len(d) > 0)
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
        
        # Add snow variables to dataframe
        for var_name, var_data in means.items():
            if var_name in SNODAS_VARIABLES and len(var_data) > 0:
                # Trim to minimum length and add to dataframe
                df[var_name] = var_data[:min_len]
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Exported snow data to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting snow data: {e}", exc_info=True)
        return False

def calculate_snow_trends(data: Dict[str, np.ndarray], start_year: int, end_year: int) -> Dict[str, Dict]:
    """
    Calculate trends in snow variables over time.
    
    Args:
        data: Dictionary with arrays for snow variables
        start_year: First year of the data
        end_year: Last year of the data
        
    Returns:
        Dictionary with trend statistics for each variable
    """
    try:
        from scipy.stats import linregress
        
        # Calculate spatial means for each variable
        means = get_snodas_spatial_means(data)
        
        if not means:
            logger.warning("No data to calculate trends")
            return {}
            
        # Create time axis (assume annual data)
        years = np.arange(start_year, end_year + 1)
        
        # Calculate trends for each variable
        trends = {}
        
        for var_name, var_data in means.items():
            if var_name in SNODAS_VARIABLES and len(var_data) > 0:
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
        logger.error(f"Error calculating snow trends: {e}", exc_info=True)
        return {}

if __name__ == "__main__":
    print("SNODAS utilities module loaded. Import to use functions.")
