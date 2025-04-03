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

def plot_snow_anomalies(data: Dict[str, Dict], start_year: int, end_year: int,
                       output_path: Optional[str] = None,
                       title: Optional[str] = None) -> bool:
    """
    Create a plot showing inter-annual anomalies for snow variables.
    
    Args:
        data: Dictionary with inter-annual variability data from analyze_interannual_variability()
        start_year: Starting year
        end_year: Ending year
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
            years = list(range(start_year, end_year + 1))
            x = np.arange(len(years))
            
            # Plot anomalies for each variable
            var_names = ['snow_water_equivalent', 'snow_layer_thickness', 'melt_rate']
            available_vars = [v for v in var_names if v in data]
            
            if not available_vars:
                logger.warning("No valid variables found for anomaly plotting")
                return False
                
            n_vars = len(available_vars)
            fig, axes = plt.subplots(n_vars, 1, figsize=(12, 3*n_vars), sharex=True)
            
            # Convert to array if single axis
            if n_vars == 1:
                axes = [axes]
            
            for i, var_name in enumerate(available_vars):
                var_info = SNODAS_VARIABLES[var_name]
                anomalies = data[var_name]['normalized_anomalies']
                
                # Create bar plot of anomalies
                bars = axes[i].bar(x[:len(anomalies)], 
                                   np.array(anomalies) * 100,  # Convert to percent
                                   color=[('blue' if val >= 0 else 'red') for val in anomalies],
                                   alpha=0.7)
                
                # Add zero line
                axes[i].axhline(y=0, color='k', linestyle='-', alpha=0.3)
                
                # Add labels
                axes[i].set_ylabel(f"{var_info['display_name']} Anomaly (%)")
                axes[i].set_title(f"{var_info['description']} Inter-Annual Variability")
                axes[i].grid(True, linestyle='--', alpha=0.3)
                
                # Add value labels to the bars
                for bar_idx, bar in enumerate(bars):
                    height = bar.get_height()
                    if abs(height) > 10:  # Only label significant anomalies
                        axes[i].annotate(f'{height:.0f}%',
                                        xy=(bar.get_x() + bar.get_width()/2, height),
                                        xytext=(0, 3 if height > 0 else -10),
                                        textcoords="offset points",
                                        ha='center', va='bottom' if height > 0 else 'top',
                                        fontsize=8)
            
            # Set x-axis labels on the bottom plot
            axes[-1].set_xlabel('Year')
            axes[-1].set_xticks(x)
            axes[-1].set_xticklabels(years, rotation=45)
            
            # Set title for the figure
            if title:
                fig.suptitle(title, fontsize=14)
            else:
                fig.suptitle(f"SNODAS Inter-Annual Variability ({start_year}-{end_year})", fontsize=14)
            
            plt.tight_layout()
            
            # Save figure if output path provided
            if output_path:
                return save_figure(fig, output_path)
            
            return True
            
    except Exception as e:
        logger.error(f"Error plotting snow anomalies: {e}", exc_info=True)
        plt.close('all')  # Emergency cleanup
        return False

def create_snow_accumulation_melt_plot(timing_data: Dict, start_year: int, end_year: int,
                                    output_path: Optional[str] = None) -> bool:
    """
    Create a plot showing the timing of snow accumulation and melt.
    
    Args:
        timing_data: Dictionary with snow timing data from detect_snow_timing()
        start_year: Starting year
        end_year: Ending year
        output_path: Path to save the plot
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if not timing_data or 'peak_swe_months' not in timing_data:
            logger.warning("No data to plot snow timing")
            return False
            
        with safe_figure(figsize=(10, 6)) as fig:
            peak_months = timing_data['peak_swe_months']
            years = timing_data['years']
            
            # Plot peak SWE month for each year
            plt.plot(years, peak_months, 'o-', color='steelblue', markersize=8)
            
            # Add mean line
            mean_peak = timing_data.get('mean_peak_month')
            if mean_peak is not None:
                plt.axhline(y=mean_peak, color='r', linestyle='--', 
                           alpha=0.7, label=f'Mean Peak Month: {mean_peak:.1f}')
            
            # Set labels and title
            plt.xlabel('Year')
            plt.ylabel('Peak SWE Month')
            plt.title('Timing of Peak Snow Water Equivalent')
            
            # Format the y-axis to show month names
            month_names = [calendar.month_abbr[m] for m in range(1, 13)]
            plt.yticks(range(1, 13), month_names)
            
            # Add grid and legend
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # If years span more than 10, format x-axis accordingly
            if len(years) > 10:
                plt.xticks(range(start_year, end_year+1, 2), rotation=45)
            
            plt.tight_layout()
            
            # Save figure if output path provided
            if output_path:
                return save_figure(fig, output_path)
            
            return True
            
    except Exception as e:
        logger.error(f"Error plotting snow timing: {e}", exc_info=True)
        plt.close('all')
        return False

def create_snow_duration_map(duration_data: Dict, output_path: Optional[str] = None,
                          title: Optional[str] = None) -> bool:
    """
    Create a spatial map of snow cover duration.
    
    Args:
        duration_data: Dictionary with snow duration data from calculate_snow_cover_duration()
        output_path: Path to save the plot
        title: Optional title for the plot
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if not duration_data or 'duration_map' not in duration_data:
            logger.warning("No data to plot snow duration")
            return False
            
        with safe_figure(figsize=(10, 8)) as fig:
            duration_map = duration_data['duration_map']
            
            # Create masked array to handle NaN values
            masked_map = np.ma.masked_invalid(duration_map)
            
            # Create plot with colormap
            im = plt.imshow(masked_map, cmap='Blues', interpolation='nearest')
            
            # Add colorbar
            cbar = plt.colorbar(im, pad=0.01)
            cbar.set_label('Snow Cover Duration (days)')
            
            # Add title
            if title:
                plt.title(title)
            else:
                plt.title(f"Snow Cover Duration Map\nMean: {duration_data['mean_duration']:.1f} days")
            
            # Remove axis ticks
            plt.xticks([])
            plt.yticks([])
            
            plt.tight_layout()
            
            # Save figure if output path provided
            if output_path:
                return save_figure(fig, output_path)
            
            return True
            
    except Exception as e:
        logger.error(f"Error plotting snow duration map: {e}", exc_info=True)
        plt.close('all')
        return False
        
def plot_snow_extreme_events(extreme_data: Dict, start_year: int, end_year: int,
                           output_path: Optional[str] = None) -> bool:
    """
    Create a plot showing extreme snow events statistics.
    
    Args:
        extreme_data: Dictionary with extreme statistics from calculate_extreme_statistics()
        start_year: Starting year
        end_year: Ending year
        output_path: Path to save the plot
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if not extreme_data:
            logger.warning("No extreme event data to plot")
            return False
            
        with safe_figure(figsize=(10, 8)) as fig:
            # Set up subplots based on available variables
            n_vars = len(extreme_data)
            fig, axes = plt.subplots(n_vars, 1, figsize=(10, 4*n_vars))
            
            # Convert to array if single axis
            if n_vars == 1:
                axes = [axes]
                
            # Plot each variable's extreme statistics
            i = 0
            for var_type, stats in extreme_data.items():
                if var_type == 'swe_extremes':
                    var_name = 'snow_water_equivalent'
                    title = 'Extreme Snow Water Equivalent Events'
                elif var_type == 'melt_extremes':
                    var_name = 'melt_rate'
                    title = 'Extreme Snowmelt Events'
                else:
                    continue
                    
                var_info = SNODAS_VARIABLES.get(var_name, {'units': ''})
                
                # Create bar chart for percentile thresholds
                percentiles = [stats['p90'], stats['p95'], stats['p99']]
                axes[i].bar(['90th', '95th', '99th'], percentiles, 
                          color='steelblue', alpha=0.7)
                
                # Add max value line
                axes[i].axhline(y=stats['max_value'], color='r', linestyle='--', 
                              label=f"Max: {stats['max_value']:.2f} {var_info['units']}")
                
                # Add text annotation for extreme days
                axes[i].annotate(f"Days above 90th percentile: {stats['extreme_days']}", 
                               xy=(0.5, 0.9), xycoords='axes fraction',
                               ha='center', va='center',
                               bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))
                
                # Set labels and title
                axes[i].set_ylabel(f"{var_info['units']}")
                axes[i].set_title(title)
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
                
                i += 1
                
            plt.tight_layout()
            
            # Save figure if output path provided
            if output_path:
                return save_figure(fig, output_path)
            
            return True
            
    except Exception as e:
        logger.error(f"Error plotting snow extreme events: {e}", exc_info=True)
        plt.close('all')
        return False

def create_snow_monthly_analysis_plot(data: Dict[str, np.ndarray], start_year: int, end_year: int,
                                   output_path: Optional[str] = None, figsize: Tuple[int, int] = (12, 8)) -> Optional[plt.Figure]:
    """
    Create a plot showing monthly averages with uncertainty ranges for SNODAS variables.
    
    Args:
        data: Dictionary with arrays for snow variables
        start_year: Starting year
        end_year: Ending year
        output_path: Path to save the plot
        figsize: Figure size as tuple (width, height)
        
    Returns:
        Matplotlib Figure object or None if error occurs
    """
    try:
        if not data:
            logger.warning("No data to plot monthly analysis")
            return None
            
        # Get the SNODAS data at monthly resolution if not already
        # We need to reshape the data by month to compute monthly statistics
        spatial_means = get_snodas_spatial_means(data)
        
        if not spatial_means:
            logger.warning("Could not calculate spatial means for monthly analysis")
            return None
            
        # Create dataframe with dates
        years = list(range(start_year, end_year + 1))
        months = range(1, 13)
        
        # Generate all possible year-month combinations
        dates = []
        for year in years:
            for month in months:
                dates.append(pd.Timestamp(year=year, month=month, day=15))
        
        # Create DataFrame with proper date indices
        df = pd.DataFrame({'date': dates})
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['month_name'] = df['date'].dt.month_name().str.slice(0, 3)
        
        # Determine which variables to plot
        plot_vars = []
        for var_name, values in spatial_means.items():
            if len(values) > 0 and var_name in SNODAS_VARIABLES:
                # For monthly data, the length should match the number of months
                # For annual data, we'll need to reshape, but we'll skip for now
                if len(values) >= 12:  # At least a year of data
                    plot_vars.append(var_name)
                    # Try to map the values to months
                    monthly_values = values[:len(df)]
                    df[var_name] = pd.Series(monthly_values)
        
        if not plot_vars:
            logger.warning("No suitable variables found for monthly analysis")
            return None
            
        # Create the figure with 2x2 subplots (or fewer if fewer variables)
        n_vars = min(len(plot_vars), 4)  # Maximum 4 variables to plot
        n_cols = min(2, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
        
        # Flatten the axes array if it's multi-dimensional
        if n_vars > 1:
            if n_rows > 1 and n_cols > 1:
                axes = axes.flatten()
            elif n_rows == 1:
                axes = [axes[i] for i in range(n_cols)]
            elif n_cols == 1:
                axes = [axes[i] for i in range(n_rows)]
        else:
            axes = [axes]
            
        # Create the monthly analysis plots
        for i, var_name in enumerate(plot_vars[:n_vars]):
            var_info = SNODAS_VARIABLES[var_name]
            ax = axes[i]
            
            # Group by month and calculate statistics
            monthly_stats = df.groupby('month')[var_name].agg(['mean', 'std', 'min', 'max']).reset_index()
            
            # Plot mean line with error band
            x = monthly_stats['month']
            mean = monthly_stats['mean']
            std = monthly_stats['std']
            min_val = monthly_stats['min']
            max_val = monthly_stats['max']
            
            # Plot mean, std range, and min-max range
            ax.plot(x, mean, 'o-', color=var_info['line_color'], linewidth=2, 
                   label=f"{var_info['display_name']} Mean")
            
            # Add standard deviation as shaded area
            ax.fill_between(x, mean - std, mean + std, color=var_info['line_color'], 
                           alpha=0.3, label='±1 Std Dev')
            
            # Add min-max range
            ax.fill_between(x, min_val, max_val, color=var_info['line_color'], 
                           alpha=0.1, label='Min-Max Range')
            
            # Add labels and customize
            ax.set_title(f"{var_info['description']} Monthly Pattern", fontweight='bold')
            ax.set_ylabel(f"{var_info['description']} ({var_info['units']})")
            ax.set_xlabel('Month')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='best')
            
            # Format x-axis with month names
            month_names = [calendar.month_abbr[m] for m in range(1, 13)]
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(month_names)
            
            # Add text annotation showing the average annual value
            annual_mean = mean.mean()
            if var_name in ['snow_water_equivalent', 'snow_layer_thickness']:
                # For stock variables, show the mean over all months
                ax.annotate(f'Annual Average: {annual_mean:.1f} {var_info["units"]}',
                           xy=(0.5, 0.03), xycoords='axes fraction',
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                           ha='center', va='bottom')
            else:
                # For flux variables, sum the monthly means to get annual total
                annual_total = mean.sum()
                ax.annotate(f'Annual Total: {annual_total:.1f} {var_info["units"]}',
                           xy=(0.5, 0.03), xycoords='axes fraction',
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                           ha='center', va='bottom')
        
        # If we have fewer than 4 variables, remove unused axes
        if n_vars < len(axes):
            for i in range(n_vars, len(axes)):
                fig.delaxes(axes[i])
        
        plt.suptitle(f"SNODAS Monthly Analysis ({start_year}-{end_year})", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save the figure if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved monthly analysis plot to {output_path}")
            
        return fig
        
    except Exception as e:
        logger.error(f"Error creating monthly analysis plot: {e}", exc_info=True)
        plt.close('all')
        return None

if __name__ == "__main__":
    print("SNODAS utilities module loaded. Import to use functions.")
