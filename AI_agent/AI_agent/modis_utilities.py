"""
MODIS data processing and visualization utilities.

This module provides functions for analyzing, visualizing, and exporting
MODIS remote sensing data extracted from HDF5 datasets.
"""
import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import logging
import calendar
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Union
from pathlib import Path

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

# MODIS product information lookup tables
MODIS_PRODUCTS = {
    'MOD13Q1_NDVI': {
        'description': 'Normalized Difference Vegetation Index',
        'units': 'NDVI',
        'scale_factor': 0.0001,
        'valid_range': (-2000, 10000),
        'color_map': 'YlGn',
        'frequency': '16-day'
    },
    'MOD13Q1_EVI': {
        'description': 'Enhanced Vegetation Index',
        'units': 'EVI',
        'scale_factor': 0.0001,
        'valid_range': (-2000, 10000),
        'color_map': 'viridis',
        'frequency': '16-day'
    },
    'MOD16A2_ET': {
        'description': 'Evapotranspiration',
        'units': 'mm/8-day',
        'scale_factor': 0.1,
        'valid_range': (0, 32700),
        'color_map': 'Blues',
        'frequency': '8-day'
    },
    'MOD15A2H_Lai_500m': {
        'description': 'Leaf Area Index',
        'units': 'm²/m²',
        'scale_factor': 0.1,
        'valid_range': (0, 1000),
        'color_map': 'Greens',
        'frequency': '8-day'
    },
    'MOD15A2H_Fpar_500m': {
        'description': 'Fraction of Photosynthetically Active Radiation',
        'units': '%',
        'scale_factor': 0.01,
        'valid_range': (0, 100),
        'color_map': 'YlGn',
        'frequency': '8-day'
    },
    'MOD09GQ_sur_refl_b01': {
        'description': 'Surface Reflectance Band 1 (Red)',
        'units': 'reflectance',
        'scale_factor': 0.0001,
        'valid_range': (0, 10000),
        'color_map': 'Reds',
        'frequency': 'daily'
    },
    'MOD09GQ_sur_refl_b02': {
        'description': 'Surface Reflectance Band 2 (NIR)',
        'units': 'reflectance',
        'scale_factor': 0.0001,
        'valid_range': (0, 10000),
        'color_map': 'RdYlBu',
        'frequency': 'daily'
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

def extract_modis_data(database_path: str, h5_group_name: str, start_year: int, end_year: int, 
                      bounding_box: Optional[Tuple[float, float, float, float]] = None) -> np.ndarray:
    """
    Extract MODIS data for a given period and region.
    
    Args:
        database_path: Path to the HDF5 file
        h5_group_name: The HDF5 group path containing the MODIS data
        start_year: First year to extract
        end_year: Last year to extract
        bounding_box: Optional [min_lon, min_lat, max_lon, max_lat] for spatial subset
        
    Returns:
        Numpy array containing the extracted data with shape (time, y, x)
    """
    extracted_data = []
    
    # Calculate bounding box indices if specified
    bounding_indices = None
    if bounding_box:
        min_lon, min_lat, max_lon, max_lat = bounding_box
        min_x, max_x, min_y, max_y = get_rowcol_range_by_latlon(
            database_path, min_lat, max_lat, min_lon, max_lon
        )
        if min_x is not None:
            logger.info(f"Spatial subset indices: {min_x}, {max_x}, {min_y}, {max_y}")
            bounding_indices = (min_x, max_x, min_y, max_y)
        else:
            logger.warning(f"Could not determine indices for bounding box {bounding_box}")

    # Open the HDF5 file and extract data
    with h5py.File(database_path, 'r') as f:
        logger.info(f"Available MODIS groups: {list(f['MODIS'].keys())}")
        
        # Get the available datasets under the product group
        datasets_path = h5_group_name
        
        # Get the list of datasets
        if datasets_path in f:
            datasets = list(f[datasets_path].keys())
            logger.info(f"Total number of available datasets: {len(datasets)}")

            # Extract data for each year and month in the specified period
            for year in range(start_year, end_year + 1):
                for month in range(1, 13):
                    # Create pattern to match datasets for this year-month
                    pattern = f"MODIS_{datasets_path.split('/')[-1]}_{year}{month:02d}"
                    
                    # Find matching datasets
                    matching_datasets = [d for d in datasets if pattern in d]
                    
                    for dataset_name in matching_datasets:
                        try:
                            if bounding_indices:  
                                min_x, max_x, min_y, max_y = bounding_indices
                                img = f[f"{datasets_path}/{dataset_name}"][min_x:max_x+1, min_y:max_y+1]
                            else:
                                img = f[f"{datasets_path}/{dataset_name}"][:]
                                
                            # Clean data - replace -999 or extreme values with NaN
                            product_key = datasets_path.split('/')[-1]
                            if product_key in MODIS_PRODUCTS:
                                valid_range = MODIS_PRODUCTS[product_key].get('valid_range', (0, np.inf))
                                img = np.where((img < valid_range[0]) | (img > valid_range[1]), np.nan, img)
                            else:
                                img = np.where(img == -999, np.nan, img)
                                
                            extracted_data.append(img)
                        except Exception as e:
                            logger.warning(f"Could not extract {dataset_name}: {e}")
        else:
            logger.error(f"Group {datasets_path} not found in file")

    # Stack data into 3D array
    if len(extracted_data) > 0:
        extracted_data = np.stack(extracted_data, axis=0)
        logger.info(f"Extracted {len(extracted_data)} images with shape {extracted_data.shape}")
    else:
        extracted_data = np.array([])
        logger.warning("No data extracted")
    
    return extracted_data

def get_modis_dates(product_name: str, start_year: int, end_year: int) -> List[datetime]:
    """
    Generate approximate dates for MODIS data based on the product and date range.
    
    Args:
        product_name: MODIS product identifier
        start_year: Starting year
        end_year: Ending year
        
    Returns:
        List of datetime objects corresponding to the MODIS data
    """
    dates = []
    
    # Get product frequency from the lookup table
    frequency = '16-day'  # Default
    if product_name in MODIS_PRODUCTS:
        frequency = MODIS_PRODUCTS[product_name].get('frequency', '16-day')
    
    # Calculate interval in days
    if frequency == 'daily':
        interval = 1
    elif frequency == '8-day':
        interval = 8
    elif frequency == '16-day':
        interval = 16
    else:
        interval = 30  # Monthly default
    
    # Generate dates
    current_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=interval)
    
    return dates

def plot_modis_timeseries(data: np.ndarray, product_name: str, start_year: int, end_year: int, 
                         output_path: Optional[str] = None, title: Optional[str] = None, 
                         figsize: Tuple[int, int] = (12, 6)) -> Optional[plt.Figure]:
    """
    Create a time series plot of MODIS data.
    
    Args:
        data: 3D numpy array of MODIS data with shape (time, y, x)
        product_name: MODIS product identifier
        start_year: Starting year of the data
        end_year: Ending year of the data
        output_path: Path to save the figure (optional)
        title: Custom title for the plot (optional)
        figsize: Figure size as (width, height) in inches
        
    Returns:
        Matplotlib Figure object or None if error occurs
    """
    if data.size == 0:
        logger.warning("No data to plot")
        return None
        
    try:
        # Calculate spatial mean for each time step
        time_series = np.nanmean(data, axis=(1, 2))
        
        # Generate approximate dates
        dates = get_modis_dates(product_name, start_year, end_year)
        
        # Trim dates to match data length if necessary
        dates = dates[:len(time_series)]
        
        # Get product metadata
        product_info = MODIS_PRODUCTS.get(product_name, {
            'description': product_name,
            'units': '',
            'scale_factor': 1.0
        })
        
        # Apply scale factor if needed
        scale_factor = product_info.get('scale_factor', 1.0)
        if scale_factor != 1.0:
            time_series = time_series * scale_factor
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the time series
        ax.plot(dates, time_series, marker='o', linestyle='-', 
                markersize=4, color='#1f77b4', linewidth=1.5)
        
        # Calculate yearly averages for trend line
        if len(dates) > 5:  # Only if we have enough data
            yearly_data = {}
            for date, value in zip(dates, time_series):
                year = date.year
                if year not in yearly_data:
                    yearly_data[year] = []
                yearly_data[year].append(value)
            
            yearly_means = {year: np.nanmean(values) for year, values in yearly_data.items()}
            yearly_x = list(yearly_means.keys())
            yearly_y = list(yearly_means.values())
            
            # Add yearly average line
            if len(yearly_x) > 1:
                ax.plot(
                    [datetime(year, 6, 15) for year in yearly_x], 
                    yearly_y, 
                    'r--', 
                    linewidth=2,
                    label='Annual average'
                )
        
        # Format the x-axis as dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        fig.autofmt_xdate()
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set axis labels and title
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(f"{product_info.get('description', product_name)} ({product_info.get('units', '')})", 
                     fontsize=12)
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title(f"{product_info.get('description', product_name)} Time Series ({start_year}-{end_year})", 
                        fontsize=14, fontweight='bold')
        
        # Add legend if we have multiple lines
        if len(dates) > 5:
            ax.legend()
        
        plt.tight_layout()
        
        # Save if output path is specified
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {output_path}")
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating time series plot: {e}")
        return None

def create_modis_spatial_plot(data: np.ndarray, product_name: str, time_index: int = 0,
                            output_path: Optional[str] = None, title: Optional[str] = None,
                            figsize: Tuple[int, int] = (8, 8)) -> Optional[plt.Figure]:
    """
    Create a spatial map visualization of MODIS data for a single time step.
    
    Args:
        data: 3D numpy array of MODIS data with shape (time, y, x)
        product_name: MODIS product identifier
        time_index: Index of time slice to visualize (default: 0 = first)
        output_path: Path to save the figure (optional)
        title: Custom title for the plot (optional)
        figsize: Figure size as (width, height) in inches
        
    Returns:
        Matplotlib Figure object or None if error occurs
    """
    if data.size == 0 or time_index >= data.shape[0]:
        logger.warning(f"Invalid data or time index {time_index} out of range")
        return None
    
    try:
        # Get single time slice
        img = data[time_index]
        
        # Get product metadata
        product_info = MODIS_PRODUCTS.get(product_name, {
            'description': product_name,
            'units': '',
            'scale_factor': 1.0,
            'color_map': 'viridis'
        })
        
        # Apply scale factor
        scale_factor = product_info.get('scale_factor', 1.0)
        if scale_factor != 1.0:
            img = img * scale_factor
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Choose colormap
        cmap_name = product_info.get('color_map', 'viridis')
        cmap = plt.get_cmap(cmap_name)
        
        # Create masked array to handle NaN values
        masked_img = np.ma.masked_invalid(img)
        
        # Determine min/max values for colorbar, excluding outliers
        valid_data = masked_img.compressed()
        if len(valid_data) > 0:
            vmin = np.nanpercentile(valid_data, 1)
            vmax = np.nanpercentile(valid_data, 99)
        else:
            vmin, vmax = 0, 1
        
        # Create image
        im = ax.imshow(masked_img, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, pad=0.01)
        cbar.set_label(f"{product_info.get('description')} ({product_info.get('units', '')})")
        
        # Set title
        if title:
            ax.set_title(title, fontsize=12, fontweight='bold')
        else:
            ax.set_title(f"{product_info.get('description')} - Image {time_index+1}", 
                        fontsize=12, fontweight='bold')
        
        # Remove axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(True)
        
        plt.tight_layout()
        
        # Save if output path is specified
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Spatial plot saved to {output_path}")
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating spatial plot: {e}")
        return None

def create_modis_seasonal_plot(data: np.ndarray, product_name: str, start_year: int, end_year: int,
                             output_path: Optional[str] = None, figsize: Tuple[int, int] = (12, 8)) -> Optional[plt.Figure]:
    """
    Create a seasonal analysis plot showing monthly averages of MODIS data.
    
    Args:
        data: 3D numpy array of MODIS data with shape (time, y, x)
        product_name: MODIS product identifier
        start_year: Starting year of the data
        end_year: Ending year of the data
        output_path: Path to save the figure (optional)
        figsize: Figure size as (width, height) in inches
        
    Returns:
        Matplotlib Figure object or None if error occurs
    """
    if data.size == 0:
        logger.warning("No data to plot")
        return None
    
    try:
        # Calculate spatial mean for each time step
        time_series = np.nanmean(data, axis=(1, 2))
        
        # Generate dates
        dates = get_modis_dates(product_name, start_year, end_year)
        dates = dates[:len(time_series)]
        
        # Get product metadata
        product_info = MODIS_PRODUCTS.get(product_name, {
            'description': product_name,
            'units': '',
            'scale_factor': 1.0
        })
        
        # Apply scale factor
        scale_factor = product_info.get('scale_factor', 1.0)
        if scale_factor != 1.0:
            time_series = time_series * scale_factor
        
        # Create a DataFrame to organize by month
        df = pd.DataFrame({
            'date': dates,
            'value': time_series
        })
        df['month'] = df['date'].apply(lambda x: x.month)
        df['year'] = df['date'].apply(lambda x: x.year)
        
        # Calculate monthly statistics
        monthly_means = df.groupby('month')['value'].mean()
        monthly_std = df.groupby('month')['value'].std()
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [2, 1]})
        
        # Plot 1: Line plot by month with error bars
        months = list(range(1, 13))
        month_names = [calendar.month_abbr[m] for m in months]
        
        ax1.errorbar(
            months, 
            monthly_means.reindex(months).values,
            yerr=monthly_std.reindex(months).values,
            marker='o',
            linestyle='-',
            elinewidth=1,
            capsize=4
        )
        
        # Set x-axis ticks to month names
        ax1.set_xticks(months)
        ax1.set_xticklabels(month_names)
        
        # Add labels and title
        ax1.set_xlabel('Month', fontsize=12)
        ax1.set_ylabel(f"{product_info.get('description')} ({product_info.get('units')})", fontsize=12)
        ax1.set_title('Monthly Average Pattern', fontsize=14, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 2: Boxplot by season
        # Define seasons: Winter (12,1,2), Spring (3,4,5), Summer (6,7,8), Fall (9,10,11)
        df['season'] = df['month'].apply(lambda m: 
            'Winter' if m in [12, 1, 2] else
            'Spring' if m in [3, 4, 5] else
            'Summer' if m in [6, 7, 8] else
            'Fall'
        )
        
        # Create box plot with custom colors
        season_order = ['Winter', 'Spring', 'Summer', 'Fall']
        season_colors = ['#74add1', '#80cdc1', '#f46d43', '#dfc27d']
        
        sns.boxplot(
            x='season',
            y='value',
            data=df,
            order=season_order,
            palette=season_colors,
            ax=ax2
        )
        
        ax2.set_xlabel('Season', fontsize=12)
        ax2.set_ylabel('', fontsize=12)  # No y-label needed on second plot
        ax2.set_title('Seasonal Variability', fontsize=14, fontweight='bold')
        
        # Add overall title
        plt.suptitle(
            f"{product_info.get('description')} Seasonal Analysis ({start_year}-{end_year})", 
            fontsize=16, 
            fontweight='bold',
            y=1.05
        )
        
        plt.tight_layout()
        
        # Save if output path is specified
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Seasonal plot saved to {output_path}")
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating seasonal plot: {e}")
        return None

def calculate_modis_anomalies(
    data: np.ndarray, 
    product_name: str, 
    start_year: int, 
    end_year: int,
    baseline_years: Optional[List[int]] = None
) -> Tuple[np.ndarray, np.ndarray, List[datetime]]:
    """
    Calculate anomalies from a baseline period.
    
    Args:
        data: 3D numpy array of MODIS data with shape (time, y, x)
        product_name: MODIS product identifier
        start_year: Starting year of the data
        end_year: Ending year of the data
        baseline_years: Optional list of years to use as baseline (uses all available years if None)
        
    Returns:
        Tuple containing:
            - 1D numpy array of time series anomalies
            - 2D numpy array of spatial mean anomalies
            - List of datetime objects corresponding to the MODIS data
    """
    try:
        # Calculate spatial mean for each time step
        time_series = np.nanmean(data, axis=(1, 2))
        
        # Generate approximate dates
        dates = get_modis_dates(product_name, start_year, end_year)
        dates = dates[:len(time_series)]
        
        # Get product scale factor
        scale_factor = MODIS_PRODUCTS.get(product_name, {}).get('scale_factor', 1.0)
        
        # Apply scale factor if needed
        if scale_factor != 1.0:
            time_series = time_series * scale_factor
        
        # Determine baseline years
        if baseline_years is None:
            # Use all years as baseline
            baseline_years = list(range(start_year, end_year + 1))
        
        # Create arrays to store month indices and years
        months = np.array([date.month for date in dates])
        years = np.array([date.year for date in dates])
        
        # Calculate monthly climatology from baseline years
        monthly_means = {}
        for month in range(1, 13):
            # For each month, get indices where:
            # 1. month matches
            # 2. year is in baseline years
            indices = [i for i, (m, y) in enumerate(zip(months, years)) 
                      if m == month and y in baseline_years]
            
            if indices:
                # Calculate mean for this month
                monthly_means[month] = np.nanmean(time_series[indices])
            else:
                monthly_means[month] = np.nan
        
        # Calculate anomalies for each time point
        anomalies = np.zeros_like(time_series)
        for i, (date, value) in enumerate(zip(dates, time_series)):
            month = date.month
            baseline = monthly_means.get(month, np.nan)
            if not np.isnan(baseline) and not np.isnan(value):
                anomalies[i] = value - baseline
            else:
                anomalies[i] = np.nan
        
        return anomalies, time_series, dates
    
    except Exception as e:
        logger.error(f"Error calculating anomalies: {e}")
        return np.array([]), np.array([]), []

def create_modis_anomaly_plot(
    data: np.ndarray,
    product_name: str, 
    start_year: int, 
    end_year: int,
    baseline_years: Optional[List[int]] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> Optional[plt.Figure]:
    """
    Create a plot showing anomalies from a baseline period.
    
    Args:
        data: 3D numpy array of MODIS data with shape (time, y, x)
        product_name: MODIS product identifier
        start_year: Starting year of the data
        end_year: Ending year of the data
        baseline_years: Optional list of years to use as baseline (uses all available years if None)
        output_path: Path to save the figure (optional)
        figsize: Figure dimensions (width, height) in inches
        
    Returns:
        Matplotlib Figure object or None if error occurs
    """
    try:
        # Calculate anomalies
        anomalies, time_series, dates = calculate_modis_anomalies(
            data, product_name, start_year, end_year, baseline_years
        )
        
        if len(anomalies) == 0:
            logger.warning("No anomaly data to plot")
            return None
        
        # Get product info
        product_info = MODIS_PRODUCTS.get(product_name, {
            'description': product_name,
            'units': '',
            'scale_factor': 1.0
        })
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create the baseline period label
        if baseline_years and len(baseline_years) > 0:
            baseline_label = f"Baseline: {min(baseline_years)}-{max(baseline_years)}"
        else:
            baseline_label = f"Baseline: {start_year}-{end_year}"
        
        # Plot the anomalies
        colormap = plt.cm.RdBu_r  # Red-Blue diverging colormap
        colors = np.array(['r' if a > 0 else 'b' for a in anomalies])
        
        bars = ax.bar(dates, anomalies, color=colors, alpha=0.7, width=10)
        
        # Add zero line
        ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
        
        # Format x-axis as dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        fig.autofmt_xdate()
        
        # Add labels and title
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(f"Anomaly ({product_info.get('units', '')})", fontsize=12)
        ax.set_title(f"{product_info.get('description')} Anomalies", fontsize=14, fontweight='bold')
        
        # Add baseline text
        ax.text(0.02, 0.95, baseline_label, transform=ax.transAxes, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round,pad=0.5'))
        
        # Add a legend explaining the colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='r', edgecolor='r', alpha=0.7, label='Above normal'),
            Patch(facecolor='b', edgecolor='b', alpha=0.7, label='Below normal')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Anomaly plot saved to {output_path}")
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating anomaly plot: {e}")
        return None

def create_modis_comparison_plot(
    data_dict: Dict[str, np.ndarray],
    product_names: Dict[str, str],
    start_year: int,
    end_year: int,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> Optional[plt.Figure]:
    """
    Create a comparison plot of multiple MODIS products.
    
    Args:
        data_dict: Dictionary mapping product keys to data arrays
        product_names: Dictionary mapping product keys to display names
        start_year: First year of data
        end_year: Last year of data
        output_path: Path to save the figure (optional)
        figsize: Figure dimensions (width, height) in inches
        
    Returns:
        Matplotlib Figure object or None if error occurs
    """
    try:
        if not data_dict:
            logger.warning("No data to plot")
            return None
            
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Store line objects for legend
        lines = []
        
        # Process each product
        for product_key, product_data in data_dict.items():
            # Calculate spatial mean
            time_series = np.nanmean(product_data, axis=(1, 2))
            
            # Get scale factor
            scale_factor = MODIS_PRODUCTS.get(product_key, {}).get('scale_factor', 1.0)
            if scale_factor != 1.0:
                time_series = time_series * scale_factor
            
            # Generate dates
            dates = get_modis_dates(product_key, start_year, end_year)
            dates = dates[:len(time_series)]
            
            # Normalize time series (Z-score) for comparison
            valid_mask = ~np.isnan(time_series)
            if np.sum(valid_mask) > 1:
                mean_val = np.nanmean(time_series)
                std_val = np.nanstd(time_series)
                if std_val > 0:
                    normalized = (time_series - mean_val) / std_val
                else:
                    normalized = time_series - mean_val
            else:
                normalized = time_series
                
            # Plot the normalized time series
            display_name = product_names.get(product_key, product_key)
            line, = ax.plot(dates, normalized, '-o', markersize=3, linewidth=1.5, label=display_name)
            lines.append(line)
        
        # Format x-axis as dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        fig.autofmt_xdate()
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Add labels and legend
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Normalized Value (Z-score)', fontsize=12)
        ax.set_title('Comparison of MODIS Products (Normalized)', fontsize=14, fontweight='bold')
        ax.legend(handles=lines, loc='upper right')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {output_path}")
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating comparison plot: {e}", exc_info=True)
        return None

def create_modis_spatial_animation(
    data: np.ndarray, 
    product_name: str,
    start_year: int,
    end_year: int,
    output_path: Optional[str] = None,
    interval: int = 300,  # Animation speed in ms
    fps: int = 5,
    dpi: int = 100,
    figsize: Tuple[int, int] = (8, 8)
) -> Optional[str]:
    """
    Create an animated GIF showing the spatial changes in MODIS data over time.
    
    Args:
        data: 3D numpy array of MODIS data with shape (time, y, x)
        product_name: MODIS product identifier
        start_year: First year of data
        end_year: Last year of data
        output_path: Path to save the animation file (optional)
        interval: Animation frame interval in milliseconds
        fps: Frames per second for the animation
        dpi: Resolution for saved animation
        figsize: Figure dimensions (width, height) in inches
        
    Returns:
        Path to the saved animation file or None if error occurs
    """
    try:
        if data.size == 0:
            logger.warning("No data to animate")
            return None
            
        import matplotlib.animation as animation
        
        # Generate dates
        dates = get_modis_dates(product_name, start_year, end_year)
        dates = dates[:data.shape[0]]
        
        # Get product metadata
        product_info = MODIS_PRODUCTS.get(product_name, {
            'description': product_name,
            'units': '',
            'scale_factor': 1.0,
            'color_map': 'viridis'
        })
        
        # Apply scale factor
        scale_factor = product_info.get('scale_factor', 1.0)
        
        # Create figure and initial plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Choose colormap
        cmap_name = product_info.get('color_map', 'viridis')
        cmap = plt.get_cmap(cmap_name)
        
        # Determine min/max values for colorbar across all frames
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            vmin = np.nanpercentile(data, 1) * scale_factor
            vmax = np.nanpercentile(data, 99) * scale_factor
        else:
            vmin, vmax = 0, 1
        
        # Initial frame
        frame_data = data[0] * scale_factor
        masked_data = np.ma.masked_invalid(frame_data)
        
        im = ax.imshow(masked_data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        title = ax.text(0.5, 1.05, f"{product_info.get('description')} - {dates[0].strftime('%Y-%m-%d')}", 
                       transform=ax.transAxes, ha="center", fontsize=12)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, pad=0.01)
        cbar.set_label(f"{product_info.get('description')} ({product_info.get('units', '')})")
        
        # Remove axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(True)
        
        # Function to update animation frame
        def update_frame(frame_num):
            if frame_num < len(dates):
                frame_data = data[frame_num] * scale_factor
                masked_data = np.ma.masked_invalid(frame_data)
                im.set_array(masked_data)
                title.set_text(f"{product_info.get('description')} - {dates[frame_num].strftime('%Y-%m-%d')}")
            return im, title
        
        # Create animation
        ani = animation.FuncAnimation(
            fig, 
            update_frame, 
            frames=min(50, len(dates)),  # Limit to max 50 frames for performance
            interval=interval, 
            blit=False
        )
        
        # Save animation if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            ani.save(output_path, writer='pillow', fps=fps, dpi=dpi)
            logger.info(f"Animation saved to {output_path}")
            plt.close(fig)
            return output_path
        
        return None
        
    except Exception as e:
        logger.error(f"Error creating animation: {e}", exc_info=True)
        plt.close()
        return None

def calculate_modis_statistics(
    data: np.ndarray,
    product_name: str,
    start_year: int,
    end_year: int
) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for MODIS data.
    
    Args:
        data: 3D numpy array of MODIS data with shape (time, y, x)
        product_name: MODIS product identifier
        start_year: First year of data
        end_year: Last year of data
        
    Returns:
        Dictionary containing various statistics
    """
    try:
        if data.size == 0:
            logger.warning("No data to calculate statistics")
            return {}
            
        # Get product metadata
        product_info = MODIS_PRODUCTS.get(product_name, {
            'description': product_name,
            'units': '',
            'scale_factor': 1.0
        })
        
        # Apply scale factor
        scale_factor = product_info.get('scale_factor', 1.0)
        
        # Initialize statistics dictionary
        stats = {
            'product': product_name,
            'description': product_info.get('description', product_name),
            'units': product_info.get('units', ''),
            'period': f"{start_year}-{end_year}"
        }
        
        # Calculate temporal mean (spatial average for each timestep)
        time_series = np.nanmean(data, axis=(1, 2)) * scale_factor
        
        # Basic statistics
        stats['mean'] = float(np.nanmean(time_series))
        stats['median'] = float(np.nanmedian(time_series))
        stats['std'] = float(np.nanstd(time_series))
        stats['min'] = float(np.nanmin(time_series))
        stats['max'] = float(np.nanmax(time_series))
        stats['range'] = stats['max'] - stats['min']
        
        # Generate dates
        dates = get_modis_dates(product_name, start_year, end_year)
        dates = dates[:len(time_series)]
        
        # Extract years and months
        years = [date.year for date in dates]
        months = [date.month for date in dates]
        
        # Calculate trend
        if len(time_series) > 2:
            try:
                from scipy import stats as scipy_stats
                # Use indices as x values (equal time steps)
                x = np.arange(len(time_series))
                mask = ~np.isnan(time_series)
                
                if np.sum(mask) > 2:  # Need at least 3 points
                    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
                        x[mask], time_series[mask])
                    
                    stats['trend'] = {
                        'slope': float(slope),
                        'intercept': float(intercept),
                        'r_value': float(r_value),
                        'r_squared': float(r_value ** 2),
                        'p_value': float(p_value),
                        'std_err': float(std_err),
                        'significant': p_value < 0.05
                    }
            except Exception as e:
                logger.warning(f"Could not calculate trend: {e}")
        
        # Monthly statistics
        monthly_stats = {}
        for month in range(1, 13):
            month_indices = [i for i, m in enumerate(months) if m == month]
            if month_indices:
                month_data = time_series[month_indices]
                if len(month_data) > 0:
                    monthly_stats[month] = {
                        'mean': float(np.nanmean(month_data)),
                        'median': float(np.nanmedian(month_data)),
                        'std': float(np.nanstd(month_data)),
                        'min': float(np.nanmin(month_data)),
                        'max': float(np.nanmax(month_data)),
                        'count': int(len(month_data))
                    }
        
        stats['monthly'] = monthly_stats
        
        # Calculate anomalies
        try:
            anomalies, _, _ = calculate_modis_anomalies(data, product_name, start_year, end_year)
            
            # Count positive and negative anomalies
            positive_count = np.sum(anomalies > 0)
            negative_count = np.sum(anomalies < 0)
            
            stats['anomalies'] = {
                'positive_count': int(positive_count),
                'negative_count': int(negative_count),
                'mean': float(np.nanmean(anomalies)),
                'std': float(np.nanstd(anomalies)),
                'max_positive': float(np.nanmax(anomalies)),
                'max_negative': float(np.nanmin(anomalies))
            }
        except Exception as e:
            logger.warning(f"Could not calculate anomalies: {e}")
        
        # Calculate spatial statistics for first timestep
        if data.shape[0] > 0:
            first_image = data[0] * scale_factor
            valid_pixels = np.sum(~np.isnan(first_image))
            total_pixels = first_image.size
            
            stats['spatial'] = {
                'spatial_mean': float(np.nanmean(first_image)),
                'spatial_median': float(np.nanmedian(first_image)),
                'spatial_std': float(np.nanstd(first_image)),
                'spatial_min': float(np.nanmin(first_image)),
                'spatial_max': float(np.nanmax(first_image)),
                'valid_pixels': int(valid_pixels),
                'total_pixels': int(total_pixels),
                'coverage_percent': float(valid_pixels / total_pixels * 100) if total_pixels > 0 else 0
            }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating statistics: {e}", exc_info=True)
        return {}
