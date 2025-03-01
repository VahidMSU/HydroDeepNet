from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

try:
    from AI_agent.config import AgentConfig
    from AI_agent.modis_utilities import (
        get_modis_dates, plot_modis_timeseries, create_modis_spatial_plot, 
        create_modis_seasonal_plot, MODIS_PRODUCTS, create_modis_anomaly_plot,
        calculate_modis_statistics, create_modis_comparison_plot,
        create_modis_spatial_animation
    )
except ImportError:
    from config import AgentConfig
    from modis_utilities import (
        get_modis_dates, plot_modis_timeseries, create_modis_spatial_plot, 
        create_modis_seasonal_plot, MODIS_PRODUCTS, create_modis_anomaly_plot,
        calculate_modis_statistics, create_modis_comparison_plot,
        create_modis_spatial_animation
    )

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def generate_modis_report(data: np.ndarray, product_name: str, start_year: int, end_year: int,
                         bounding_box: Optional[Tuple[float, float, float, float]] = None,
                         output_dir: str = 'modis_report') -> str:
    """
    Generate a comprehensive MODIS data analysis report with visualizations.
    
    Args:
        data: 3D numpy array of MODIS data with shape (time, y, x)
        product_name: MODIS product identifier
        start_year: Starting year of the data
        end_year: Ending year of the data
        bounding_box: Optional [min_lon, min_lat, max_lon, max_lat] of the region
        output_dir: Directory to save report files
        
    Returns:
        Path to the generated report file
    """
    if data.size == 0:
        logger.warning("No data to generate report")
        return ""
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get product metadata
        product_info = MODIS_PRODUCTS.get(product_name, {
            'description': product_name,
            'units': '',
            'scale_factor': 1.0
        })
        
        # Define file paths
        timeseries_path = os.path.join(output_dir, f"{product_name}_timeseries.png")
        spatial_path = os.path.join(output_dir, f"{product_name}_spatial.png")
        seasonal_path = os.path.join(output_dir, f"{product_name}_seasonal.png")
        report_path = os.path.join(output_dir, f"{product_name}_report.md")
        stats_path = os.path.join(output_dir, f"{product_name}_stats.csv")
        
        # Generate visualizations
        plot_modis_timeseries(data, product_name, start_year, end_year, timeseries_path)
        
        # If we have at least one data point, create a spatial visualization of the first one
        if data.shape[0] > 0:
            create_modis_spatial_plot(data, product_name, 0, spatial_path)
            
        create_modis_seasonal_plot(data, product_name, start_year, end_year, seasonal_path)
        
        # Calculate statistics
        spatial_mean = np.nanmean(data, axis=(1, 2)) * product_info.get('scale_factor', 1.0)
        
        # Prepare statistics table
        dates = get_modis_dates(product_name, start_year, end_year)
        dates = dates[:len(spatial_mean)]
        
        stats_df = pd.DataFrame({
            'Date': dates,
            'Mean': spatial_mean,
            'Min': np.nanmin(data, axis=(1, 2)) * product_info.get('scale_factor', 1.0),
            'Max': np.nanmax(data, axis=(1, 2)) * product_info.get('scale_factor', 1.0),
            'StdDev': np.nanstd(data, axis=(1, 2)) * product_info.get('scale_factor', 1.0)
        })
        
        # Add year and month columns
        stats_df['Year'] = stats_df['Date'].dt.year
        stats_df['Month'] = stats_df['Date'].dt.month
        
        # Save statistics to CSV
        stats_df.to_csv(stats_path, index=False)
        
        # Generate markdown report
        with open(report_path, 'w') as f:
            # Header
            f.write(f"# MODIS {product_info.get('description')} Analysis Report\n\n")
            
            # Basic information
            f.write("## Overview\n\n")
            f.write(f"**Product:** {product_name} - {product_info.get('description')}\n\n")
            f.write(f"**Period:** {start_year} to {end_year}\n\n")
            
            if bounding_box:
                f.write(f"**Region:** Lat [{bounding_box[1]:.4f}, {bounding_box[3]:.4f}], ")
                f.write(f"Lon [{bounding_box[0]:.4f}, {bounding_box[2]:.4f}]\n\n")
                f.write(f"**Units:** {product_info.get('units', '')}\n\n")
                f.write(f"**Data points:** {len(spatial_mean)}\n\n")
            
            # Summary statistics
            f.write("## Summary Statistics\n\n")
            f.write(f"**Mean:** {np.nanmean(spatial_mean):.4f} {product_info.get('units', '')}\n\n")
            f.write(f"**Minimum:** {np.nanmin(spatial_mean):.4f} {product_info.get('units', '')}\n\n")
            f.write(f"**Maximum:** {np.nanmax(spatial_mean):.4f} {product_info.get('units', '')}\n\n")
            f.write(f"**Standard Deviation:** {np.nanstd(spatial_mean):.4f} {product_info.get('units', '')}\n\n")
            
            # Yearly averages
            yearly_stats = stats_df.groupby('Year')['Mean'].agg(['mean', 'min', 'max', 'std']).reset_index()
            
            f.write("## Yearly Averages\n\n")
            f.write("| Year | Mean | Min | Max | Std Dev |\n")
            f.write("|------|------|-----|-----|---------|\n")
            for _, row in yearly_stats.iterrows():
                f.write(f"| {int(row['Year'])} | {row['mean']:.4f} | {row['min']:.4f} | {row['max']:.4f} | {row['std']:.4f} |\n")
            f.write("\n")

            # Seasonal patterns
            f.write("## Seasonal Patterns\n\n")
            f.write(f"The seasonal analysis shows how {product_info.get('description')} values vary throughout the year.\n\n")
            f.write(f"![Seasonal Analysis]({os.path.basename(seasonal_path)})\n\n")
            
            # Try to add some interpretation of seasonal patterns
            try:
                season_means = stats_df.copy()
                season_means['Season'] = season_means['Month'].apply(lambda m: 
                    'Winter' if m in [12, 1, 2] else
                    'Spring' if m in [3, 4, 5] else
                    'Summer' if m in [6, 7, 8] else
                    'Fall'
                )
                season_avg = season_means.groupby('Season')['Mean'].mean()
                highest_season = season_avg.idxmax()
                lowest_season = season_avg.idxmin()
                seasonal_range = season_avg.max() - season_avg.min()
                seasonal_range_pct = (seasonal_range / season_avg.mean()) * 100
                
                f.write(f"The highest values typically occur during **{highest_season}**, while the lowest values are in **{lowest_season}**. ")
                f.write(f"The seasonal range is approximately {seasonal_range:.4f} {product_info.get('units', '')}, ")
                f.write(f"representing a {seasonal_range_pct:.1f}% variation from the annual mean.\n\n")
            except Exception as e:
                logger.warning(f"Could not generate seasonal interpretation: {e}")
            
            # Time series visualization
            f.write("## Time Series Analysis\n\n")
            f.write(f"The time series shows the change in {product_info.get('description')} over the entire period.\n\n")
            f.write(f"![Time Series]({os.path.basename(timeseries_path)})\n\n")
            
            # Calculate and report trends if we have enough years
            if end_year - start_year > 1:
                try:
                    yearly_means = stats_df.groupby('Year')['Mean'].mean().reset_index()
                    if len(yearly_means) > 2:
                        from scipy.stats import linregress
                        x = yearly_means['Year']
                        y = yearly_means['Mean']
                        slope, intercept, r_value, p_value, std_err = linregress(x, y)
                        
                        trend_direction = "increasing" if slope > 0 else "decreasing"
                        significance = "significant" if p_value < 0.05 else "not statistically significant"
                        change_per_year = slope
                        total_change = slope * (end_year - start_year)
                        total_change_pct = (total_change / (intercept + slope * start_year)) * 100
                        
                        f.write("### Trend Analysis\n\n")
                        f.write(f"The data shows a {trend_direction} trend of {abs(change_per_year):.4f} {product_info.get('units', '')} per year ")
                        f.write(f"(p-value: {p_value:.4f}, {significance}).\n\n")
                        f.write(f"Over the entire {end_year-start_year+1} year period, this represents a change of ")
                        f.write(f"{abs(total_change):.4f} {product_info.get('units', '')} ({abs(total_change_pct):.1f}%).\n\n")
                except Exception as e:
                    logger.warning(f"Could not generate trend analysis: {e}")
            
            # Spatial visualization
            f.write("## Spatial Distribution\n\n")
            f.write(f"The spatial map shows the distribution of {product_info.get('description')} across the study area.\n\n")
            f.write(f"![Spatial Distribution]({os.path.basename(spatial_path)})\n\n")
            
            # Add pixel statistics
            f.write("### Pixel-level Statistics\n\n")
            valid_pixels = np.sum(~np.isnan(data[0]))
            total_pixels = data[0].size
            coverage_pct = (valid_pixels / total_pixels) * 100 if total_pixels > 0 else 0
            
            f.write(f"**Valid data pixels:** {valid_pixels} out of {total_pixels} ({coverage_pct:.1f}%)\n\n")
            
            if valid_pixels > 0:
                # Calculate spatial variability
                spatial_std = np.nanstd(data[0] * product_info.get('scale_factor', 1.0))
                spatial_cv = (spatial_std / np.nanmean(data[0] * product_info.get('scale_factor', 1.0))) * 100
                
                f.write(f"**Spatial variability:** {spatial_std:.4f} {product_info.get('units', '')} ")
                f.write(f"(coefficient of variation: {spatial_cv:.1f}%)\n\n")
            
            # Applications based on product type
            f.write("## Potential Applications\n\n")
            
            if "NDVI" in product_name or "EVI" in product_name:
                f.write("This vegetation index data can be used for:\n\n")
                f.write("- Monitoring crop health and agricultural productivity\n")
                f.write("- Assessing drought impacts on vegetation\n")
                f.write("- Tracking seasonal vegetation phenology\n")
                f.write("- Land cover change detection\n")
            elif "ET" in product_name:
                f.write("This evapotranspiration data can be used for:\n\n")
                f.write("- Agricultural water use monitoring\n")
                f.write("- Drought assessment and water stress detection\n")
                f.write("- Hydrological modeling inputs\n")
                f.write("- Water resource management\n")
            elif "Lai" in product_name:
                f.write("This leaf area index data can be used for:\n\n")
                f.write("- Ecosystem productivity modeling\n")
                f.write("- Canopy structure analysis\n")
                f.write("- Carbon sequestration estimation\n")
                f.write("- Climate change impact assessment\n")
            elif "Fpar" in product_name:
                f.write("This fraction of photosynthetically active radiation data can be used for:\n\n")
                f.write("- Primary productivity modeling\n")
                f.write("- Radiation use efficiency analysis\n")
                f.write("- Terrestrial ecosystem modeling\n")
                f.write("- Carbon cycle studies\n")
            elif "refl" in product_name:
                f.write("This surface reflectance data can be used for:\n\n")
                f.write("- Land cover classification\n")
                f.write("- Surface feature detection\n")
                f.write("- Creating custom spectral indices\n")
                f.write("- Input for radiative transfer models\n")
            
            # Data source and methodology
            f.write("\n## Data Source and Methodology\n\n")
            f.write("This analysis is based on the MODIS (Moderate Resolution Imaging Spectroradiometer) satellite data. ")
            f.write("The original MODIS data products are provided by NASA's Earth Observing System Data and Information System (EOSDIS) ")
            f.write("and have been processed to extract time series for the specified geographic region.\n\n")
            
            f.write("**Processing steps:**\n\n")
            f.write("1. Extraction of raw data from HDF5 database\n")
            f.write("2. Spatial subsetting to the region of interest\n")
            f.write("3. Application of quality filters and scaling factors\n")
            f.write("4. Statistical analysis and visualization\n\n")
            
            # Export information
            f.write("## Data Export\n\n")
            f.write(f"The complete dataset has been exported to CSV format. Access the data at: [{os.path.basename(stats_path)}]({os.path.basename(stats_path)})\n\n")
            
            # Report generation information
            f.write("---\n\n")
            f.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M')}*\n")
        
        logger.info(f"Report successfully generated: {report_path}")
        return report_path
        
    except Exception as e:
        logger.error(f"Error generating MODIS report: {e}", exc_info=True)
        return ""

def export_modis_data_to_csv(data: np.ndarray, product_name: str, start_year: int, end_year: int,
                           output_path: str) -> bool:
    """
    Export processed MODIS data to CSV format with dates.
    
    Args:
        data: 3D numpy array of MODIS data with shape (time, y, x)
        product_name: MODIS product identifier
        start_year: First year of the data
        end_year: Last year of the data
        output_path: Path to save the CSV file
        
    Returns:
        Boolean indicating success
    """
    try:
        if data.size == 0:
            logger.warning("No data to export")
            return False
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get spatial means for each time point
        means = np.nanmean(data, axis=(1, 2))
        
        # Get additional statistics
        mins = np.nanmin(data, axis=(1, 2))
        maxs = np.nanmax(data, axis=(1, 2))
        stds = np.nanstd(data, axis=(1, 2))
        
        # Get dates
        dates = get_modis_dates(product_name, start_year, end_year)
        dates = dates[:len(means)]
        
        # Get scale factor
        scale_factor = MODIS_PRODUCTS.get(product_name, {}).get('scale_factor', 1.0)
        
        # Create DataFrame with scaled values
        df = pd.DataFrame({
            'Date': dates,
            'Mean': means * scale_factor,
            'Min': mins * scale_factor,
            'Max': maxs * scale_factor,
            'StdDev': stds * scale_factor
        })
        
        # Add year and month columns
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Data exported to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        return False

def batch_process_modis(config: Dict[str, Any], output_dir: str) -> List[str]:
    """
    Batch process multiple MODIS products and generate reports.
    
    Args:
        config: Configuration dictionary with processing parameters
        output_dir: Directory to save all outputs
        
    Returns:
        List of paths to generated reports
    """
    from AI_agent.modis_utilities import extract_modis_data
    
    reports = []
    
    try:
        # Create base output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get basic parameters
        database_path = config.get('database_path', AgentConfig.HydroGeoDataset_ML_250_path)
        start_year = config.get('start_year', 2000)
        end_year = config.get('end_year', 2020)
        bounding_box = config.get('bounding_box')
        
        # Get products to process (default to all available)
        products = config.get('products', list(MODIS_PRODUCTS.keys()))
        
        for product_name in products:
            logger.info(f"Processing MODIS product: {product_name}")
            
            # Create product-specific directory
            product_dir = os.path.join(output_dir, product_name)
            os.makedirs(product_dir, exist_ok=True)
            
            try:
                # Extract data
                h5_group_name = f"MODIS/{product_name}"
                data = extract_modis_data(
                    database_path=database_path,
                    h5_group_name=h5_group_name,
                    start_year=start_year,
                    end_year=end_year,
                    bounding_box=bounding_box
                )
                
                if data.size > 0:
                    # Generate report
                    report_path = generate_modis_report(
                        data=data,
                        product_name=product_name,
                        start_year=start_year,
                        end_year=end_year,
                        bounding_box=bounding_box,
                        output_dir=product_dir
                    )
                    
                    if report_path:
                        reports.append(report_path)
                else:
                    logger.warning(f"No data extracted for {product_name}")
            except Exception as e:
                logger.error(f"Error processing {product_name}: {e}")
                
        return reports
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        return reports

def generate_comprehensive_modis_report(
    data: Dict[str, np.ndarray],  # Dictionary of product_name: data
    products_config: Dict[str, Dict],
    region_name: Optional[str] = None,
    bounding_box: Optional[Tuple[float, float, float, float]] = None,
    output_dir: str = 'modis_comprehensive_report',
    include_animations: bool = False,
    baseline_years: Optional[Dict[str, List[int]]] = None,
    cross_product_analysis: bool = True
) -> str:
    """
    Generate a comprehensive MODIS data analysis report with multiple products and visualizations.
    
    Args:
        data: Dictionary mapping product names to data arrays
        products_config: Dictionary with config for each product (start/end years)
        region_name: Name of the region (optional)
        bounding_box: Optional [min_lon, min_lat, max_lon, max_lat] of the region
        output_dir: Directory to save report files
        include_animations: Whether to include GIF animations (resource intensive)
        baseline_years: Dictionary mapping product names to lists of baseline years
        cross_product_analysis: Whether to analyze relationships between products
        
    Returns:
        Path to the generated report file
    """
    if not data:
        logger.warning("No data to generate report")
        return ""
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize paths for report components
        report_path = os.path.join(output_dir, "modis_comprehensive_report.md")
        summary_path = os.path.join(output_dir, "modis_data_summary.csv")
        
        # File paths for visualizations
        visualization_paths = {}
        stats_data = {}
        
        # Process each product
        for product_name, product_data in data.items():
            if product_data.size == 0:
                logger.warning(f"No data available for {product_name}")
                continue
                
            # Get product config
            config = products_config.get(product_name, {})
            start_year = config.get('start_year', 2000)
            end_year = config.get('end_year', 2020)
            
            # Create product directory
            product_dir = os.path.join(output_dir, product_name)
            os.makedirs(product_dir, exist_ok=True)
            
            # Initialize paths for this product
            product_paths = {
                'timeseries': os.path.join(product_dir, f"{product_name}_timeseries.png"),
                'spatial': os.path.join(product_dir, f"{product_name}_spatial.png"),
                'seasonal': os.path.join(product_dir, f"{product_name}_seasonal.png"),
                'anomaly': os.path.join(product_dir, f"{product_name}_anomaly.png"),
                'stats': os.path.join(product_dir, f"{product_name}_stats.csv")
            }
            
            # Generate visualizations
            plot_modis_timeseries(
                product_data, product_name, start_year, end_year, 
                product_paths['timeseries']
            )
            
            create_modis_spatial_plot(
                product_data, product_name, 0, product_paths['spatial']
            )
            
            create_modis_seasonal_plot(
                product_data, product_name, start_year, end_year, 
                product_paths['seasonal']
            )
            
            # Get baseline years for this product
            product_baseline = baseline_years.get(product_name) if baseline_years else None
            
            # Create anomaly plot
            create_modis_anomaly_plot(
                product_data, product_name, start_year, end_year,
                product_baseline, product_paths['anomaly']
            )
            
            # Generate animation if requested (resource intensive)
            if include_animations:
                animation_path = os.path.join(product_dir, f"{product_name}_animation.gif")
                create_modis_spatial_animation(
                    product_data, product_name, start_year, end_year, animation_path
                )
                product_paths['animation'] = animation_path
            
            # Calculate statistics
            stats = calculate_modis_statistics(
                product_data, product_name, start_year, end_year
            )
            stats_data[product_name] = stats
            
            # Export time series to CSV
            export_modis_data_to_csv(
                product_data, product_name, start_year, end_year, 
                product_paths['stats']
            )
            
            visualization_paths[product_name] = product_paths
        
        # Cross-product analysis if we have multiple products
        if cross_product_analysis and len(data) > 1:
            cross_dir = os.path.join(output_dir, "cross_product")
            os.makedirs(cross_dir, exist_ok=True)
            
            # Generate comparison plot
            comparison_path = os.path.join(cross_dir, "products_comparison.png")
            product_names = {k: MODIS_PRODUCTS.get(k, {}).get('description', k) for k in data.keys()}
            
            # Find common time period
            min_start = max(cfg.get('start_year', 2000) for cfg in products_config.values())
            max_end = min(cfg.get('end_year', 2020) for cfg in products_config.values())
            
            create_modis_comparison_plot(
                data, product_names, min_start, max_end, comparison_path
            )
            
            # Calculate correlations between products
            try:
                correlation_path = os.path.join(cross_dir, "product_correlations.png")
                
                # Extract time series for each product
                series_data = {}
                for product_name, product_data in data.items():
                    config = products_config.get(product_name, {})
                    start_year = config.get('start_year', 2000)
                    end_year = config.get('end_year', 2020)
                    scale_factor = MODIS_PRODUCTS.get(product_name, {}).get('scale_factor', 1.0)
                    
                    # Calculate spatial mean
                    time_series = np.nanmean(product_data, axis=(1, 2)) * scale_factor
                    
                    # Get product dates and create DataFrame
                    dates = get_modis_dates(product_name, start_year, end_year)
                    dates = dates[:len(time_series)]
                    df = pd.DataFrame({
                        'date': dates[:len(time_series)],
                        'value': time_series,
                        'year': [d.year for d in dates[:len(time_series)]],
                        'month': [d.month for d in dates[:len(time_series)]]
                    })
                    series_data[product_name] = df
                
                # Create combined data for correlation analysis
                combined_data = {}
                for name, df in series_data.items():
                    combined_data[name] = df.set_index('date')['value']
                
                # Create a combined DataFrame
                combined_df = pd.DataFrame(combined_data)
                
                # Calculate correlation matrix
                corr_matrix = combined_df.corr()
                
                # Plot correlation matrix
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(
                    corr_matrix, 
                    annot=True, 
                    cmap='coolwarm', 
                    vmin=-1, 
                    vmax=1,
                    ax=ax
                )
                plt.title('Correlation Between MODIS Products', fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig(correlation_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                visualization_paths['cross_product'] = {
                    'comparison': comparison_path,
                    'correlation': correlation_path
                }
            except Exception as e:
                logger.warning(f"Could not create correlation analysis: {e}")
        
        # Generate the comprehensive markdown report
        with open(report_path, 'w') as f:
            # Header
            f.write("# Comprehensive MODIS Data Analysis Report\n\n")
            
            # Region information
            if region_name:
                f.write(f"**Region:** {region_name}\n\n")
                
            if bounding_box:
                f.write(f"**Coordinates:** Lat [{bounding_box[1]:.4f}, {bounding_box[3]:.4f}], ")
                f.write(f"Lon [{bounding_box[0]:.4f}, {bounding_box[2]:.4f}]\n\n")
            
            # Table of contents
            f.write("## Table of Contents\n\n")
            f.write("1. [Executive Summary](#executive-summary)\n")
            f.write("2. [Products Overview](#products-overview)\n")
            
            product_counter = 3
            for product_name in data.keys():
                safe_name = product_name.replace("/", "-")
                f.write(f"{product_counter}. [{MODIS_PRODUCTS.get(product_name, {}).get('description', product_name)}](#product-{safe_name})\n")
                product_counter += 1
            
            if cross_product_analysis and len(data) > 1:
                f.write(f"{product_counter}. [Cross-Product Analysis](#cross-product-analysis)\n")
                product_counter += 1
                
            f.write(f"{product_counter}. [Methodology](#methodology)\n")
            product_counter += 1
            f.write(f"{product_counter}. [Data Sources](#data-sources)\n")
            product_counter += 1
            f.write(f"{product_counter}. [Limitations](#limitations)\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write("This report presents a comprehensive analysis of MODIS remote sensing data for ")
            if region_name:
                f.write(f"the {region_name} region. ")
            else:
                f.write("the specified region. ")
                
            f.write("The analysis includes temporal trends, spatial patterns, seasonal variations, ")
            f.write("and anomaly detection for multiple MODIS products.\n\n")
            
            # Add key findings
            f.write("### Key Findings\n\n")
            
            # Generate key findings based on statistics
            for product_name, stats in stats_data.items():
                product_info = MODIS_PRODUCTS.get(product_name, {'description': product_name})
                f.write(f"- **{product_info.get('description')}**: ")
                
                # Add trend information if available
                if 'trend' in stats:
                    trend = stats['trend']
                    trend_direction = "increasing" if trend['slope'] > 0 else "decreasing"
                    if trend['significant']:
                        f.write(f"Shows a statistically significant {trend_direction} trend ")
                        f.write(f"({abs(trend['slope']):.4f} units/year). ")
                    else:
                        f.write(f"Shows a non-significant {trend_direction} trend. ")
                
                # Add seasonal information if available
                if 'monthly' in stats:
                    monthly_means = {m: v['mean'] for m, v in stats['monthly'].items()}
                    if monthly_means:
                        max_month = max(monthly_means.items(), key=lambda x: x[1])[0]
                        min_month = min(monthly_means.items(), key=lambda x: x[1])[0]
                        month_names = {
                            1: 'January', 2: 'February', 3: 'March', 4: 'April',
                            5: 'May', 6: 'June', 7: 'July', 8: 'August',
                            9: 'September', 10: 'October', 11: 'November', 12: 'December'
                        }
                        f.write(f"Peaks in {month_names.get(max_month, max_month)}, ")
                        f.write(f"lowest in {month_names.get(min_month, min_month)}. ")
                
                f.write("\n")
                
            f.write("\n")
            
            # Products Overview
            f.write("## Products Overview\n\n")
            f.write("| Product | Description | Period | Resolution | Units |\n")
            f.write("|---------|-------------|--------|------------|-------|\n")
            
            for product_name in data.keys():
                config = products_config.get(product_name, {})
                start_year = config.get('start_year', 2000)
                end_year = config.get('end_year', 2020)
                product_info = MODIS_PRODUCTS.get(product_name, {
                    'description': product_name,
                    'units': '-',
                    'frequency': '-'
                })
                
                f.write(f"| {product_name} | {product_info.get('description')} | ")
                f.write(f"{start_year}-{end_year} | {product_info.get('frequency')} | ")
                f.write(f"{product_info.get('units')} |\n")
                
            f.write("\n")
            
            # Individual Product Sections
            for product_name, product_data in data.items():
                if product_name not in visualization_paths:
                    continue
                    
                # Create section anchor
                safe_name = product_name.replace("/", "-")
                f.write(f"<a id='product-{safe_name}'></a>\n")
                
                # Get product info
                product_info = MODIS_PRODUCTS.get(product_name, {'description': product_name})
                
                # Section header
                f.write(f"## {product_info.get('description')}\n\n")
                
                # Summary statistics if available
                if product_name in stats_data:
                    stats = stats_data[product_name]
                    f.write("### Summary Statistics\n\n")
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    
                    # Basic statistics
                    f.write(f"| Mean | {stats.get('mean', 'N/A'):.4f} {product_info.get('units', '')} |\n")
                    f.write(f"| Median | {stats.get('median', 'N/A'):.4f} {product_info.get('units', '')} |\n")
                    f.write(f"| Standard Deviation | {stats.get('std', 'N/A'):.4f} {product_info.get('units', '')} |\n")
                    f.write(f"| Min | {stats.get('min', 'N/A'):.4f} {product_info.get('units', '')} |\n")
                    f.write(f"| Max | {stats.get('max', 'N/A'):.4f} {product_info.get('units', '')} |\n")
                    
                    # Trend information if available
                    if 'trend' in stats:
                        trend = stats['trend']
                        trend_direction = "Increasing" if trend['slope'] > 0 else "Decreasing"
                        significance = "Significant" if trend['significant'] else "Not significant"
                        
                        f.write(f"| Trend | {trend_direction} ({abs(trend['slope']):.4f} units/year) |\n")
                        f.write(f"| Trend Significance | {significance} (p={trend['p_value']:.4f}) |\n")
                        f.write(f"| R-squared | {trend['r_squared']:.4f} |\n")
                    
                    f.write("\n")
                
                # Time Series Analysis
                paths = visualization_paths[product_name]
                
                f.write("### Time Series Analysis\n\n")
                if 'timeseries' in paths:
                    f.write(f"![Time Series]({os.path.relpath(paths['timeseries'], output_dir)})\n\n")
                
                # Anomaly Analysis
                if 'anomaly' in paths:
                    f.write("### Anomaly Analysis\n\n")
                    f.write(f"![Anomalies]({os.path.relpath(paths['anomaly'], output_dir)})\n\n")
                    
                    # Add anomaly interpretation if available
                    if product_name in stats_data and 'anomalies' in stats_data[product_name]:
                        anomaly_stats = stats_data[product_name]['anomalies']
                        pos_count = anomaly_stats.get('positive_count', 0)
                        neg_count = anomaly_stats.get('negative_count', 0)
                        total_count = pos_count + neg_count
                        
                        if total_count > 0:
                            pos_pct = pos_count / total_count * 100
                            neg_pct = neg_count / total_count * 100
                            
                            f.write(f"The data shows positive anomalies in {pos_count} timepoints ({pos_pct:.1f}%) ")
                            f.write(f"and negative anomalies in {neg_count} timepoints ({neg_pct:.1f}%). ")
                            
                            if abs(pos_count - neg_count) > 0.2 * total_count:
                                if pos_count > neg_count:
                                    f.write("This indicates predominantly above-average conditions during the observed period.")
                                else:
                                    f.write("This indicates predominantly below-average conditions during the observed period.")
                            else:
                                f.write("The distribution of anomalies is relatively balanced.")
                            
                            f.write("\n\n")
                
                # Seasonal Pattern
                if 'seasonal' in paths:
                    f.write("### Seasonal Pattern\n\n")
                    f.write(f"![Seasonal Pattern]({os.path.relpath(paths['seasonal'], output_dir)})\n\n")
                    
                    # Add seasonal interpretation based on statistics
                    if product_name in stats_data and 'monthly' in stats_data[product_name]:
                        monthly_stats = stats_data[product_name]['monthly']
                        monthly_means = {m: v['mean'] for m, v in monthly_stats.items()}
                        
                        if monthly_means:
                            max_month = max(monthly_means.items(), key=lambda x: x[1])[0]
                            min_month = min(monthly_means.items(), key=lambda x: x[1])[0]
                            month_names = {
                                1: 'January', 2: 'February', 3: 'March', 4: 'April',
                                5: 'May', 6: 'June', 7: 'July', 8: 'August',
                                9: 'September', 10: 'October', 11: 'November', 12: 'December'
                            }
                            
                            seasonal_range = monthly_means[max_month] - monthly_means[min_month]
                            mean_val = np.mean(list(monthly_means.values()))
                            seasonal_variability = seasonal_range / mean_val * 100 if mean_val > 0 else 0
                            
                            f.write(f"The {product_info.get('description')} shows strong seasonal variability ")
                            f.write(f"with peak values in {month_names.get(max_month, max_month)} ")
                            f.write(f"and minimum values in {month_names.get(min_month, min_month)}. ")
                            f.write(f"The seasonal range is {seasonal_range:.4f} {product_info.get('units', '')}, ")
                            f.write(f"representing {seasonal_variability:.1f}% of the annual mean.\n\n")
                
                # Spatial Distribution
                if 'spatial' in paths:
                    f.write("### Spatial Distribution\n\n")
                    f.write(f"![Spatial Distribution]({os.path.relpath(paths['spatial'], output_dir)})\n\n")
                    
                    # Add spatial statistics if available
                    if product_name in stats_data and 'spatial' in stats_data[product_name]:
                        spatial_stats = stats_data[product_name]['spatial']
                        f.write("**Spatial Statistics:**\n\n")
                        f.write(f"- Spatial Mean: {spatial_stats.get('spatial_mean', 'N/A'):.4f} {product_info.get('units', '')}\n")
                        f.write(f"- Spatial Standard Deviation: {spatial_stats.get('spatial_std', 'N/A'):.4f} {product_info.get('units', '')}\n")
                        f.write(f"- Data Coverage: {spatial_stats.get('coverage_percent', 'N/A'):.1f}% ({spatial_stats.get('valid_pixels', 0)} valid pixels)\n\n")
                
                # Animation (if included)
                if include_animations and 'animation' in paths:
                    f.write("### Temporal Animation\n\n")
                    f.write(f"![Animation]({os.path.relpath(paths['animation'], output_dir)})\n\n")
                    f.write("The animation shows the dynamic changes in spatial patterns over time.\n\n")
                
                # Add link to data export
                if 'stats' in paths:
                    f.write("### Data Export\n\n")
                    f.write(f"Complete time series data for {product_info.get('description')} is available ")
                    f.write(f"[here]({os.path.relpath(paths['stats'], output_dir)}).\n\n")
                
                # Add separator between products
                f.write("---\n\n")
            
            # Cross-Product Analysis
            if cross_product_analysis and len(data) > 1 and 'cross_product' in visualization_paths:
                f.write("<a id='cross-product-analysis'></a>\n")
                f.write("## Cross-Product Analysis\n\n")
                
                cross_paths = visualization_paths['cross_product']
                
                # Comparison plot
                if 'comparison' in cross_paths:
                    f.write("### Product Comparison\n\n")
                    f.write("The following chart shows normalized values (z-scores) of different MODIS products ")
                    f.write("to facilitate comparison of their temporal patterns:\n\n")
                    f.write(f"![Product Comparison]({os.path.relpath(cross_paths['comparison'], output_dir)})\n\n")
                    f.write("Normalization allows comparison of products with different units and magnitudes.\n\n")
                
                # Correlation analysis
                if 'correlation' in cross_paths:
                    f.write("### Product Correlations\n\n")
                    f.write("The following heatmap shows the Pearson correlation coefficients between different MODIS products:\n\n")
                    f.write(f"![Product Correlations]({os.path.relpath(cross_paths['correlation'], output_dir)})\n\n")
                    
                    # Add correlation interpretation
                    f.write("Strong positive correlations suggest that the products vary similarly over time, ")
                    f.write("while negative correlations indicate inverse relationships. ")
                    f.write("Products with correlation close to zero vary independently of each other.\n\n")
                    
                f.write("---\n\n")
            
            # Methodology Section
            f.write("<a id='methodology'></a>\n")
            f.write("## Methodology\n\n")
            f.write("This analysis follows these processing steps:\n\n")
            f.write("1. **Data Extraction**: Raw MODIS data is extracted from the HDF5 database\n")
            f.write("2. **Quality Control**: Invalid values are masked and scale factors applied\n")
            f.write("3. **Statistical Analysis**: Time series and spatial statistics are calculated\n")
            f.write("4. **Anomaly Detection**: Data is compared to baseline climatology\n")
            f.write("5. **Visualization**: Multiple visualization types are generated to highlight patterns\n")
            
            if cross_product_analysis and len(data) > 1:
                f.write("6. **Cross-product Analysis**: Products are compared and correlation analyzed\n")
                
            f.write("\n### Data Processing\n\n")
            f.write("For each MODIS product, the following processing is applied:\n\n")
            f.write("- Spatial subsetting to the region of interest\n")
            f.write("- Application of appropriate scaling factors\n")
            f.write("- Removal of invalid data points (marked as -999 or outside valid ranges)\n")
            f.write("- Calculation of spatial statistics over the region\n")
            f.write("- Temporal aggregation for annual and seasonal patterns\n")
            f.write("- Trend and anomaly analysis\n\n")
            
            # Data Sources Section
            f.write("<a id='data-sources'></a>\n")
            f.write("## Data Sources\n\n")
            f.write("This analysis uses the following MODIS (Moderate Resolution Imaging Spectroradiometer) products:\n\n")
            
            for product_name in data.keys():
                product_info = MODIS_PRODUCTS.get(product_name, {'description': product_name})
                f.write(f"- **{product_name}**: {product_info.get('description')}\n")
                
            f.write("\nMODIS data is collected by sensors aboard NASA's Terra and Aqua satellites. ")
            f.write("The original data products are provided by NASA's Earth Observing System ")
            f.write("Data and Information System (EOSDIS).\n\n")
            
            # Limitations Section
            f.write("<a id='limitations'></a>\n")
            f.write("## Limitations\n\n")
            f.write("This analysis has several limitations to consider:\n\n")
            f.write("- **Spatial Resolution**: The MODIS data used has a resolution of 250-500m, which may not capture fine-scale patterns\n")
            f.write("- **Cloud Interference**: Despite quality control, some cloud effects may remain in the data\n")
            f.write("- **Temporal Coverage**: The analysis is limited to the available time period in the database\n")
            f.write("- **Algorithmic Uncertainties**: Each MODIS product has inherent uncertainties in its retrieval algorithm\n")
            f.write("- **Regional Biases**: The accuracy of satellite retrievals can vary by region and land cover type\n\n")
            
            # Footer
            f.write("---\n\n")
            f.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M')}*\n")
            
        logger.info(f"Comprehensive report successfully generated at {report_path}")
        return report_path
        
    except Exception as e:
        logger.error(f"Error generating comprehensive MODIS report: {e}", exc_info=True)
        return ""

def analyze_modis_environmental_indicators(
    modis_data: Dict[str, np.ndarray],
    config: Dict[str, Any],
    output_dir: str = 'modis_indicators'
) -> Dict[str, Any]:
    """
    Analyze MODIS data to derive environmental indicators.
    
    Args:
        modis_data: Dictionary mapping product names to data arrays
        config: Configuration dictionary with parameters
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary of environmental indicators and metrics
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        indicators = {}
        vegetation_products = ['MOD13Q1_NDVI', 'MOD13Q1_EVI', 'MOD15A2H_Lai_500m']
        water_products = ['MOD16A2_ET']
        
        # Calculate vegetation health indicators if available
        veg_products = [p for p in vegetation_products if p in modis_data]
        if veg_products:
            indicators['vegetation'] = {}
            
            # Use NDVI or EVI for vegetation health
            if 'MOD13Q1_NDVI' in modis_data:
                product_name = 'MOD13Q1_NDVI'
                data = modis_data[product_name]
                
                # Get dates and calculate trend
                start_year = config.get('start_year', 2000)
                end_year = config.get('end_year', 2020)
                scale_factor = MODIS_PRODUCTS.get(product_name, {}).get('scale_factor', 0.0001)
                
                # Calculate vegetation health metrics
                spatial_mean = np.nanmean(data, axis=(1, 2)) * scale_factor
                
                # Calculate trend
                try:
                    from scipy import stats
                    x = np.arange(len(spatial_mean))
                    mask = ~np.isnan(spatial_mean)
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        x[mask], spatial_mean[mask]
                    )
                    
                    indicators['vegetation']['trend'] = {
                        'slope': float(slope),
                        'p_value': float(p_value),
                        'r_squared': float(r_value ** 2),
                        'direction': 'improving' if slope > 0 else 'degrading',
                        'significant': p_value < 0.05
                    }
                except Exception as e:
                    logger.warning(f"Could not calculate vegetation trend: {e}")
                
                # Calculate variability
                indicators['vegetation']['variability'] = {
                    'cv': float(np.nanstd(spatial_mean) / np.nanmean(spatial_mean) * 100) if np.nanmean(spatial_mean) > 0 else 0,
                    'range': float(np.nanmax(spatial_mean) - np.nanmin(spatial_mean)),
                    'std': float(np.nanstd(spatial_mean))
                }
                
                # Calculate average condition
                indicators['vegetation']['condition'] = {
                    'mean': float(np.nanmean(spatial_mean)),
                    'recent_mean': float(np.nanmean(spatial_mean[-min(12, len(spatial_mean)):])),
                    'min': float(np.nanmin(spatial_mean)),
                    'max': float(np.nanmax(spatial_mean))
                }
                
                # Generate visualization
                plt.figure(figsize=(10, 6))
                plt.plot(spatial_mean, marker='o', markersize=4)
                plt.axhline(np.nanmean(spatial_mean), color='r', linestyle='--', label='Mean')
                plt.title('Vegetation Health Index')
                plt.ylabel('NDVI')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                veg_plot_path = os.path.join(output_dir, "vegetation_health.png")
                plt.savefig(veg_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                indicators['vegetation']['plot_path'] = veg_plot_path
        
        # Calculate evapotranspiration and water indicators if available
        water_available = [p for p in water_products if p in modis_data]
        if water_available:
            indicators['water'] = {}
            
            if 'MOD16A2_ET' in modis_data:
                product_name = 'MOD16A2_ET'
                data = modis_data[product_name]
                scale_factor = MODIS_PRODUCTS.get(product_name, {}).get('scale_factor', 0.1)
                
                # Calculate ET metrics
                spatial_mean = np.nanmean(data, axis=(1, 2)) * scale_factor
                
                # Calculate water use intensity
                indicators['water']['annual_et'] = float(np.nanmean(spatial_mean) * 365 / 8)  # Convert 8-day to annual
                indicators['water']['variability'] = {
                    'cv': float(np.nanstd(spatial_mean) / np.nanmean(spatial_mean) * 100) if np.nanmean(spatial_mean) > 0 else 0,
                    'seasonal_range': float(np.nanmax(spatial_mean) - np.nanmin(spatial_mean))
                }
                
                # Calculate trend
                try:
                    from scipy import stats
                    x = np.arange(len(spatial_mean))
                    mask = ~np.isnan(spatial_mean)
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        x[mask], spatial_mean[mask]
                    )
                    
                    indicators['water']['trend'] = {
                        'slope': float(slope),
                        'p_value': float(p_value),
                        'r_squared': float(r_value ** 2),
                        'direction': 'increasing' if slope > 0 else 'decreasing',
                        'significant': p_value < 0.05
                    }
                except Exception as e:
                    logger.warning(f"Could not calculate water trend: {e}")
                
                # Generate visualization
                plt.figure(figsize=(10, 6))
                plt.plot(spatial_mean, marker='o', markersize=4, color='blue')
                plt.axhline(np.nanmean(spatial_mean), color='r', linestyle='--', label='Mean')
                plt.title('Evapotranspiration')
                plt.ylabel('ET (mm/8-day)')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                
                water_plot_path = os.path.join(output_dir, "evapotranspiration.png")
                plt.savefig(water_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                indicators['water']['plot_path'] = water_plot_path
        
        # Generate environmental indicator report
        report_path = os.path.join(output_dir, "environmental_indicators.md")
        with open(report_path, 'w') as f:
            f.write("# Environmental Indicators from MODIS Data\n\n")
            
            if 'vegetation' in indicators:
                f.write("## Vegetation Health\n\n")
                f.write(f"![Vegetation Health]({os.path.basename(indicators['vegetation'].get('plot_path', ''))})\n\n")
                
                f.write("### Vegetation Metrics\n\n")
                f.write("| Indicator | Value | Interpretation |\n")
                f.write("|-----------|-------|----------------|\n")
                
                cond = indicators['vegetation'].get('condition', {})
                if cond:
                    mean_ndvi = cond.get('mean', 0)
                    condition_text = (
                        "Excellent (dense vegetation)" if mean_ndvi > 0.7 else
                        "Good (healthy vegetation)" if mean_ndvi > 0.5 else
                        "Moderate (mixed/sparse vegetation)" if mean_ndvi > 0.3 else
                        "Poor (sparse vegetation or stressed)" if mean_ndvi > 0.2 else
                        "Very poor (bare soil or extreme stress)"
                    )
                    f.write(f"| Mean NDVI | {mean_ndvi:.4f} | {condition_text} |\n")
                    
                    recent_mean = cond.get('recent_mean', 0)
                    change = recent_mean - mean_ndvi
                    change_pct = (change / mean_ndvi * 100) if mean_ndvi > 0 else 0
                    recent_text = (
                        f"Improving (+{change_pct:.1f}% above average)" if change > 0.02 else
                        f"Declining ({change_pct:.1f}% below average)" if change < -0.02 else
                        "Stable (near average)"
                    )
                    f.write(f"| Recent Condition | {recent_mean:.4f} | {recent_text} |\n")
                
                var = indicators['vegetation'].get('variability', {})
                if var:
                    cv = var.get('cv', 0)
                    stability_text = (
                        "Very stable" if cv < 10 else
                        "Moderately stable" if cv < 20 else
                        "Variable" if cv < 30 else
                        "Highly variable"
                    )
                    f.write(f"| Variability (CV) | {cv:.1f}% | {stability_text} |\n")
                
                trend = indicators['vegetation'].get('trend', {})
                if trend:
                    slope = trend.get('slope', 0)
                    direction = trend.get('direction', '')
                    significance = trend.get('significant', False)
                    p_value = trend.get('p_value', 1.0)
                    r_squared = trend.get('r_squared', 0)
                    
                    trend_text = (
                        f"Significant {direction} trend (p={p_value:.4f})" if significance else
                        f"Non-significant {direction} trend (p={p_value:.4f})"
                    )
                    f.write(f"| Trend | {slope:.6f}/period | {trend_text} |\n")
                    f.write(f"| Trend Strength | R²={r_squared:.4f} | " + 
                           ("Strong" if r_squared > 0.6 else 
                            "Moderate" if r_squared > 0.3 else 
                            "Weak") + " relationship |\n")
                
                f.write("\n")
                
            if 'water' in indicators:
                f.write("## Water Use Intensity\n\n")
                f.write(f"![Evapotranspiration]({os.path.basename(indicators['water'].get('plot_path', ''))})\n\n")
                
                f.write("### Water Metrics\n\n")
                f.write("| Indicator | Value | Interpretation |\n")
                f.write("|-----------|-------|----------------|\n")
                
                annual_et = indicators['water'].get('annual_et', 0)
                et_category = (
                    "Very high" if annual_et > 1000 else
                    "High" if annual_et > 750 else
                    "Moderate" if annual_et > 500 else
                    "Low" if annual_et > 250 else
                    "Very low"
                )
                f.write(f"| Annual ET | {annual_et:.1f} mm | {et_category} water use |\n")
                
                var = indicators['water'].get('variability', {})
                if var:
                    cv = var.get('cv', 0)
                    stability_text = (
                        "Very stable" if cv < 15 else
                        "Moderately stable" if cv < 30 else
                        "Variable" if cv < 45 else
                        "Highly variable"
                    )
                    f.write(f"| Variability (CV) | {cv:.1f}% | {stability_text} |\n")
                
                trend = indicators['water'].get('trend', {})
                if trend:
                    slope = trend.get('slope', 0)
                    direction = trend.get('direction', '')
                    significance = trend.get('significant', False)
                    p_value = trend.get('p_value', 1.0)
                    
                    if direction == 'increasing' and significance:
                        implication = "Increasing water demand or available moisture"
                    elif direction == 'decreasing' and significance:
                        implication = "Decreasing water demand or potential drought stress"
                    else:
                        implication = "No significant change in water use patterns"
                        
                    trend_text = f"Significant {direction} trend (p={p_value:.4f})" if significance else f"Non-significant {direction} trend"
                    f.write(f"| Trend | {slope:.4f}/period | {trend_text} |\n")
                    f.write(f"| Implication | | {implication} |\n")
                
                f.write("\n")
            
            # Environmental health assessment    
            f.write("## Overall Environmental Health Assessment\n\n")
            
            # Combined assessment based on available indicators
            if 'vegetation' in indicators and 'water' in indicators:
                veg_trend = indicators['vegetation'].get('trend', {}).get('direction', 'neutral')
                veg_trend_sig = indicators['vegetation'].get('trend', {}).get('significant', False)
                water_trend = indicators['water'].get('trend', {}).get('direction', 'neutral')
                water_trend_sig = indicators['water'].get('trend', {}).get('significant', False)
                
                if veg_trend == 'improving' and veg_trend_sig:
                    if water_trend == 'increasing' and water_trend_sig:
                        assessment = "Ecosystem shows improving vegetation health with increasing water use, suggesting favorable growing conditions but potentially higher resource demand."
                    elif water_trend == 'decreasing' and water_trend_sig:
                        assessment = "Ecosystem shows improving vegetation health with decreasing water use, suggesting improved water use efficiency and positive ecosystem development."
                    else:
                        assessment = "Ecosystem shows improving vegetation health with stable water use patterns."
                elif veg_trend == 'degrading' and veg_trend_sig:
                    if water_trend == 'increasing' and water_trend_sig:
                        assessment = "Ecosystem shows declining vegetation health despite increasing water use, suggesting possible disturbance, disease, or inefficient water utilization."
                    elif water_trend == 'decreasing' and water_trend_sig:
                        assessment = "Ecosystem shows declining vegetation health with decreasing water use, suggesting possible drought stress or water limitation."
                    else:
                        assessment = "Ecosystem shows declining vegetation health with stable water use patterns, suggesting non-water factors affecting plant health."
                else:
                    assessment = "Ecosystem shows relatively stable vegetation conditions."
                    
                f.write(f"{assessment}\n\n")
            elif 'vegetation' in indicators:
                veg_condition = indicators['vegetation'].get('condition', {}).get('mean', 0)
                veg_trend = indicators['vegetation'].get('trend', {}).get('direction', 'neutral')
                veg_trend_sig = indicators['vegetation'].get('trend', {}).get('significant', False)
                
                if veg_condition > 0.5:
                    if veg_trend == 'improving' and veg_trend_sig:
                        assessment = "Ecosystem shows healthy vegetation with positive development trends."
                    elif veg_trend == 'degrading' and veg_trend_sig:
                        assessment = "Ecosystem shows currently healthy vegetation but with concerning degradation trends that may indicate emerging stress."
                    else:
                        assessment = "Ecosystem shows healthy and stable vegetation conditions."
                else:
                    if veg_trend == 'improving' and veg_trend_sig:
                        assessment = "Ecosystem shows moderate to poor vegetation health but with positive improvement trends."
                    elif veg_trend == 'degrading' and veg_trend_sig:
                        assessment = "Ecosystem shows poor vegetation health with continued degradation, suggesting chronic stress or disturbance."
                    else:
                        assessment = "Ecosystem shows moderate to poor vegetation health with stable conditions."
                        
                f.write(f"{assessment}\n\n")
            
            # Recommendations
            f.write("## Management Recommendations\n\n")
            recommendations = []
            
            if 'vegetation' in indicators:
                veg_trend = indicators['vegetation'].get('trend', {}).get('direction', 'neutral')
                veg_trend_sig = indicators['vegetation'].get('trend', {}).get('significant', False)
                veg_condition = indicators['vegetation'].get('condition', {}).get('mean', 0)
                
                if veg_trend == 'degrading' and veg_trend_sig:
                    recommendations.append("Investigate causes of vegetation decline through field assessment")
                    recommendations.append("Consider implementing stress mitigation practices appropriate for the ecosystem type")
                
                if veg_condition < 0.4:
                    recommendations.append("Evaluate potential for restoration or rehabilitation of vegetation cover")
                    recommendations.append("Assess if management practices need modification to improve vegetation health")
            
            if 'water' in indicators:
                water_trend = indicators['water'].get('trend', {}).get('direction', 'neutral')
                water_trend_sig = indicators['water'].get('trend', {}).get('significant', False)
                annual_et = indicators['water'].get('annual_et', 0)
                
                if water_trend == 'increasing' and water_trend_sig and annual_et > 800:
                    recommendations.append("Evaluate water use efficiency and consider water conservation practices")
                    recommendations.append("Monitor for potential water stress if increasing demand exceeds supply")
                
                if water_trend == 'decreasing' and water_trend_sig:
                    recommendations.append("Assess if decreased water use indicates drought stress requiring intervention")
                    recommendations.append("Consider drought-tolerant species or practices if trend continues")
            
            # Add general recommendations
            recommendations.append("Continue monitoring environmental indicators to track ecosystem responses")
            recommendations.append("Conduct field validation to confirm remote sensing observations")
            
            # Write recommendations
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")
                
            f.write("\n---\n\n")
            f.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d')}*\n")
            
        indicators['report_path'] = report_path
        return indicators
        
    except Exception as e:
        logger.error(f"Error analyzing environmental indicators: {e}", exc_info=True)
        return {}

def create_integrated_landcover_report(
    modis_data: Dict[str, np.ndarray],
    cdl_data: Dict[int, Dict[str, Any]],
    output_dir: str = 'integrated_report'
) -> str:
    """
    Create an integrated report combining MODIS and CDL data analysis.
    
    Args:
        modis_data: Dictionary mapping product names to MODIS data arrays
        cdl_data: Dictionary mapping years to CDL land use data
        output_dir: Directory to save outputs
        
    Returns:
        Path to the generated report file
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Define report path
        report_path = os.path.join(output_dir, "integrated_landcover_report.md")
        integrated_fig_path = os.path.join(output_dir, "integrated_analysis.png")
        
        # Check if we have both MODIS and CDL data
        if not modis_data or not cdl_data:
            logger.warning("Missing either MODIS or CDL data for integrated report")
            return ""
        
        # Get available years in both datasets
        cdl_years = sorted(cdl_data.keys())
        
        # Extract NDVI if available for correlation with land use
        ndvi_data = None
        ndvi_years = []
        ndvi_means = []
        
        if 'MOD13Q1_NDVI' in modis_data:
            ndvi_data = modis_data['MOD13Q1_NDVI']
            scale_factor = MODIS_PRODUCTS.get('MOD13Q1_NDVI', {}).get('scale_factor', 0.0001)
            
            # Calculate yearly NDVI averages to match with CDL data
            for year in cdl_years:
                # Get the dates for NDVI data
                all_dates = get_modis_dates('MOD13Q1_NDVI', year, year)
                
                # Find indices of dates in this year
                year_indices = [i for i, date in enumerate(all_dates) if date.year == year]
                
                if year_indices and year_indices[0] < ndvi_data.shape[0]:
                    # Extract data for this year and calculate mean
                    year_data = ndvi_data[year_indices, :, :]
                    year_mean = np.nanmean(year_data) * scale_factor
                    
                    ndvi_years.append(year)
                    ndvi_means.append(year_mean)
        
        # Create integrated visualization if we have matching years
        if ndvi_years and len(ndvi_years) > 0:
            # Calculate agricultural area percentage for each year
            ag_pcts = []
            for year in ndvi_years:
                if year in cdl_data:
                    year_data = cdl_data[year]
                    total_area = year_data.get("Total Area", 0)
                    
                    # Calculate agricultural area (excluding non-agricultural classes)
                    ag_area = sum(
                        year_data.get(crop, 0) for crop in year_data 
                        if crop not in ["Total Area", "unit", "Developed", "Water", 
                                        "Forest", "Wetlands", "Barren"]
                    )
                    
                    ag_pct = (ag_area / total_area * 100) if total_area > 0 else 0
                    ag_pcts.append(ag_pct)
                else:
                    ag_pcts.append(np.nan)
            
            # Create integrated visualization
            fig, ax1 = plt.subplots(figsize=(12, 7))
            
            # Plot NDVI trend
            color = 'tab:green'
            ax1.set_xlabel('Year')
            ax1.set_ylabel('NDVI', color=color)
            ax1.plot(ndvi_years, ndvi_means, marker='o', color=color, label='NDVI')
            ax1.tick_params(axis='y', labelcolor=color)
            
            # Add second y-axis for agricultural percentage
            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('Agricultural Area (%)', color=color)
            ax2.plot(ndvi_years, ag_pcts, marker='s', color=color, label='Agricultural %')
            ax2.tick_params(axis='y', labelcolor=color)
            
            # Add grid and title
            ax1.grid(True, alpha=0.3)
            plt.title('Integrated Analysis: Vegetation Health and Agricultural Land Use')
            
            # Add legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            plt.tight_layout()
            plt.savefig(integrated_fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Calculate correlation between NDVI and agricultural percentage
            try:
                from scipy import stats
                # Filter out NaN values
                valid_indices = [i for i, (n, a) in enumerate(zip(ndvi_means, ag_pcts)) 
                               if not (np.isnan(n) or np.isnan(a))]
                
                if len(valid_indices) >= 3:  # Need at least 3 points for meaningful correlation
                    valid_ndvi = [ndvi_means[i] for i in valid_indices]
                    valid_ag = [ag_pcts[i] for i in valid_indices]
                    
                    r, p_value = stats.pearsonr(valid_ndvi, valid_ag)
                    correlation = {
                        'r': float(r),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05,
                        'direction': 'positive' if r > 0 else 'negative'
                    }
                else:
                    correlation = None
            except Exception as e:
                logger.warning(f"Could not calculate correlation: {e}")
                correlation = None
        
        # Generate the report
        with open(report_path, 'w') as f:
            f.write("# Integrated Land Cover and Vegetation Analysis\n\n")
            
            # Overview section
            f.write("## Overview\n\n")
            f.write("This report integrates analysis from multiple remote sensing sources:\n\n")
            f.write("1. **MODIS Vegetation Indices** - Providing continuous vegetation health monitoring\n")
            f.write("2. **Cropland Data Layer (CDL)** - Providing detailed land use and crop type information\n\n")
            
            # Combined temporal trends
            f.write("## Integrated Temporal Analysis\n\n")
            if ndvi_years and len(ndvi_years) > 0:
                f.write(f"![Integrated Analysis]({os.path.basename(integrated_fig_path)})\n\n")
                
                f.write("The graph above shows the relationship between vegetation health (NDVI) ")
                f.write("and agricultural land use percentage over time.\n\n")
                
                # Correlation interpretation
                if correlation:
                    f.write("### Statistical Relationship\n\n")
                    f.write(f"Correlation coefficient (r): {correlation['r']:.4f} ({correlation['direction']})\n\n")
                    
                    if correlation['significant']:
                        f.write(f"**Significant relationship** (p-value: {correlation['p_value']:.4f})\n\n")
                        
                        if correlation['r'] > 0.7:
                            f.write("There is a strong positive relationship between vegetation health and agricultural area, ")
                            f.write("suggesting that agricultural vegetation is contributing significantly to the overall greenness ")
                            f.write("of the landscape.\n\n")
                        elif correlation['r'] > 0:
                            f.write("There is a moderate positive relationship between vegetation health and agricultural area, ")
                            f.write("suggesting that agricultural practices are generally maintaining good vegetation health.\n\n")
                        elif correlation['r'] > -0.7:
                            f.write("There is a moderate negative relationship between vegetation health and agricultural area, ")
                            f.write("suggesting that increases in agricultural area may be associated with some reduction ")
                            f.write("in overall vegetation health, possibly due to conversion of natural vegetation.\n\n")
                        else:
                            f.write("There is a strong negative relationship between vegetation health and agricultural area, ")
                            f.write("suggesting that agricultural expansion may be associated with significant reduction ")
                            f.write("in vegetation health, possibly due to conversion of dense natural vegetation or ")
                            f.write("intensive practices with periods of bare soil.\n\n")
                    else:
                        f.write(f"No significant relationship detected (p-value: {correlation['p_value']:.4f})\n\n")
                        f.write("The lack of significant correlation suggests that changes in agricultural area ")
                        f.write("and vegetation health are likely influenced by different factors or occur at different scales.\n\n")
            else:
                f.write("Insufficient matching data to perform integrated temporal analysis.\n\n")
            
            # Land cover and management implications
            f.write("## Land Cover Management Implications\n\n")
            
            # Get CDL data for latest year
            if cdl_years:
                latest_year = max(cdl_years)
                latest_data = cdl_data[latest_year]
                
                # Extract top crops
                crop_items = [(k, v) for k, v in latest_data.items() 
                             if k not in ["Total Area", "unit"] and not k.endswith("(%)")]
                top_crops = sorted(crop_items, key=lambda x: x[1], reverse=True)[:5]
                
                f.write(f"### Current Land Cover ({latest_year})\n\n")
                f.write("| Rank | Crop/Land Cover | Area (ha) | Percentage |\n")
                f.write("|------|----------------|-----------|------------|\n")
                
                total_area = latest_data.get("Total Area", 0)
                for i, (crop, area) in enumerate(top_crops, 1):
                    pct = (area / total_area * 100) if total_area > 0 else 0
                    f.write(f"| {i} | {crop} | {area:,.2f} | {pct:.2f}% |\n")
                
                f.write("\n")
            
            # Management recommendations
            f.write("### Management Recommendations\n\n")
            
            if ndvi_data is not None and ndvi_years and len(ndvi_years) >= 2:
                # Check for NDVI trend
                first_ndvi = ndvi_means[0]
                last_ndvi = ndvi_means[-1]
                ndvi_change = last_ndvi - first_ndvi
                ndvi_trend = "increasing" if ndvi_change > 0.01 else "decreasing" if ndvi_change < -0.01 else "stable"
                
                # Get agricultural land change if available
                ag_change_pct = None
                if len(ag_pcts) >= 2 and not np.isnan(ag_pcts[0]) and not np.isnan(ag_pcts[-1]):
                    ag_change_pct = ag_pcts[-1] - ag_pcts[0]
                
                # Generate recommendations based on trends
                recommendations = []
                
                if ndvi_trend == "increasing":
                    recommendations.append("Continue current vegetation management practices that support positive trends in vegetation health.")
                    if ag_change_pct is not None and ag_change_pct > 5:
                        recommendations.append("Agricultural expansion appears to be maintaining or improving vegetation health. Consider documenting sustainable practices for broader application.")
                elif ndvi_trend == "decreasing":
                    recommendations.append("Investigate causes of declining vegetation health through field assessment.")
                    if ag_change_pct is not None and ag_change_pct > 5:
                        recommendations.append("Consider whether agricultural expansion is contributing to vegetation stress and evaluate more sustainable practices.")
                    recommendations.append("Evaluate potential adoption of conservation practices such as cover crops or reduced tillage.")
                else:
                    recommendations.append("Vegetation health appears stable. Continue monitoring for any emerging trends.")
                
                # Add general recommendations
                recommendations.append("Perform periodic ground truthing to validate remote sensing observations.")
                recommendations.append("Consider seasonal timing of agricultural activities to optimize vegetation growth periods.")
                
                # Write recommendations
                for rec in recommendations:
                    f.write(f"- {rec}\n")
            else:
                f.write("Insufficient data to generate specific management recommendations based on trends.\n")
                f.write("- Consider field-based assessments to validate remote sensing observations\n")
                f.write("- Implement best management practices appropriate for the dominant land cover types\n")
                f.write("- Monitor vegetation health and agricultural productivity regularly\n")
            
            # Data integration importance
            f.write("\n## Data Integration Value\n\n")
            f.write("The integration of MODIS vegetation indices with CDL land cover data provides several benefits:\n\n")
            f.write("1. **Contextual understanding**: Vegetation health metrics can be interpreted in the context of specific land use types\n")
            f.write("2. **Change attribution**: Changes in vegetation indices can be attributed to specific land cover changes\n")
            f.write("3. **Management precision**: More targeted management recommendations based on both vegetation condition and land use\n")
            f.write("4. **Validation**: Multiple data sources provide cross-validation of observed patterns\n\n")
            
            # Limitations section
            f.write("## Limitations\n\n")
            f.write("This integrated analysis has the following limitations:\n\n")
            f.write("- **Scale differences**: MODIS (250-500m) and CDL (30m) have different spatial resolutions\n")
            f.write("- **Temporal alignment**: MODIS provides higher temporal frequency than the annual CDL data\n")
            f.write("- **Causality**: Correlation between vegetation health and land cover does not imply causation\n")
            f.write("- **Mixed pixels**: Remote sensing data may include mixed land cover types within pixels\n\n")
            
            # Report footer
            f.write("---\n\n")
            f.write(f"*Integrated report generated on {datetime.now().strftime('%Y-%m-%d')}*\n")
        
        logger.info(f"Integrated landcover report generated at {report_path}")
        return report_path
        
    except Exception as e:
        logger.error(f"Error creating integrated report: {e}", exc_info=True)
        return ""

def create_modis_climate_comparison(
    modis_data: Dict[str, np.ndarray],
    climate_data: pd.DataFrame,
    output_dir: str = 'climate_comparison'
) -> str:
    """
    Create an analysis comparing MODIS data with climate variables.
    
    Args:
        modis_data: Dictionary mapping product names to MODIS data arrays
        climate_data: DataFrame with climate variables (must have 'date' column)
        output_dir: Directory to save outputs
        
    Returns:
        Path to the generated report file
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Define report path
        report_path = os.path.join(output_dir, "climate_vegetation_report.md")
        
        # Check if we have both MODIS and climate data
        if not modis_data or climate_data.empty:
            logger.warning("Missing either MODIS or climate data for comparison")
            return ""
        
        # Process climate data
        if 'date' not in climate_data.columns:
            logger.error("Climate data missing required 'date' column")
            return ""
            
        # Ensure date column is datetime type
        climate_data['date'] = pd.to_datetime(climate_data['date'])
        
        # Find available climate variables (exclude date column)
        climate_vars = [col for col in climate_data.columns if col != 'date']
        
        if not climate_vars:
            logger.error("No climate variables found in data")
            return ""
            
        # Extract NDVI or EVI if available
        veg_data = None
        veg_product = None
        
        # Look for vegetation indices in priority order
        for product in ['MOD13Q1_NDVI', 'MOD13Q1_EVI', 'MOD15A2H_Lai_500m']:
            if product in modis_data:
                veg_data = modis_data[product]
                veg_product = product
                break
                
        if veg_data is None:
            logger.warning("No vegetation index found in MODIS data")
            return ""
            
        # Get product info
        product_info = MODIS_PRODUCTS.get(veg_product, {'description': veg_product, 'scale_factor': 1.0})
        scale_factor = product_info.get('scale_factor', 1.0)
        
        # Get the years range based on MODIS data
        years = set()
        for product, data_array in modis_data.items():
            config = product_info.get('config', {})
            start_year = config.get('start_year', 2000)
            end_year = config.get('end_year', 2020)
            years.update(range(start_year, end_year + 1))
            
        start_year, end_year = min(years), max(years)
        
        # Generate MODIS dates
        veg_dates = get_modis_dates(veg_product, start_year, end_year)
        veg_dates = veg_dates[:veg_data.shape[0]]
        
        # Calculate vegetation index spatial mean
        veg_means = np.nanmean(veg_data, axis=(1, 2)) * scale_factor
        
        # Create a DataFrame for vegetation data
        veg_df = pd.DataFrame({
            'date': veg_dates[:len(veg_means)],
            'value': veg_means
        })
        
        # Create a figure for each climate variable correlation
        correlation_plots = {}
        correlation_stats = {}
        
        from scipy import stats
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        for climate_var in climate_vars:
            # Ensure the climate variable is numeric
            if not np.issubdtype(climate_data[climate_var].dtype, np.number):
                continue
                
            # Merge vegetation and climate data on nearest date
            merged_data = pd.merge_asof(
                veg_df.sort_values('date'),
                climate_data[['date', climate_var]].sort_values('date'),
                on='date',
                direction='nearest',
                tolerance=pd.Timedelta('15 days')
            )
            
            # Drop rows with missing values
            merged_data = merged_data.dropna()
            
            if len(merged_data) < 5:
                logger.warning(f"Insufficient matching data points for {climate_var}")
                continue
                
            # Calculate correlation
            r, p_value = stats.pearsonr(merged_data['value'], merged_data[climate_var])
            correlation_stats[climate_var] = {
                'r': r,
                'p_value': p_value,
                'n': len(merged_data),
                'significant': p_value < 0.05
            }
            
            # Create scatter plot with regression line
            plot_path = os.path.join(output_dir, f"{climate_var}_correlation.png")
            
            plt.figure(figsize=(10, 6))
            sns.regplot(
                x=climate_var,
                y='value',
                data=merged_data,
                scatter_kws={'alpha': 0.6},
                line_kws={'color': 'red'}
            )
            
            plt.title(f"Relationship between {product_info.get('description')} and {climate_var}")
            plt.xlabel(climate_var)
            plt.ylabel(product_info.get('description'))
            
            # Add correlation info
            plt.annotate(
                f"r = {r:.3f}, p = {p_value:.4f}\nn = {len(merged_data)}",
                xy=(0.05, 0.95),
                xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8)
            )
            
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            correlation_plots[climate_var] = plot_path
        
        # If there are no correlations to report, exit
        if not correlation_plots:
            logger.warning("No valid correlations could be calculated")
            return ""
        
        # Generate the report
        with open(report_path, 'w') as f:
            f.write("# Climate-Vegetation Relationship Analysis\n\n")
            
            # Basic information
            f.write("## Overview\n\n")
            f.write("This report analyzes the relationship between vegetation health metrics from MODIS ")
            f.write(f"and climate variables for the period {start_year}-{end_year}.\n\n")
            
            # Summary of correlations
            f.write("## Climate-Vegetation Correlations\n\n")
            f.write("| Climate Variable | Correlation (r) | p-value | Significance | Relationship |\n")
            f.write("|-----------------|----------------|---------|--------------|-------------|\n")
            
            for var, stats in correlation_stats.items():
                r = stats['r']
                p = stats['p_value']
                sig = "Significant" if stats['significant'] else "Not significant"
                
                relationship = (
                    "Strong positive" if r > 0.7 else
                    "Moderate positive" if r > 0.3 else
                    "Weak positive" if r > 0 else
                    "Strong negative" if r < -0.7 else
                    "Moderate negative" if r < -0.3 else
                    "Weak negative"
                )
                
                f.write(f"| {var} | {r:.3f} | {p:.4f} | {sig} | {relationship} |\n")
                
            f.write("\n")
            
            # Individual correlation plots and interpretation
            for var, plot_path in correlation_plots.items():
                f.write(f"## {var} Relationship\n\n")
                f.write(f"![{var} Correlation]({os.path.basename(plot_path)})\n\n")
                
                # Add interpretation based on correlation statistics
                stats = correlation_stats[var]
                r = stats['r']
                p = stats['p_value']
                
                if stats['significant']:
                    f.write(f"There is a statistically significant relationship (p={p:.4f}) between ")
                    f.write(f"{product_info.get('description')} and {var}. ")
                    
                    if r > 0:
                        f.write(f"The positive correlation (r={r:.3f}) indicates that vegetation health tends to ")
                        f.write(f"increase with higher {var}. ")
                        
                        if r > 0.7:
                            f.write("This strong relationship suggests that this climate variable may be a ")
                            f.write("key driver of vegetation dynamics in the region.\n\n")
                        else:
                            f.write("This relationship explains part of the observed vegetation variability, ")
                            f.write("though other factors also play important roles.\n\n")
                    else:
                        f.write(f"The negative correlation (r={r:.3f}) indicates that vegetation health tends to ")
                        f.write(f"decrease with higher {var}. ")
                        
                        if r < -0.7:
                            f.write("This strong inverse relationship suggests that this climate variable may ")
                            f.write("contribute to vegetation stress in the region.\n\n")
                        else:
                            f.write("This relationship explains part of the observed vegetation variability, ")
                            f.write("though other factors also play important roles.\n\n")
                else:
                    f.write(f"No statistically significant relationship (p={p:.4f}) was found between ")
                    f.write(f"{product_info.get('description')} and {var}. This suggests that other factors ")
                    f.write("may be more important in determining vegetation health in this region.\n\n")
            
            # Final interpretations and limitations
            f.write("## Summary and Implications\n\n")
            
            # Count significant correlations
            sig_count = sum(1 for stats in correlation_stats.values() if stats['significant'])
            
            if sig_count > 0:
                f.write(f"Of the {len(correlation_stats)} climate variables analyzed, ")
                f.write(f"{sig_count} showed significant relationships with vegetation health. ")
                
                # Find the strongest correlation
                strongest_var = max(correlation_stats.items(), 
                                   key=lambda x: abs(x[1]['r']) if x[1]['significant'] else 0)
                
                if strongest_var[1]['significant']:
                    var_name = strongest_var[0]
                    r_value = strongest_var[1]['r']
                    f.write(f"The strongest relationship was with {var_name} (r={r_value:.3f}), ")
                    f.write("suggesting this may be a particularly important climate driver for vegetation in this region.\n\n")
            else:
                f.write("None of the analyzed climate variables showed significant relationships with vegetation health. ")
                f.write("This could indicate that vegetation in this region is resilient to climate variations, ")
                f.write("or that other factors not included in this analysis are more important drivers.\n\n")
            
            # Limitations
            f.write("### Limitations\n\n")
            f.write("- Correlation does not necessarily imply causation\n")
            f.write("- Time lags between climate events and vegetation response may not be captured in this analysis\n")
            f.write("- Local microclimates may differ from the climate data used\n")
            f.write("- Complex interactions between multiple climate variables are not accounted for\n\n")
            
            # Report footer
            f.write("---\n\n")
            f.write(f"*Climate-Vegetation analysis generated on {datetime.now().strftime('%Y-%m-%d')}*\n")
        
        logger.info(f"Climate-vegetation report generated at {report_path}")
        return report_path
        
    except Exception as e:
        logger.error(f"Error creating climate comparison report: {e}", exc_info=True)
        return ""




if __name__ == "__main__":
    ### generate report
    config = { 
        "RESOLUTION": 250,
        "start_year": 2000,
        "end_year": 2003,
        'bounding_box': [-85.444332, 43.658148, -85.239256, 44.164683],
    }


