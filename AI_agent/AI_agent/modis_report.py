import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime
import matplotlib.pyplot as plt
import logging
from pathlib import Path

try:
    from AI_agent.config import AgentConfig
    from AI_agent.modis_utilities import (
        get_modis_dates, plot_modis_timeseries, create_modis_spatial_plot, 
        create_modis_seasonal_plot, MODIS_PRODUCTS
    )
except ImportError:
    from config import AgentConfig
    from modis_utilities import (
        get_modis_dates, plot_modis_timeseries, create_modis_spatial_plot, 
        create_modis_seasonal_plot, MODIS_PRODUCTS
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

if __name__ == "__main__":
    # Example usage
    from modis_utilities import extract_modis_data
    
    config = {
        "bounding_box": [-85.444332, 43.158148, -84.239256, 44.164683],
        "start_year": 2010,
        "end_year": 2016,
        "product": "MOD13Q1_NDVI"
    }
    
    # Extract data
    data = extract_modis_data(
        database_path=AgentConfig.HydroGeoDataset_ML_250_path,
        h5_group_name=f"MODIS/{config['product']}",
        start_year=config['start_year'],
        end_year=config['end_year'],
        bounding_box=config['bounding_box']
    )
    
    if data.size > 0:
        # Generate report
        report_path = generate_modis_report(
            data=data,
            product_name=config['product'],
            start_year=config['start_year'],
            end_year=config['end_year'],
            bounding_box=config['bounding_box'],
            output_dir='modis_results'
        )
        
        if report_path:
            print(f"Report generated successfully: {report_path}")
    else:
        print("No data was extracted")
