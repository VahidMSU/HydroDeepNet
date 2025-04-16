"""
PRISM climate data analysis and report generation.

This module provides functionality to generate comprehensive reports analyzing
PRISM (Parameter-elevation Regressions on Independent Slopes Model) climate data.
"""
import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import calendar

try:
    from config import AgentConfig
    from prism_utilities import (
        extract_prism_data, get_prism_spatial_means, create_period_labels,
        plot_climate_timeseries, create_climate_spatial_plot, create_climate_seasonal_plot,
        export_climate_data_to_csv, calculate_climate_trends, PRISM_VARIABLES
    )
except ImportError:
    from GeoReporter.config import AgentConfig
    from .prism_utilities import (
        extract_prism_data, get_prism_spatial_means, create_period_labels,
        plot_climate_timeseries, create_climate_spatial_plot, create_climate_seasonal_plot,
        export_climate_data_to_csv, calculate_climate_trends, PRISM_VARIABLES
    )

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def generate_prism_report(data: Dict[str, np.ndarray], start_year: int, end_year: int,
                         bounding_box: Optional[Tuple[float, float, float, float]] = None,
                         aggregation: str = 'annual', output_dir: str = 'prism_report') -> str:
    """
    Generate a comprehensive PRISM climate data analysis report with visualizations.
    
    Args:
        data: Dictionary containing arrays for each climate variable
        start_year: Starting year of the data
        end_year: Ending year of the data
        bounding_box: Optional [min_lon, min_lat, max_lon, max_lat] of the region
        aggregation: Temporal aggregation ('daily', 'monthly', 'seasonal', 'annual')
        output_dir: Directory to save report files
        
    Returns:
        Path to the generated report file
    """
    if not data:
        logger.warning("No data to generate report")
        return ""
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Define file paths
        timeseries_path = os.path.join(output_dir, "prism_timeseries.png")
        spatial_path = os.path.join(output_dir, "prism_spatial.png")
        seasonal_path = os.path.join(output_dir, "prism_seasonal.png")
        report_path = os.path.join(output_dir, "prism_report.md")
        stats_path = os.path.join(output_dir, "prism_stats.csv")
        
        # Generate visualizations
        plot_climate_timeseries(
            data=get_prism_spatial_means(data),
            start_year=start_year,
            end_year=end_year,
            aggregation=aggregation,
            output_path=timeseries_path,
            title=f"PRISM Climate Data ({start_year}-{end_year})"
        )
        
        # If we have at least one data point, create a spatial visualization of the first one
        for var_name, var_data in data.items():
            if var_data.size > 0:
                create_climate_spatial_plot(
                    data=data,
                    time_index=0,
                    output_path=spatial_path,
                    title=f"PRISM Spatial Distribution ({start_year}-{end_year})"
                )
                break
            
        create_climate_seasonal_plot(
            data=data,
            start_year=start_year,
            end_year=end_year,
            output_path=seasonal_path
        )
        
        # Calculate statistics for each variable
        spatial_means = get_prism_spatial_means(data)
        
        # Prepare statistics table
        period_labels = create_period_labels(start_year, end_year, aggregation)
        
        # Create DataFrame with stats
        stats_df = pd.DataFrame({'Period': period_labels})
        
        # Add climate variables to dataframe
        for var_name, var_data in spatial_means.items():
            if var_name in PRISM_VARIABLES and len(var_data) > 0:
                # Trim to match period labels length
                stats_df[var_name] = var_data[:len(period_labels)]
        
        # Add date parsing columns based on aggregation
        if aggregation == 'monthly':
            stats_df['Year'] = [int(p.split('-')[0]) for p in period_labels]
            stats_df['Month'] = [int(p.split('-')[1]) for p in period_labels]
        elif aggregation == 'seasonal':
            stats_df['Year'] = [int(p.split('-')[0]) for p in period_labels]
            stats_df['Season'] = [p.split('-')[1] for p in period_labels]
        elif aggregation == 'annual':
            stats_df['Year'] = [int(p) for p in period_labels]
        
        # Save statistics to CSV
        stats_df.to_csv(stats_path, index=False)
        
        # Calculate trends
        trends = calculate_climate_trends(data, start_year, end_year)
        
        # Generate markdown report
        with open(report_path, 'w') as f:
            # Header
            f.write("# PRISM Climate Data Analysis Report\n\n")
            
            # Basic information
            f.write("## Overview\n\n")
            f.write(f"**Period:** {start_year} to {end_year}\n\n")
            f.write(f"**Temporal Resolution:** {aggregation}\n\n")
            
            if bounding_box:
                f.write(f"**Region:** Lat [{bounding_box[1]:.4f}, {bounding_box[3]:.4f}], ")
                f.write(f"Lon [{bounding_box[0]:.4f}, {bounding_box[2]:.4f}]\n\n")
            
            # Data availability
            f.write("**Available Variables:**\n\n")
            for var_name, var_data in data.items():
                if var_name in PRISM_VARIABLES and var_data.size > 0:
                    f.write(f"- {PRISM_VARIABLES[var_name]['description']} ({PRISM_VARIABLES[var_name]['units']})\n")
            f.write("\n")
            
            # Summary statistics for each variable
            f.write("## Summary Statistics\n\n")
            
            for var_name, var_data in spatial_means.items():
                if var_name in PRISM_VARIABLES and len(var_data) > 0:
                    var_info = PRISM_VARIABLES[var_name]
                    
                    f.write(f"### {var_info['description']}\n\n")
                    f.write(f"**Mean:** {np.nanmean(var_data):.2f} {var_info['units']}\n\n")
                    f.write(f"**Minimum:** {np.nanmin(var_data):.2f} {var_info['units']}\n\n")
                    f.write(f"**Maximum:** {np.nanmax(var_data):.2f} {var_info['units']}\n\n")
                    f.write(f"**Standard Deviation:** {np.nanstd(var_data):.2f} {var_info['units']}\n\n")
                    
                    # Add trend information if available
                    if var_name in trends:
                        trend = trends[var_name]
                        trend_dir = "increasing" if trend['slope'] > 0 else "decreasing"
                        significance = "significant" if trend['significant'] else "not statistically significant"
                        
                        f.write("**Trend Analysis:**\n\n")
                        f.write(f"- {trend_dir.capitalize()} at {abs(trend['slope']):.3f} {var_info['units']}/year ")
                        f.write(f"(p-value: {trend['p_value']:.4f}, {significance})\n")
                        f.write(f"- Total change: {abs(trend['total_change']):.2f} {var_info['units']} ")
                        
                        if trend['percent_change'] != float('inf'):
                            f.write(f"({abs(trend['percent_change']):.1f}%)\n")
                        f.write(f"- R-squared: {trend['r_squared']:.3f}\n\n")
            
            # Yearly averages if applicable
            if aggregation == 'annual' or aggregation == 'monthly':
                f.write("## Yearly Averages\n\n")
                
                # For temperature variables
                if 'tmean' in spatial_means:
                    yearly_temps = stats_df.groupby('Year')['tmean'].agg(['mean', 'min', 'max', 'std']).reset_index()
                    f.write("### Mean Temperature (°C)\n\n")
                    f.write("| Year | Mean | Min | Max | Std Dev |\n")
                    f.write("|------|------|-----|-----|---------|\n")
                    for _, row in yearly_temps.iterrows():
                        f.write(f"| {int(row['Year'])} | {row['mean']:.2f} | {row['min']:.2f} | {row['max']:.2f} | {row['std']:.2f} |\n")
                    f.write("\n")
                
                # For precipitation
                if 'ppt' in spatial_means:
                    yearly_precip = stats_df.groupby('Year')['ppt'].agg(['sum', 'mean', 'min', 'max', 'std']).reset_index()
                    f.write("### Precipitation (mm)\n\n")
                    f.write("| Year | Total | Mean | Min | Max | Std Dev |\n")
                    f.write("|------|-------|------|-----|-----|---------|\n")
                    for _, row in yearly_precip.iterrows():
                        f.write(f"| {int(row['Year'])} | {row['sum']:.1f} | {row['mean']:.1f} | {row['min']:.1f} | {row['max']:.1f} | {row['std']:.1f} |\n")
                    f.write("\n")
            
            # Seasonal patterns
            f.write("## Seasonal Patterns\n\n")
            f.write(f"The seasonal analysis shows how climate variables vary throughout the year.\n\n")
            f.write(f"![Seasonal Analysis]({os.path.basename(seasonal_path)})\n\n")
            
            # Add some interpretation of seasonal patterns for temperature
            if 'tmean' in spatial_means and aggregation == 'monthly':
                try:
                    monthly_data = stats_df.copy()
                    temp_by_month = monthly_data.groupby('Month')['tmean'].mean()
                    hottest_month = temp_by_month.idxmax()
                    coldest_month = temp_by_month.idxmin()
                    temp_range = temp_by_month.max() - temp_by_month.min()
                    
                    f.write("### Temperature Seasonality\n\n")
                    f.write(f"The highest temperatures typically occur in **{calendar.month_name[hottest_month]}**, ")
                    f.write(f"while the lowest temperatures are in **{calendar.month_name[coldest_month]}**. ")
                    f.write(f"The seasonal temperature range is approximately {temp_range:.1f}°C.\n\n")
                except Exception as e:
                    logger.warning(f"Could not generate temperature seasonality interpretation: {e}")
            
            # Add some interpretation of seasonal patterns for precipitation
            if 'ppt' in spatial_means and aggregation == 'monthly':
                try:
                    monthly_data = stats_df.copy()
                    precip_by_month = monthly_data.groupby('Month')['ppt'].mean()
                    wettest_month = precip_by_month.idxmax()
                    driest_month = precip_by_month.idxmin()
                    
                    wet_season = []
                    dry_season = []
                    
                    # Identify wet/dry seasons (months above/below average)
                    avg_precip = precip_by_month.mean()
                    for month, precip in precip_by_month.items():
                        if precip > avg_precip:
                            wet_season.append(calendar.month_name[month])
                        else:
                            dry_season.append(calendar.month_name[month])
                    
                    f.write("### Precipitation Seasonality\n\n")
                    f.write(f"The wettest month is typically **{calendar.month_name[wettest_month]}** ")
                    f.write(f"({precip_by_month[wettest_month]:.1f} mm), while the driest month ")
                    f.write(f"is **{calendar.month_name[driest_month]}** ({precip_by_month[driest_month]:.1f} mm).\n\n")
                    
                    if wet_season and len(wet_season) < 8:
                        f.write(f"The wet season generally includes: {', '.join(wet_season)}.\n\n")
                    
                except Exception as e:
                    logger.warning(f"Could not generate precipitation seasonality interpretation: {e}")
            
            # Time series analysis
            f.write("## Time Series Analysis\n\n")
            f.write(f"The time series shows the change in climate variables over the entire period.\n\n")
            f.write(f"![Time Series]({os.path.basename(timeseries_path)})\n\n")
            
            # Add trend analysis
            if trends:
                f.write("### Climate Trends\n\n")
                f.write("| Variable | Annual Change | Total Change | % Change | P-value | Significant? |\n")
                f.write("|----------|--------------|--------------|----------|---------|-------------|\n")
                
                for var_name, trend in trends.items():
                    if var_name in PRISM_VARIABLES:
                        var_info = PRISM_VARIABLES[var_name]
                        direction = "+" if trend['slope'] > 0 else ""
                        
                        pct_change = f"{abs(trend['percent_change']):.1f}%" if trend['percent_change'] != float('inf') else "N/A"
                        
                        f.write(f"| {var_info['description']} | {direction}{trend['slope']:.3f} {var_info['units']}/yr | ")
                        f.write(f"{direction}{abs(trend['total_change']):.2f} {var_info['units']} | ")
                        f.write(f"{pct_change} | {trend['p_value']:.4f} | {'Yes' if trend['significant'] else 'No'} |\n")
                f.write("\n")
            
            # Spatial distribution
            f.write("## Spatial Distribution\n\n")
            f.write(f"The spatial maps show the distribution of climate variables across the study area.\n\n")
            f.write(f"![Spatial Distribution]({os.path.basename(spatial_path)})\n\n")
            
            # Climate implications
            f.write("## Climate Implications\n\n")
            
            # Automatically determine if there are significant changes
            has_warming_trend = False
            has_cooling_trend = False
            has_precip_increase = False
            has_precip_decrease = False
            
            if trends:
                if 'tmean' in trends:
                    if trends['tmean']['significant']:
                        if trends['tmean']['slope'] > 0:
                            has_warming_trend = True
                        else:
                            has_cooling_trend = True
                
                if 'ppt' in trends:
                    if trends['ppt']['significant']:
                        if trends['ppt']['slope'] > 0:
                            has_precip_increase = True
                        else:
                            has_precip_decrease = True
            
            # Generate climate implications based on trends
            if has_warming_trend:
                f.write("The data shows a **significant warming trend** over the analyzed period. ")
                f.write("This warming may impact:\n\n")
                f.write("- Growing season length and crop selection options\n")
                f.write("- Evapotranspiration rates and irrigation requirements\n")
                f.write("- Heat stress on crops during critical growth stages\n")
                f.write("- Pest and disease prevalence and distribution\n\n")
            
            elif has_cooling_trend:
                f.write("The data shows a **significant cooling trend** over the analyzed period. ")
                f.write("This cooling may impact:\n\n")
                f.write("- Growing season length and frost risk\n")
                f.write("- Crop selection limitations\n")
                f.write("- Reduced evapotranspiration and potential water stress\n\n")
            
            if has_precip_increase:
                f.write("The analysis indicates a **significant increase in precipitation**. ")
                f.write("Increased precipitation may lead to:\n\n")
                f.write("- Higher soil moisture availability\n")
                f.write("- Potential increases in flooding and erosion risk\n")
                f.write("- Changes in nutrient leaching and water quality\n")
                f.write("- Potential delays in field operations during wet periods\n\n")
            
            elif has_precip_decrease:
                f.write("The analysis indicates a **significant decrease in precipitation**. ")
                f.write("Decreased precipitation may lead to:\n\n")
                f.write("- Increased drought risk and water stress\n")
                f.write("- Higher irrigation requirements\n")
                f.write("- Changes in suitable crop varieties\n")
                f.write("- Potential yield reductions in non-irrigated systems\n\n")
            
            if not (has_warming_trend or has_cooling_trend or has_precip_increase or has_precip_decrease):
                f.write("No statistically significant climate trends were detected in the analyzed period. ")
                f.write("However, short-term variability and extreme events should still be considered ")
                f.write("in agricultural planning and management decisions.\n\n")
            
            # Add seasonal considerations
            f.write("### Seasonal Considerations\n\n")
            f.write("Agricultural planning should take into account the seasonal patterns observed in the data, ")
            f.write("particularly the timing of temperature extremes and precipitation. ")
            f.write("Management practices should be adapted to the local climate seasonality to optimize ")
            f.write("planting dates, irrigation scheduling, and harvest timing.\n\n")
            
            # Water resource implications
            f.write("## Water Resource Implications\n\n")
            
            if 'ppt' in spatial_means and 'tmean' in spatial_means:
                # Calculate average annual precipitation
                annual_precip = np.nanmean(spatial_means['ppt'])
                if aggregation == 'monthly':
                    annual_precip *= 12
                elif aggregation == 'daily':
                    annual_precip *= 365
                
                # Estimate basic water balance implications
                f.write(f"The average annual precipitation for the area is approximately {annual_precip:.1f} mm. ")
                
                avg_temp = np.nanmean(spatial_means['tmean'])
                if avg_temp > 10:
                    f.write("With the observed temperature regime, this suggests:\n\n")
                    
                    if annual_precip < 500:
                        f.write("- **Water limited conditions**: The region likely experiences significant water ")
                        f.write("constraints for agriculture, potentially requiring irrigation for optimal crop production.\n")
                    elif annual_precip < 800:
                        f.write("- **Moderate water availability**: Depending on seasonal distribution, most crops ")
                        f.write("can be grown, but supplemental irrigation may be beneficial during dry periods.\n")
                    else:
                        f.write("- **Adequate water availability**: The region generally receives sufficient ")
                        f.write("precipitation for most agricultural activities, though seasonal water management ")
                        f.write("may still be necessary.\n")
                    
                    # Add note about temperature effects on water demand
                    if has_warming_trend:
                        f.write("\nThe observed warming trend suggests increasing evapotranspiration demand, ")
                        f.write("potentially offsetting precipitation and reducing water availability for crops.\n")
            
            # Applications and recommendations section
            f.write("## Applications and Recommendations\n\n")
            
            f.write("### Agricultural Applications\n\n")
            f.write("- **Crop selection**: Choose varieties adapted to the local temperature and precipitation patterns\n")
            f.write("- **Planting dates**: Schedule planting based on seasonal temperature and precipitation trends\n")
            f.write("- **Water management**: Design irrigation systems and scheduling based on climate patterns\n")
            f.write("- **Risk management**: Plan for climate variability and extremes identified in the analysis\n\n")
            
            f.write("### Hydrological Applications\n\n")
            f.write("- **Water resource planning**: Use precipitation patterns to inform water allocation decisions\n")
            f.write("- **Flood risk assessment**: Consider precipitation intensity and seasonality\n")
            f.write("- **Drought monitoring**: Compare current conditions to historical patterns\n")
            f.write("- **Watershed management**: Design based on typical precipitation regimes and potential changes\n\n")
            
            # Data source and methodology
            f.write("## Data Source and Methodology\n\n")
            f.write("This analysis is based on PRISM (Parameter-elevation Regressions on Independent Slopes Model) climate data. ")
            f.write("PRISM is a sophisticated interpolation method that uses point measurements of climate data and digital ")
            f.write("elevation models to generate gridded estimates of climate parameters.\n\n")
            
            f.write("**Processing steps:**\n\n")
            f.write("1. Extraction of raw data from HDF5 database\n")
            f.write("2. Spatial subsetting to the region of interest\n")
            f.write("3. Temporal aggregation to " + aggregation + " values\n")
            f.write("4. Statistical analysis and visualization\n")
            f.write("5. Trend detection using linear regression methods\n\n")
            
            # Limitations
            f.write("### Limitations\n\n")
            f.write("- The analysis is limited by the temporal range and resolution of the available data\n")
            f.write("- Spatial interpolation may introduce uncertainties, especially in areas with complex terrain\n")
            f.write("- The analysis focuses on average conditions and may not fully represent extreme events\n")
            f.write("- Future climate conditions may differ from historical trends due to ongoing climate change\n\n")
            
            # Data export information
            f.write("## Data Export\n\n")
            f.write(f"The complete dataset has been exported to CSV format. Access the data at: [{os.path.basename(stats_path)}]({os.path.basename(stats_path)})\n\n")
            
            # Report generation information
            f.write("---\n\n")
            f.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M')}*\n")

        logger.info(f"Report successfully generated: {report_path}")
        return report_path
        
    except Exception as e:
        logger.error(f"Error generating PRISM report: {e}", exc_info=True)
        return ""

def batch_process_prism(config: Dict[str, Any], output_dir: str) -> str:
    """
    Process PRISM data and generate a report based on configuration.
    
    Args:
        config: Configuration dictionary with processing parameters
        output_dir: Directory to save outputs
        
    Returns:
        Path to the generated report file
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get basic parameters
        prism_path = config.get('prism_path', AgentConfig.PRISM_PATH)
        base_path = config.get('base_path', AgentConfig.HydroGeoDataset_ML_250_path)
        start_year = config.get('start_year', 2010)
        end_year = config.get('end_year', 2019)
        aggregation = config.get('aggregation', 'annual')
        bounding_box = config.get('bounding_box')
        resolution = config.get('RESOLUTION', 250)
        
        # Extract PRISM data
        logger.info(f"Extracting PRISM data for {start_year}-{end_year}")
        climate_data = extract_prism_data(
            prism_path=prism_path,
            base_path=base_path,
            start_year=start_year,
            end_year=end_year,
            bounding_box=bounding_box,
            aggregation=aggregation,
            resolution=resolution
        )
        
        if not climate_data:
            logger.error("Failed to extract PRISM data")
            return ""
        
        # Generate report
        logger.info("Generating PRISM climate report")
        report_path = generate_prism_report(
            data=climate_data,
            start_year=start_year,
            end_year=end_year,
            bounding_box=bounding_box,
            aggregation=aggregation,
            output_dir=output_dir
        )
        
        return report_path
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}", exc_info=True)
        return ""

def export_prism_data(data: Dict[str, np.ndarray], start_year: int, end_year: int,
                     aggregation: str, output_dir: str) -> str:
    """
    Export PRISM data to CSV and generate basic visualization.
    
    Args:
        data: Dictionary with arrays for climate variables
        start_year: First year of the data
        end_year: Last year of the data
        aggregation: Temporal aggregation type
        output_dir: Directory to save outputs
        
    Returns:
        Path to the exported CSV file
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Define file paths
        csv_path = os.path.join(output_dir, "prism_data.csv")
        timeseries_path = os.path.join(output_dir, "prism_timeseries.png")
        
        # Export data to CSV
        export_success = export_climate_data_to_csv(
            data=data,
            start_year=start_year,
            end_year=end_year,
            aggregation=aggregation,
            output_path=csv_path
        )
        
        # Create basic visualization
        plot_climate_timeseries(
            data=get_prism_spatial_means(data),
            start_year=start_year,
            end_year=end_year,
            aggregation=aggregation,
            output_path=timeseries_path,
            title=f"PRISM Climate Data ({start_year}-{end_year})"
        )
        
        return csv_path if export_success else ""
        
    except Exception as e:
        logger.error(f"Error exporting PRISM data: {e}", exc_info=True)
        return ""

if __name__ == "__main__":
    # Example usage
    try:
        from prism_utilities import extract_prism_data, get_prism_spatial_means
        
        config = {
            "bounding_box": [-85.444332, 43.658148, -85.239256, 44.164683],
            "start_year": 2010,
            "end_year": 2015,
            "aggregation": "monthly"
        }
        
        # Extract data
        print("Extracting PRISM data...")
        climate_data = extract_prism_data(
            prism_path=AgentConfig.PRISM_PATH,
            base_path=AgentConfig.HydroGeoDataset_ML_250_path,
            start_year=config['start_year'],
            end_year=config['end_year'],
            bounding_box=config['bounding_box'],
            aggregation=config['aggregation']
        )
        
        if climate_data:
            # Generate report
            output_dir = 'prism_results'
            report_path = generate_prism_report(
                data=climate_data,
                start_year=config['start_year'],
                end_year=config['end_year'],
                bounding_box=config['bounding_box'],
                aggregation=config['aggregation'],
                output_dir=output_dir
            )
            
            if report_path:
                print(f"Report generated successfully: {report_path}")
        else:
            print("No data was extracted")
            
    except Exception as e:
        print(f"Error in example execution: {e}")