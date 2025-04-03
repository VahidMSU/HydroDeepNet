"""
SNODAS data analysis and report generation.

This module provides functionality to generate comprehensive reports analyzing
SNODAS (Snow Data Assimilation System) data.
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
from config import AgentConfig

try:
    from snodas.snowdas_utils import (
        get_snodas_spatial_means, create_period_labels,
        plot_snow_timeseries, create_snow_spatial_plot, create_snow_seasonal_plot,
        export_snow_data_to_csv, calculate_snow_trends, SNODAS_VARIABLES,
        create_snow_monthly_analysis_plot
    )
except ImportError:
    from datasets.snodas.snowdas_utils import (
        get_snodas_spatial_means, create_period_labels,
        plot_snow_timeseries, create_snow_spatial_plot, create_snow_seasonal_plot,
        export_snow_data_to_csv, calculate_snow_trends, SNODAS_VARIABLES,
        create_snow_monthly_analysis_plot
    )

try:
    from snodas.snowdas import SNODAS_Dataset
except ImportError:
    from datasets.snodas.snowdas import SNODAS_Dataset


# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def generate_snodas_report(data: Dict[str, np.ndarray], start_year: int, end_year: int,
                         bounding_box: Optional[Tuple[float, float, float, float]] = None,
                         aggregation: str = 'annual', output_dir: str = 'snodas_report') -> str:
    """
    Generate a comprehensive SNODAS data analysis report with visualizations.
    
    Args:
        data: Dictionary containing arrays for each snow variable
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
        timeseries_path = os.path.join(output_dir, "snodas_timeseries.png")
        spatial_path = os.path.join(output_dir, "snodas_spatial.png")
        seasonal_path = os.path.join(output_dir, "snodas_seasonal.png")
        monthly_analysis_path = os.path.join(output_dir, "snodas_monthly_analysis.png")
        report_path = os.path.join(output_dir, "snodas_report.md")
        stats_path = os.path.join(output_dir, "snodas_stats.csv")
        
        # Generate visualizations
        plot_snow_timeseries(
            data=get_snodas_spatial_means(data),
            start_year=start_year,
            end_year=end_year,
            aggregation=aggregation,
            output_path=timeseries_path,
            title=f"SNODAS Data ({start_year}-{end_year})"
        )
        
        # If we have at least one data point, create a spatial visualization of the first one
        for var_name, var_data in data.items():
            if var_data.size > 0:
                create_snow_spatial_plot(
                    data=data,
                    time_index=0,
                    output_path=spatial_path,
                    title=f"SNODAS Spatial Distribution ({start_year}-{end_year})"
                )
                break
            
        create_snow_seasonal_plot(
            data=data,
            start_year=start_year,
            end_year=end_year,
            output_path=seasonal_path
        )
        
        # Generate the new monthly analysis plot
        create_snow_monthly_analysis_plot(
            data=data,
            start_year=start_year,
            end_year=end_year,
            output_path=monthly_analysis_path
        )
        
        # Calculate statistics for each variable
        spatial_means = get_snodas_spatial_means(data)
        
        # Prepare statistics table
        period_labels = create_period_labels(start_year, end_year, aggregation)
        
        # Create DataFrame with stats
        stats_df = pd.DataFrame({'Period': period_labels})
        
        # Add snow variables to dataframe
        for var_name, var_data in spatial_means.items():
            if var_name in SNODAS_VARIABLES and len(var_data) > 0:
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
        trends = calculate_snow_trends(data, start_year, end_year)
        
        # Generate markdown report
        with open(report_path, 'w') as f:
            # Header
            f.write("# SNODAS Snow Data Analysis Report\n\n")
            
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
                if var_name in SNODAS_VARIABLES and var_data.size > 0:
                    f.write(f"- {SNODAS_VARIABLES[var_name]['description']} ({SNODAS_VARIABLES[var_name]['units']})\n")
            f.write("\n")
            
            # Summary statistics for each variable
            f.write("## Summary Statistics\n\n")
            
            for var_name, var_data in spatial_means.items():
                if var_name in SNODAS_VARIABLES and len(var_data) > 0:
                    var_info = SNODAS_VARIABLES[var_name]
                    
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
                f.write("## Annual Snow Patterns\n\n")
                
                # For SWE
                if 'snow_water_equivalent' in spatial_means:
                    yearly_swe = stats_df.groupby('Year')['snow_water_equivalent'].agg(['mean', 'min', 'max', 'std']).reset_index()
                    f.write("### Snow Water Equivalent (mm)\n\n")
                    f.write("| Year | Mean | Min | Max | Std Dev |\n")
                    f.write("|------|------|-----|-----|---------|\n")
                    for _, row in yearly_swe.iterrows():
                        f.write(f"| {int(row['Year'])} | {row['mean']:.2f} | {row['min']:.2f} | {row['max']:.2f} | {row['std']:.2f} |\n")
                    f.write("\n")
                
                # For Snow Depth
                if 'snow_layer_thickness' in spatial_means:
                    yearly_depth = stats_df.groupby('Year')['snow_layer_thickness'].agg(['mean', 'min', 'max', 'std']).reset_index()
                    f.write("### Snow Depth (mm)\n\n")
                    f.write("| Year | Mean | Min | Max | Std Dev |\n")
                    f.write("|------|------|-----|-----|---------|\n")
                    for _, row in yearly_depth.iterrows():
                        f.write(f"| {int(row['Year'])} | {row['mean']:.1f} | {row['min']:.1f} | {row['max']:.1f} | {row['std']:.1f} |\n")
                    f.write("\n")
                    
                # For Melt Rate (if available)
                if 'melt_rate' in spatial_means:
                    yearly_melt = stats_df.groupby('Year')['melt_rate'].agg(['sum', 'mean', 'max', 'std']).reset_index()
                    f.write("### Snowmelt (mm)\n\n")
                    f.write("| Year | Total | Mean | Max | Std Dev |\n")
                    f.write("|------|-------|------|-----|---------|\n")
                    for _, row in yearly_melt.iterrows():
                        f.write(f"| {int(row['Year'])} | {row['sum']:.1f} | {row['mean']:.2f} | {row['max']:.2f} | {row['std']:.2f} |\n")
                    f.write("\n")
            
            # Seasonal patterns
            f.write("## Seasonal Snow Patterns\n\n")
            f.write(f"The seasonal analysis shows how snow variables vary throughout the year.\n\n")
            f.write(f"![Seasonal Analysis]({os.path.basename(seasonal_path)})\n\n")
            
            # New monthly analysis section
            f.write("## Monthly Snow Analysis\n\n")
            f.write(f"The monthly analysis shows the average patterns and variability of snow variables by month over the {start_year}-{end_year} period.\n\n")
            f.write(f"![Monthly Analysis]({os.path.basename(monthly_analysis_path)})\n\n")
            f.write("The plots show the mean monthly values (line), the standard deviation range (darker shading), and the minimum-maximum range (lighter shading) over the analyzed period.\n\n")
            f.write("This visualization helps identify:\n\n")
            f.write("- The typical seasonal cycle of snow variables\n")
            f.write("- The months with highest uncertainty/variability\n")
            f.write("- The overall pattern of snow accumulation and melt\n\n")
            
            # Add some interpretation of seasonal patterns for SWE
            if 'snow_water_equivalent' in spatial_means and aggregation == 'monthly':
                try:
                    monthly_data = stats_df.copy()
                    swe_by_month = monthly_data.groupby('Month')['snow_water_equivalent'].mean()
                    peak_month = swe_by_month.idxmax()
                    lowest_month = swe_by_month.idxmin()
                    
                    f.write("### Snow Water Equivalent Seasonality\n\n")
                    f.write(f"The highest SWE typically occurs in **{calendar.month_name[peak_month]}**, ")
                    f.write(f"while the lowest SWE is in **{calendar.month_name[lowest_month]}**. ")
                    
                    # Determine snow season months (months with significant SWE)
                    snow_season = []
                    for month, swe in swe_by_month.items():
                        if swe > 0.2 * swe_by_month.max():  # 20% of peak as threshold
                            snow_season.append(calendar.month_name[month])
                    
                    if snow_season:
                        f.write(f"The primary snow season includes: {', '.join(snow_season)}.\n\n")
                    else:
                        f.write(f"The snowpack shows significant seasonal variability.\n\n")
                except Exception as e:
                    logger.warning(f"Could not generate SWE seasonality interpretation: {e}")
            
            # Add some interpretation of seasonal patterns for snowmelt
            if 'melt_rate' in spatial_means and aggregation == 'monthly':
                try:
                    monthly_data = stats_df.copy()
                    melt_by_month = monthly_data.groupby('Month')['melt_rate'].mean()
                    peak_melt_month = melt_by_month.idxmax()
                    
                    f.write("### Snowmelt Seasonality\n\n")
                    f.write(f"The highest snowmelt rates typically occur in **{calendar.month_name[peak_melt_month]}** ")
                    f.write(f"({melt_by_month[peak_melt_month]:.2f} mm/day), which corresponds to the primary snowmelt season.\n\n")
                    
                    # Determine primary melt season (months with significant melt)
                    melt_season = []
                    for month, melt in melt_by_month.items():
                        if melt > 0.2 * melt_by_month.max():  # 20% of peak as threshold
                            melt_season.append(calendar.month_name[month])
                    
                    if melt_season and len(melt_season) < 12:
                        f.write(f"The primary snowmelt season includes: {', '.join(melt_season)}.\n\n")
                    
                except Exception as e:
                    logger.warning(f"Could not generate snowmelt seasonality interpretation: {e}")
            
            # Time series analysis
            f.write("## Time Series Analysis\n\n")
            f.write(f"The time series shows the change in snow variables over the entire period.\n\n")
            f.write(f"![Time Series]({os.path.basename(timeseries_path)})\n\n")
            
            # Add trend analysis
            if trends:
                f.write("### Snow Trends\n\n")
                f.write("| Variable | Annual Change | Total Change | % Change | P-value | Significant? |\n")
                f.write("|----------|--------------|--------------|----------|---------|-------------|\n")
                
                for var_name, trend in trends.items():
                    if var_name in SNODAS_VARIABLES:
                        var_info = SNODAS_VARIABLES[var_name]
                        direction = "+" if trend['slope'] > 0 else ""
                        
                        pct_change = f"{abs(trend['percent_change']):.1f}%" if trend['percent_change'] != float('inf') else "N/A"
                        
                        f.write(f"| {var_info['description']} | {direction}{trend['slope']:.3f} {var_info['units']}/yr | ")
                        f.write(f"{direction}{abs(trend['total_change']):.2f} {var_info['units']} | ")
                        f.write(f"{pct_change} | {trend['p_value']:.4f} | {'Yes' if trend['significant'] else 'No'} |\n")
                f.write("\n")
            
            # Spatial distribution
            f.write("## Spatial Distribution\n\n")
            f.write(f"The spatial maps show the distribution of snow variables across the study area.\n\n")
            f.write(f"![Spatial Distribution]({os.path.basename(spatial_path)})\n\n")
            
            # Snow implications
            f.write("## Hydrological Implications\n\n")
            
            # Automatically determine if there are significant changes
            has_increasing_swe = False
            has_decreasing_swe = False
            has_increasing_melt = False
            has_decreasing_melt = False
            
            if trends:
                if 'snow_water_equivalent' in trends:
                    if trends['snow_water_equivalent']['significant']:
                        if trends['snow_water_equivalent']['slope'] > 0:
                            has_increasing_swe = True
                        else:
                            has_decreasing_swe = True
                
                if 'melt_rate' in trends:
                    if trends['melt_rate']['significant']:
                        if trends['melt_rate']['slope'] > 0:
                            has_increasing_melt = True
                        else:
                            has_decreasing_melt = True
            
            # Generate snowpack implications based on trends
            f.write("### Snowpack Implications\n\n")
            
            if has_increasing_swe:
                f.write("The data shows a **significant increasing trend in snow water equivalent** over the analyzed period. ")
                f.write("This increase may impact:\n\n")
                f.write("- Extended snowmelt periods in spring\n")
                f.write("- Increased water availability for spring and summer runoff\n")
                f.write("- Potential for increased flooding during rapid melt events\n")
                f.write("- Changes to groundwater recharge timing and volume\n\n")
            
            elif has_decreasing_swe:
                f.write("The data shows a **significant decreasing trend in snow water equivalent** over the analyzed period. ")
                f.write("This decrease may impact:\n\n")
                f.write("- Reduced spring and summer water availability\n")
                f.write("- Earlier snowmelt timing and shorter snow season\n")
                f.write("- Potential water scarcity during dry seasons\n")
                f.write("- Changes to ecosystems dependent on snowmelt\n\n")
            
            if has_increasing_melt:
                f.write("The analysis indicates a **significant increase in snowmelt rates**. ")
                f.write("Increased snowmelt may lead to:\n\n")
                f.write("- Higher peak flows in rivers and streams\n")
                f.write("- Potential increases in spring flooding risk\n")
                f.write("- Changes in the timing of water availability\n")
                f.write("- Possible impacts on aquatic ecosystems due to altered flow regimes\n\n")
            
            elif has_decreasing_melt:
                f.write("The analysis indicates a **significant decrease in snowmelt rates**. ")
                f.write("Decreased snowmelt may lead to:\n\n")
                f.write("- More gradual spring runoff\n")
                f.write("- Potentially extended periods of snowmelt contribution to streamflow\n")
                f.write("- Reduced peak flows during spring\n")
                f.write("- Changes in seasonal water availability patterns\n\n")
            
            if not (has_increasing_swe or has_decreasing_swe or has_increasing_melt or has_decreasing_melt):
                f.write("No statistically significant trends in snow variables were detected in the analyzed period. ")
                f.write("However, year-to-year variability in snowpack conditions remains an important consideration ")
                f.write("for water resource planning and management.\n\n")
            
            # Add seasonal considerations
            f.write("### Seasonal Water Resource Considerations\n\n")
            f.write("Water resource planning should take into account the seasonal patterns observed in the snow data, ")
            f.write("particularly the timing of peak SWE and snowmelt. ")
            f.write("Management practices should be adapted to the local snow seasonality to optimize ")
            f.write("water storage, flood control, and water supply allocation.\n\n")
            
            # Water resource implications
            f.write("## Water Supply Implications\n\n")
            
            if 'snow_water_equivalent' in spatial_means and 'melt_rate' in spatial_means:
                # Calculate peak SWE and total melt
                try:
                    peak_swe = np.nanmax(spatial_means['snow_water_equivalent'])
                    total_melt = np.nansum(spatial_means['melt_rate'])
                    
                    f.write(f"The maximum snow water equivalent for the area is approximately {peak_swe:.1f} mm, ")
                    f.write(f"representing significant water storage in the snowpack. ")
                    
                    f.write(f"The estimated total snowmelt contribution is {total_melt:.1f} mm");
                    
                    if aggregation == 'annual':
                        f.write(" per year on average")
                    elif aggregation == 'monthly':
                        f.write(" over the entire period")
                    
                    f.write(", which suggests:\n\n")
                    
                    if peak_swe < 50:
                        f.write("- **Limited snowpack contribution**: The region has relatively minimal snowpack storage, ")
                        f.write("suggesting only minor contributions to the water supply from snowmelt.\n")
                    elif peak_swe < 200:
                        f.write("- **Moderate snowpack contribution**: Snowmelt provides a noticeable seasonal water source, ")
                        f.write("but may not dominate the annual water budget.\n")
                    else:
                        f.write("- **Significant snowpack contribution**: The region has substantial snow water storage, ")
                        f.write("making snowmelt a critical component of the annual water budget and seasonal water availability.\n")
                    
                    # Add note about snowmelt timing
                    if 'melt_rate' in spatial_means:
                        f.write("\nUnderstanding the timing and rate of snowmelt is crucial for water resource management ")
                        f.write("in this region, including reservoir operations, flood control, and water supply planning.\n")
                except Exception as e:
                    logger.warning(f"Could not calculate water supply implications: {e}")
            
            # Applications and recommendations section
            f.write("## Applications and Recommendations\n\n")
            
            f.write("### Water Management Applications\n\n")
            f.write("- **Reservoir operations**: Adjust storage and release schedules based on snowpack conditions and melt timing\n")
            f.write("- **Flood forecasting**: Use SWE and melt rate data to predict spring runoff volumes and timing\n")
            f.write("- **Drought planning**: Monitor snowpack as an early indicator of potential water scarcity\n")
            f.write("- **Water allocation**: Plan water deliveries based on projected snowmelt contributions\n\n")
            
            f.write("### Hydrological Applications\n\n")
            f.write("- **Streamflow forecasting**: Use SWE data to predict spring and summer runoff volumes\n")
            f.write("- **Hydropower planning**: Schedule generation based on expected snowmelt patterns\n")
            f.write("- **Ecological considerations**: Manage for environmental flows based on natural snowmelt regimes\n")
            f.write("- **Climate change assessment**: Monitor trends in snowpack as indicators of changing conditions\n\n")
            
            # Data source and methodology
            f.write("## Data Source and Methodology\n\n")
            f.write("This analysis is based on SNODAS (Snow Data Assimilation System) data. ")
            f.write("SNODAS is a modeling and data assimilation system developed by the National Weather Service's National ")
            f.write("Operational Hydrologic Remote Sensing Center (NOHRSC) to provide estimates of snow cover and associated parameters.\n\n")
            
            f.write("**Processing steps:**\n\n")
            f.write("1. Extraction of raw data from HDF5 database\n")
            f.write("2. Spatial subsetting to the region of interest\n")
            f.write("3. Temporal aggregation to " + aggregation + " values\n")
            f.write("4. Statistical analysis and visualization\n")
            f.write("5. Trend detection using linear regression methods\n\n")
            
            # Limitations
            f.write("### Limitations\n\n")
            f.write("- The analysis is limited by the temporal range and resolution of the available data\n")
            f.write("- SNODAS combines model and observational data, introducing some uncertainty\n")
            f.write("- Spatial resolution may not capture fine-scale variability in complex terrain\n")
            f.write("- Snow processes are complex and influenced by many factors not fully represented in this analysis\n\n")
            
            # Data export information
            f.write("## Data Export\n\n")
            f.write(f"The complete dataset has been exported to CSV format. Access the data at: [{os.path.basename(stats_path)}]({os.path.basename(stats_path)})\n\n")
            
            # Report generation information
            f.write("---\n\n")
            f.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M')}*\n")

        logger.info(f"Report successfully generated: {report_path}")
        return report_path
        
    except Exception as e:
        logger.error(f"Error generating SNODAS report: {e}", exc_info=True)
        return ""

def batch_process_snodas(config: Dict[str, Any], output_dir: str) -> str:
    """
    Process SNODAS data and generate a report based on configuration.
    
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
        snodas_path = config.get('snodas_path', AgentConfig.SNODAS_PATH if hasattr(AgentConfig, 'SNODAS_PATH') else '/data/SWATGenXApp/GenXAppData/HydroGeoDataset/SNODAS.h5')
        base_path = config.get('base_path', AgentConfig.HydroGeoDataset_ML_250_path)
        start_year = config.get('start_year', 2010)
        end_year = config.get('end_year', 2019)
        aggregation = config.get('aggregation', 'annual')
        bounding_box = config.get('bounding_box')
        
        # Create SNODAS dataset and extract data
        logger.info(f"Creating SNODAS dataset for {start_year}-{end_year}")

        snodas_dataset = SNODAS_Dataset(config)
        snow_data = snodas_dataset.get_data(start_year, end_year)
        
        if not snow_data:
            logger.error("Failed to extract SNODAS data")
            return ""
        
        # Generate report
        logger.info("Generating SNODAS report")
        report_path = generate_snodas_report(
            data=snow_data,
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

def export_snodas_data(data: Dict[str, np.ndarray], start_year: int, end_year: int,
                     aggregation: str, output_dir: str) -> str:
    """
    Export SNODAS data to CSV and generate basic visualization.
    
    Args:
        data: Dictionary with arrays for snow variables
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
        csv_path = os.path.join(output_dir, "snodas_data.csv")
        timeseries_path = os.path.join(output_dir, "snodas_timeseries.png")
        
        # Export data to CSV
        export_success = export_snow_data_to_csv(
            data=data,
            start_year=start_year,
            end_year=end_year,
            aggregation=aggregation,
            output_path=csv_path
        )
        
        # Create basic visualization
        plot_snow_timeseries(
            data=get_snodas_spatial_means(data),
            start_year=start_year,
            end_year=end_year,
            aggregation=aggregation,
            output_path=timeseries_path,
            title=f"SNODAS Data ({start_year}-{end_year})"
        )
        
        return csv_path if export_success else ""
        
    except Exception as e:
        logger.error(f"Error exporting SNODAS data: {e}", exc_info=True)
        return ""

if __name__ == "__main__":
    # Example usage
    try:
        config = {
            "RESOLUTION": 250,
            "aggregation": "annual",
            "start_year": 2010,
            "end_year": 2015,
            'bounding_box': [-85.444332, 43.658148, -85.239256, 44.164683],
        }
        
        # Process data and generate report
        output_dir = 'snodas_results'
        report_path = batch_process_snodas(config, output_dir)
        
        if report_path:
            print(f"Report generated successfully: {report_path}")
        else:
            print("Report generation failed")
            
    except Exception as e:
        print(f"Error in example execution: {e}")
