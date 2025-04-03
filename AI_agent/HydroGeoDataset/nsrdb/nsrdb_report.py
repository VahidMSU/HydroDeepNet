"""
NSRDB solar and meteorological data analysis and report generation.

This module provides functionality to generate comprehensive reports analyzing
NSRDB (National Solar Radiation Database) data, including solar radiation, 
wind speed, and humidity variables.
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
import geopandas as gpd
from config import AgentConfig
from utils.plot_utils import safe_figure, save_figure

try:
    from HydroGeoDataset.nsrdb.nsrdb_utilities import (
        extract_nsrdb_data, extract_nsrdb_multiyear, get_coordinates_from_bbox,
        create_interpolated_grid, create_nsrdb_timeseries, create_nsrdb_map,
        aggregate_nsrdb_daily, extract_for_swat, calculate_statistics,
        calculate_monthly_averages, export_data_to_csv, NSRDB_VARIABLES, 
        save_as_raster
    )
    
    from HydroGeoDataset.nsrdb.nsrdb_solar_analysis import (
        calculate_heat_wave_statistics, calculate_solar_energy_potential,
        plot_heat_wave_analysis, plot_solar_energy_potential,
        calculate_radiation_extremes, analyze_climate_correlations,
        plot_climate_correlations, calculate_pv_performance_metrics,
        simulate_pv_output, plot_pv_simulation
    )
except ImportError:
    from HydroGeoDataset.nsrdb.nsrdb_utilities import (
        extract_nsrdb_data, extract_nsrdb_multiyear, get_coordinates_from_bbox,
        create_interpolated_grid, create_nsrdb_timeseries, create_nsrdb_map,
        aggregate_nsrdb_daily, extract_for_swat, calculate_statistics,
        calculate_monthly_averages, export_data_to_csv, NSRDB_VARIABLES,
        save_as_raster
    )
    from HydroGeoDataset.nsrdb.nsrdb_solar_analysis import (
        calculate_heat_wave_statistics, calculate_solar_energy_potential,
        plot_heat_wave_analysis, plot_solar_energy_potential,
        calculate_radiation_extremes, analyze_climate_correlations,
        plot_climate_correlations, calculate_pv_performance_metrics,
        simulate_pv_output, plot_pv_simulation
    )

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def generate_nsrdb_report(data: Dict[str, np.ndarray], start_year: int, end_year: int,
                          bbox: List[float], coordinates_index: gpd.GeoDataFrame,
                          output_dir: str = 'nsrdb_report') -> str:
    """
    Generate a comprehensive NSRDB data analysis report with visualizations.
    
    Args:
        data: Dictionary containing arrays for each NSRDB variable
        start_year: Starting year of the data
        end_year: Ending year of the data
        bbox: Bounding box as [lon_min, lat_min, lon_max, lat_max]
        coordinates_index: GeoDataFrame with coordinate information
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
        timeseries_path = os.path.join(output_dir, "nsrdb_timeseries.png")
        spatial_path = os.path.join(output_dir, "nsrdb_spatial.png")
        report_path = os.path.join(output_dir, "nsrdb_report.md")
        stats_path = os.path.join(output_dir, "nsrdb_stats.csv")
        
        # Advanced solar analysis paths
        solar_dir = os.path.join(output_dir, "solar_analysis")
        os.makedirs(solar_dir, exist_ok=True)
        
        heat_wave_path = os.path.join(solar_dir, "heat_wave_analysis.png")
        energy_potential_path = os.path.join(solar_dir, "solar_energy_potential.png")
        pv_simulation_path = os.path.join(solar_dir, "pv_simulation.png")
        climate_correlations_path = os.path.join(solar_dir, "climate_correlations.png")
        
        # Generate daily aggregated data for analysis
        daily_data = aggregate_nsrdb_daily(data)
        
        if not daily_data:
            logger.warning("Failed to aggregate daily data for report")
            return ""
        
        # Create spatial averages time series visualization
        create_nsrdb_timeseries(
            data=daily_data,
            start_year=start_year,
            aggregation='daily',
            output_path=timeseries_path
        )
        
        # Create spatial maps for visualization
        grid_data = create_interpolated_grid(
            data=data,
            coordinates_index=coordinates_index,
            bbox=bbox
        )
        
        if grid_data:
            create_nsrdb_map(
                grid_data=grid_data,
                bbox=bbox,
                output_path=spatial_path
            )
        
        # Calculate basic statistics
        stats = calculate_statistics(daily_data)
        
        # Calculate monthly averages for seasonal analysis
        monthly_avgs = calculate_monthly_averages(daily_data, start_year)
        
        # Export data to CSV
        export_data_to_csv(
            data=data, 
            coordinates_index=coordinates_index,
            output_path=stats_path,
            start_year=start_year
        )
        
        # Create date range for advanced analysis
        days = daily_data['ghi'].shape[0]
        dates = pd.date_range(start=f"{start_year}-01-01", periods=days)
        
        # Calculate spatial mean for solar radiation
        spatial_mean_ghi = np.nanmean(daily_data['ghi'], axis=1)
        
        # ADVANCED SOLAR ANALYSIS
        
        # 1. Heat wave analysis
        heat_wave_stats = calculate_heat_wave_statistics(
            spatial_mean_ghi, dates, threshold_percentile=90, min_consecutive_days=3
        )
        plot_heat_wave_analysis(spatial_mean_ghi, dates, heat_wave_stats, heat_wave_path)
        
        # 2. Solar energy potential analysis
        energy_potential = calculate_solar_energy_potential(spatial_mean_ghi)
        plot_solar_energy_potential(spatial_mean_ghi, dates, energy_potential, energy_potential_path)
        
        # 3. PV system performance metrics
        pv_metrics = calculate_pv_performance_metrics(spatial_mean_ghi)
        
        # 4. Radiation extremes
        radiation_extremes = calculate_radiation_extremes(spatial_mean_ghi, dates)
        
        # 5. PV system simulation
        pv_simulation = simulate_pv_output(spatial_mean_ghi, dates, system_capacity_kw=10.0)
        plot_pv_simulation(pv_simulation, pv_simulation_path)
        
        # 6. Climate variable correlations (if other variables are available)
        if 'relative_humidity' in daily_data and 'wind_speed' in daily_data:
            spatial_mean_hum = np.nanmean(daily_data['relative_humidity'], axis=1)
            spatial_mean_wind = np.nanmean(daily_data['wind_speed'], axis=1)
            
            correlation_stats = analyze_climate_correlations(
                spatial_mean_ghi, spatial_mean_hum, spatial_mean_wind, dates
            )
            
            plot_climate_correlations(
                spatial_mean_ghi, spatial_mean_hum, spatial_mean_wind, 
                dates, correlation_stats, climate_correlations_path
            )
        else:
            correlation_stats = None
        
        # Generate markdown report
        with open(report_path, 'w') as f:
            # Header
            f.write("# NSRDB Climate Data Analysis Report\n\n")
            
            # Basic information
            f.write("## Overview\n\n")
            f.write(f"**Period:** {start_year} to {end_year}\n\n")
            f.write(f"**Data Source:** National Solar Radiation Database (NSRDB)\n\n")
            
            f.write(f"**Region:** Lat [{bbox[1]:.4f}, {bbox[3]:.4f}], ")
            f.write(f"Lon [{bbox[0]:.4f}, {bbox[2]:.4f}]\n\n")
            
            # Data availability
            f.write("**Available Variables:**\n\n")
            for var_name in data.keys():
                if var_name in NSRDB_VARIABLES:
                    f.write(f"- {NSRDB_VARIABLES[var_name]['description']} ({NSRDB_VARIABLES[var_name]['units']})\n")
            f.write("\n")
            
            # Summary statistics for each variable
            f.write("## Summary Statistics\n\n")
            
            for var_name, var_stats in stats.items():
                if var_name in NSRDB_VARIABLES:
                    var_info = NSRDB_VARIABLES[var_name]
                    
                    f.write(f"### {var_info['description']}\n\n")
                    f.write(f"**Mean:** {var_stats['mean']:.2f} {var_info['units']}\n\n")
                    f.write(f"**Minimum:** {var_stats['min']:.2f} {var_info['units']}\n\n")
                    f.write(f"**Maximum:** {var_stats['max']:.2f} {var_info['units']}\n\n")
                    f.write(f"**Standard Deviation:** {var_stats['std']:.2f} {var_info['units']}\n\n")
                    f.write(f"**Temporal Variability:** {var_stats['temporal_variability']:.3f}\n\n")
                    f.write(f"**Spatial Variability:** {var_stats['spatial_variability']:.3f}\n\n")
            
            # Add seasonal (monthly) patterns section if available
            if monthly_avgs:
                f.write("## Seasonal Patterns\n\n")
                
                for var_name, monthly_data in monthly_avgs.items():
                    if var_name in NSRDB_VARIABLES:
                        var_info = NSRDB_VARIABLES[var_name]
                        
                        f.write(f"### {var_info['description']} by Month\n\n")
                        f.write("| Year | Month | Mean | Min | Max | Std Dev |\n")
                        f.write("|------|-------|------|-----|-----|---------|\n")
                        
                        # Only show first 12 months (one year) to keep the report concise
                        for i, row in monthly_data.head(12).iterrows():
                            month_name = calendar.month_name[int(row['month'])]
                            f.write(f"| {int(row['year'])} | {month_name} | {row['mean']:.2f} | ")
                            f.write(f"{row['min']:.2f} | {row['max']:.2f} | {row['std']:.2f} |\n")
                        f.write("\n")
                
                # Add seasonal analysis text
                f.write("### Seasonal Analysis\n\n")
                
                # Solar radiation seasonal patterns
                if 'ghi' in monthly_avgs:
                    ghi_by_month = monthly_avgs['ghi'].groupby('month')['mean'].mean().reset_index()
                    max_ghi_month = ghi_by_month.loc[ghi_by_month['mean'].idxmax(), 'month']
                    min_ghi_month = ghi_by_month.loc[ghi_by_month['mean'].idxmin(), 'month']
                    
                    f.write("**Solar Radiation:**\n\n")
                    f.write(f"The highest solar radiation typically occurs in **{calendar.month_name[int(max_ghi_month)]}**, ")
                    f.write(f"while the lowest is in **{calendar.month_name[int(min_ghi_month)]}**. This pattern ")
                    f.write(f"follows the expected seasonal variation based on day length and sun angle.\n\n")
                
                # Wind speed seasonal patterns
                if 'wind_speed' in monthly_avgs:
                    wind_by_month = monthly_avgs['wind_speed'].groupby('month')['mean'].mean().reset_index()
                    max_wind_month = wind_by_month.loc[wind_by_month['mean'].idxmax(), 'month']
                    min_wind_month = wind_by_month.loc[wind_by_month['mean'].idxmin(), 'month']
                    
                    f.write("**Wind Speed:**\n\n")
                    f.write(f"Wind speeds are typically highest in **{calendar.month_name[int(max_wind_month)]}** ")
                    f.write(f"and lowest in **{calendar.month_name[int(min_wind_month)]}**. ")
                    f.write(f"This seasonal pattern should be considered for wind energy applications and agricultural planning.\n\n")
            
            # Advanced Solar Analysis Section
            f.write("## Advanced Solar Radiation Analysis\n\n")
            
            # Solar energy potential subsection
            f.write("### Solar Energy Potential\n\n")
            f.write(f"The study area receives an average of **{energy_potential['daily_mean_kwh']:.2f} kWh/m²/day** ")
            f.write(f"of solar energy, equivalent to **{energy_potential['annual_kwh_per_m2']:.1f} kWh/m²/year**. ")
            f.write(f"The variability in solar resources (coefficient of variation) is {energy_potential['variability']:.2f}.\n\n")
            
            f.write("**Key solar metrics:**\n\n")
            f.write(f"- Daily minimum: {energy_potential['daily_min_kwh']:.2f} kWh/m²/day\n")
            f.write(f"- Daily maximum: {energy_potential['daily_max_kwh']:.2f} kWh/m²/day\n")
            f.write(f"- 90th percentile: {energy_potential['percentile_90_kwh']:.2f} kWh/m²/day\n")
            f.write(f"- 10th percentile: {energy_potential['percentile_10_kwh']:.2f} kWh/m²/day\n\n")
            
            f.write("![Solar Energy Potential](solar_analysis/solar_energy_potential.png)\n\n")
            
            # PV system performance subsection
            f.write("### Photovoltaic System Performance\n\n")
            f.write("For a standard 1 kW photovoltaic system:\n\n")
            f.write(f"- Annual production: **{pv_metrics['annual_output_kwh']:.1f} kWh**\n")
            f.write(f"- Capacity factor: {pv_metrics['capacity_factor_percent']:.1f}%\n")
            f.write(f"- Performance ratio: {pv_metrics['performance_ratio']:.2f}\n")
            f.write(f"- Estimated annual value: ${pv_metrics['annual_value_usd']:.2f}\n")
            f.write(f"- Simple payback period: {pv_metrics['simple_payback_years']:.1f} years\n\n")
            
            f.write("A simulation of a 10 kW system shows the following performance characteristics:\n\n")
            
            f.write("![PV System Simulation](solar_analysis/pv_simulation.png)\n\n")
            
            # Heat wave analysis subsection
            f.write("### High Solar Radiation Periods\n\n")
            f.write("Extended periods of high solar radiation ('heat waves') were analyzed using ")
            f.write(f"a threshold of the 90th percentile ({heat_wave_stats['threshold_value']:.1f} W/m²):\n\n")
            
            f.write(f"- Number of heat wave events: **{heat_wave_stats['count']}**\n")
            f.write(f"- Average duration: {heat_wave_stats['avg_duration']:.1f} days\n")
            f.write(f"- Maximum duration: {heat_wave_stats['max_duration']} days\n")
            f.write(f"- Percentage of days in heat wave: {heat_wave_stats['probability']:.1f}%\n\n")
            
            f.write("**Monthly distribution of heat wave events:**\n\n")
            f.write("| Month | Number of Events |\n")
            f.write("|-------|----------------|\n")
            for month, count in heat_wave_stats['freq_by_month'].items():
                f.write(f"| {month} | {count} |\n")
            f.write("\n")
            
            f.write("![Heat Wave Analysis](solar_analysis/heat_wave_analysis.png)\n\n")
            
            # Radiation extremes subsection
            f.write("### Solar Radiation Extremes\n\n")
            f.write("Analysis of extreme solar radiation values revealed:\n\n")
            
            f.write(f"- Extreme high days per year: {radiation_extremes['extreme_high_days_per_year']:.1f}\n")
            f.write(f"- Extreme low days per year: {radiation_extremes['extreme_low_days_per_year']:.1f}\n")
            f.write(f"- Longest high radiation run: {radiation_extremes['longest_high_extreme_run']} days\n")
            f.write(f"- Longest low radiation run: {radiation_extremes['longest_low_extreme_run']} days\n\n")
            
            f.write("**Top 5 highest radiation days:**\n\n")
            f.write("| Date | Radiation (W/m²) |\n")
            f.write("|------|----------------|\n")
            for day in radiation_extremes['highest_radiation_days']:
                date_str = pd.Timestamp(day['date']).strftime('%Y-%m-%d')
                f.write(f"| {date_str} | {day['radiation']:.2f} |\n")
            f.write("\n")
            
            # Add climate correlations section if available
            if correlation_stats:
                f.write("### Climate Variable Correlations\n\n")
                f.write("The analysis of correlations between solar radiation and other climate variables shows:\n\n")
                
                f.write(f"- Solar radiation vs. humidity: **{correlation_stats['correlation_radiation_humidity']:.2f}**\n")
                f.write(f"- Solar radiation vs. wind speed: **{correlation_stats['correlation_radiation_wind']:.2f}**\n")
                f.write(f"- Solar aridity index: {correlation_stats['solar_aridity_index_mean']:.2f}\n\n")
                
                f.write("![Climate Correlations](solar_analysis/climate_correlations.png)\n\n")
            
            # Time series visualization section
            f.write("## Time Series Analysis\n\n")
            f.write(f"The time series shows the change in climate variables over the entire period.\n\n")
            f.write(f"![Time Series]({os.path.basename(timeseries_path)})\n\n")
            
            # Spatial distribution
            if os.path.exists(spatial_path):
                f.write("## Spatial Distribution\n\n")
                f.write(f"The spatial maps show the distribution of climate variables across the study area.\n\n")
                f.write(f"![Spatial Distribution]({os.path.basename(spatial_path)})\n\n")
            
            # Applications section
            f.write("## Applications\n\n")
            
            # Solar energy section
            f.write("### Solar Energy Applications\n\n")
            f.write("The solar radiation data can be used for:\n\n")
            f.write("- Solar energy potential assessment\n")
            f.write("- Photovoltaic system design and optimization\n")
            f.write("- Solar resource forecasting\n")
            f.write("- Agricultural solar applications (solar drying, greenhouse operations)\n\n")
            
            # Solar design recommendations based on analysis
            f.write("**Solar design recommendations:**\n\n")
            if pv_metrics['capacity_factor_percent'] > 18:
                f.write("- **Excellent solar potential:** This location is well-suited for solar PV installations with above-average energy yield.\n")
            elif pv_metrics['capacity_factor_percent'] > 15:
                f.write("- **Good solar potential:** This location has favorable conditions for solar PV installations.\n")
            else:
                f.write("- **Moderate solar potential:** Solar installations are feasible but will have lower yields compared to sunnier regions.\n")
            
            if heat_wave_stats['probability'] > 15:
                f.write("- **Consider heat management:** The high frequency of intense solar radiation periods may require additional cooling for PV systems.\n")
            
            if energy_potential['variability'] > 0.5:
                f.write("- **High variability noted:** Consider energy storage solutions to manage the variable solar resource.\n")
            else:
                f.write("- **Relatively stable solar resource:** The area shows consistent solar radiation patterns.\n\n")
            
            # Wind energy section
            f.write("### Wind Energy Applications\n\n")
            f.write("The wind speed data can be used for:\n\n")
            f.write("- Wind farm siting and energy production estimation\n")
            f.write("- Small-scale wind energy applications\n")
            f.write("- Agricultural applications (windbreaks, ventilation)\n\n")
            
            # Agricultural section
            f.write("### Agricultural Applications\n\n")
            f.write("The NSRDB data can inform agricultural decisions related to:\n\n")
            f.write("- Evapotranspiration modeling using solar radiation data\n")
            f.write("- Crop growth modeling incorporating radiation and humidity\n")
            f.write("- Planning for wind-sensitive operations\n")
            f.write("- Irrigation scheduling\n\n")
            
            # SWAT model applications
            f.write("### Hydrological Modeling Applications\n\n")
            f.write("The NSRDB data can be used in SWAT+ and other hydrological models for:\n\n")
            f.write("- Improved evapotranspiration calculations\n")
            f.write("- Energy balance modeling\n")
            f.write("- Snow accumulation and melt modeling\n\n")
            
            # Data source and methodology
            f.write("## Data Source and Methodology\n\n")
            f.write("This analysis is based on the National Solar Radiation Database (NSRDB), which provides ")
            f.write("solar radiation, meteorological data, and other environmental parameters. ")
            f.write("The NSRDB uses a combination of satellite imagery, ground measurements, and physical models ")
            f.write("to generate gridded estimates of solar radiation and related variables.\n\n")
            
            f.write("**Variable details:**\n\n")
            f.write("| Variable | Description | Units | Scaling |\n")
            f.write("|----------|-------------|-------|--------|\n")
            f.write("| ghi | Global Horizontal Irradiance | W/m² | 1.0 |\n")
            f.write("| wind_speed | Wind Speed | m/s | 10.0 |\n")
            f.write("| relative_humidity | Relative Humidity | % | 100.0 |\n\n")
            
            f.write("**Processing steps:**\n\n")
            f.write("1. Extraction of raw data from HDF5 database\n")
            f.write("2. Application of correct scaling factors\n")
            f.write("3. Spatial subsetting to the region of interest\n")
            f.write("4. Temporal aggregation from 30-minute to daily values\n")
            f.write("5. Statistical analysis and visualization\n")
            f.write("6. Advanced solar resource assessment and PV performance modeling\n")
            f.write("7. Analysis of extreme events and climate correlations\n\n")
            
            # Limitations
            f.write("### Limitations\n\n")
            f.write("- The NSRDB data has a spatial resolution that may not capture fine-scale variations\n")
            f.write("- The analysis is limited by the temporal range of the available data\n")
            f.write("- Local terrain effects and microclimates may not be fully represented\n")
            f.write("- Cloud cover and atmospheric conditions can affect data accuracy\n")
            f.write("- PV performance estimates use standard assumptions and may differ from actual system performance\n\n")
            
            # Data export information
            f.write("## Data Export\n\n")
            f.write(f"The complete dataset has been exported to CSV format. Access the data at: ")
            f.write(f"[{os.path.basename(stats_path)}]({os.path.basename(stats_path)})\n\n")
            
            # Report generation information
            f.write("---\n\n")
            f.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M')}*\n")
        
        logger.info(f"Report successfully generated: {report_path}")
        return report_path
    
    except Exception as e:
        logger.error(f"Error generating NSRDB report: {e}", exc_info=True)
        return ""

def batch_process_nsrdb(config: Dict[str, Any], output_dir: str) -> str:
    """
    Process NSRDB data and generate a report based on configuration.
    
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
        coor_path = config.get('coor_path', '/data/SWATGenXApp/GenXAppData/NSRDB/CONUS_coordinates_index/CONUS_coordinates_index.shp')
        start_year = config.get('start_year', 2019)
        end_year = config.get('end_year', 2019)
        bbox = config.get('bbox', [-85.444332, 43.658148, -85.239256, 44.164683])
        nsrdb_path_template = config.get('nsrdb_path_template', 
                                        '/data/SWATGenXApp/GenXAppData/NSRDB/nsrdb_{}_full_filtered.h5')
        
        # Get coordinates within the bounding box
        coordinates_index = get_coordinates_from_bbox(coor_path, bbox)
        
        if coordinates_index.empty:
            logger.error("No coordinates found within the bounding box")
            return ""
        
        # Extract NSRDB data
        logger.info(f"Extracting NSRDB data for {start_year}-{end_year}")
        
        if start_year == end_year:
            nsrdb_data = extract_nsrdb_data(
                year=start_year,
                coordinates_index=coordinates_index,
                nsrdb_path_template=nsrdb_path_template
            )
        else:
            years = list(range(start_year, end_year + 1))
            nsrdb_data = extract_nsrdb_multiyear(
                years=years,
                coordinates_index=coordinates_index,
                nsrdb_path_template=nsrdb_path_template
            )
        
        if not nsrdb_data:
            logger.error("Failed to extract NSRDB data")
            return ""
        
        # Generate report
        logger.info("Generating NSRDB report")
        report_path = generate_nsrdb_report(
            data=nsrdb_data,
            start_year=start_year,
            end_year=end_year,
            bbox=bbox,
            coordinates_index=coordinates_index,
            output_dir=output_dir
        )
        
        # Optionally extract data for SWAT models if requested
        if config.get('extract_for_swat', False):
            swat_output_dir = os.path.join(output_dir, 'swat_files')
            extract_for_swat(
                nsrdb_data=nsrdb_data,
                coordinates_index=coordinates_index,
                output_dir=swat_output_dir,
                start_year=start_year
            )
            
        return report_path
    
    except Exception as e:
        logger.error(f"Error in batch processing: {e}", exc_info=True)
        return ""

def create_nsrdb_visualization(data: Dict[str, np.ndarray], 
                              coordinates_index: gpd.GeoDataFrame,
                              bbox: List[float],
                              output_dir: str = 'nsrdb_visualization') -> bool:
    """
    Create visualization files for NSRDB data without generating a full report.
    
    Args:
        data: Dictionary with NSRDB variable arrays
        coordinates_index: GeoDataFrame with coordinate information
        bbox: Bounding box as [lon_min, lat_min, lon_max, lat_max]
        output_dir: Directory to save visualizations
        
    Returns:
        Boolean indicating success
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Define file paths
        timeseries_path = os.path.join(output_dir, "nsrdb_timeseries.png")
        spatial_path = os.path.join(output_dir, "nsrdb_spatial.png")
        
        # Generate daily aggregated data for analysis
        daily_data = aggregate_nsrdb_daily(data)
        
        if not daily_data:
            logger.warning("Failed to aggregate daily data for visualization")
            return False
        
        # Create spatial averages time series visualization
        create_nsrdb_timeseries(
            data=daily_data,
            start_year=datetime.now().year,  # Default to current year if unknown
            aggregation='daily',
            output_path=timeseries_path
        )
        
        # Create spatial maps for visualization
        grid_data = create_interpolated_grid(
            data=data,
            coordinates_index=coordinates_index,
            bbox=bbox
        )
        
        if grid_data:
            create_nsrdb_map(
                grid_data=grid_data,
                bbox=bbox,
                output_path=spatial_path
            )
            
        logger.info(f"Created NSRDB visualizations in {output_dir}")
        return True
    
    except Exception as e:
        logger.error(f"Error creating NSRDB visualization: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    # Example usage
    try:
        config = {
            "bbox": [-85.444332, 43.658148, -85.239256, 44.164683],
            "start_year": 2010,
            "end_year": 2019
        }
        
        output_dir = 'nsrdb_results'
        report_path = batch_process_nsrdb(config, output_dir)
        
        if report_path:
            print(f"Report generated successfully: {report_path}")
        else:
            print("Failed to generate report")
            
    except Exception as e:
        print(f"Error in example execution: {e}")
