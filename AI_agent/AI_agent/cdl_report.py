"""
CDL (Cropland Data Layer) analysis and report generation.

This module provides functionality to generate comprehensive reports analyzing
USDA Cropland Data Layer (CDL) data, including crop distribution, trends, and changes.
"""
import os
import sys
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# Add parent directory to path to help with imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from AI_agent.config import AgentConfig
    from AI_agent.cdl import CDL_dataset
    from AI_agent.cdl_utilities import (
        plot_cdl_trends, calculate_crop_changes, create_crop_change_plot,
        create_crop_composition_pie, create_crop_diversity_plot, create_crop_rotation_heatmap,
        export_cdl_data, calculate_agricultural_intensity, get_crop_categories
    )
except ImportError:
    try:
        from config import AgentConfig
        from cdl import CDL_dataset
        from cdl_utilities import (
            plot_cdl_trends, calculate_crop_changes, create_crop_change_plot,
            create_crop_composition_pie, create_crop_diversity_plot, create_crop_rotation_heatmap,
            export_cdl_data, calculate_agricultural_intensity, get_crop_categories
        )
    except ImportError:
        # If we're in the same directory as the files
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(current_dir)
        try:
            from config import AgentConfig
        except ImportError:
            # Create a basic config if we can't import
            class AgentConfig:
                HydroGeoDataset_ML_250_path = "/data/SWATGenXApp/HydroGeoDataset_ML_250.h5"
                CDL_CODES_path = "/data/SWATGenXApp/cdl_classes.csv"
        
        from cdl_utilities import (
            plot_cdl_trends, calculate_crop_changes, create_crop_change_plot,
            create_crop_composition_pie, create_crop_diversity_plot, create_crop_rotation_heatmap,
            export_cdl_data, calculate_agricultural_intensity, get_crop_categories
        )
        from cdl import CDL_dataset

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def generate_cdl_detailed_report(
    cdl_data: Dict[int, Dict[str, Any]],
    bounding_box: Tuple[float, float, float, float],
    start_year: int,
    end_year: int,
    output_dir: str = 'cdl_report',
    advanced_analysis: bool = True,
    export_formats: List[str] = ['csv', 'markdown']
) -> str:
    """
    Generate a comprehensive CDL analysis report with extended analysis and visualizations.
    
    Args:
        cdl_data: Dictionary mapping years to land use data
        bounding_box: Tuple of (min_lon, min_lat, max_lon, max_lat)
        start_year: First year of data
        end_year: Last year of data
        output_dir: Directory to save report files
        advanced_analysis: Whether to include advanced analyses like diversity and rotation
        export_formats: List of formats to export data (csv, markdown)
        
    Returns:
        Path to the generated report file
    """
    if not cdl_data:
        logger.warning("No data to generate report")
        return ""
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Define file paths
        trends_path = os.path.join(output_dir, "cdl_trends.png")
        changes_path = os.path.join(output_dir, "cdl_changes.png")
        composition_path = os.path.join(output_dir, "cdl_composition.png")
        diversity_path = os.path.join(output_dir, "cdl_diversity.png")
        rotation_path = os.path.join(output_dir, "cdl_rotation.png")
        report_path = os.path.join(output_dir, "cdl_report.md")
        csv_path = os.path.join(output_dir, "cdl_data.csv")
        
        # Generate visualizations
        plot_cdl_trends(
            cdl_data=cdl_data,
            output_path=trends_path,
            title=f"Crop Distribution ({start_year}-{end_year})"
        )
        
        create_crop_change_plot(
            cdl_data=cdl_data,
            output_path=changes_path
        )
        
        # Use the most recent year for composition
        latest_year = max(cdl_data.keys())
        create_crop_composition_pie(
            cdl_data=cdl_data,
            year=latest_year,
            output_path=composition_path
        )
        
        # Advanced analyses
        if advanced_analysis:
            create_crop_diversity_plot(
                cdl_data=cdl_data,
                output_path=diversity_path
            )
            
            create_crop_rotation_heatmap(
                cdl_data=cdl_data,
                output_path=rotation_path
            )
        
        # Export data to CSV
        if 'csv' in export_formats:
            export_cdl_data(
                cdl_data=cdl_data,
                output_path=csv_path,
                include_percentages=True
            )
        
        # Calculate changes
        changes, change_df = calculate_crop_changes(cdl_data)
        
        # Calculate agricultural intensity
        intensity_data = calculate_agricultural_intensity(cdl_data)
        
        # Get crop categories
        categories = get_crop_categories(cdl_data)
        
        # Generate markdown report
        with open(report_path, 'w') as f:
            # Header
            f.write("# Cropland Data Layer (CDL) Analysis Report\n\n")
            
            # Basic information
            f.write("## Overview\n\n")
            f.write(f"**Period:** {start_year} to {end_year}\n\n")
            f.write(f"**Region:** Lat [{bounding_box[1]:.4f}, {bounding_box[3]:.4f}], ")
            f.write(f"Lon [{bounding_box[0]:.4f}, {bounding_box[2]:.4f}]\n\n")
            
            # Available years
            available_years = sorted(cdl_data.keys())
            f.write(f"**Available Data Years:** {', '.join(map(str, available_years))}\n\n")
            
            # Summary of agricultural land
            f.write("## Agricultural Land Summary\n\n")
            f.write("| Year | Total Area (ha) | Agricultural Area (ha) | Agricultural Percentage | Number of Crop Types |\n")
            f.write("|------|----------------|------------------------|-------------------------|---------------------|\n")
            
            for year in available_years:
                year_data = cdl_data[year]
                total_area = year_data.get("Total Area", 0)
                
                # Calculate agricultural area (excluding non-agricultural classes)
                ag_area = sum(year_data.get(crop, 0) for crop in year_data 
                             if crop not in ["Total Area", "unit", "Developed", "Water", "Forest", "Wetlands", "Barren"])
                
                ag_percentage = (ag_area / total_area * 100) if total_area > 0 else 0
                
                crop_count = len([k for k in year_data.keys() 
                                if k not in ["Total Area", "unit"] and not k.endswith("(%)")])
                
                f.write(f"| {year} | {total_area:,.2f} | {ag_area:,.2f} | {ag_percentage:.2f}% | {crop_count} |\n")
            
            f.write("\n")
            
            # Crop diversity and rotation
            if intensity_data:
                f.write("## Agricultural Intensity\n\n")
                f.write("### Intensity Metrics\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|------|\n")
                
                for metric, value in intensity_data.items():
                    if isinstance(value, float):
                        f.write(f"| {metric} | {value:.2f} |\n")
                    else:
                        f.write(f"| {metric} | {value} |\n")
                
                f.write("\n")
            
            # Top crops for the latest year
            f.write(f"## Crop Composition ({latest_year})\n\n")
            
            latest_data = cdl_data[latest_year]
            crop_items = [(k, v) for k, v in latest_data.items() 
                         if k not in ["Total Area", "unit"] and not k.endswith("(%)")]
            top_crops = sorted(crop_items, key=lambda x: x[1], reverse=True)[:10]
            
            f.write("| Rank | Crop | Area (ha) | Percentage |\n")
            f.write("|------|------|-----------|------------|\n")
            
            total_area = latest_data.get("Total Area", 0)
            for i, (crop, area) in enumerate(top_crops, 1):
                pct = (area / total_area * 100) if total_area > 0 else 0
                f.write(f"| {i} | {crop} | {area:,.2f} | {pct:.2f}% |\n")
            
            f.write("\n")
            f.write(f"![Crop Composition {latest_year}]({os.path.basename(composition_path)})\n\n")
            
            # Crop categories
            if categories:
                f.write("## Crop Categories\n\n")
                f.write("| Category | Area (ha) | Percentage |\n")
                f.write("|----------|-----------|------------|\n")
                
                total_area = sum(categories.values())
                for category, area in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                    pct = (area / total_area * 100) if total_area > 0 else 0
                    f.write(f"| {category} | {area:,.2f} | {pct:.2f}% |\n")
                
                f.write("\n")
            
            # Crop trends over time
            f.write("## Crop Trends\n\n")
            f.write("The following chart shows the trends in major crop types over the analyzed period:\n\n")
            f.write(f"![Crop Trends]({os.path.basename(trends_path)})\n\n")
            
            # Significant changes
            f.write("## Crop Changes\n\n")
            f.write("### Major Changes Between First and Last Year\n\n")
            f.write(f"![Crop Changes]({os.path.basename(changes_path)})\n\n")
            
            # Filter top 5 increases and decreases
            increases = change_df[change_df["Change (ha)"] > 0].head(5)
            decreases = change_df[change_df["Change (ha)"] < 0].head(5)
            
            if not increases.empty:
                f.write("### Largest Increases\n\n")
                f.write("| Crop | Change (ha) | Change (%) |\n")
                f.write("|------|------------|------------|\n")
                
                for _, row in increases.iterrows():
                    f.write(f"| {row['Crop']} | +{row['Change (ha)']:,.2f} | {row['Status']} |\n")
                f.write("\n")
            
            if not decreases.empty:
                f.write("### Largest Decreases\n\n") 
                f.write("| Crop | Change (ha) | Change (%) |\n")
                f.write("|------|------------|------------|\n")
                
                for _, row in decreases.iterrows():
                    f.write(f"| {row['Crop']} | {row['Change (ha)']:,.2f} | {row['Status']} |\n")
                f.write("\n")
            
            # Advanced analyses sections
            if advanced_analysis:
                # Crop diversity
                f.write("## Crop Diversity Analysis\n\n")
                f.write("Crop diversity is an important indicator of agricultural resilience and ecosystem health. ")
                f.write("Higher diversity can reduce pest and disease pressure and improve soil health.\n\n")
                f.write(f"![Crop Diversity]({os.path.basename(diversity_path)})\n\n")
                
                # Crop rotation
                f.write("## Crop Rotation Patterns\n\n")
                f.write("The heatmap below shows common crop rotations observed in the data, ")
                f.write("indicating which crops tend to follow others in sequence:\n\n")
                f.write(f"![Crop Rotation]({os.path.basename(rotation_path)})\n\n")
            
            # Implications
            f.write("## Agricultural Implications\n\n")
            
            # Generate implications based on data
            if changes:
                has_corn_increase = changes.get("Corn", 0) > 0
                has_soybean_increase = changes.get("Soybeans", 0) > 0
                has_wheat_decrease = changes.get("Winter Wheat", 0) < 0 or changes.get("Spring Wheat", 0) < 0
                has_pasture_decrease = changes.get("Pasture/Grass", 0) < 0 or changes.get("Grassland/Pasture", 0) < 0
                
                f.write("### Observed Trends and Their Implications\n\n")
                
                if has_corn_increase or has_soybean_increase:
                    f.write("- **Increasing row crop production**: ")
                    if has_corn_increase and has_soybean_increase:
                        f.write("Both corn and soybean acreage have increased, suggesting a focus on commodity crops. ")
                    elif has_corn_increase:
                        f.write("Corn acreage has increased, potentially reflecting favorable market conditions or ethanol demand. ")
                    else:
                        f.write("Soybean acreage has increased, potentially reflecting favorable market conditions or export demand. ")
                    f.write("This may indicate intensification of agricultural production, which could have implications for soil health and nutrient management.\n\n")
                
                if has_wheat_decrease:
                    f.write("- **Declining wheat production**: The decrease in wheat acreage may indicate shifting market conditions, ")
                    f.write("climate factors, or changes in farm management strategies. Wheat is often a key component of crop rotations, ")
                    f.write("so its reduction might affect overall rotation diversity.\n\n")
                
                if has_pasture_decrease:
                    f.write("- **Conversion of grassland/pasture**: The reduction in pasture/grassland suggests potential conversion to cropland, ")
                    f.write("which could have implications for soil erosion, carbon sequestration, and wildlife habitat.\n\n")
                
                # Check for diversity changes
                first_year = min(cdl_data.keys())
                last_year = max(cdl_data.keys())
                
                first_year_crop_count = len([k for k in cdl_data[first_year].keys() 
                                          if k not in ["Total Area", "unit"] and not k.endswith("(%)")])
                
                last_year_crop_count = len([k for k in cdl_data[last_year].keys() 
                                         if k not in ["Total Area", "unit"] and not k.endswith("(%)")])
                
                if last_year_crop_count > first_year_crop_count * 1.2:
                    f.write("- **Increasing crop diversity**: There has been an expansion in the variety of crops grown in the region, ")
                    f.write("which may contribute to reduced pest pressure, improved soil health, and greater agricultural resilience.\n\n")
                elif last_year_crop_count < first_year_crop_count * 0.8:
                    f.write("- **Decreasing crop diversity**: There has been a reduction in the variety of crops grown in the region, ")
                    f.write("which may increase vulnerability to pests and diseases, and could impact long-term soil health.\n\n")
            
            f.write("### Management Recommendations\n\n")
            f.write("Based on the observed crop patterns and changes, the following management practices may be beneficial:\n\n")
            f.write("1. **Diversified crop rotations**: Incorporate a wider variety of crops to improve soil health and reduce pest pressure\n")
            f.write("2. **Cover crops**: Implement cover crops during fallow periods to protect soil, fix nitrogen, and add organic matter\n")
            f.write("3. **Conservation practices**: Consider conservation tillage, buffer strips, and contour farming in areas with high erosion risk\n")
            f.write("4. **Precision agriculture**: Use precision technology to optimize inputs and reduce environmental impacts\n")
            f.write("5. **Integrated pest management**: Adopt IPM practices to reduce pesticide use while maintaining effective pest control\n\n")
            
            # Data source and methodology
            f.write("## Data Source and Methodology\n\n")
            f.write("This report is based on the USDA National Agricultural Statistics Service (NASS) Cropland Data Layer (CDL), ")
            f.write("which provides geo-referenced, crop-specific land cover data at 30-meter resolution. ")
            f.write("The CDL is created annually using satellite imagery and extensive ground truth verification.\n\n")
            
            f.write("**Processing steps:**\n\n")
            f.write("1. Extraction of CDL data for the specified region and time period\n")
            f.write("2. Aggregation and calculation of area statistics by crop type\n")
            f.write("3. Analysis of temporal trends, crop changes, and diversity metrics\n")
            f.write("4. Visualization of key patterns and relationships\n\n")
            
            # Limitations
            f.write("### Limitations\n\n")
            f.write("- CDL accuracy varies by crop type and region (typically 85-95% for major crops)\n")
            f.write("- Small fields or mixed plantings may not be accurately represented\n")
            f.write("- Analysis is limited to the temporal range of available data\n")
            f.write("- Local factors affecting crop decisions (e.g., specific markets, infrastructure) are not captured\n\n")
            
            # Data export information
            f.write("## Data Export\n\n")
            f.write(f"The complete dataset has been exported to CSV format. Access the data at: [{os.path.basename(csv_path)}]({os.path.basename(csv_path)})\n\n")
            
            # Report generation information
            f.write("---\n\n")
            f.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M')}*\n")

        logger.info(f"Report successfully generated: {report_path}")
        return report_path
        
    except Exception as e:
        logger.error(f"Error generating CDL report: {e}", exc_info=True)
        return ""

def analyze_cdl_data(config: Dict[str, Any], output_dir: str) -> str:
    """
    Process CDL data and generate a comprehensive report.
    
    Args:
        config: Configuration dictionary with processing parameters
        output_dir: Directory to save outputs
        
    Returns:
        Path to the generated report file
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract basic parameters
        bounding_box = config.get('bounding_box')
        start_year = config.get('start_year', 2010)
        end_year = config.get('end_year', 2020)
        advanced_analysis = config.get('advanced_analysis', True)
        
        if not bounding_box:
            logger.error("Bounding box not specified in configuration")
            return ""
        
        # Initialize CDL dataset handler
        cdl = CDL_dataset(config)
        
        # Extract CDL data
        logger.info(f"Extracting CDL data for {start_year}-{end_year}")
        cdl_data = cdl.cdl_trends()
        
        if not cdl_data:
            logger.error("Failed to extract CDL data")
            return ""
        
        # Generate report
        logger.info("Generating comprehensive CDL report")
        report_path = generate_cdl_detailed_report(
            cdl_data=cdl_data,
            bounding_box=bounding_box,
            start_year=start_year,
            end_year=end_year,
            output_dir=output_dir,
            advanced_analysis=advanced_analysis
        )
        
        return report_path
        
    except Exception as e:
        logger.error(f"Error in CDL data analysis: {e}", exc_info=True)
        return ""

def export_cdl_summary(cdl_data: Dict[int, Dict[str, Any]], output_dir: str) -> str:
    """
    Export a simple CDL summary with basic visualizations.
    
    Args:
        cdl_data: Dictionary mapping years to land use data
        output_dir: Directory to save outputs
        
    Returns:
        Path to the exported CSV file
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Define file paths
        csv_path = os.path.join(output_dir, "cdl_summary.csv")
        plot_path = os.path.join(output_dir, "cdl_summary.png")
        
        # Export data to CSV
        export_success = export_cdl_data(
            cdl_data=cdl_data,
            output_path=csv_path,
            include_percentages=True
        )
        
        # Create basic visualization
        plot_cdl_trends(
            cdl_data=cdl_data,
            output_path=plot_path,
            top_n=5
        )
        
        return csv_path if export_success else ""
        
    except Exception as e:
        logger.error(f"Error exporting CDL summary: {e}", exc_info=True)
        return ""

if __name__ == "__main__":
    # Example usage
    try:
        config = {
            "bounding_box": [-85.444332, 43.158148, -84.239256, 44.164683],
            "start_year": 2010,
            "end_year": 2020,
            "RESOLUTION": 250
        }
        
        # Process data and generate report
        report_path = analyze_cdl_data(
            config=config,
            output_dir="cdl_results"
        )
        
        if report_path:
            print(f"Report generated successfully: {report_path}")
        else:
            print("Failed to generate report")
            
    except Exception as e:
        logger.error(f"Error in example execution: {e}", exc_info=True)
