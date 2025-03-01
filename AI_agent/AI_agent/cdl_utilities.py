"""
CDL (Cropland Data Layer) visualization and analysis utilities.

This module provides functions for analyzing, visualizing, and exporting
CDL (Cropland Data Layer) data extracted from the CDL dataset.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
import os
import logging
from pathlib import Path
import seaborn as sns

# Configure logger
logger = logging.getLogger(__name__)

# Set plotting style for consistent visuals
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

def plot_cdl_trends(
    cdl_data: Dict[int, Dict[str, Any]], 
    top_n: int = 5, 
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    title: str = 'Crop Distribution Over Time',
    dpi: int = 300
) -> Optional[plt.Figure]:
    """
    Create a stacked bar plot showing crop area trends over time.
    
    Args:
        cdl_data: Dictionary mapping years to land use data
        top_n: Number of top crops to display individually (others grouped as "Other")
        output_path: Path to save the figure (optional)
        figsize: Figure dimensions (width, height) in inches
        title: Plot title
        dpi: Resolution for saved image
        
    Returns:
        Matplotlib Figure object or None if error occurs
    """
    try:
        if not cdl_data:
            logger.warning("No data to plot")
            return None
            
        # Extract available years in ascending order
        years = sorted(cdl_data.keys())
        
        # Collect all unique crop types across years (excluding metadata fields)
        all_crops = set()
        for year_data in cdl_data.values():
            all_crops.update([
                crop for crop in year_data.keys() 
                if crop not in ["Total Area", "unit"] and not crop.endswith("(%)")
            ])
        
        # Calculate total area for each crop across all years
        crop_totals = {}
        for year_data in cdl_data.values():
            for crop in all_crops:
                if crop in year_data:
                    crop_totals[crop] = crop_totals.get(crop, 0) + year_data[crop]
        
        # Identify top N crops by total area
        top_crops = sorted(crop_totals.items(), key=lambda x: x[1], reverse=True)
        top_crop_names = [crop for crop, _ in top_crops[:top_n]]
        
        # Prepare data matrix for stacked bar chart
        data_matrix = np.zeros((len(years), len(top_crop_names) + 1))  # +1 for "Other"
        
        for i, year in enumerate(years):
            year_data = cdl_data[year]
            
            # Extract top crop values
            for j, crop in enumerate(top_crop_names):
                data_matrix[i, j] = year_data.get(crop, 0)
            
            # Sum remaining crops as "Other"
            data_matrix[i, -1] = sum(
                year_data.get(crop, 0) for crop in all_crops 
                if crop not in top_crop_names
            )
        
        # Create plot with improved styling
        fig, ax = plt.subplots(figsize=figsize)
        
        # Generate color map for consistent colors
        cmap = plt.get_cmap('viridis')
        colors = [cmap(i/len(top_crop_names)) for i in range(len(top_crop_names))]
        colors.append((0.7, 0.7, 0.7, 1.0))  # Gray for "Other"
        
        # Create stacked bars
        bottom = np.zeros(len(years))
        labels = top_crop_names + ["Other"]
        
        for i, (label, color) in enumerate(zip(labels, colors)):
            ax.bar(years, data_matrix[:, i], bottom=bottom, 
                   label=label, color=color, edgecolor='white', linewidth=0.5)
            bottom += data_matrix[:, i]
        
        # Style the plot
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Area (hectares)', fontsize=12)
        ax.tick_params(axis='both', labelsize=10)
        
        # Improve legend placement and styling
        legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1), 
                    frameon=True, fontsize=10)
        legend.set_title("Crop Types", prop={'size': 11, 'weight': 'bold'})
        
        # Add data labels if there aren't too many years
        if len(years) <= 5:
            for i, year in enumerate(years):
                total = sum(data_matrix[i, :])
                ax.text(year, total + (total*0.02), f"{int(total):,}", 
                        ha='center', fontsize=9, fontweight='bold')
        
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save figure if path provided
        if output_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Plot saved to {output_path}")
            
        return fig
        
    except Exception as e:
        logger.error(f"Error creating CDL trends plot: {e}", exc_info=True)
        return None

def calculate_crop_changes(
    cdl_data: Dict[int, Dict[str, Any]],
    custom_years: Optional[Tuple[int, int]] = None
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Calculate changes in crop areas between two years.
    
    Args:
        cdl_data: Dictionary mapping years to land use data
        custom_years: Optional tuple of (start_year, end_year) to compare;
                      if not provided, uses first and last available years
        
    Returns:
        Tuple containing:
            - Dictionary of crop changes (hectares)
            - Pandas DataFrame with detailed change statistics
    """
    if len(cdl_data) < 2:
        logger.warning("Need at least two years of data to calculate changes")
        return {}, pd.DataFrame()
        
    try:
        available_years = sorted(cdl_data.keys())
        
        if custom_years and len(custom_years) == 2:
            first_year, last_year = custom_years
            if first_year not in cdl_data or last_year not in cdl_data:
                logger.warning(f"Custom years {custom_years} not available in data. Using first and last years.")
                first_year, last_year = available_years[0], available_years[-1]
        else:
            first_year, last_year = available_years[0], available_years[-1]
        
        first_data = cdl_data[first_year]
        last_data = cdl_data[last_year]
        
        # Get all unique crop types from both years
        all_crops = set()
        for crop in list(first_data.keys()) + list(last_data.keys()):
            if crop not in ["Total Area", "unit"] and not crop.endswith("(%)"):
                all_crops.add(crop)
        
        # Calculate changes and prepare DataFrame data
        changes = {}
        df_data = []
        
        for crop in all_crops:
            first_value = first_data.get(crop, 0)
            last_value = last_data.get(crop, 0)
            change = last_value - first_value
            changes[crop] = change
            
            # Calculate percentage change with proper handling of edge cases
            if first_value > 0:
                percent_change = (change / first_value) * 100
            elif last_value > 0:
                percent_change = float('inf')  # New crop appeared
            else:
                percent_change = 0  # No change (both zero)
                
            # Format percentage for display
            if percent_change == float('inf'):
                pct_display = "New crop"
            elif percent_change == -100:
                pct_display = "Disappeared"
            else:
                pct_display = f"{percent_change:.1f}%"
                
            df_data.append({
                'Crop': crop,
                f'Area {first_year} (ha)': round(first_value, 2),
                f'Area {last_year} (ha)': round(last_value, 2),
                'Change (ha)': round(change, 2),
                'Change (%)': round(percent_change, 1) if percent_change != float('inf') else percent_change,
                'Status': pct_display
            })
            
        # Create and format DataFrame
        df = pd.DataFrame(df_data)
        df = df.sort_values('Change (ha)', ascending=False)
        
        return changes, df
        
    except Exception as e:
        logger.error(f"Error calculating crop changes: {e}", exc_info=True)
        return {}, pd.DataFrame()

def export_cdl_data(
    cdl_data: Dict[int, Dict[str, Any]], 
    output_path: str,
    include_percentages: bool = False
) -> bool:
    """
    Export CDL data to CSV file.
    
    Args:
        cdl_data: Dictionary mapping years to land use data
        output_path: Path to save CSV file
        include_percentages: Whether to include percentage columns
        
    Returns:
        Boolean indicating success
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        years = sorted(cdl_data.keys())
        
        # Collect all unique crops (excluding metadata fields)
        all_crops = set()
        for year_data in cdl_data.values():
            crop_keys = [k for k in year_data.keys() 
                        if k not in ["Total Area", "unit"] and not k.endswith("(%)")]
            all_crops.update(crop_keys)
        all_crops = sorted(all_crops)
        
        # Create DataFrame
        df = pd.DataFrame(index=all_crops)
        
        # Add area columns for each year
        for year in years:
            df[f"{year}"] = df.index.map(lambda crop: cdl_data[year].get(crop, 0))
            
            # Add percentage columns if requested
            if include_percentages and "Total Area" in cdl_data[year]:
                total = cdl_data[year]["Total Area"]
                if total > 0:
                    df[f"{year} (%)"] = df[f"{year}"].map(lambda x: round((x / total) * 100, 2))
        
        # Add summary statistics
        df_with_totals = df.copy()
        df_with_totals.loc['Total Area', :] = df.sum()
        
        # Save to CSV with proper formatting
        df_with_totals.to_csv(output_path, float_format='%.2f')
        logger.info(f"Data exported to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting CDL data: {e}", exc_info=True)
        return False

def create_crop_change_plot(
    cdl_data: Dict[int, Dict[str, Any]],
    top_n: int = 10,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> Optional[plt.Figure]:
    """
    Create a horizontal bar chart showing the biggest crop changes over time.
    
    Args:
        cdl_data: Dictionary mapping years to land use data
        top_n: Number of top changing crops to display
        output_path: Path to save the figure (optional)
        figsize: Figure dimensions (width, height) in inches
        
    Returns:
        Matplotlib Figure object or None if error occurs
    """
    try:
        changes, df = calculate_crop_changes(cdl_data)
        if df.empty:
            return None
            
        # Select top N changes (absolute value)
        df['Abs Change'] = df['Change (ha)'].abs()
        top_changes = df.nlargest(top_n, 'Abs Change')
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar chart
        bars = ax.barh(
            y=top_changes['Crop'],
            width=top_changes['Change (ha)'],
            color=[
                'forestgreen' if x > 0 else 'firebrick' 
                for x in top_changes['Change (ha)']
            ],
            edgecolor='white',
            linewidth=0.5
        )
        
        # Add data labels
        for bar in bars:
            width = bar.get_width()
            label_x = width + (width * 0.02) if width > 0 else width - (abs(width) * 0.08)
            ax.text(
                label_x, 
                bar.get_y() + bar.get_height()/2, 
                f"{width:,.1f}",
                va='center',
                fontsize=9,
                color='black' if width > 0 else 'white'
            )
        
        # Determine year range for title
        years = sorted(cdl_data.keys())
        year_range = f"{years[0]} to {years[-1]}"
        
        # Style the plot
        ax.set_title(f'Top {top_n} Crop Changes ({year_range})', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Change in Area (hectares)', fontsize=12)
        ax.tick_params(axis='both', labelsize=10)
        
        # Add a vertical line at x=0
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        
        # Add grid lines
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Crop change plot saved to {output_path}")
            
        return fig
        
    except Exception as e:
        logger.error(f"Error creating crop change plot: {e}", exc_info=True)
        return None

def create_crop_composition_pie(
    cdl_data: Dict[int, Dict[str, Any]],
    year: Optional[int] = None,
    top_n: int = 5,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> Optional[plt.Figure]:
    """
    Create a pie chart showing crop composition for a specific year.
    
    Args:
        cdl_data: Dictionary mapping years to land use data
        year: Year to visualize (uses latest year if None)
        top_n: Number of top crops to display individually
        output_path: Path to save the figure (optional)
        figsize: Figure dimensions (width, height) in inches
        
    Returns:
        Matplotlib Figure object or None if error occurs
    """
    try:
        if not cdl_data:
            logger.warning("No data to visualize")
            return None
            
        # Select year (use latest if not specified)
        if year is None or year not in cdl_data:
            year = max(cdl_data.keys())
            
        year_data = cdl_data[year]
        
        # Get crop data (excluding metadata fields)
        crop_data = {
            k: v for k, v in year_data.items() 
            if k not in ["Total Area", "unit"] and not k.endswith("%")
        }
        
        if not crop_data:
            logger.warning(f"No crop data found for year {year}")
            return None
            
        # Sort crops by area
        sorted_crops = sorted(crop_data.items(), key=lambda x: x[1], reverse=True)
        
        # Select top N crops and group the rest as "Other"
        top_crops = sorted_crops[:top_n]
        other_sum = sum(area for _, area in sorted_crops[top_n:])
        
        # Prepare data for pie chart
        labels = [crop for crop, _ in top_crops]
        if other_sum > 0:
            labels.append("Other")
        
        sizes = [area for _, area in top_crops]
        if other_sum > 0:
            sizes.append(other_sum)
            
        # Calculate percentages for labels
        total = sum(sizes)
        pcts = [100 * s / total for s in sizes]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create pie chart with percentage labels
        wedges, texts, autotexts = ax.pie(
            sizes, 
            labels=None,
            autopct='',
            startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1},
            explode=[0.05] * len(sizes)
        )
        
        # Create legend with percentages
        legend_labels = [f"{l} ({p:.1f}%)" for l, p in zip(labels, pcts)]
        ax.legend(
            wedges, legend_labels,
            title="Crop Types",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1)
        )
        
        # Set equal aspect ratio to ensure circular pie
        ax.set_aspect('equal')
        
        # Add title
        ax.set_title(f'Crop Composition for {year}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Pie chart saved to {output_path}")
            
        return fig
        
    except Exception as e:
        logger.error(f"Error creating pie chart: {e}", exc_info=True)
        return None

def generate_cdl_report(
    cdl_data: Dict[int, Dict[str, Any]],
    output_dir: str,
    report_name: str = "CDL_Analysis",
    include_plots: bool = True
) -> str:
    """
    Generate a comprehensive CDL analysis report with data and visualizations.
    
    Args:
        cdl_data: Dictionary mapping years to land use data
        output_dir: Directory to save the report and visualizations
        report_name: Base name for report files
        include_plots: Whether to include plots in the report
        
    Returns:
        Path to the generated report file
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Base paths
        csv_path = os.path.join(output_dir, f"{report_name}_data.csv")
        trend_plot_path = os.path.join(output_dir, f"{report_name}_trends.png")
        change_plot_path = os.path.join(output_dir, f"{report_name}_changes.png")
        pie_plot_path = os.path.join(output_dir, f"{report_name}_composition.png")
        report_path = os.path.join(output_dir, f"{report_name}.md")
        
        # Export data to CSV
        export_cdl_data(cdl_data, csv_path, include_percentages=True)
        
        # Generate plots if requested
        if include_plots:
            plot_cdl_trends(cdl_data, output_path=trend_plot_path)
            create_crop_change_plot(cdl_data, output_path=change_plot_path)
            create_crop_composition_pie(cdl_data, output_path=pie_plot_path)
            
        # Calculate changes for report
        _, change_df = calculate_crop_changes(cdl_data)
        
        # Generate markdown report
        with open(report_path, 'w') as f:
            years = sorted(cdl_data.keys())
            
            # Header
            f.write(f"# CDL Analysis Report\n\n")
            f.write(f"Analysis period: {years[0]} - {years[-1]}\n\n")
            
            # Summary statistics
            f.write("## Summary\n\n")
            f.write("| Year | Total Area (ha) | Number of Crop Types |\n")
            f.write("|------|----------------|----------------------|\n")
            
            for year in years:
                year_data = cdl_data[year]
                total_area = year_data.get("Total Area", 0)
                crop_count = len([k for k in year_data.keys() 
                                if k not in ["Total Area", "unit"] and not k.endswith("(%)")])
                f.write(f"| {year} | {total_area:,.2f} | {crop_count} |\n")
            f.write("\n")
            
            # Top crops for the latest year
            latest_year = years[-1]
            latest_data = cdl_data[latest_year]
            crop_items = [(k, v) for k, v in latest_data.items() 
                         if k not in ["Total Area", "unit"] and not k.endswith("(%)")]
            top_crops = sorted(crop_items, key=lambda x: x[1], reverse=True)[:10]
            
            f.write(f"## Top Crops ({latest_year})\n\n")
            f.write("| Rank | Crop | Area (ha) | Percentage |\n")
            f.write("|------|------|-----------|------------|\n")
            
            total_area = latest_data.get("Total Area", 0)
            for i, (crop, area) in enumerate(top_crops, 1):
                pct = (area / total_area * 100) if total_area > 0 else 0
                f.write(f"| {i} | {crop} | {area:,.2f} | {pct:.2f}% |\n")
            f.write("\n")
            
            # Significant changes
            f.write("## Significant Changes\n\n")
            
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
            
            # Add plots if generated
            if include_plots:
                f.write("## Visualizations\n\n")
                
                f.write("### Crop Area Trends\n\n")
                f.write(f"![Crop Trends]({os.path.basename(trend_plot_path)})\n\n")
                
                f.write("### Crop Area Changes\n\n")
                f.write(f"![Crop Changes]({os.path.basename(change_plot_path)})\n\n")
                
                f.write(f"### Crop Composition ({latest_year})\n\n")
                f.write(f"![Crop Composition]({os.path.basename(pie_plot_path)})\n\n")
            
            # Footer
            f.write("---\n")
            f.write("Report generated automatically. ")
            f.write(f"Complete data available in [{os.path.basename(csv_path)}]({os.path.basename(csv_path)}).\n")
        
        logger.info(f"Report generated successfully at {report_path}")
        return report_path
        
    except Exception as e:
        logger.error(f"Error generating CDL report: {e}", exc_info=True)
        return ""

if __name__ == "__main__":
    print("CDL utilities module loaded. Import to use functions.")
