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
try:
    from utils.plot_utils import safe_figure, save_figure, close_all_figures
except ImportError:
    from GeoReporter.utils.plot_utils import safe_figure, save_figure, close_all_figures
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
    title: str = 'Landcover Distribution Over Time',
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

def create_crop_composition_pie(cdl_data: Dict[int, Dict[str, Any]], year: int, 
                              output_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (10, 8)) -> bool:
    """
    Create a pie chart showing crop composition for a specific year.
    
    Args:
        cdl_data: Dictionary mapping years to land use data
        year: Year to visualize
        output_path: Path to save the figure (optional)
        figsize: Figure dimensions (width, height)
        
    Returns:
        Boolean indicating success
    """
    try:
        # Make sure any previous figures are closed
        close_all_figures()
        
        if year not in cdl_data:
            logger.warning(f"Year {year} not found in data")
            return False
        
        # Get data for the specified year
        year_data = cdl_data[year]
        total_area = year_data.get("Total Area", 0)
        
        if total_area <= 0:
            logger.warning(f"No area data for year {year}")
            return False
        
        # Filter items that are numeric values and not metadata or percentages
        items = [(k, v) for k, v in year_data.items() 
                if (isinstance(v, (int, float)) and 
                    k not in ["Total Area", "unit"] and 
                    not k.endswith("(%)") and
                    v > 0)]
        
        if not items:
            logger.warning(f"No crop data for year {year}")
            return False
        
        # Sort by area (descending) and get top items
        items = sorted(items, key=lambda x: x[1], reverse=True)
        top_items = items[:8]  # Top 8 crops
        
        # Combine remaining items into "Other"
        other_area = sum(v for _, v in items[8:])
        if other_area > 0:
            top_items.append(("Other", other_area))
        
        # Extract labels and values
        labels = [k for k, _ in top_items]
        values = [v for _, v in top_items]
        
        # Calculate percentages for labels
        percentages = [v / total_area * 100 for v in values]
        labels = [f"{l} ({p:.1f}%)" for l, p in zip(labels, percentages)]
        
        # Create a unique figure
        with safe_figure(figsize=figsize) as fig:
            ax = fig.add_subplot(111)
            
            # Create pie chart
            explode = [0.05 if i == 0 else 0 for i in range(len(top_items))]
            wedges, texts, autotexts = ax.pie(
                values, 
                explode=explode, 
                labels=None,  # We'll add a legend instead
                autopct=lambda p: f'{p:.1f}%' if p > 3 else '',
                shadow=False, 
                startangle=90,
                wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
            )
            
            # Add legend
            ax.legend(wedges, labels, title="Crop Types", 
                    loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
            
            # Set title
            ax.set_title(f"Crop Composition in {year}", fontsize=14, fontweight='bold')
            
            # Ensure the pie chart is drawn as a circle
            ax.axis('equal')
            
            # Set tight layout
            fig.tight_layout()
            
            # Save if path provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                # Save with explicit path that includes CDL identifier to avoid conflicts
                unique_path = output_path  
                save_figure(fig, unique_path)
                logger.info(f"Crop composition pie chart saved to {unique_path}")
        
        # Make sure figure is closed
        close_all_figures()
        return True
        
    except Exception as e:
        logger.error(f"Error creating crop composition pie: {e}", exc_info=True)
        close_all_figures()  # Make sure to clean up
        return False

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

def create_crop_diversity_plot(
    cdl_data: Dict[int, Dict[str, Any]],
    metric: str = "shannon",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> Optional[plt.Figure]:
    """
    Create a plot showing crop diversity metrics over time.
    
    Args:
        cdl_data: Dictionary mapping years to land use data
        metric: Diversity metric to use ('shannon' or 'richness')
        output_path: Path to save the figure (optional)
        figsize: Figure dimensions (width, height) in inches
        
    Returns:
        Matplotlib Figure object or None if error occurs
    """
    try:
        if not cdl_data:
            logger.warning("No data to plot diversity metrics")
            return None
            
        # Extract available years in ascending order
        years = sorted(cdl_data.keys())
        
        # Calculate diversity metrics for each year
        diversity_values = []
        richness_values = []
        
        for year in years:
            year_data = cdl_data[year]
            # Get crop data (excluding metadata fields)
            crop_data = {
                k: v for k, v in year_data.items() 
                if k not in ["Total Area", "unit"] and not k.endswith("(%)")
            }
            
            # Skip if no crops found
            if not crop_data:
                diversity_values.append(0)
                richness_values.append(0)
                continue
            
            # Calculate species richness (number of crop types)
            richness = len(crop_data)
            richness_values.append(richness)
            
            # Calculate Shannon diversity index
            total_area = sum(crop_data.values())
            if total_area > 0:
                proportions = [area / total_area for area in crop_data.values()]
                shannon = -sum(p * np.log(p) for p in proportions if p > 0)
                diversity_values.append(shannon)
            else:
                diversity_values.append(0)
        
        # Create plot
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # Plot the selected metric
        if metric.lower() == "shannon":
            color = "tab:blue"
            ax1.set_ylabel("Shannon Diversity Index", fontsize=12, color=color)
            line1 = ax1.plot(years, diversity_values, marker='o', linestyle='-', 
                     color=color, linewidth=2, markersize=8)
            ax1.tick_params(axis='y', labelcolor=color)
            y_values = diversity_values
            metric_name = "Shannon Diversity Index"
        else:
            color = "tab:green"
            ax1.set_ylabel("Crop Richness (Count)", fontsize=12, color=color)
            line1 = ax1.plot(years, richness_values, marker='o', linestyle='-', 
                     color=color, linewidth=2, markersize=8)
            ax1.tick_params(axis='y', labelcolor=color)
            y_values = richness_values
            metric_name = "Crop Richness"
        
        # Add a secondary axis for the other metric if they differ significantly
        if metric.lower() == "shannon" and max(richness_values) > 0:
            ax2 = ax1.twinx()
            color = "tab:green"
            ax2.set_ylabel("Crop Richness (Count)", fontsize=12, color=color)
            line2 = ax2.plot(years, richness_values, marker='s', linestyle='--', 
                     color=color, linewidth=1.5, markersize=6)
            ax2.tick_params(axis='y', labelcolor=color)
            ax1.legend(line1 + line2, ["Shannon Index", "Crop Count"], loc="upper left")
        
        # Set x-axis label and title
        ax1.set_xlabel("Year", fontsize=12)
        ax1.set_title(f"Crop Diversity Trends ({years[0]}-{years[-1]})", fontsize=14, fontweight='bold')
        
        # Make x-axis ticks for all years
        ax1.set_xticks(years)
        ax1.set_xticklabels(years, rotation=45)
        
        # Add grid
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Add data labels
        for i, (year, val) in enumerate(zip(years, y_values)):
            ax1.annotate(f"{val:.2f}", (year, val), 
                        textcoords="offset points", 
                        xytext=(0, 10), 
                        ha='center',
                        fontsize=9)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Diversity plot saved to {output_path}")
            
        return fig
        
    except Exception as e:
        logger.error(f"Error creating crop diversity plot: {e}", exc_info=True)
        return None

def create_crop_rotation_heatmap(
    cdl_data: Dict[int, Dict[str, Any]],
    top_n: int = 10,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> Optional[plt.Figure]:
    """
    Create a heatmap showing crop rotation patterns between consecutive years.
    
    Args:
        cdl_data: Dictionary mapping years to land use data
        top_n: Number of top crops to include in the rotation analysis
        output_path: Path to save the figure (optional)
        figsize: Figure dimensions (width, height) in inches
        
    Returns:
        Matplotlib Figure object or None if error occurs
    """
    try:
        if len(cdl_data) < 2:
            logger.warning("Need at least two consecutive years of data to analyze rotations")
            return None
            
        # Get consecutive years only
        years = sorted(cdl_data.keys())
        
        # Find top N crops across all years
        all_crops = {}
        for year_data in cdl_data.values():
            for crop, area in year_data.items():
                if crop not in ["Total Area", "unit"] and not crop.endswith("(%)"):
                    all_crops[crop] = all_crops.get(crop, 0) + area
        
        top_crops = [crop for crop, _ in sorted(all_crops.items(), 
                                              key=lambda x: x[1], 
                                              reverse=True)[:top_n]]
        
        # Initialize rotation matrix
        rotation_matrix = np.zeros((len(top_crops), len(top_crops)))
        
        # Count transitions between consecutive years
        for i in range(len(years) - 1):
            year1, year2 = years[i], years[i+1]
            year1_data = cdl_data[year1]
            year2_data = cdl_data[year2]
            
            # Build rotation count matrix for top crops
            for from_idx, from_crop in enumerate(top_crops):
                from_area = year1_data.get(from_crop, 0)
                if from_area == 0:
                    continue
                    
                for to_idx, to_crop in enumerate(top_crops):
                    to_area = year2_data.get(to_crop, 0)
                    if to_area == 0:
                        continue
                    
                    # Estimate rotation based on relative areas
                    # This is a simplification since we don't have parcel-level data
                    rotation_area = min(from_area, to_area)
                    rotation_matrix[from_idx, to_idx] += rotation_area
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        # Normalize data for better visualization (percentage of source crop)
        normalized_matrix = np.zeros_like(rotation_matrix)
        row_sums = rotation_matrix.sum(axis=1)
        
        for i in range(rotation_matrix.shape[0]):
            if row_sums[i] > 0:
                normalized_matrix[i, :] = rotation_matrix[i, :] / row_sums[i] * 100
        
        # Create heatmap with customized colormap
        cmap = plt.cm.YlOrRd
        im = ax.imshow(normalized_matrix, cmap=cmap, aspect='auto')
        
        # Customize colorbar
        cbar = fig.colorbar(im, ax=ax, pad=0.01)
        cbar.set_label('Rotation Percentage (%)', rotation=270, labelpad=15)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(top_crops)))
        ax.set_yticks(np.arange(len(top_crops)))
        ax.set_xticklabels(top_crops, rotation=45, ha='right')
        ax.set_yticklabels(top_crops)
        
        # Add labels and title
        ax.set_xlabel('Following Year Crop', fontsize=12)
        ax.set_ylabel('Previous Year Crop', fontsize=12)
        ax.set_title(f'Crop Rotation Patterns ({years[0]}-{years[-1]})', 
                    fontsize=14, fontweight='bold')
        
        # Add text annotations for significant rotations (>10%)
        threshold = 10.0
        for i in range(len(top_crops)):
            for j in range(len(top_crops)):
                if normalized_matrix[i, j] > threshold:
                    ax.text(j, i, f"{normalized_matrix[i, j]:.1f}%", 
                           ha="center", va="center", 
                           color="white" if normalized_matrix[i, j] > 50 else "black",
                           fontsize=9)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Crop rotation heatmap saved to {output_path}")
            
        return fig
        
    except Exception as e:
        logger.error(f"Error creating crop rotation heatmap: {e}", exc_info=True)
        return None

def calculate_agricultural_intensity(
    cdl_data: Dict[int, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculate metrics of agricultural intensity from CDL data.
    
    Args:
        cdl_data: Dictionary mapping years to land use data
        
    Returns:
        Dictionary of agricultural intensity metrics
    """
    try:
        if not cdl_data:
            logger.warning("No data to calculate agricultural intensity")
            return {}
            
        years = sorted(cdl_data.keys())
        
        # Define crop categories
        row_crops = ["Corn", "Soybeans", "Cotton", "Rice", "Sunflower", "Peanuts", "Tobacco", "Sweet Corn",
                     "Sorghum", "Barley", "Millet", "Speltz", "Canola", "Flaxseed", "Safflower", 
                     "Mustard", "Sugarcane"]
                     
        small_grains = ["Durum Wheat", "Spring Wheat", "Winter Wheat", "Other Small Grains", "Rye", "Oats"]
        
        perennial_crops = ["Alfalfa", "Other Hay/Non Alfalfa", "Clover/Wildflowers", 
                         "Sod/Grass Seed", "Switchgrass", "Fallow/Idle Cropland", 
                         "Grassland/Pasture", "Woody Wetlands", "Herbaceous Wetlands"]
                         
        specialty_crops = ["Fruits", "Vegetables", "Berries", "Cherries", "Peaches", "Apples", 
                          "Grapes", "Christmas Trees", "Other Tree Crops", "Citrus", 
                          "Pecans", "Almonds", "Walnuts", "Pistachios", "Oranges",
                          "Pears", "Blueberries", "Cranberries"]
        
        # Initialize metrics
        metrics = {
            "Agricultural Intensity Index": 0.0,
            "Row Crop Percentage": 0.0,
            "Small Grains Percentage": 0.0,
            "Perennial Cover Percentage": 0.0,
            "Specialty Crops Percentage": 0.0,
            "Dominant Crop": "",
            "Crop Diversity (Shannon)": 0.0
        }
        
        # Calculate metrics for most recent year
        latest_year = years[-1]
        latest_data = cdl_data[latest_year]
        
        # Extract crop areas (excluding metadata)
        crop_data = {
            k: v for k, v in latest_data.items() 
            if k not in ["Total Area", "unit"] and not k.endswith("(%)")
        }
        
        total_area = sum(crop_data.values())
        if total_area == 0:
            logger.warning("No crop area data available for analysis")
            return metrics
        
        # Calculate row crop percentage
        row_crop_area = sum(crop_data.get(crop, 0) for crop in row_crops)
        row_crop_pct = row_crop_area / total_area * 100
        metrics["Row Crop Percentage"] = row_crop_pct
        
        # Calculate small grains percentage
        small_grain_area = sum(crop_data.get(crop, 0) for crop in small_grains)
        small_grain_pct = small_grain_area / total_area * 100
        metrics["Small Grains Percentage"] = small_grain_pct
        
        # Calculate perennial cover percentage
        perennial_area = sum(crop_data.get(crop, 0) for crop in perennial_crops)
        perennial_pct = perennial_area / total_area * 100
        metrics["Perennial Cover Percentage"] = perennial_pct
        
        # Calculate specialty crop percentage
        specialty_area = sum(crop_data.get(crop, 0) for crop in specialty_crops)
        specialty_pct = specialty_area / total_area * 100
        metrics["Specialty Crops Percentage"] = specialty_pct
        
        # Find dominant crop
        if crop_data:
            dominant_crop = max(crop_data.items(), key=lambda x: x[1])
            metrics["Dominant Crop"] = f"{dominant_crop[0]} ({dominant_crop[1]/total_area*100:.1f}%)"
        
        # Calculate Shannon diversity
        proportions = [area / total_area for area in crop_data.values() if area > 0]
        shannon_index = -sum(p * np.log(p) for p in proportions)
        metrics["Crop Diversity (Shannon)"] = shannon_index
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating agricultural intensity: {e}", exc_info=True)
        return {}

def get_crop_categories(
    cdl_data: Dict[int, Dict[str, Any]],
    year: Optional[int] = None,
    custom_categories: Optional[Dict[str, List[str]]] = None
) -> Dict[str, float]:
    """
    Categorize land cover types into logical groups and calculate area for each category.
    
    Args:
        cdl_data: Dictionary mapping years to land use data
        year: Year to analyze (uses latest year if None)
        custom_categories: Optional custom category definitions
        
    Returns:
        Dictionary mapping category names to total areas
    """
    try:
        if not cdl_data:
            logger.warning("No data to categorize")
            return {}
            
        # Select year (use latest if not specified)
        if year is None or year not in cdl_data:
            year = max(cdl_data.keys())
            
        year_data = cdl_data[year]
        
        # Define default land cover categories
        default_categories = {
            "Row Crops": [
                "Corn", "Soybeans", "Cotton", "Rice", "Sunflower", "Peanuts", 
                "Tobacco", "Sweet Corn", "Popcorn or Ornamental Corn", "Sorghum", 
                "Barley", "Millet", "Speltz", "Canola", "Flaxseed", "Safflower", 
                "Mustard", "Sugarcane", "Sugar Beets"
            ],
            "Small Grains": [
                "Durum Wheat", "Spring Wheat", "Winter Wheat", "Other Small Grains", 
                "Rye", "Oats", "Triticale", "Buckwheat"
            ],
            "Perennial & Forage": [
                "Alfalfa", "Other Hay/Non Alfalfa", "Clover/Wildflowers", 
                "Sod/Grass Seed", "Switchgrass", "Fallow/Idle Cropland", 
                "Grassland/Pasture", "Pasture/Hay", "Pasture/Grass"
            ],
            "Fruits & Nuts": [
                "Cherries", "Peaches", "Apples", "Grapes", "Citrus", 
                "Pecans", "Almonds", "Walnuts", "Pistachios", "Oranges",
                "Pears", "Plums", "Olives", "Avocados", "Nectarines", 
                "Prunes", "Pomegranates", "Kiwi", "Apricots", "Other Tree Fruits",
                "Blueberries", "Cranberries", "Other Tree Crops",
                "Strawberries", "Raspberries", "Other Berries"
            ],
            "Vegetables": [
                "Vegetables", "Potatoes", "Onions", "Tomatoes", "Peppers", 
                "Lettuce", "Broccoli", "Carrots", "Celery", "Radishes", 
                "Cucumbers", "Greens", "Garlic", "Beans", "Squash", 
                "Asparagus", "Watermelons", "Pumpkins", "Cabbage"
            ],
            "Specialty Crops": [
                "Herbs", "Christmas Trees", "Hops", "Mint", "Ginger", 
                "Pineapple", "Flowers", "Gourds", "Coffee", "Ginseng",
                "Nursery", "Greenhouse"
            ],
            "Forest & Woodland": [
                "Forest", "Mixed Forest", "Deciduous Forest", "Evergreen Forest",
                "Shrubland", "Woody Wetlands"
            ],
            "Wetlands & Water": [
                "Wetlands", "Herbaceous Wetlands", "Aquaculture", "Open Water",
                "Water"
            ],
            "Developed & Other": [
                "Developed", "Developed/Open Space", "Developed/Low Intensity",
                "Developed/Med Intensity", "Developed/High Intensity",
                "Barren", "Fallow"
            ]
        }
        
        # Use custom categories if provided, otherwise use default
        categories = custom_categories if custom_categories is not None else default_categories
        
        # Get land cover data (excluding metadata fields)
        land_cover_data = {
            k: v for k, v in year_data.items() 
            if k not in ["Total Area", "unit"] and not k.endswith("(%)")
        }
        
        if not land_cover_data:
            logger.warning(f"No land cover data found for year {year}")
            return {}
            
        # Calculate area for each category
        result = {}
        for category, cover_list in categories.items():
            category_area = 0.0
            for cover_type in cover_list:
                # Look for exact matches as well as partial matches (for flexibility)
                exact_match = land_cover_data.get(cover_type, 0)
                category_area += exact_match
                
                # Add partial matches if applicable
                if exact_match == 0:
                    for cover_name, area in land_cover_data.items():
                        if cover_type in cover_name and cover_name != cover_type:
                            category_area += area
            
            if category_area > 0:
                result[category] = category_area
        
        # Create an "Other" category for crops not captured in the above categories
        categorized_crops = set()
        for crop_list in categories.values():
            categorized_crops.update(crop_list)
            
        other_area = sum(area for crop, area in land_cover_data.items() 
                        if not any(cat_crop in crop for cat_crop in categorized_crops))
        
        if other_area > 0:
            result["Other"] = other_area
            
        return result
        
    except Exception as e:
        logger.error(f"Error categorizing crops: {e}", exc_info=True)
        return {}

if __name__ == "__main__":
    print("CDL utilities module loaded. Import to use functions.")
