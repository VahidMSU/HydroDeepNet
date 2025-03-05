"""
gSSURGO soil data analysis and report generation.

This module provides functionality to generate comprehensive reports analyzing
soil data from the Gridded Soil Survey Geographic (gSSURGO) database.
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
from matplotlib import cm
import seaborn as sns

# Add parent directory to path to help with imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from AI_agent.config import AgentConfig
    from AI_agent.gssurgo_utilities import (
        calculate_soil_statistics, plot_soil_parameter_distribution,
        create_soil_correlation_matrix, classify_soil_texture,
        create_soil_texture_pie_chart, create_soil_parameter_maps,
        analyze_soil_fertility, generate_soil_summary,
        export_soil_data_as_csv, analyze_soil_limitations,
        SOIL_PARAMETERS
    )
    from AI_agent.gssurgo_dataset import extract_gssurgo_data
except ImportError:
    try:
        from config import AgentConfig
        from gssurgo_utilities import (
            calculate_soil_statistics, plot_soil_parameter_distribution,
            create_soil_correlation_matrix, classify_soil_texture,
            create_soil_texture_pie_chart, create_soil_parameter_maps,
            analyze_soil_fertility, generate_soil_summary,
            export_soil_data_as_csv, analyze_soil_limitations,
            SOIL_PARAMETERS
        )
        from gssurgo_dataset import extract_gssurgo_data
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
        
        from gssurgo_utilities import (
            calculate_soil_statistics, plot_soil_parameter_distribution,
            create_soil_correlation_matrix, classify_soil_texture,
            create_soil_texture_pie_chart, create_soil_parameter_maps,
            analyze_soil_fertility, generate_soil_summary,
            export_soil_data_as_csv, analyze_soil_limitations,
            SOIL_PARAMETERS
        )
        from gssurgo_dataset import extract_gssurgo_data

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def create_composite_distribution_image(
    soil_data: Dict[str, np.ndarray],
    output_path: str,
    max_params: int = 9
) -> bool:
    """
    Create a composite image with multiple soil parameter distributions.
    
    Args:
        soil_data: Dictionary containing arrays for each soil parameter
        output_path: Path to save the composite image
        max_params: Maximum number of parameters to include
        
    Returns:
        Boolean indicating success
    """
    try:
        # Filter to include only valid soil parameters
        available_params = [p for p in soil_data.keys() if p in SOIL_PARAMETERS]
        
        # Limit to max_params
        if len(available_params) > max_params:
            # Prioritize important parameters
            priority_params = ['ph', 'clay', 'sand', 'silt', 'awc', 'carbon', 'bd', 'dp_tot', 'rock']
            selected_params = []
            
            # Add priority parameters first
            for param in priority_params:
                if param in available_params and len(selected_params) < max_params:
                    selected_params.append(param)
                    
            # Fill remaining slots with other parameters
            for param in available_params:
                if param not in selected_params and len(selected_params) < max_params:
                    selected_params.append(param)
        else:
            selected_params = available_params
        
        # Calculate grid dimensions
        n_params = len(selected_params)
        if n_params <= 3:
            n_rows, n_cols = 1, n_params
        elif n_params <= 6:
            n_rows, n_cols = 2, 3
        else:
            n_rows, n_cols = 3, 3
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3))
        
        # Flatten axes array for easier indexing
        if n_params > 1:
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]
        
        # Plot each parameter distribution
        for i, param in enumerate(selected_params):
            if i >= len(axes_flat):
                break
                
            ax = axes_flat[i]
            data = soil_data[param]
            flat_data = data.flatten()
            clean_data = flat_data[~np.isnan(flat_data)]
            
            if len(clean_data) > 0:
                # Get parameter info
                param_info = SOIL_PARAMETERS.get(param, {'description': param, 'units': ''})
                
                # Plot histogram with KDE
                sns.histplot(clean_data, kde=True, ax=ax)
                
                # Set labels and title
                ax.set_title(f"{param_info['description']}")
                ax.set_xlabel(f"{param_info['units']}")
                
                # Add statistics in text box
                stats_text = (
                    f"Mean: {np.mean(clean_data):.2f}\n"
                    f"Median: {np.median(clean_data):.2f}\n"
                    f"Std: {np.std(clean_data):.2f}"
                )
                
                # Position the text box in the upper right
                props = dict(boxstyle='round', facecolor='white', alpha=0.7)
                ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                        fontsize=8, verticalalignment='top', horizontalalignment='right',
                        bbox=props)
        
        # Hide any unused subplots
        for j in range(i+1, len(axes_flat)):
            axes_flat[j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating composite distribution image: {e}", exc_info=True)
        return False

def create_composite_parameter_maps(
    soil_data: Dict[str, np.ndarray],
    output_path: str,
    max_params: int = 9
) -> bool:
    """
    Create a composite image with multiple soil parameter maps.
    
    Args:
        soil_data: Dictionary containing arrays for each soil parameter
        output_path: Path to save the composite image
        max_params: Maximum number of parameters to include
        
    Returns:
        Boolean indicating success
    """
    try:
        # Filter to include only valid soil parameters
        available_params = [p for p in soil_data.keys() if p in SOIL_PARAMETERS]
        
        # Limit to max_params
        if len(available_params) > max_params:
            # Prioritize important parameters
            priority_params = ['ph', 'clay', 'sand', 'silt', 'awc', 'carbon', 'bd', 'dp_tot', 'rock']
            selected_params = []
            
            # Add priority parameters first
            for param in priority_params:
                if param in available_params and len(selected_params) < max_params:
                    selected_params.append(param)
                    
            # Fill remaining slots with other parameters
            for param in available_params:
                if param not in selected_params and len(selected_params) < max_params:
                    selected_params.append(param)
        else:
            selected_params = available_params
        
        # Calculate grid dimensions
        n_params = len(selected_params)
        if n_params <= 3:
            n_rows, n_cols = 1, n_params
        elif n_params <= 6:
            n_rows, n_cols = 2, 3
        else:
            n_rows, n_cols = 3, 3
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
        
        # Flatten axes array for easier indexing
        if n_params > 1:
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]
        
        # Plot each parameter map
        for i, param in enumerate(selected_params):
            if i >= len(axes_flat):
                break
                
            ax = axes_flat[i]
            data = soil_data[param]
            
            if not np.all(np.isnan(data)):
                # Get parameter info
                param_info = SOIL_PARAMETERS.get(param, {'description': param, 'units': ''})
                
                # Create custom colormap with NaN values as white
                cmap = plt.cm.viridis.copy()
                cmap.set_bad('white', 1.0)
                
                # Plot data
                im = ax.imshow(data, cmap=cmap, interpolation='nearest')
                
                # Add colorbar
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(f"{param_info['units']}", fontsize=8)
                cbar.ax.tick_params(labelsize=6)
                
                # Set title
                ax.set_title(f"{param_info['description']}", fontsize=10)
                
                # Remove axis ticks
                ax.set_xticks([])
                ax.set_yticks([])
        
        # Hide any unused subplots
        for j in range(i+1, len(axes_flat)):
            axes_flat[j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating composite parameter maps: {e}", exc_info=True)
        return False

def generate_soil_report(
    soil_data: Dict[str, np.ndarray], 
    bounding_box: List[float], 
    output_dir: str = 'soil_report',
    advanced_analysis: bool = True
) -> str:
    """
    Generate a comprehensive soil analysis report with visualizations.
    
    Args:
        soil_data: Dictionary containing arrays for each soil parameter
        bounding_box: [min_lon, min_lat, max_lon, max_lat]
        output_dir: Directory to save report files
        advanced_analysis: Whether to include advanced analyses
        
    Returns:
        Path to the generated report file
    """
    if not soil_data:
        logger.warning("No data to generate report")
        return ""
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories
        maps_dir = os.path.join(output_dir, "parameter_maps")
        os.makedirs(maps_dir, exist_ok=True)
        
        distributions_dir = os.path.join(output_dir, "distributions")
        os.makedirs(distributions_dir, exist_ok=True)
        
        # Define file paths
        report_path = os.path.join(output_dir, "soil_report.md")
        texture_path = os.path.join(output_dir, "soil_texture.png")
        correlation_path = os.path.join(output_dir, "soil_correlation.png")
        csv_path = os.path.join(output_dir, "soil_data.csv")
        composite_distributions_path = os.path.join(output_dir, "soil_distributions_composite.png")
        composite_maps_path = os.path.join(output_dir, "soil_maps_composite.png")
        
        # Create individual distributions for reference (saved in the distributions folder)
        for param in soil_data.keys():
            if param in SOIL_PARAMETERS:
                plot_soil_parameter_distribution(
                    soil_data=soil_data,
                    parameter=param,
                    output_path=os.path.join(distributions_dir, f"{param}_distribution.png")
                )
        
        # Generate correlation matrix
        corr_matrix, corr_success = create_soil_correlation_matrix(
            soil_data=soil_data,
            output_path=correlation_path
        )
        
        # Create texture pie chart if texture data available
        if all(x in soil_data for x in ['sand', 'silt', 'clay']):
            texture_pct, texture_success = create_soil_texture_pie_chart(
                soil_data=soil_data,
                output_path=texture_path
            )
        else:
            texture_success = False
        
        # Create individual parameter maps (saved in the parameter_maps folder)
        create_soil_parameter_maps(
            soil_data=soil_data,
            output_dir=maps_dir
        )
        
        # Create composite distribution image
        create_composite_distribution_image(
            soil_data=soil_data,
            output_path=composite_distributions_path
        )
        
        # Create composite parameter maps
        create_composite_parameter_maps(
            soil_data=soil_data,
            output_path=composite_maps_path
        )
        
        # Calculate statistics
        stats = calculate_soil_statistics(soil_data)
        
        # Analyze soil fertility
        fertility = analyze_soil_fertility(soil_data)
        
        # Analyze soil limitations
        limitations = analyze_soil_limitations(soil_data)
        
        # Export data to CSV
        export_soil_data_as_csv(
            soil_data=soil_data,
            output_path=csv_path
        )
        
        # Generate comprehensive soil summary
        summary = generate_soil_summary(soil_data)
        
        # Generate markdown report
        with open(report_path, 'w') as f:
            # Header
            f.write("# gSSURGO Soil Analysis Report\n\n")
            
            # Basic information
            f.write("## Overview\n\n")
            f.write(f"**Region:** Lat [{bounding_box[1]:.4f}, {bounding_box[3]:.4f}], ")
            f.write(f"Lon [{bounding_box[0]:.4f}, {bounding_box[2]:.4f}]\n\n")
            
            # Data availability
            available_params = [p for p in soil_data.keys() if p in SOIL_PARAMETERS]
            f.write("**Available Soil Parameters:**\n\n")
            for param in available_params:
                param_info = SOIL_PARAMETERS[param]
                f.write(f"- {param_info['description']} ({param_info['units']})\n")
            f.write("\n")
            
            # Overall soil characteristics summary
            f.write("## Soil Characteristics Summary\n\n")
            
            # Add texture information if available
            if texture_success and 'texture' in summary:
                dominant_texture = summary['texture']['dominant']
                f.write(f"**Dominant Soil Texture:** {dominant_texture}\n\n")
                f.write("**Texture Composition:**\n\n")
                f.write("![Soil Texture Composition](soil_texture.png)\n\n")
            
            # Add fertility information if available
            if 'fertility' in summary and 'overall' in summary['fertility']:
                f.write(f"**Soil Fertility Level:** {summary['fertility']['overall']['level']}\n\n")
                f.write(f"**Description:** {summary['fertility']['overall']['description']}\n\n")
                
                # Add specific fertility component information
                f.write("### Soil Fertility Components\n\n")
                f.write("| Parameter | Value | Rating | Implication |\n")
                f.write("|-----------|-------|--------|-------------|\n")
                
                if 'ph' in summary['fertility']:
                    ph_info = summary['fertility']['ph']
                    f.write(f"| pH | {ph_info['mean']:.2f} | {ph_info['category']} | {ph_info['implication']} |\n")
                
                if 'carbon' in summary['fertility']:
                    carbon_info = summary['fertility']['carbon']
                    f.write(f"| Organic Carbon | {carbon_info['mean']:.2f}% | {carbon_info['category']} | {carbon_info['implication']} |\n")
                    
                if 'awc' in summary['fertility']:
                    awc_info = summary['fertility']['awc']
                    f.write(f"| Available Water Capacity | {awc_info['mean']:.3f} cm/cm | {awc_info['category']} | {awc_info['implication']} |\n")
                
                if 'texture' in summary['fertility']:
                    texture_info = summary['fertility']['texture']
                    dom_texture = texture_info['dominant_texture']
                    drainage_info = texture_info['drainage']
                    retention_info = texture_info['nutrient_retention']
                    
                    f.write(f"| Soil Texture | {dom_texture} | - | Affects drainage and nutrient retention |\n")
                    f.write(f"| Drainage | - | {drainage_info['category']} | {drainage_info['implication']} |\n")
                    f.write(f"| Nutrient Retention | - | {retention_info['category']} | {retention_info['implication']} |\n")
                
                f.write("\n")
            
            # Soil Parameter Statistics
            f.write("## Soil Parameter Statistics\n\n")
            f.write("The following table presents key statistics for the analyzed soil parameters:\n\n")
            
            f.write("| Parameter | Mean | Median | Min | Max | Std Dev | CV |\n")
            f.write("|-----------|------|--------|-----|-----|---------|----|\n")
            
            for param in available_params:
                if param in stats:
                    param_stats = stats[param]
                    param_info = SOIL_PARAMETERS[param]
                    
                    # Only include parameters with valid data
                    if not np.isnan(param_stats['mean']):
                        f.write(f"| {param_info['description']} | {param_stats['mean']:.3f} | ")
                        f.write(f"{param_stats['median']:.3f} | {param_stats['min']:.3f} | ")
                        f.write(f"{param_stats['max']:.3f} | {param_stats['std']:.3f} | ")
                        f.write(f"{param_stats['cv']:.3f} |\n")
            
            f.write("\n")
            f.write("*CV: Coefficient of Variation (Std Dev / Mean)*\n\n")
            
            # Add composite parameter distributions
            f.write("## Soil Parameter Distributions\n\n")
            f.write("The distributions of key soil parameters are shown below:\n\n")
            f.write(f"![Soil Parameter Distributions](soil_distributions_composite.png)\n\n")
            f.write("*Individual parameter distributions can be found in the 'distributions' folder.*\n\n")
            
            # Add composite parameter maps
            f.write("## Spatial Distribution Maps\n\n") 
            f.write("The spatial distribution of key soil parameters across the study area is shown below:\n\n")
            f.write(f"![Soil Parameter Maps](soil_maps_composite.png)\n\n")
            f.write("*Individual parameter maps can be found in the 'parameter_maps' folder.*\n\n")
            
            # Soil Limitations section
            if limitations:
                f.write("## Soil Limitations\n\n")
                for lim_type, lim_data in limitations.items():
                    if lim_type == 'shallow_depth':
                        f.write("### Soil Depth Limitations\n\n")
                        f.write(f"- **Mean soil depth:** {lim_data['mean_depth_cm']:.1f} cm\n")
                        f.write(f"- **Areas with shallow soil (<50cm):** {lim_data['shallow_areas_pct']:.1f}%\n")
                        f.write(f"- **Limitation level:** {lim_data['limitation_level']}\n")
                        f.write(f"- **Recommendation:** {lim_data['recommendation']}\n\n")
                        
                    elif lim_type == 'rockiness':
                        f.write("### Rock Content Limitations\n\n")
                        f.write(f"- **Mean rock content:** {lim_data['mean_rock_content_pct']:.1f}%\n")
                        f.write(f"- **Areas with high rock content (>15%):** {lim_data['high_rock_areas_pct']:.1f}%\n")
                        f.write(f"- **Limitation level:** {lim_data['limitation_level']}\n")
                        f.write(f"- **Recommendation:** {lim_data['recommendation']}\n\n")
                        
                    elif lim_type == 'drainage':
                        f.write("### Drainage Limitations\n\n")
                        f.write(f"- **Areas with poor drainage:** {lim_data['poor_drainage_areas_pct']:.1f}%\n")
                        f.write(f"- **Limitation level:** {lim_data['limitation_level']}\n")
                        f.write(f"- **Recommendation:** {lim_data['recommendation']}\n\n")
                        
                    elif lim_type == 'ph_extremes':
                        f.write("### pH Limitations\n\n")
                        f.write(f"- **Areas with acidic soils (pH<5.5):** {lim_data['acidic_areas_pct']:.1f}%\n")
                        f.write(f"- **Areas with alkaline soils (pH>7.5):** {lim_data['alkaline_areas_pct']:.1f}%\n")
                        f.write(f"- **Limitation level:** {lim_data['limitation_level']}\n")
                        f.write(f"- **Recommendation:** {lim_data['recommendation']}\n\n")
                        
                    elif lim_type == 'salinity':
                        f.write("### Salinity Limitations\n\n")
                        f.write(f"- **Mean electrical conductivity:** {lim_data['mean_ec_ds_m']:.2f} dS/m\n")
                        f.write(f"- **Areas with high salinity (>4 dS/m):** {lim_data['high_salinity_areas_pct']:.1f}%\n")
                        f.write(f"- **Limitation level:** {lim_data['limitation_level']}\n")
                        f.write(f"- **Recommendation:** {lim_data['recommendation']}\n\n")
            
            # Soil Management Recommendations
            if 'recommendations' in summary and summary['recommendations']:
                f.write("## Soil Management Recommendations\n\n")
                f.write("Based on the analysis of soil properties, the following management practices are recommended:\n\n")
                
                for i, recommendation in enumerate(summary['recommendations'], 1):
                    f.write(f"{i}. {recommendation}\n")
                
                f.write("\n")
            
            # Parameter Correlations
            if corr_success:
                f.write("## Parameter Correlations\n\n")
                f.write("The correlation matrix below shows relationships between soil parameters. ")
                f.write("Strong positive correlations appear in red, while strong negative correlations appear in blue.\n\n")
                f.write("![Soil Parameter Correlation Matrix](soil_correlation.png)\n\n")
                
                # Add interpretation of key correlations if available
                if isinstance(corr_matrix, pd.DataFrame):
                    # Find strongest positive and negative correlations
                    corr_pairs = []
                    for i, param1 in enumerate(corr_matrix.columns):
                        for j, param2 in enumerate(corr_matrix.columns):
                            if i < j:  # Upper triangle only
                                corr_value = corr_matrix.iloc[i, j]
                                if not np.isnan(corr_value) and abs(corr_value) > 0.5:
                                    corr_pairs.append((param1, param2, corr_value))
                    
                    # Sort by absolute correlation value
                    corr_pairs = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)
                    
                    if corr_pairs:
                        f.write("### Key Parameter Relationships\n\n")
                        f.write("| Parameter 1 | Parameter 2 | Correlation | Interpretation |\n")
                        f.write("|------------|------------|-------------|---------------|\n")
                        
                        for param1, param2, corr_value in corr_pairs[:5]:  # Show top 5 correlations
                            if param1 in SOIL_PARAMETERS and param2 in SOIL_PARAMETERS:
                                desc1 = SOIL_PARAMETERS[param1]['description']
                                desc2 = SOIL_PARAMETERS[param2]['description']
                                
                                # Generate interpretation
                                if corr_value > 0.7:
                                    interp = "Strong positive relationship"
                                elif corr_value > 0.5:
                                    interp = "Moderate positive relationship"
                                elif corr_value < -0.7:
                                    interp = "Strong negative relationship"
                                elif corr_value < -0.5:
                                    interp = "Moderate negative relationship"
                                else:
                                    interp = "Weak relationship"
                                
                                f.write(f"| {desc1} | {desc2} | {corr_value:.2f} | {interp} |\n")
                        
                        f.write("\n")
            
            # Data Source and Methodology
            f.write("## Data Source and Methodology\n\n")
            f.write("This report is based on the Gridded Soil Survey Geographic (gSSURGO) database, ")
            f.write("which provides soil information at a 250m resolution. The gSSURGO database ")
            f.write("combines data from soil surveys conducted by the USDA Natural Resources Conservation Service (NRCS).\n\n")
            
            f.write("**Processing steps:**\n\n")
            f.write("1. Extraction of soil data for the specified region\n")
            f.write("2. Statistical analysis of soil parameters\n")
            f.write("3. Soil texture classification based on sand, silt, and clay content\n")
            f.write("4. Assessment of soil fertility and limitations\n")
            f.write("5. Generation of visualizations and spatial maps\n\n")
            
            # Limitations
            f.write("### Limitations\n\n")
            f.write("- The analysis is based on the resolution of the gSSURGO data (250m), which may not capture fine-scale soil variations\n")
            f.write("- Some parameters may have missing data in certain areas\n")
            f.write("- Local soil conditions may vary and on-site testing is recommended for specific applications\n\n")
            
            # Data Export
            f.write("## Data Export\n\n")
            f.write(f"The complete dataset has been exported to CSV format for further analysis: ")
            f.write(f"[{os.path.basename(csv_path)}]({os.path.basename(csv_path)})\n\n")
            
            # Report generation information
            f.write("---\n\n")
            f.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M')}*\n")
        
        logger.info(f"Report successfully generated: {report_path}")
        return report_path
        
    except Exception as e:
        logger.error(f"Error generating soil report: {e}", exc_info=True)
        return ""

def process_soil_data(config: Dict[str, Any], output_dir: str) -> str:
    """
    Process soil data and generate a comprehensive report.
    
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
        advanced_analysis = config.get('advanced_analysis', True)
        
        if not bounding_box:
            logger.error("Bounding box not specified in configuration")
            return ""
        
        # Extract gSSURGO data
        logger.info(f"Extracting gSSURGO data for bounding box: {bounding_box}")
        soil_data = extract_gssurgo_data(bounding_box)
        
        if not soil_data:
            logger.error("Failed to extract gSSURGO data")
            return ""
        
        # Generate report
        logger.info("Generating soil report")
        report_path = generate_soil_report(
            soil_data=soil_data,
            bounding_box=bounding_box,
            output_dir=output_dir,
            advanced_analysis=advanced_analysis
        )
        
        return report_path
        
    except Exception as e:
        logger.error(f"Error processing soil data: {e}", exc_info=True)
        return ""

def export_soil_summary(soil_data: Dict[str, np.ndarray], output_dir: str) -> str:
    """
    Export a simple soil data summary with basic visualizations.
    
    Args:
        soil_data: Dictionary of soil parameter arrays
        output_dir: Directory to save outputs
        
    Returns:
        Path to the exported CSV file
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Define file paths
        csv_path = os.path.join(output_dir, "soil_summary.csv")
        texture_path = os.path.join(output_dir, "soil_texture.png")
        
        # Calculate statistics
        stats = calculate_soil_statistics(soil_data)
        
        # Export data to CSV
        export_success = export_soil_data_as_csv(
            soil_data=soil_data,
            output_path=csv_path
        )
        
        # Create soil texture visualization if possible
        if all(x in soil_data for x in ['sand', 'silt', 'clay']):
            texture_success = create_soil_texture_pie_chart(
                soil_data=soil_data,
                output_path=texture_path
            )[1]
        else:
            texture_success = False
        
        # Create statistics summary CSV
        stats_path = os.path.join(output_dir, "soil_statistics.csv")
        stats_df = pd.DataFrame()
        
        for param, param_stats in stats.items():
            if param in SOIL_PARAMETERS:
                param_info = SOIL_PARAMETERS[param]
                param_df = pd.DataFrame({
                    'Parameter': [param_info['description']],
                    'Units': [param_info['units']],
                    'Mean': [param_stats['mean']],
                    'Median': [param_stats['median']],
                    'Min': [param_stats['min']],
                    'Max': [param_stats['max']],
                    'StdDev': [param_stats['std']],
                    'CV': [param_stats['cv']]
                })
                stats_df = pd.concat([stats_df, param_df], ignore_index=True)
        
        if not stats_df.empty:
            stats_df.to_csv(stats_path, index=False)
        
        return csv_path if export_success else ""
        
    except Exception as e:
        logger.error(f"Error exporting soil summary: {e}", exc_info=True)
        return ""

if __name__ == "__main__":
    # Example usage
    try:
        config = {
            "bounding_box": [-85.444332, 43.158148, -84.239256, 44.164683]
        }
        
        # Process data and generate report
        report_path = process_soil_data(
            config=config,
            output_dir="soil_results"
        )
        
        if report_path:
            print(f"Soil report generated successfully: {report_path}")
        else:
            print("Failed to generate soil report")
            
    except Exception as e:
        logger.error(f"Error in example execution: {e}", exc_info=True)