"""
gSSURGO soil data analysis utilities.

This module provides functions for analyzing and visualizing soil data from
the Gridded Soil Survey Geographic (gSSURGO) database.
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Define soil parameter descriptions and units
SOIL_PARAMETERS = {
    'alb': {'description': 'Albedo', 'units': 'fraction', 'range': [0, 1]},
    'awc': {'description': 'Available Water Capacity', 'units': 'cm/cm', 'range': [0, 1]},
    'bd': {'description': 'Bulk Density', 'units': 'g/cm³', 'range': [0.5, 2.0]},
    'caco3': {'description': 'Calcium Carbonate', 'units': '%', 'range': [0, 100]},
    'carbon': {'description': 'Organic Carbon', 'units': '%', 'range': [0, 20]},
    'clay': {'description': 'Clay Content', 'units': '%', 'range': [0, 100]},
    'dp': {'description': 'Depth', 'units': 'cm', 'range': [0, 200]},
    'dp_tot': {'description': 'Total Soil Depth', 'units': 'cm', 'range': [0, 200]},
    'ec': {'description': 'Electrical Conductivity', 'units': 'dS/m', 'range': [0, 20]},
#    'gSURRGO_swat_250m': {'description': 'SWAT Soil Classification', 'units': 'class', 'range': [1, 10000]},
    'ph': {'description': 'Soil pH', 'units': 'pH', 'range': [3, 10]},
    'rock': {'description': 'Rock Fragment Content', 'units': '%', 'range': [0, 100]},
    'sand': {'description': 'Sand Content', 'units': '%', 'range': [0, 100]},
    'silt': {'description': 'Silt Content', 'units': '%', 'range': [0, 100]},
    'soil_k': {'description': 'Hydraulic Conductivity', 'units': 'mm/hr', 'range': [0, 1000]}
}

def calculate_soil_statistics(soil_data: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """
    Calculate basic statistics for each soil parameter.
    
    Args:
        soil_data: Dictionary of soil parameter arrays
        
    Returns:
        Dictionary with statistics for each parameter
    """
    stats_dict = {}
    
    for param, data in soil_data.items():
        if param in SOIL_PARAMETERS:
            # Convert to 1D array and remove NaN values
            flat_data = data.flatten()
            clean_data = flat_data[~np.isnan(flat_data)]
            
            if len(clean_data) > 0:
                stats_dict[param] = {
                    'mean': float(np.mean(clean_data)),
                    'median': float(np.median(clean_data)),
                    'min': float(np.min(clean_data)),
                    'max': float(np.max(clean_data)),
                    'std': float(np.std(clean_data)),
                    'cv': float(np.std(clean_data) / np.mean(clean_data) if np.mean(clean_data) != 0 else 0),
                    'q25': float(np.percentile(clean_data, 25)),
                    'q75': float(np.percentile(clean_data, 75)),
                    'valid_cells': int(len(clean_data)),
                    'nan_cells': int(np.sum(np.isnan(flat_data)))
                }
            else:
                stats_dict[param] = {
                    'mean': np.nan, 'median': np.nan, 'min': np.nan, 'max': np.nan,
                    'std': np.nan, 'cv': np.nan, 'q25': np.nan, 'q75': np.nan,
                    'valid_cells': 0, 'nan_cells': int(len(flat_data))
                }
    
    return stats_dict

def plot_soil_parameter_distribution(
    soil_data: Dict[str, np.ndarray],
    parameter: str,
    output_path: str = None,
    title: str = None
) -> bool:
    """
    Create histogram of soil parameter distribution.
    
    Args:
        soil_data: Dictionary of soil parameter arrays
        parameter: Soil parameter to plot
        output_path: Path to save the plot
        title: Custom title for the plot
        
    Returns:
        Boolean indicating success
    """
    if parameter not in soil_data:
        logger.warning(f"Parameter '{parameter}' not found in soil data")
        return False
        
    try:
        data = soil_data[parameter]
        flat_data = data.flatten()
        clean_data = flat_data[~np.isnan(flat_data)]
        
        if len(clean_data) == 0:
            logger.warning(f"No valid data for parameter '{parameter}'")
            return False
            
        param_info = SOIL_PARAMETERS.get(parameter, 
                                        {'description': parameter, 'units': ''})
        
        plt.figure(figsize=(10, 6))
        sns.histplot(clean_data, kde=True)
        
        if title:
            plt.title(title)
        else:
            plt.title(f"Distribution of {param_info['description']}")
            
        plt.xlabel(f"{param_info['description']} ({param_info['units']})")
        plt.ylabel("Frequency")
        
        # Add statistics to the plot
        stats_text = (
            f"Mean: {np.mean(clean_data):.2f}\n"
            f"Median: {np.median(clean_data):.2f}\n"
            f"Std Dev: {np.std(clean_data):.2f}\n"
            f"Min: {np.min(clean_data):.2f}\n"
            f"Max: {np.max(clean_data):.2f}"
        )
        
        # Position the text box in the upper right in axes coords
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=props)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300)
            plt.close()
            return True
        else:
            plt.show()
            plt.close()
            return True
            
    except Exception as e:
        logger.error(f"Error plotting soil parameter distribution: {e}", exc_info=True)
        return False

def create_soil_correlation_matrix(
    soil_data: Dict[str, np.ndarray],
    output_path: str = None
) -> Tuple[Optional[pd.DataFrame], bool]:
    """
    Create correlation matrix for soil parameters.
    
    Args:
        soil_data: Dictionary of soil parameter arrays
        output_path: Path to save the correlation matrix plot
        
    Returns:
        Tuple of (correlation dataframe, success boolean)
    """
    try:
        # Convert soil data to pandas DataFrame
        data_dict = {}
        for param, array in soil_data.items():
            if param in SOIL_PARAMETERS:
                flat_data = array.flatten()
                data_dict[param] = flat_data
        
        # Create DataFrame with aligned indices
        df = pd.DataFrame(data_dict)
        
        # Calculate correlation matrix
        corr_matrix = df.corr(method='pearson')
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
        plt.title("Soil Parameter Correlation Matrix")
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300)
            plt.close()
        else:
            plt.show()
            plt.close()
            
        return corr_matrix, True
        
    except Exception as e:
        logger.error(f"Error creating soil correlation matrix: {e}", exc_info=True)
        return None, False

def classify_soil_texture(
    sand_pct: np.ndarray,
    silt_pct: np.ndarray,
    clay_pct: np.ndarray
) -> np.ndarray:
    """
    Classify soil texture based on USDA soil texture triangle.
    
    Args:
        sand_pct: Sand content percentage array
        silt_pct: Silt content percentage array
        clay_pct: Clay content percentage array
        
    Returns:
        Array of soil texture classifications
    """
    # Create output array with same shape as inputs
    texture_class = np.full(sand_pct.shape, "Unknown", dtype=object)
    
    # Flatten arrays for element-wise processing
    sand_flat = sand_pct.flatten()
    silt_flat = silt_pct.flatten()
    clay_flat = clay_pct.flatten()
    
    # Initialize flattened output
    texture_flat = np.full(sand_flat.shape, "Unknown", dtype=object)
    
    # Create mask for valid data (no NaNs and percentages sum approximately to 100)
    valid_mask = ~np.isnan(sand_flat) & ~np.isnan(silt_flat) & ~np.isnan(clay_flat)
    total_pct = sand_flat + silt_flat + clay_flat
    valid_mask = valid_mask & (total_pct > 95) & (total_pct < 105)
    
    # Apply classification rules for valid data
    # This is a simplified version of the USDA texture triangle
    
    # Clay
    clay_mask = valid_mask & (clay_flat >= 40)
    texture_flat[clay_mask] = "Clay"
    
    # Sand
    sand_mask = valid_mask & (sand_flat >= 85) & (clay_flat < 10)
    texture_flat[sand_mask] = "Sand"
    
    # Silt
    silt_mask = valid_mask & (silt_flat >= 80) & (clay_flat < 12)
    texture_flat[silt_mask] = "Silt"
    
    # Sandy Clay
    sandy_clay_mask = valid_mask & (sand_flat >= 45) & (clay_flat >= 35)
    texture_flat[sandy_clay_mask] = "Sandy Clay"
    
    # Silty Clay
    silty_clay_mask = valid_mask & (silt_flat >= 40) & (clay_flat >= 40)
    texture_flat[silty_clay_mask] = "Silty Clay"
    
    # Clay Loam
    clay_loam_mask = valid_mask & (clay_flat >= 27) & (clay_flat < 40) & (sand_flat >= 20) & (sand_flat < 45)
    texture_flat[clay_loam_mask] = "Clay Loam"
    
    # Sandy Clay Loam
    sandy_clay_loam_mask = valid_mask & (clay_flat >= 20) & (clay_flat < 35) & (sand_flat >= 45)
    texture_flat[sandy_clay_loam_mask] = "Sandy Clay Loam"
    
    # Silty Clay Loam
    silty_clay_loam_mask = valid_mask & (clay_flat >= 27) & (clay_flat < 40) & (silt_flat >= 40)
    texture_flat[silty_clay_loam_mask] = "Silty Clay Loam"
    
    # Loam
    loam_mask = valid_mask & (clay_flat >= 7) & (clay_flat < 27) & (silt_flat >= 28) & (silt_flat < 50) & (sand_flat < 52)
    texture_flat[loam_mask] = "Loam"
    
    # Sandy Loam
    sandy_loam_mask = valid_mask & (clay_flat < 20) & (silt_flat < 50) & (sand_flat >= 43) & (sand_flat < 85)
    texture_flat[sandy_loam_mask] = "Sandy Loam"
    
    # Silt Loam
    silt_loam_mask = valid_mask & (clay_flat < 27) & (silt_flat >= 50) & (silt_flat < 80)
    texture_flat[silt_loam_mask] = "Silt Loam"
    
    # Apply any remaining valid but unclassified cells as Loam
    unclassified_mask = valid_mask & (texture_flat == "Unknown")
    texture_flat[unclassified_mask] = "Loam"
    
    # Reshape result back to original shape
    texture_class = texture_flat.reshape(sand_pct.shape)
    
    return texture_class

def create_soil_texture_pie_chart(
    soil_data: Dict[str, np.ndarray],
    output_path: str = None
) -> Tuple[Optional[Dict[str, float]], bool]:
    """
    Create bar chart showing soil texture composition.
    
    Args:
        soil_data: Dictionary of soil parameter arrays
        output_path: Path to save the plot
        
    Returns:
        Tuple of (texture percentages dict, success boolean)
    """
    try:
        if not all(param in soil_data for param in ['sand', 'silt', 'clay']):
            logger.warning("Sand, silt, or clay data missing")
            return None, False
            
        # Get texture classifications
        textures = classify_soil_texture(
            soil_data['sand'], 
            soil_data['silt'], 
            soil_data['clay']
        )
        
        # Count occurrences of each texture class
        texture_flat = textures.flatten()
        texture_counts = {}
        for texture in np.unique(texture_flat):
            if texture != "Unknown":
                count = np.sum(texture_flat == texture)
                texture_counts[texture] = count
        
        total_valid = sum(texture_counts.values())
        if total_valid == 0:
            logger.warning("No valid soil texture classifications")
            return None, False
            
        # Calculate percentages
        texture_pct = {k: v/total_valid*100 for k, v in texture_counts.items()}
        
        # Sort items by percentage (descending)
        sorted_items = sorted(texture_pct.items(), key=lambda x: x[1], reverse=True)
        labels = [item[0] for item in sorted_items]
        sizes = [item[1] for item in sorted_items]
        
        # Create figure with appropriate size
        plt.figure(figsize=(12, 8))
        
        # Define soil-specific colors - earthy and visually distinct
        soil_colors = {
            "Clay": "#8B4513",  # SaddleBrown
            "Silty Clay": "#A0522D",  # Sienna
            "Sandy Clay": "#CD853F",  # Peru
            "Clay Loam": "#D2691E",  # Chocolate
            "Silty Clay Loam": "#DEB887",  # BurlyWood
            "Sandy Clay Loam": "#E9967A",  # DarkSalmon
            "Loam": "#F5DEB3",  # Wheat
            "Silt Loam": "#FFE4B5",  # Moccasin
            "Sandy Loam": "#FFDEAD",  # NavajoWhite
            "Silt": "#EEE8AA",  # PaleGoldenrod
            "Sand": "#F5F5DC",  # Beige
            "Loamy Sand": "#FAFAD2",  # LightGoldenrodYellow
        }
        
        # Match colors to labels
        colors = [soil_colors.get(texture, "#D2B48C") for texture in labels]  # Default to Tan
        
        # Create horizontal bar chart
        bars = plt.barh(labels, sizes, color=colors, edgecolor='white', linewidth=0.8)
        
        # Add percentage labels to the bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                     ha='left', va='center', fontweight='bold')
        
        # Add title and labels
        plt.title('Soil Texture Composition', fontsize=16)
        plt.xlabel('Percentage (%)')
        plt.ylabel('Soil Texture Class')
        
        # Add grid lines for better readability
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add total area and dominant texture as annotation
        dominant_texture = labels[0]  # First label is the most common texture
        dominant_pct = sizes[0]
        
        plt.figtext(0.5, 0.01, 
                   f"Dominant texture: {dominant_texture} ({dominant_pct:.1f}%) | Based on USDA Soil Texture Classification", 
                   ha="center", fontsize=10, fontstyle="italic")
        
        # Adjust layout
        plt.tight_layout(pad=3)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            plt.close()
        
        return texture_pct, True
        
    except Exception as e:
        logger.error(f"Error creating soil texture bar chart: {e}", exc_info=True)
        return None, False

def create_soil_parameter_maps(
    soil_data: Dict[str, np.ndarray],
    parameters: List[str] = None,
    output_dir: str = None
) -> bool:
    """
    Create spatial maps for soil parameters.
    
    Args:
        soil_data: Dictionary of soil parameter arrays
        parameters: List of parameters to map (if None, maps all)
        output_dir: Directory to save the maps
        
    Returns:
        Boolean indicating success
    """
    try:
        if parameters is None:
            parameters = [p for p in soil_data.keys() if p in SOIL_PARAMETERS]
            
        if not parameters:
            logger.warning("No valid parameters to map")
            return False
            
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        success = False
        
        for param in parameters:
            if param not in soil_data:
                continue
                
            data = soil_data[param]
            
            if np.all(np.isnan(data)):
                continue
                
            # Get parameter information
            param_info = SOIL_PARAMETERS.get(param, {'description': param, 'units': ''})
            
            # Create figure
            plt.figure(figsize=(12, 10))
            
            # Create custom colormap with NaN values as white
            cmap = plt.cm.viridis.copy()
            cmap.set_bad('white', 1.0)
            
            # Plot data
            im = plt.imshow(data, cmap=cmap, interpolation='nearest')
            
            # Add colorbar
            cbar = plt.colorbar(im)
            cbar.set_label(f"{param_info['description']} ({param_info['units']})")
            
            # Add title
            plt.title(f"Spatial Distribution of {param_info['description']}")
            
            # Remove axis labels
            plt.axis('off')
            
            # Save or show
            if output_dir:
                output_path = os.path.join(output_dir, f"{param}_map.png")
                plt.savefig(output_path, bbox_inches='tight', dpi=300)
                plt.close()
                success = True
            else:
                plt.show()
                plt.close()
                success = True
                
        return success
        
    except Exception as e:
        logger.error(f"Error creating soil parameter maps: {e}", exc_info=True)
        return False

def analyze_soil_fertility(soil_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Analyze soil fertility based on available parameters.
    
    Args:
        soil_data: Dictionary of soil parameter arrays
        
    Returns:
        Dictionary with fertility assessments
    """
    fertility_assessment = {}
    
    try:
        # Extract necessary parameters if available
        ph_data = soil_data.get('ph')
        carbon_data = soil_data.get('carbon')
        clay_data = soil_data.get('clay')
        awc_data = soil_data.get('awc')
        
        # Analyze pH if available
        if ph_data is not None:
            flat_ph = ph_data.flatten()
            valid_ph = flat_ph[~np.isnan(flat_ph)]
            
            if len(valid_ph) > 0:
                mean_ph = np.mean(valid_ph)
                
                # Categorize pH
                if mean_ph < 5.5:
                    ph_category = "Acidic"
                    ph_implication = "May require liming; potential nutrient deficiencies (P, Ca, Mg)"
                elif mean_ph >= 5.5 and mean_ph <= 7.2:
                    ph_category = "Optimal"
                    ph_implication = "Favorable for most crops; good nutrient availability"
                else:
                    ph_category = "Alkaline"
                    ph_implication = "Potential micronutrient deficiencies (Fe, Mn, Zn)"
                
                # Calculate percentage in optimal range
                optimal_range_pct = np.sum((valid_ph >= 5.5) & (valid_ph <= 7.2)) / len(valid_ph) * 100
                
                fertility_assessment['ph'] = {
                    'mean': float(mean_ph),
                    'category': ph_category,
                    'implication': ph_implication,
                    'optimal_range_percentage': float(optimal_range_pct)
                }
        
        # Analyze organic carbon if available
        if carbon_data is not None:
            flat_carbon = carbon_data.flatten()
            valid_carbon = flat_carbon[~np.isnan(flat_carbon)]
            
            if len(valid_carbon) > 0:
                mean_carbon = np.mean(valid_carbon)
                
                # Categorize organic carbon
                if mean_carbon < 1.0:
                    carbon_category = "Low"
                    carbon_implication = "Low fertility; poor soil structure; low water retention"
                elif mean_carbon >= 1.0 and mean_carbon <= 3.0:
                    carbon_category = "Moderate"
                    carbon_implication = "Moderate fertility; adequate soil structure"
                else:
                    carbon_category = "High"
                    carbon_implication = "High fertility; good soil structure; high water retention"
                
                fertility_assessment['carbon'] = {
                    'mean': float(mean_carbon),
                    'category': carbon_category,
                    'implication': carbon_implication
                }
        
        # Analyze water holding capacity if available
        if awc_data is not None:
            flat_awc = awc_data.flatten()
            valid_awc = flat_awc[~np.isnan(flat_awc)]
            
            if len(valid_awc) > 0:
                mean_awc = np.mean(valid_awc)
                
                # Categorize AWC
                if mean_awc < 0.10:
                    awc_category = "Low"
                    awc_implication = "Drought-prone; requires frequent irrigation"
                elif mean_awc >= 0.10 and mean_awc <= 0.20:
                    awc_category = "Moderate"
                    awc_implication = "Moderate water retention; average irrigation needs"
                else:
                    awc_category = "High"
                    awc_implication = "Good water retention; drought resistant"
                
                fertility_assessment['awc'] = {
                    'mean': float(mean_awc),
                    'category': awc_category,
                    'implication': awc_implication
                }
        
        # Analyze texture-based properties if available
        if all(x in soil_data for x in ['sand', 'silt', 'clay']):
            # Get soil texture classifications
            textures = classify_soil_texture(
                soil_data['sand'],
                soil_data['silt'],
                soil_data['clay']
            )
            
            # Analyze drainage based on texture
            texture_flat = textures.flatten()
            texture_counts = {}
            for texture in np.unique(texture_flat):
                if texture != "Unknown":
                    count = np.sum(texture_flat == texture)
                    texture_counts[texture] = count
            
            total_valid = sum(texture_counts.values())
            if total_valid > 0:
                texture_pct = {k: v/total_valid*100 for k, v in texture_counts.items()}
                
                # Determine dominant texture
                dominant_texture = max(texture_counts.items(), key=lambda x: x[1])[0]
                
                # Assess drainage
                good_drainage_textures = ["Sand", "Sandy Loam", "Loamy Sand"]
                poor_drainage_textures = ["Clay", "Silty Clay", "Clay Loam"]
                
                good_drainage_pct = sum(texture_pct.get(t, 0) for t in good_drainage_textures)
                poor_drainage_pct = sum(texture_pct.get(t, 0) for t in poor_drainage_textures)
                
                if good_drainage_pct > poor_drainage_pct:
                    drainage_category = "Good"
                    drainage_implication = "Well-drained; low risk of waterlogging"
                elif good_drainage_pct < poor_drainage_pct:
                    drainage_category = "Poor"
                    drainage_implication = "Poorly drained; potential waterlogging issues"
                else:
                    drainage_category = "Moderate"
                    drainage_implication = "Moderate drainage; potential seasonal waterlogging"
                
                # Assess nutrient retention
                high_retention_textures = ["Clay", "Silty Clay", "Clay Loam", "Silty Clay Loam"]
                low_retention_textures = ["Sand", "Loamy Sand"]
                
                high_retention_pct = sum(texture_pct.get(t, 0) for t in high_retention_textures)
                low_retention_pct = sum(texture_pct.get(t, 0) for t in low_retention_textures)
                
                if high_retention_pct > low_retention_pct:
                    retention_category = "High"
                    retention_implication = "Good nutrient retention; efficient fertilizer use"
                elif high_retention_pct < low_retention_pct:
                    retention_category = "Low"
                    retention_implication = "Poor nutrient retention; potential leaching issues"
                else:
                    retention_category = "Moderate"
                    retention_implication = "Moderate nutrient retention"
                
                fertility_assessment['texture'] = {
                    'dominant_texture': dominant_texture,
                    'drainage': {
                        'category': drainage_category,
                        'implication': drainage_implication
                    },
                    'nutrient_retention': {
                        'category': retention_category,
                        'implication': retention_implication
                    }
                }
        
        # Overall fertility assessment
        fertility_rating = 0
        rating_count = 0
        
        # pH contribution
        if 'ph' in fertility_assessment:
            if fertility_assessment['ph']['category'] == 'Optimal':
                fertility_rating += 3
            elif fertility_assessment['ph']['category'] == 'Acidic':
                fertility_rating += 1
            else:  # Alkaline
                fertility_rating += 2
            rating_count += 1
        
        # Carbon contribution
        if 'carbon' in fertility_assessment:
            if fertility_assessment['carbon']['category'] == 'High':
                fertility_rating += 3
            elif fertility_assessment['carbon']['category'] == 'Moderate':
                fertility_rating += 2
            else:  # Low
                fertility_rating += 1
            rating_count += 1
        
        # AWC contribution
        if 'awc' in fertility_assessment:
            if fertility_assessment['awc']['category'] == 'High':
                fertility_rating += 3
            elif fertility_assessment['awc']['category'] == 'Moderate':
                fertility_rating += 2
            else:  # Low
                fertility_rating += 1
            rating_count += 1
        
        # Nutrient retention contribution
        if 'texture' in fertility_assessment:
            if fertility_assessment['texture']['nutrient_retention']['category'] == 'High':
                fertility_rating += 3
            elif fertility_assessment['texture']['nutrient_retention']['category'] == 'Moderate':
                fertility_rating += 2
            else:  # Low
                fertility_rating += 1
            rating_count += 1
        
        # Calculate average rating
        if rating_count > 0:
            avg_rating = fertility_rating / rating_count
            
            if avg_rating >= 2.5:
                fertility_level = "High"
                fertility_description = "Fertile soils suitable for a wide range of crops with minimal amendments."
            elif avg_rating >= 1.5:
                fertility_level = "Moderate"
                fertility_description = "Moderately fertile soils requiring standard amendments for good productivity."
            else:
                fertility_level = "Low"
                fertility_description = "Low fertility soils requiring significant amendments for agricultural use."
            
            fertility_assessment['overall'] = {
                'rating': float(avg_rating),
                'level': fertility_level,
                'description': fertility_description
            }
    
    except Exception as e:
        logger.error(f"Error analyzing soil fertility: {e}", exc_info=True)
    
    return fertility_assessment

def generate_soil_summary(soil_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Generate a comprehensive summary of soil properties.
    
    Args:
        soil_data: Dictionary of soil parameter arrays
        
    Returns:
        Dictionary with soil summary information
    """
    summary = {}
    
    try:
        # Calculate statistics for all parameters
        stats = calculate_soil_statistics(soil_data)
        summary['statistics'] = stats
        
        # Analyze soil fertility
        fertility = analyze_soil_fertility(soil_data)
        summary['fertility'] = fertility
        
        # Analyze soil texture if data available
        if all(x in soil_data for x in ['sand', 'silt', 'clay']):
            # Get soil texture classifications
            textures = classify_soil_texture(
                soil_data['sand'],
                soil_data['silt'],
                soil_data['clay']
            )
            
            # Count occurrences of each texture class
            texture_flat = textures.flatten()
            texture_counts = {}
            for texture in np.unique(texture_flat):
                if texture != "Unknown":
                    count = np.sum(texture_flat == texture)
                    texture_counts[texture] = count
            
            total_valid = sum(texture_counts.values())
            if total_valid > 0:
                texture_pct = {k: v/total_valid*100 for k, v in texture_counts.items()}
                summary['texture'] = {
                    'counts': texture_counts,
                    'percentages': texture_pct,
                    'dominant': max(texture_counts.items(), key=lambda x: x[1])[0]
                }
        
        # Generate management recommendations
        recommendations = []
        
        # pH recommendations
        if 'ph' in fertility:
            ph_category = fertility['ph']['category']
            if (ph_category == 'Acidic'):
                recommendations.append("Apply lime to increase soil pH for better nutrient availability.")
            elif (ph_category == 'Alkaline'):
                recommendations.append("Consider adding organic matter or soil amendments to reduce soil pH.")
        
        # Organic matter recommendations
        if 'carbon' in fertility:
            carbon_category = fertility['carbon']['category']
            if (carbon_category == 'Low'):
                recommendations.append("Add organic matter (compost, cover crops, reduced tillage) to improve soil health.")
            elif (carbon_category == 'Moderate'):
                recommendations.append("Maintain organic matter through crop residue retention and minimal tillage.")
        
        # Water management recommendations
        if 'awc' in fertility:
            awc_category = fertility['awc']['category']
            if (awc_category == 'Low'):
                recommendations.append("Implement irrigation systems and add organic matter to improve water retention.")
            elif (awc_category == 'High'):
                recommendations.append("Monitor irrigation to prevent waterlogging in areas with high water retention.")
        
        # Texture-based recommendations
        if 'texture' in fertility:
            drainage = fertility['texture']['drainage']['category']
            if (drainage == 'Poor'):
                recommendations.append("Install drainage systems in areas with poor drainage to prevent root diseases.")
            
            retention = fertility['texture']['nutrient_retention']['category']
            if (retention == 'Low'):
                recommendations.append("Apply fertilizers in smaller, more frequent doses to prevent nutrient leaching.")
        
        summary['recommendations'] = recommendations
        
    except Exception as e:
        logger.error(f"Error generating soil summary: {e}", exc_info=True)
    
    return summary

def export_soil_data_as_csv(soil_data: Dict[str, np.ndarray], output_path: str) -> bool:
    """
    Export soil data to CSV format.
    
    Args:
        soil_data: Dictionary of soil parameter arrays
        output_path: Path to save the CSV file
        
    Returns:
        Boolean indicating success
    """
    try:
        # Create a list to hold all parameter values
        data_dict = {}
        
        # Process each parameter
        for param, array in soil_data.items():
            if param in SOIL_PARAMETERS:
                # Flatten array and add to dictionary
                flat_array = array.flatten()
                data_dict[param] = flat_array
        
        # Create DataFrame
        df = pd.DataFrame(data_dict)
        
        # Add column with coordinates or cell indices
        df['cell_index'] = df.index
        
        # Export to CSV
        df.to_csv(output_path, index=False)
        
        return True
    except Exception as e:
        logger.error(f"Error exporting soil data to CSV: {e}", exc_info=True)
        return False

def analyze_soil_limitations(soil_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Analyze soil limitations for agricultural use.
    
    Args:
        soil_data: Dictionary of soil parameter arrays
        
    Returns:
        Dictionary with soil limitations assessment
    """
    limitations = {}
    
    try:
        # Check for shallow depth limitations
        if 'dp_tot' in soil_data:
            depth_array = soil_data['dp_tot']
            flat_depth = depth_array.flatten()
            valid_depth = flat_depth[~np.isnan(flat_depth)]
            
            if len(valid_depth) > 0:
                mean_depth = np.mean(valid_depth)
                shallow_pct = np.sum(valid_depth < 50) / len(valid_depth) * 100
                
                limitations['shallow_depth'] = {
                    'mean_depth_cm': float(mean_depth),
                    'shallow_areas_pct': float(shallow_pct),
                    'limitation_level': 'Severe' if shallow_pct > 30 else 'Moderate' if shallow_pct > 10 else 'Slight',
                    'recommendation': 'Choose shallow-rooted crops or implement raised beds in affected areas.'
                }
        
        # Check for high rock content limitations
        if 'rock' in soil_data:
            rock_array = soil_data['rock']
            flat_rock = rock_array.flatten()
            valid_rock = flat_rock[~np.isnan(flat_rock)]
            
            if len(valid_rock) > 0:
                mean_rock = np.mean(valid_rock)
                high_rock_pct = np.sum(valid_rock > 15) / len(valid_rock) * 100
                
                limitations['rockiness'] = {
                    'mean_rock_content_pct': float(mean_rock),
                    'high_rock_areas_pct': float(high_rock_pct),
                    'limitation_level': 'Severe' if high_rock_pct > 30 else 'Moderate' if high_rock_pct > 10 else 'Slight',
                    'recommendation': 'Consider rock removal or selecting crops tolerant of rocky soils.'
                }
        
        # Check for drainage limitations
        if all(x in soil_data for x in ['sand', 'clay']):
            sand_array = soil_data['sand']
            clay_array = soil_data['clay']
            
            # Create masks for poorly drained areas (high clay, low sand)
            poor_drainage_mask = (clay_array > 40) & (sand_array < 30)
            
            # Count cells with poor drainage
            total_cells = np.sum(~np.isnan(clay_array) & ~np.isnan(sand_array))
            poor_drainage_cells = np.sum(poor_drainage_mask)
            
            if total_cells > 0:
                poor_drainage_pct = poor_drainage_cells / total_cells * 100
                
                limitations['drainage'] = {
                    'poor_drainage_areas_pct': float(poor_drainage_pct),
                    'limitation_level': 'Severe' if poor_drainage_pct > 30 else 'Moderate' if poor_drainage_pct > 10 else 'Slight',
                    'recommendation': 'Install drainage systems or select water-tolerant crops in poorly drained areas.'
                }
        
        # Check for pH limitations
        if 'ph' in soil_data:
            ph_array = soil_data['ph']
            flat_ph = ph_array.flatten()
            valid_ph = flat_ph[~np.isnan(flat_ph)]
            
            if len(valid_ph) > 0:
                acidic_pct = np.sum(valid_ph < 5.5) / len(valid_ph) * 100
                alkaline_pct = np.sum(valid_ph > 7.5) / len(valid_ph) * 100
                
                limitations['ph_extremes'] = {
                    'acidic_areas_pct': float(acidic_pct),
                    'alkaline_areas_pct': float(alkaline_pct),
                    'limitation_level': 'Severe' if max(acidic_pct, alkaline_pct) > 50 else 'Moderate' if max(acidic_pct, alkaline_pct) > 20 else 'Slight',
                    'recommendation': 'Apply lime in acidic areas or sulfur in alkaline areas to adjust pH.'
                }
        
        # Check for salinity limitations
        if 'ec' in soil_data:
            ec_array = soil_data['ec']
            flat_ec = ec_array.flatten()
            valid_ec = flat_ec[~np.isnan(flat_ec)]
            
            if len(valid_ec) > 0:
                mean_ec = np.mean(valid_ec)
                high_salinity_pct = np.sum(valid_ec > 4.0) / len(valid_ec) * 100
                
                limitations['salinity'] = {
                    'mean_ec_ds_m': float(mean_ec),
                    'high_salinity_areas_pct': float(high_salinity_pct),
                    'limitation_level': 'Severe' if high_salinity_pct > 30 else 'Moderate' if high_salinity_pct > 10 else 'Slight',
                    'recommendation': 'Leach salts with irrigation, use salt-tolerant crops in affected areas.'
                }
    
    except Exception as e:
        logger.error(f"Error analyzing soil limitations: {e}", exc_info=True)
    
    return limitations