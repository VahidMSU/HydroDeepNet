"""
US Governmental Units analysis and report generation.

This module provides functionality to extract, analyze, and visualize
governmental boundaries (counties, municipalities, etc.) for a given area.
"""
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import contextily as ctx
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import numpy as np
from pathlib import Path
try:
    from config import AgentConfig
except ImportError:
    from AI_agent.config import AgentConfig
# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Layers of interest in the governmental units database
GOVERNMENTAL_LAYERS = {
    'GU_CountyOrEquivalent': {
        'description': 'Counties',
        'key_fields': ['PERMANENT_IDENTIFIER', 'STATE_NAME', 'COUNTY_NAME', 'POPULATION', 'AREASQKM', 'GNIS_NAME'],
        'plot_color': 'lightblue',
        'plot_order': 1
    },
    'GU_StateOrTerritory': {
        'description': 'States',
        'key_fields': ['STATE_NAME', 'POPULATION', 'AREASQKM', 'GNIS_NAME'],
        'plot_color': 'none',  # Just show outline
        'plot_order': 0
    },
    'GU_IncorporatedPlace': {
        'description': 'Cities and Towns',
        'key_fields': ['STATE_NAME', 'PLACE_NAME', 'POPULATION', 'AREASQKM', 'GNIS_NAME'],
        'plot_color': 'lightgreen',
        'plot_order': 2
    },
    'GU_MinorCivilDivision': {
        'description': 'Townships',
        'key_fields': ['STATE_NAME', 'MINORCIVILDIVISION_NAME', 'POPULATION', 'AREASQKM', 'GNIS_NAME'],
        'plot_color': 'lightyellow',
        'plot_order': 3
    },
    'GU_NativeAmericanArea': {
        'description': 'Tribal Areas',
        'key_fields': ['NAME', 'POPULATION', 'AREASQKM', 'ADMINTYPE'],
        'plot_color': 'lightcoral',
        'plot_order': 4
    }
}

def acres_to_sq_miles(acres: float) -> float:
    """Convert acres to square miles."""
    return acres / 640.0

def extract_governmental_units(gdb_path: str, 
                              bounding_box: Tuple[float, float, float, float],
                              layers: Optional[List[str]] = None) -> Dict[str, gpd.GeoDataFrame]:
    """
    Extract governmental units within a bounding box.
    
    Args:
        gdb_path: Path to the governmental units geodatabase
        bounding_box: Tuple of (min_lon, min_lat, max_lon, max_lat)
        layers: List of layer names to extract (defaults to GOVERNMENTAL_LAYERS keys)
        
    Returns:
        Dictionary mapping layer names to GeoDataFrames
    """
    if not os.path.exists(gdb_path):
        logger.error(f"Geodatabase not found: {gdb_path}")
        return {}
        
    # Use default layers if none specified
    if layers is None:
        layers = list(GOVERNMENTAL_LAYERS.keys())
        
    # Create a bounding box geometry in WGS84
    from shapely.geometry import box
    bbox_geom = box(bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3])
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs="EPSG:4326")
    
    result = {}
    
    try:
        for layer_name in layers:
            if layer_name not in GOVERNMENTAL_LAYERS:
                logger.warning(f"Layer {layer_name} not in supported layers list")
                continue
                
            try:
                logger.info(f"Reading layer: {layer_name}")
                # Read the layer with pyogrio for better performance
                gdf = gpd.read_file(gdb_path, layer=layer_name)
                
                # Print layer info for debugging
                logger.info(f"Layer {layer_name} has {len(gdf)} features with columns: {list(gdf.columns)}")
                
                # Convert to WGS84 if needed
                if gdf.crs != "EPSG:4326":
                    logger.info(f"Converting {layer_name} from {gdf.crs} to EPSG:4326")
                    gdf = gdf.to_crs("EPSG:4326")
                
                # Clip to bounding box
                logger.info(f"Clipping {layer_name} to bounding box")
                gdf_clipped = gpd.clip(gdf, bbox_gdf)
                
                # Skip if no features within the bounding box
                if gdf_clipped.empty:
                    logger.warning(f"No {layer_name} features found in the bounding box")
                    continue
                
                # List the most important fields for this layer
                logger.info(f"Important fields in {layer_name}:")
                key_fields = GOVERNMENTAL_LAYERS[layer_name]['key_fields']
                available_fields = [f for f in key_fields if f in gdf_clipped.columns]
                
                # Show sample values
                if len(gdf_clipped) > 0:
                    sample = gdf_clipped.iloc[0]
                    for field in available_fields:
                        if field in sample:
                            logger.info(f"  {field}: {sample[field]}")
                
                # Keep only the key fields plus geometry
                if available_fields:
                    gdf_clipped = gdf_clipped[available_fields + ['geometry']]
                
                # Add layer to results
                result[layer_name] = gdf_clipped
                logger.info(f"Successfully extracted {len(gdf_clipped)} {layer_name} features")
                
            except Exception as e:
                logger.error(f"Error extracting {layer_name}: {e}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error extracting governmental units: {e}")
        return {}

def create_governmental_map(data: Dict[str, gpd.GeoDataFrame], 
                           output_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (12, 10),
                           add_basemap: bool = True) -> plt.Figure:
    """
    Create a map showing governmental units.
    
    Args:
        data: Dictionary mapping layer names to GeoDataFrames
        output_path: Path to save the figure (optional)
        figsize: Figure dimensions (width, height) in inches
        add_basemap: Whether to add a web tile basemap
        
    Returns:
        Matplotlib Figure object
    """
    try:
        # Explicitly close any existing figures to prevent mix-ups
        plt.close('all')
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sort layers by plot order
        layer_items = sorted(
            [(name, gdf) for name, gdf in data.items() if name in GOVERNMENTAL_LAYERS],
            key=lambda x: GOVERNMENTAL_LAYERS[x[0]]['plot_order']
        )
        
        # Plot each layer
        for layer_name, gdf in layer_items:
            layer_info = GOVERNMENTAL_LAYERS[layer_name]
            color = layer_info['plot_color']
            
            # Set edgecolor and facecolor based on the plot_color
            edgecolor = 'black'
            facecolor = color if color != 'none' else None
            alpha = 0.6 if facecolor else None
            
            # Plot the layer
            gdf.plot(
                ax=ax,
                edgecolor=edgecolor,
                facecolor=facecolor,
                alpha=alpha,
                linewidth=1 if layer_name == 'GU_StateOrTerritory' else 0.5
            )
        
        # Add basemap if requested
        if add_basemap:
            try:
                ctx.add_basemap(
                    ax, 
                    source=ctx.providers.OpenStreetMap.Mapnik,
                    zoom=10
                )
            except Exception as e:
                logger.warning(f"Could not add basemap: {e}")
        
        # Add title and labels
        layers_shown = [GOVERNMENTAL_LAYERS[name]['description'] for name, _ in layer_items]
        ax.set_title(f"US Governmental Units\n({', '.join(layers_shown)})", fontsize=14)
        
        # Remove axis labels and ticks for map
        ax.set_axis_off()
        
        # Add legend if multiple layers
        if len(layer_items) > 1:
            from matplotlib.patches import Patch
            legend_elements = []
            
            for layer_name, _ in layer_items:
                layer_info = GOVERNMENTAL_LAYERS[layer_name]
                color = layer_info['plot_color']
                
                if color == 'none':
                    # Special case for outline-only layers
                    legend_elements.append(
                        Patch(edgecolor='black', facecolor='white', label=layer_info['description'])
                    )
                else:
                    legend_elements.append(
                        Patch(facecolor=color, edgecolor='black', alpha=0.6, label=layer_info['description'])
                    )
            
            ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        # Tight layout
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # Make filename unique to avoid collisions
            unique_path = output_path
            plt.savefig(unique_path, dpi=300, bbox_inches='tight')
            logger.info(f"Map saved to {unique_path}")
            
            # Explicitly close the figure after saving
            plt.close(fig)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating governmental map: {e}", exc_info=True)
        plt.close('all')  # Make sure to clean up
        return None

def extract_unit_statistics(data: Dict[str, gpd.GeoDataFrame]) -> Dict[str, Dict[str, Any]]:
    """
    Extract statistics about governmental units.
    
    Args:
        data: Dictionary mapping layer names to GeoDataFrames
        
    Returns:
        Dictionary with statistics for each layer
    """
    stats = {}
    
    for layer_name, gdf in data.items():
        if layer_name not in GOVERNMENTAL_LAYERS:
            continue
        
        layer_stats = {
            'count': len(gdf),
            'description': GOVERNMENTAL_LAYERS[layer_name]['description'],
            'units': []
        }
        
        # Log column names for debugging
        logger.info(f"Columns in {layer_name}: {list(gdf.columns)}")
        
        # Calculate total area
        if 'AREASQKM' in gdf.columns:
            try:
                # Convert sq km to sq miles
                total_area = gdf['AREASQKM'].sum() * 0.386102
                layer_stats['total_area_sqmi'] = total_area
                logger.info(f"{layer_name} total area: {total_area:.2f} sq mi")
            except Exception as e:
                logger.error(f"Error calculating area for {layer_name}: {e}")
        
        # Include individual unit details
        for idx, row in gdf.iterrows():
            try:
                unit = {}
                
                # Add name based on layer type
                if layer_name == 'GU_CountyOrEquivalent' and 'COUNTY_NAME' in gdf.columns:
                    unit['name'] = row['COUNTY_NAME']
                elif layer_name == 'GU_IncorporatedPlace' and 'PLACE_NAME' in gdf.columns:
                    unit['name'] = row['PLACE_NAME']
                elif layer_name == 'GU_MinorCivilDivision' and 'MINORCIVILDIVISION_NAME' in gdf.columns:
                    unit['name'] = row['MINORCIVILDIVISION_NAME']
                elif layer_name == 'GU_NativeAmericanArea' and 'NAME' in gdf.columns:
                    unit['name'] = row['NAME']
                elif layer_name == 'GU_StateOrTerritory' and 'STATE_NAME' in gdf.columns:
                    unit['name'] = row['STATE_NAME']
                elif 'NAME' in gdf.columns:
                    unit['name'] = row['NAME']
                
                # Add common fields
                if 'STATE_NAME' in gdf.columns:
                    unit['state'] = row['STATE_NAME']
                
                if 'GNIS_NAME' in gdf.columns:
                    unit['gnis_name'] = row['GNIS_NAME']
                
                if 'POPULATION' in gdf.columns:
                    try:
                        unit['population'] = int(row['POPULATION'])
                    except (ValueError, TypeError):
                        pass
                
                # Process area
                if 'AREASQKM' in gdf.columns:
                    try:
                        unit['area_sqmi'] = row['AREASQKM'] * 0.386102
                    except (ValueError, TypeError):
                        pass

                if 'ADMINTYPE' in gdf.columns:
                    unit['admin_type'] = row['ADMINTYPE']
                
                layer_stats['units'].append(unit)
            except Exception as e:
                logger.error(f"Error processing unit {idx} in {layer_name}: {e}")
        
        stats[layer_name] = layer_stats
    
    return stats

def generate_governmental_report(data: Dict[str, gpd.GeoDataFrame], 
                               stats: Dict[str, Dict[str, Any]],
                               bounding_box: Tuple[float, float, float, float],
                               output_dir: str = 'governmental_report') -> str:
    """
    Generate a report on governmental units for the area.
    
    Args:
        data: Dictionary mapping layer names to GeoDataFrames
        stats: Dictionary with statistics for each layer
        bounding_box: Tuple of (min_lon, min_lat, max_lon, max_lat)
        output_dir: Directory to save report files
        
    Returns:
        Path to the generated report file
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create report filename
    report_path = os.path.join(output_dir, "governmental_units_report.md")
    map_path = os.path.join(output_dir, "governmental_units_map.png")
    
    try:
        # Generate map with explicit figure management
        plt.close('all')  # Close any existing figures
        create_governmental_map(data, output_path=map_path)
        plt.close('all')  # Close again to ensure cleanup
        
        # Check if the map was created
        if not os.path.exists(map_path):
            logger.error(f"Failed to create governmental units map at {map_path}")
            
        # Rest of the report generation
        # ...existing code...
    except Exception as e:
        logger.error(f"Error in map generation: {e}")
        plt.close('all')  # Ensure cleanup on error
    
    # Open report file for writing
    with open(report_path, "w") as f:
        # Header
        f.write("# US Governmental Units Analysis Report\n\n")
        
        # Region information
        f.write("## Study Area\n\n")
        f.write(f"**Bounding Box:** Longitude: {bounding_box[0]:.4f}° to {bounding_box[2]:.4f}°, ")
        f.write(f"Latitude: {bounding_box[1]:.4f}° to {bounding_box[3]:.4f}°\n\n")
        
        # Map
        f.write("## Governmental Boundaries Map\n\n")
        f.write(f"![Governmental Units Map]({os.path.basename(map_path)})\n\n")
        
        # Summary of layers
        available_layers = [name for name in data.keys() if name in GOVERNMENTAL_LAYERS]
        if available_layers:
            f.write("## Summary\n\n")
            f.write("| Administrative Unit Type | Count | Total Area (sq mi) | Population |\n")
            f.write("|--------------------------|-------|-------------------|------------|\n")
            
            for layer_name in available_layers:
                if layer_name not in stats:
                    continue
                    
                layer_stats = stats[layer_name]
                description = layer_stats['description']
                count = layer_stats['count']
                
                # Area information (if available)
                total_area = layer_stats.get('total_area_sqmi', "N/A")
                
                # Calculate total population if available
                total_pop = "N/A"
                try:
                    population_values = [unit.get('population', 0) for unit in layer_stats.get('units', [])]
                    if population_values and any(isinstance(p, (int, float)) for p in population_values):
                        total_pop = sum(p for p in population_values if isinstance(p, (int, float)))
                except Exception:
                    pass
                
                # Format numbers
                if isinstance(total_area, (int, float)):
                    total_area = f"{total_area:,.2f}"
                if isinstance(total_pop, (int, float)):
                    total_pop = f"{total_pop:,}"
                
                f.write(f"| {description} | {count} | {total_area} | {total_pop} |\n")
            
            f.write("\n")
        
        # Details for each layer
        for layer_name in available_layers:
            if layer_name not in stats:
                continue
                
            layer_stats = stats[layer_name]
            description = layer_stats['description']
            
            f.write(f"## {description} Details\n\n")
            
            units = layer_stats.get('units', [])
            if units:
                # Different table format based on layer type
                if layer_name == 'GU_CountyOrEquivalent':
                    f.write("| County | State | Area (sq mi) | Population |\n")
                    f.write("|--------|-------|--------------|------------|\n")
                    
                    for unit in units:
                        name = unit.get('name', "N/A")
                        state = unit.get('state', "N/A")
                        area = unit.get('area_sqmi', "N/A")
                        population = unit.get('population', "N/A")
                        
                        # Format numbers
                        if isinstance(area, (int, float)):
                            area = f"{area:,.2f}"
                        if isinstance(population, (int, float)):
                            population = f"{population:,}"
                            
                        f.write(f"| {name} | {state} | {area} | {population} |\n")
                
                elif layer_name == 'GU_IncorporatedPlace':
                    f.write("| City/Town | State | Area (sq mi) | Population |\n")
                    f.write("|-----------|-------|--------------|------------|\n")
                    
                    for unit in units:
                        name = unit.get('name', "N/A")
                        state = unit.get('state', "N/A")
                        area = unit.get('area_sqmi', "N/A")
                        population = unit.get('population', "N/A")
                        
                        # Format numbers
                        if isinstance(area, (int, float)):
                            area = f"{area:,.2f}"
                        if isinstance(population, (int, float)):
                            population = f"{population:,}"
                            
                        f.write(f"| {name} | {state} | {area} | {population} |\n")
                
                elif layer_name == 'GU_StateOrTerritory':
                    f.write("| State | Area (sq mi) | Population |\n")
                    f.write("|-------|--------------|------------|\n")
                    
                    for unit in units:
                        name = unit.get('name', "N/A")
                        area = unit.get('area_sqmi', "N/A")
                        population = unit.get('population', "N/A")
                        
                        # Format numbers
                        if isinstance(area, (int, float)):
                            area = f"{area:,.2f}"
                        if isinstance(population, (int, float)):
                            population = f"{population:,}"
                            
                        f.write(f"| {name} | {area} | {population} |\n")
                
                elif layer_name == 'GU_MinorCivilDivision':
                    f.write("| Township | State | Area (sq mi) | Population |\n")
                    f.write("|----------|-------|--------------|------------|\n")
                    
                    for unit in units:
                        name = unit.get('name', "N/A")
                        state = unit.get('state', "N/A")
                        area = unit.get('area_sqmi', "N/A")
                        population = unit.get('population', "N/A")
                        
                        # Format numbers
                        if isinstance(area, (int, float)):
                            area = f"{area:,.2f}"
                        if isinstance(population, (int, float)):
                            population = f"{population:,}"
                            
                        f.write(f"| {name} | {state} | {area} | {population} |\n")
                
                elif layer_name == 'GU_NativeAmericanArea':
                    f.write("| Name | Type | Area (sq mi) | Population |\n")
                    f.write("|------|------|--------------|------------|\n")
                    
                    for unit in units:
                        name = unit.get('name', "N/A")
                        admin_type = unit.get('admin_type', "N/A")
                        area = unit.get('area_sqmi', "N/A")
                        population = unit.get('population', "N/A")
                        
                        # Format numbers
                        if isinstance(area, (int, float)):
                            area = f"{area:,.2f}"
                        if isinstance(population, (int, float)):
                            population = f"{population:,}"
                            
                        f.write(f"| {name} | {admin_type} | {area} | {population} |\n")
                
                else:
                    # Generic table for other layers
                    f.write("| Name | Area (sq mi) | Population |\n")
                    f.write("|------|--------------|------------|\n")
                    
                    for unit in units:
                        name = unit.get('name', unit.get('gnis_name', "N/A"))
                        area = unit.get('area_sqmi', "N/A")
                        population = unit.get('population', "N/A")
                        
                        # Format numbers
                        if isinstance(area, (int, float)):
                            area = f"{area:,.2f}"
                        if isinstance(population, (int, float)):
                            population = f"{population:,}"
                            
                        f.write(f"| {name} | {area} | {population} |\n")
            
            f.write("\n")
        
        # Add information about the region
        total_population = 0
        total_area = 0
        
        # Collect county information for region summary
        if 'GU_CountyOrEquivalent' in stats:
            county_units = stats['GU_CountyOrEquivalent'].get('units', [])
            county_names = [unit.get('name', '') for unit in county_units if 'name' in unit]
            
            # Calculate total population and area
            for unit in county_units:
                if 'population' in unit and isinstance(unit['population'], (int, float)):
                    total_population += unit['population']
                if 'area_sqmi' in unit and isinstance(unit['area_sqmi'], (int, float)):
                    total_area += unit['area_sqmi']
            
            if county_names:
                f.write("## Regional Summary\n\n")
                f.write(f"This report covers {len(county_names)} counties in ")
                
                # Get state name if available
                state_name = county_units[0].get('state', 'the United States') if county_units else 'the United States'
                f.write(f"{state_name}: {', '.join(county_names)}.\n\n")
                
                # Add population and area info
                if total_population > 0:
                    f.write(f"**Total Population:** {total_population:,}\n\n")
                if total_area > 0:
                    f.write(f"**Total Land Area:** {total_area:,.2f} square miles\n\n")
                    if total_population > 0:
                        density = total_population / total_area
                        f.write(f"**Population Density:** {density:.1f} people per square mile\n\n")
        
        # Source information
        f.write("## Data Source\n\n")
        f.write("This report uses data from the United States Census Bureau's Governmental Units database, ")
        f.write("which provides authoritative geographic boundaries for government administrative units ")
        f.write("including states, counties, incorporated places (cities and towns), and other divisions.\n\n")
        
        # Applications
        f.write("## Applications\n\n")
        f.write("Administrative boundaries are useful for:\n\n")
        f.write("- Watershed management planning\n")
        f.write("- Identifying jurisdictional responsibilities\n")
        f.write("- Analyzing land use patterns and regulations\n")
        f.write("- Coordinating between multiple authorities for environmental management\n")
        f.write("- Determining census and demographic area statistics\n")
        f.write("- Planning infrastructure projects across jurisdictional boundaries\n")
        
    logger.info(f"Report generated: {report_path}")
    return report_path

def analyze_governmental_units(gdb_path: str, bounding_box: Tuple[float, float, float, float],
                              output_dir: str = 'governmental_analysis',
                              layers: Optional[List[str]] = None) -> str:
    """
    Analyze governmental units and generate a report.
    
    Args:
        gdb_path: Path to the governmental units geodatabase
        bounding_box: Tuple of (min_lon, min_lat, max_lon, max_lat) 
        output_dir: Directory to save outputs
        layers: List of layers to analyze (defaults to all in GOVERNMENTAL_LAYERS)
        
    Returns:
        Path to the generated report
    """
    # Extract data
    data = extract_governmental_units(gdb_path, bounding_box, layers)
    
    if not data:
        logger.error("No governmental unit data extracted")
        return ""
    
    # Calculate statistics
    stats = extract_unit_statistics(data)
    
    # Generate report
    report_path = generate_governmental_report(
        data, stats, bounding_box, output_dir
    )
    
    return report_path

if __name__ == "__main__":
    # Example usage
    try:
        # Define bounding box and paths
        bbox = [-85.444332, 43.158148, -84.239256, 44.164683]
        gdb_path = AgentConfig.USGS_governmental_path
        
        # Create output directory
        output_dir = "governmental_analysis"
        
        # Analyze and generate report
        report_path = analyze_governmental_units(
            gdb_path=gdb_path,
            bounding_box=bbox,
            output_dir=output_dir
        )
        
        if report_path:
            print(f"Report generated: {report_path}")
        else:
            print("Failed to generate report")
            
    except Exception as e:
        logger.error(f"Error in governmental units analysis: {e}", exc_info=True)
