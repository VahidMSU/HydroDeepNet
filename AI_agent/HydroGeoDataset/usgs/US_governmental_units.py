"""
Extract and analyze US governmental units from the USGS Governmental Units database.
"""
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import box

try:
    from HydroGeoDataset.config import AgentConfig
    from HydroGeoDataset.governmental_units_report import analyze_governmental_units
except ImportError:
    from config import AgentConfig
    from governmental_units_report import analyze_governmental_units

# Available layers in the governmental units database
AVAILABLE_LAYERS = [
    'GU_CountyOrEquivalent',
    'GU_IncorporatedPlace',
    'GU_InternationalBoundaryLine',
    'GU_Jurisdictional',
    'GU_MinorCivilDivision',
    'GU_NativeAmericanArea',
    'GU_PLSSTownship',
    'GU_Reserve',
    'GU_StateOrTerritory',
    'GU_UnincorporatedPlace'
]

def extract_layer_by_bbox(gdb_path, layer_name, bbox):
    """
    Extract a specific layer from the geodatabase and clip by bounding box.
    
    Args:
        gdb_path: Path to the geodatabase
        layer_name: Name of the layer to extract
        bbox: Bounding box as [min_lon, min_lat, max_lon, max_lat]
        
    Returns:
        GeoDataFrame of the clipped layer
    """
    # Read the layer
    print(f"Reading layer {layer_name}...")
    gdf = gpd.read_file(gdb_path, layer=layer_name)
    
    # Print column information
    print(f"\nColumns in {layer_name}:")
    for col in gdf.columns:
        # Show column name and data type
        print(f"  - {col} ({gdf[col].dtype})")
        
        # For non-geometry columns, show some sample values if available
        if col != 'geometry' and len(gdf) > 0:
            sample_values = gdf[col].dropna().unique()[:3]  # Show up to 3 unique values
            if len(sample_values) > 0:
                print(f"    Sample values: {sample_values}")
    
    # Convert to WGS84 if needed
    if gdf.crs != "EPSG:4326":
        print(f"Converting from {gdf.crs} to EPSG:4326...")
        gdf = gdf.to_crs("EPSG:4326")
    
    # Create bounding box geometry
    bbox_poly = box(*bbox)
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_poly], crs="EPSG:4326")
    
    # Clip to bounding box
    print(f"Clipping to bounding box...")
    gdf_clipped = gpd.clip(gdf, bbox_gdf)
    
    print(f"Found {len(gdf_clipped)} features in bounding box")
    
    # Print a sample record if available
    if len(gdf_clipped) > 0:
        print("\nSample record:")
        sample = gdf_clipped.iloc[0]
        for col in sample.index:
            if col != 'geometry':
                print(f"  {col}: {sample[col]}")
    
    return gdf_clipped

if __name__ == "__main__":
    # Define parameters
    config = { 
        "RESOLUTION": 250,
        "huc8": None,
        "video": False,
        "aggregation": "annual",
        "start_year": 2000,
        "end_year": 2003,
        'bounding_box': [-85.444332, 43.158148, -84.239256, 44.164683],
    }
    
    gdb_path = AgentConfig.USGS_governmental_path
    bbox = config['bounding_box']
    
    print(f"Using governmental units database: {gdb_path}")
    print(f"Using bounding box: {bbox}")
    
    # List all available layers in the geodatabase
    print("\nAvailable layers in geodatabase:")
    for layer in AVAILABLE_LAYERS:
        print(f"- {layer}")
    
    # Extract counties first to see details
    print("\nExtracting counties for detailed inspection...")
    counties = extract_layer_by_bbox(gdb_path, 'GU_CountyOrEquivalent', bbox)
    
    # Extract other key layers to understand their structure
    print("\nExtracting state information...")
    states = extract_layer_by_bbox(gdb_path, 'GU_StateOrTerritory', bbox)
    
    print("\nExtracting incorporated places information...")
    places = extract_layer_by_bbox(gdb_path, 'GU_IncorporatedPlace', bbox)
    
    # Generate a comprehensive report (using all default layers)
    print("\nGenerating comprehensive report...")
    output_dir = os.path.join(os.getcwd(), "governmental_results")
    
    report_path = analyze_governmental_units(
        gdb_path=gdb_path,
        bounding_box=bbox,
        output_dir=output_dir,
        layers=['GU_CountyOrEquivalent', 'GU_StateOrTerritory', 'GU_IncorporatedPlace', 'GU_MinorCivilDivision', 'GU_NativeAmericanArea']
    )
    
    if report_path:
        print(f"\nReport generated successfully: {report_path}")
    else:
        print("\nFailed to generate report")
    
    # Example: Extract and display counties
    print("\nExtracting counties for demonstration...")
    counties = extract_layer_by_bbox(gdb_path, 'GU_CountyOrEquivalent', bbox)
    
    if not counties.empty:
        print("\nCounties in the area:")
        for _, county in counties.iterrows():
            if 'NAME' in county and 'STATE_NAME' in county:
                print(f"- {county['NAME']}, {county.get('STATE_NAME', '')}")
