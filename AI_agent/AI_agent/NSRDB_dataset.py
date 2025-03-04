import h5py 
import numpy as np
import geopandas as gpd
from scipy.interpolate import griddata
import rasterio
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
import os
import argparse
from typing import List, Dict, Optional

# Import the utility functions we just created
try:
    from AI_agent.NSRDB_utilities import (
        get_coordinates_from_bbox, extract_nsrdb_data, create_interpolated_grid,
        save_as_raster, aggregate_nsrdb_daily, extract_nsrdb_multiyear
    )
    from AI_agent.NSRDB_report import batch_process_nsrdb
except ImportError:
    # For direct execution
    from NSRDB_utilities import (
        get_coordinates_from_bbox, extract_nsrdb_data, create_interpolated_grid,
        save_as_raster, aggregate_nsrdb_daily, extract_nsrdb_multiyear
    )
    from NSRDB_report import batch_process_nsrdb

def process_bbox(bbox: List[float], 
                year: int = 2019,
                start_year: Optional[int] = None,
                end_year: Optional[int] = None,
                output_dir: str = None,
                generate_report: bool = False) -> Dict[str, np.ndarray]:
    """
    Process NSRDB data for a given bounding box, for a single year or a range of years.
    
    Args:
        bbox: Bounding box as [lon_min, lat_min, lon_max, lat_max]
        year: Year to process (used if start_year and end_year are None)
        start_year: Starting year for multi-year processing
        end_year: Ending year for multi-year processing
        output_dir: Output directory (created if doesn't exist)
        generate_report: Whether to generate a full report
        
    Returns:
        Dictionary with processed NSRDB data
    """
    # Default paths
    coor_path = "/data/SWATGenXApp/GenXAppData/NSRDB/CONUS_coordinates_index/CONUS_coordinates_index.shp"
    nsrdb_path_template = "/data/SWATGenXApp/GenXAppData/NSRDB/nsrdb_{}_full_filtered.h5"
    
    # Handle year range logic
    if start_year is None and end_year is None:
        # Use single year mode with the provided year parameter
        start_year = year
        end_year = year
        year_label = str(year)
    else:
        # Use the provided year range
        start_year = start_year if start_year is not None else year
        end_year = end_year if end_year is not None else start_year
        year_label = f"{start_year}-{end_year}"
    
    # Ensure start_year <= end_year
    if start_year > end_year:
        start_year, end_year = end_year, start_year
        
    # Create output directory if specified and doesn't exist
    if output_dir is None:
        output_dir = f"/data/SWATGenXApp/codes/AI_agent/output/nsrdb_{year_label}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get coordinates within the bounding box
    print(f"Extracting coordinates for bbox: {bbox}")
    coordinates_index = get_coordinates_from_bbox(coor_path, bbox)
    
    if coordinates_index.empty:
        print("No coordinates found within the bounding box")
        return {}
    
    print(f"Found {len(coordinates_index)} coordinates in the bbox")
    
    # Extract NSRDB data (single year or multi-year)
    if start_year == end_year:
        # Single year extraction
        print(f"Extracting NSRDB data for year {start_year}")
        nsrdb_data = extract_nsrdb_data(
            year=start_year,
            coordinates_index=coordinates_index,
            nsrdb_path_template=nsrdb_path_template
        )
    else:
        # Multi-year extraction
        print(f"Extracting NSRDB data for years {start_year}-{end_year}")
        years = list(range(start_year, end_year + 1))
        nsrdb_data = extract_nsrdb_multiyear(
            years=years,
            coordinates_index=coordinates_index,
            nsrdb_path_template=nsrdb_path_template
        )
    
    if not nsrdb_data:
        print("Failed to extract NSRDB data")
        return {}
    
    # Process a specific time step for visualization (for multi-year data, choose a middle time step)
    total_timesteps = nsrdb_data[list(nsrdb_data.keys())[0]].shape[0]
    time_step = min(1000, total_timesteps // 2)  # Choose middle or 1000, whichever is smaller
    print(f"Using time step {time_step} for visualization (out of {total_timesteps} total steps)")
    
    # Create interpolated grids for each variable
    grid_data = {}
    
    # Process each variable separately to avoid opening many files at once
    for var_name in ['ghi', 'wind_speed', 'relative_humidity']:
        if var_name in nsrdb_data:
            var_data = nsrdb_data[var_name]
            
            # Create points for interpolation
            points = np.column_stack((coordinates_index.longitude, coordinates_index.latitude))
            
            # Apply correct scaling based on variable
            if var_name == 'ghi':
                scale_factor = 1.0  # No scaling needed
            elif var_name == 'wind_speed':
                scale_factor = 0.1  # Divide by 10
            elif var_name == 'relative_humidity':
                scale_factor = 0.01  # Divide by 100
            else:
                scale_factor = 1.0  # Default
            
            values = var_data[time_step, :] * scale_factor
            
            # Calculate grid resolution
            lons = sorted(coordinates_index.longitude.unique())
            lats = sorted(coordinates_index.latitude.unique())
            lon_res = min(np.diff(lons)) if len(lons) > 1 else 0.04
            lat_res = min(np.diff(lats)) if len(lats) > 1 else 0.04
            
            # Create meshgrid
            grid_lon, grid_lat = np.meshgrid(
                np.arange(bbox[0], bbox[2] + lon_res, lon_res),
                np.arange(bbox[1], bbox[3] + lat_res, lat_res)
            )
            
            # Interpolate
            grid = griddata(points, values, (grid_lon, grid_lat), method='linear')
            grid_data[var_name] = grid
            
            # Save as raster
            transform = from_origin(bbox[0], bbox[3], lon_res, lat_res)
            save_as_raster(
                grid, 
                os.path.join(output_dir, f"{var_name}_timestep_{time_step}.tif"), 
                bbox, 
                coordinates_index.crs
            )
    
    # Create visualization with plots in a single row
    plt.figure(figsize=(18, 6))  # Wider figure to accommodate horizontal layout
    
    for i, (var_name, grid) in enumerate(grid_data.items()):
        # Create subplot in horizontal layout (1 row, multiple columns)
        plt.subplot(1, len(grid_data), i+1)
        
        # Plot data
        im = plt.imshow(grid, origin='lower', extent=[bbox[0], bbox[2], bbox[1], bbox[3]])
        plt.colorbar(im, label=f'{var_name}')
        
        # Add descriptive labels with correct units
        if var_name == 'ghi':
            title = 'Solar Radiation (W/m²)'
        elif var_name == 'wind_speed':
            title = 'Wind Speed (m/s)'
        elif var_name == 'relative_humidity':
            title = 'Relative Humidity (%)'
        else:
            title = f'{var_name.capitalize()}'
            
        plt.title(title)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"visualization_timestep_{time_step}.png"))
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")
    
    # Generate comprehensive report if requested
    if generate_report:
        config = {
            "bbox": bbox,
            "start_year": start_year,
            "end_year": end_year,
            "extract_for_swat": True
        }
        
        report_dir = os.path.join(output_dir, "report")
        report_path = batch_process_nsrdb(config, report_dir)
        
        if report_path:
            print(f"Report generated: {report_path}")
    
    return nsrdb_data

if __name__ == "__main__":
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Process NSRDB data for a bounding box')
    parser.add_argument('--bbox', type=float, nargs=4, 
                      default=[-85.444332, 43.658148, -85.239256, 44.164683],
                      help='Bounding box as lon_min lat_min lon_max lat_max')
    parser.add_argument('--year', type=int, default=2019,
                      help='Year to process (used if start_year and end_year are not provided)')
    parser.add_argument('--start-year', type=int, default=None,
                      help='Starting year for multi-year processing')
    parser.add_argument('--end-year', type=int, default=None,
                      help='Ending year for multi-year processing')
    parser.add_argument('--output', type=str, default=None,
                      help='Output directory')
    parser.add_argument('--report', action='store_true',
                      help='Generate comprehensive report')
    
    args = parser.parse_args()
    
    # Process the data
    data = process_bbox(
        bbox=args.bbox,
        year=args.year,
        start_year=args.start_year,
        end_year=args.end_year,
        output_dir=args.output,
        generate_report=args.report
    )
    
    if data:
        print("NSRDB data processing completed successfully")
    else:
        print("NSRDB data processing failed")


