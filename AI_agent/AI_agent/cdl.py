"""
CDL (Cropland Data Layer) dataset handling module.
Provides functionality to extract and process agricultural land use data.
"""
import h5py
import numpy as np
from scipy.spatial import cKDTree
import pandas as pd
import os
import time
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from functools import lru_cache
import matplotlib.pyplot as plt

try:
    from AI_agent.config import AgentConfig
    from AI_agent.cdl_utilities import (plot_cdl_trends, calculate_crop_changes, 
                                      export_cdl_data, create_crop_change_plot,
                                      create_crop_composition_pie, generate_cdl_report)
except ImportError:
    from config import AgentConfig
    try:
        from cdl_utilities import (plot_cdl_trends, calculate_crop_changes, 
                                export_cdl_data, create_crop_change_plot,
                                create_crop_composition_pie, generate_cdl_report)
    except ImportError:
        logging.warning("CDL utilities module not found. Visualization functions will not be available.")

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('CDL_Dataset')

class CDL_dataset:
    """Handles extraction and processing of Cropland Data Layer (CDL) data."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the CDL dataset handler.
        
        Args:
            config: Configuration dictionary containing:
                - bounding_box: [min_lon, min_lat, max_lon, max_lat]
                - start_year: First year to extract data for
                - end_year: Last year to extract data for
                - RESOLUTION: Spatial resolution in meters
                - aggregation: Temporal aggregation type
        """
        self.config = config
        self.data_path = AgentConfig.HydroGeoDataset_ML_250_path
        self.cdl_codes_path = AgentConfig.CDL_CODES_path
        self._validate_config()
        self.extracted_data = {}
        
    def _validate_config(self) -> None:
        """Validate configuration parameters and convert types if needed."""
        # Check if bounding box is valid
        if 'bounding_box' not in self.config or len(self.config['bounding_box']) != 4:
            raise ValueError("Config must include valid bounding_box with 4 values [min_lon, min_lat, max_lon, max_lat]")
            
        # Ensure years are integers
        for year_key in ['start_year', 'end_year']:
            if year_key in self.config:
                self.config[year_key] = int(self.config[year_key])
            else:
                raise ValueError(f"Config missing required '{year_key}' parameter")
        
        # Check if data file exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"HDF5 data file not found: {self.data_path}")
        
        if not os.path.exists(self.cdl_codes_path):
            raise FileNotFoundError(f"CDL codes file not found: {self.cdl_codes_path}")
    
    def cdl_trends(self) -> Dict[int, Dict[str, Any]]:
        """
        Extract CDL data for the specified years and region.
        
        Returns:
            Dictionary mapping years to land use data with area in hectares.
        """
        start_time = time.time()
        logger.info("Starting CDL data extraction...")
        
        lat_range = (self.config['bounding_box'][1], self.config['bounding_box'][3])
        lon_range = (self.config['bounding_box'][0], self.config['bounding_box'][2])
        
        years = list(range(self.config['start_year'], self.config['end_year'] + 1))
        logger.info(f"Querying years: {years}")

        # Determine available years in the dataset
        available_years = self._get_available_years(years)
        if not available_years:
            logger.warning("No requested years are available in the dataset")
            return {}

        # Extract data for each available year
        self.extracted_data = {}
        for year in available_years:
            year_start = time.time()
            try:
                data = self.read_h5_file(
                    lat_range=lat_range,
                    lon_range=lon_range,
                    address=f"CDL/{year}",
                )
                if data:
                    self.extracted_data[int(year)] = data
                    logger.info(f"Year {year} processed in {time.time() - year_start:.2f} seconds")
            except Exception as e:
                logger.error(f"Error extracting CDL data for year {year}: {e}")

        total_time = time.time() - start_time
        logger.info(f"Total CDL data extraction completed in {total_time:.2f} seconds")
        
        if not self.extracted_data:
            logger.warning("No data could be extracted for any year")
        else:
            logger.info(f"Successfully extracted data for years: {list(self.extracted_data.keys())}")

        return self.extracted_data

    def _get_available_years(self, requested_years: List[int]) -> List[int]:
        """
        Check which years are available in the dataset.
        
        Args:
            requested_years: List of years to check for availability
            
        Returns:
            List of available years that were requested
        """
        available_years = []
        try:
            with h5py.File(self.data_path, 'r') as f:
                # Check which requested years are available
                for year in requested_years:
                    if f"CDL/{year}" in f:
                        available_years.append(year)
                    else:
                        logger.warning(f"Year {year} not found in dataset")

                # If none of the requested years are available, suggest alternatives
                if not available_years and 'CDL' in f:
                    all_years = [int(y) for y in list(f['CDL'].keys()) if y.isdigit()]
                    logger.info(f"Available years in dataset: {all_years}")
                    if all_years:
                        closest_years = sorted(all_years, 
                                             key=lambda y: min(abs(y - yr) for yr in requested_years))
                        use_years = closest_years[:3]  # Use up to 3 closest years
                        logger.info(f"Using fallback years: {use_years}")
                        available_years = use_years
        except Exception as e:
            logger.error(f"Error checking available years: {e}")
            
        return available_years

    @lru_cache(maxsize=128)
    def CDL_lookup(self, code: int) -> str:
        """
        Look up the crop name for a given CDL code.
        
        Args:
            code: CDL numeric code
            
        Returns:
            Crop name corresponding to the code
        """
        try:
            df = pd.read_csv(self.cdl_codes_path)
            result = df[df['CODE'] == code]['NAME'].values
            if len(result) > 0:
                return result[0]
            return f"Unknown code: {code}"
        except Exception as e:
            logger.error(f"Error looking up CDL code {code}: {e}")
            return f"Error: {code}"

    def get_coordinates(self, lat=None, lon=None, lat_range=None, lon_range=None) -> Tuple[Optional[Tuple], Optional[Tuple]]:
        """
        Get array indices from geographic coordinates.
        
        Args:
            lat: Target latitude for point query
            lon: Target longitude for point query
            lat_range: (min_lat, max_lat) for area query
            lon_range: (min_lon, max_lon) for area query
            
        Returns:
            Tuple containing:
                - For point query: (row_index, col_index), None
                - For area query: None, (min_row, max_row, min_col, max_col)
        """
        try:
            with h5py.File(self.data_path, 'r') as f:
                lat_ = f["geospatial/lat_250m"][:]
                lon_ = f["geospatial/lon_250m"][:]
                lat_ = np.where(lat_ == -999, np.nan, lat_)
                lon_ = np.where(lon_ == -999, np.nan, lon_)

                if lat is not None and lon is not None:
                    valid_mask = ~np.isnan(lat_) & ~np.isnan(lon_)
                    coordinates = np.column_stack((lat_[valid_mask], lon_[valid_mask]))
                    _, idx = cKDTree(coordinates).query([lat, lon])
                    valid_indices = np.where(valid_mask)
                    return (valid_indices[0][idx], valid_indices[1][idx]), None

                if lat_range and lon_range:
                    min_lat, max_lat = lat_range
                    min_lon, max_lon = lon_range
                    mask = (lat_ >= min_lat) & (lat_ <= max_lat) & (lon_ >= min_lon) & (lon_ <= max_lon)
                    if np.any(mask):
                        rows, cols = np.where(mask)
                        return None, (np.min(rows), np.max(rows), np.min(cols), np.max(cols))
                    else:
                        logger.warning("No data points found within the specified geographic range")
        except Exception as e:
            logger.error(f"Error determining coordinates: {e}")
            
        return None, None

    def read_h5_file(self, address, lat=None, lon=None, lat_range=None, lon_range=None, path=None) -> Optional[Dict]:
        """
        Read data from HDF5 file for specified coordinates or region.
        
        Args:
            address: HDF5 dataset path within the file
            lat: Target latitude (optional)
            lon: Target longitude (optional)
            lat_range: (min_lat, max_lat) tuple (optional)
            lon_range: (min_lon, max_lon) tuple (optional)
            path: Alternative path to HDF5 file (optional)
            
        Returns:
            Dictionary containing extracted data or None if error
        """
        path = path or self.data_path
        
        if not os.path.exists(path):
            logger.error(f"File not found: {path}")
            return None

        try:
            point_idx, range_idx = self.get_coordinates(lat, lon, lat_range, lon_range)
            with h5py.File(path, 'r') as f:
                if point_idx:
                    data = f[address][point_idx[0], point_idx[1]]
                    return {"value": self.CDL_lookup(data) if "CDL" in address else data}
                
                if range_idx:
                    min_row, max_row, min_col, max_col = range_idx
                    data = f[address][min_row:max_row+1, min_col:max_col+1]
                    data = np.where(data == -999, np.nan, data)
                    return self.process_data(data, address)
                
                return {"value": f[address][:]}
        except KeyError as e:
            logger.error(f"Dataset {address} not found in file: {e}")
            return None
        except Exception as e:
            logger.error(f"Error reading data from {address}: {e}")
            return None

    def process_data(self, data: np.ndarray, address: str) -> Dict[str, Any]:
        """
        Process extracted array data into meaningful statistics or summaries.
        
        Args:
            data: NumPy array of extracted data
            address: HDF5 dataset path from which data was extracted
            
        Returns:
            Dictionary with processed data metrics or land use statistics
        """
        # Skip processing if data is empty or all NaN
        if data.size == 0 or np.all(np.isnan(data)):
            logger.warning("Data array is empty or contains only NaN values")
            return {"error": "No valid data points"}
            
        if "CDL" in address:
            # Calculate area for each crop type
            cell_area_ha = 6.25  # Area of one 250m cell in hectares
            
            # Handle non-finite values
            valid_data = data[np.isfinite(data)]
            if valid_data.size == 0:
                return {"error": "No valid data points"}
                
            unique, counts = np.unique(valid_data, return_counts=True)
            
            # Create dictionary with land use types and areas
            result = {}
            total_area = 0.0
            
            for code, count in zip(unique, counts):
                if not np.isnan(code):
                    crop_name = self.CDL_lookup(int(code))
                    area = float(count * cell_area_ha)
                    result[crop_name] = area
                    total_area += area
            
            # Add summary statistics
            result["Total Area"] = total_area
            result["unit"] = "hectares"
            
            # Add percentage for each crop type
            if total_area > 0:
                for crop_name in list(result.keys()):
                    if crop_name not in ["Total Area", "unit"]:
                        result[f"{crop_name} (%)"] = round(result[crop_name] / total_area * 100, 2)
            
            return result
        else:
            # For non-CDL data, return general statistics
            valid_data = data[np.isfinite(data)]
            if valid_data.size == 0:
                return {"error": "No valid data points"}
                
            return {
                "number of cells": int(valid_data.size),
                "median": float(np.nanmedian(valid_data).round(2)),
                "max": float(np.nanmax(valid_data).round(2)),
                "min": float(np.nanmin(valid_data).round(2)),
                "mean": float(np.nanmean(valid_data).round(2)),
                "std": float(np.nanstd(valid_data).round(2)),
            }

    def plot_trends(self, output_path: Optional[str] = None, 
                   top_n: int = 5, title: str = None) -> Optional[plt.Figure]:
        """
        Create a visualization of CDL trends over time.
        
        Args:
            output_path: Path to save the figure (optional)
            top_n: Number of top crops to display individually
            title: Custom title for the plot (uses default if None)
            
        Returns:
            Matplotlib Figure object or None if error occurs
        """
        if not self.extracted_data:
            logger.warning("No data available for plotting. Run cdl_trends() first.")
            return None
            
        try:
            # Check if cdl_utilities module is available
            if 'plot_cdl_trends' in globals():
                # Create a default title if not provided
                if title is None:
                    years = sorted(self.extracted_data.keys())
                    year_range = f"{years[0]}-{years[-1]}"
                    bbox = self.config['bounding_box']
                    title = f"Crop Distribution ({year_range}): {bbox[1]:.2f}°N, {bbox[0]:.2f}°E to {bbox[3]:.2f}°N, {bbox[2]:.2f}°E"
                
                return plot_cdl_trends(
                    cdl_data=self.extracted_data,
                    top_n=top_n,
                    output_path=output_path,
                    title=title
                )
            else:
                logger.error("CDL utilities module not available. Cannot create plot.")
                return None
        except Exception as e:
            logger.error(f"Error creating trends plot: {e}", exc_info=True)
            return None

    def export_data(self, output_path: str) -> bool:
        """
        Export extracted CDL data to a CSV file.
        
        Args:
            output_path: Path to save the CSV file
            
        Returns:
            Boolean indicating success
        """
        if not self.extracted_data:
            logger.warning("No data available for export. Run cdl_trends() first.")
            return False
            
        try:
            if 'export_cdl_data' in globals():
                return export_cdl_data(
                    cdl_data=self.extracted_data,
                    output_path=output_path,
                    include_percentages=True
                )
            else:
                logger.error("CDL utilities module not available. Cannot export data.")
                return False
        except Exception as e:
            logger.error(f"Error exporting data: {e}", exc_info=True)
            return False

    def generate_report(self, output_dir: str, 
                       report_name: str = None) -> str:
        """
        Generate a comprehensive CDL analysis report including visualizations.
        
        Args:
            output_dir: Directory to save the report and visualization files
            report_name: Base name for report files (uses area name if None)
            
        Returns:
            Path to the generated report file or empty string if error occurs
        """
        if not self.extracted_data:
            logger.warning("No data available for reporting. Run cdl_trends() first.")
            return ""
            
        try:
            if 'generate_cdl_report' in globals():
                # Create a default report name if not provided
                if report_name is None:
                    # Generate a name based on the bounding box
                    bbox = self.config['bounding_box']
                    report_name = f"CDL_Report_{bbox[1]:.2f}N_{bbox[0]:.2f}E"
                
                return generate_cdl_report(
                    cdl_data=self.extracted_data,
                    output_dir=output_dir,
                    report_name=report_name,
                    include_plots=True
                )
            else:
                logger.error("CDL utilities module not available. Cannot generate report.")
                return ""
        except Exception as e:
            logger.error(f"Error generating report: {e}", exc_info=True)
            return ""
            
    def analyze_changes(self, custom_years: Optional[Tuple[int, int]] = None) -> pd.DataFrame:
        """
        Analyze changes in crop distribution between years.
        
        Args:
            custom_years: Optional tuple of (start_year, end_year) to compare
                         (uses first and last available years if None)
                         
        Returns:
            Pandas DataFrame with detailed change statistics
        """
        if not self.extracted_data or len(self.extracted_data) < 2:
            logger.warning("Insufficient data available for change analysis. Need at least two years.")
            return pd.DataFrame()
            
        try:
            if 'calculate_crop_changes' in globals():
                _, df = calculate_crop_changes(
                    cdl_data=self.extracted_data,
                    custom_years=custom_years
                )
                return df
            else:
                logger.error("CDL utilities module not available. Cannot analyze changes.")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error analyzing changes: {e}", exc_info=True)
            return pd.DataFrame()
                

if __name__ == "__main__":
    config = {
        "RESOLUTION": 250,
        "aggregation": "annual",
        "start_year": 2010,
        "end_year": 2012,
        "bounding_box": [-85.444332, 43.658148, -85.239256, 44.164683],
    }
    cdl = CDL_dataset(config)
    data = cdl.cdl_trends()
    
    # Example of using the utility functions
    if data:
        print(f"Extracted data for {len(data)} years")
        
        # Create output directory
        output_dir = os.path.join(os.getcwd(), "cdl_results")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate plots and export data
        cdl.plot_trends(output_path=os.path.join(output_dir, "cdl_trends.png"))
        cdl.export_data(output_path=os.path.join(output_dir, "cdl_data.csv"))
        
        # Analyze and report
        changes_df = cdl.analyze_changes()
        if not changes_df.empty:
            print("\nTop 5 crop changes:")
            print(changes_df.head(5)[['Crop', 'Change (ha)', 'Status']])
        
        # Generate comprehensive report
        report_path = cdl.generate_report(output_dir)
        if report_path:
            print(f"\nReport generated: {report_path}")
    
    print("Done.")