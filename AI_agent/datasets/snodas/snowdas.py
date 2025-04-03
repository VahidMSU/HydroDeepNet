import numpy as np
import h5py
import pandas as pd
import time
import os
import logging
try:
    from datasets.config import AgentConfig
except ImportError:
    from config import AgentConfig

class SNODAS_Dataset:
    def __init__(self, config):
        self.config = config
        self.database_path = AgentConfig.HydroGeoDataset_ML_250_path
        self.snodas_path = AgentConfig.SNODAS_PATH if hasattr(AgentConfig, 'SNODAS_PATH') else '/data/SWATGenXApp/GenXAppData/HydroGeoDataset/SNODAS.h5'
        self.base_mask = self._get_mask()
        
        # Logger setup
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _get_mask(self):
        with h5py.File(self.database_path, 'r') as f:
            DEM_ = f[f"geospatial/BaseRaster_{self.config['RESOLUTION']}m"][:]
            return np.where(DEM_ == -999, 0, 1)

    def get_rowcol_range_by_latlon(self, desired_min_lat, desired_max_lat, desired_min_lon, desired_max_lon):
        with h5py.File(self.database_path, 'r') as f:
            lat_ = f["geospatial/lat_250m"][:]
            lon_ = f["geospatial/lon_250m"][:]
            
            # Clean up invalid values
            lat_ = np.where(lat_ == -999, np.nan, lat_)
            lon_ = np.where(lon_ == -999, np.nan, lon_)

            # Add buffer to improve matching
            buffer = 0.01  # ~1km buffer
            lat_mask = (lat_ >= (desired_min_lat - buffer)) & (lat_ <= (desired_max_lat + buffer))
            lon_mask = (lon_ >= (desired_min_lon - buffer)) & (lon_ <= (desired_max_lon + buffer))
            combined_mask = lat_mask & lon_mask

            if not np.any(combined_mask):
                self.logger.warning("No exact matches found, expanding search area...")
                buffer = 0.05  # Expand buffer to ~5km
                lat_mask = (lat_ >= (desired_min_lat - buffer)) & (lat_ <= (desired_max_lat + buffer))
                lon_mask = (lon_ >= (desired_min_lon - buffer)) & (lon_ <= (desired_max_lon + buffer))
                combined_mask = lat_mask & lon_mask

            if np.any(combined_mask):
                row_indices, col_indices = np.where(combined_mask)
                return (
                    np.min(row_indices),
                    np.max(row_indices),
                    np.min(col_indices),
                    np.max(col_indices)
                )
            
            self.logger.error("Could not find valid points for the given coordinates")
            return None, None, None, None

    def aggregate_temporal_data(self, snow_data_dict) -> dict:
        """
        Aggregate SNODAS data temporally based on configuration.
        
        Args:
            snow_data_dict: Dictionary with arrays for snow variables
            
        Returns:
            Dictionary with aggregated arrays for snow variables
        """
        # Ensure we have data
        if not snow_data_dict or not any(v.size > 0 for v in snow_data_dict.values()):
            self.logger.warning("No data to aggregate")
            return snow_data_dict

        # Get reference variable with data to determine total days
        sample_var = next((v for v in snow_data_dict.values() if v.size > 0), None)
        if sample_var is None:
            return snow_data_dict
            
        total_days = sample_var.shape[0]
        start_date = f"{self.config['start_year']}-01-01"
        dates = pd.date_range(start=start_date, periods=total_days)
        
        self.logger.info(f"#################Aggregation method: {self.config['aggregation']}#################")
        
        aggregated_data = {}
        
        if self.config['aggregation'] == 'monthly':
            for var_name, var_data in snow_data_dict.items():
                if var_data.size == 0:
                    aggregated_data[var_name] = np.array([])
                    continue
                    
                # Determine aggregation method based on variable
                if var_name in ['snow_water_equivalent', 'snow_layer_thickness']:
                    # Use mean for stock variables (state variables)
                    var_monthly = [var_data[(dates.year == year) & (dates.month == month), :, :].mean(axis=0)
                                for year in np.unique(dates.year)
                                for month in range(1, 13) if np.sum((dates.year == year) & (dates.month == month)) > 0]
                else:
                    # Use sum for flux variables (rate variables)
                    var_monthly = [var_data[(dates.year == year) & (dates.month == month), :, :].sum(axis=0)
                                for year in np.unique(dates.year)
                                for month in range(1, 13) if np.sum((dates.year == year) & (dates.month == month)) > 0]
                
                aggregated_data[var_name] = np.array(var_monthly)
                self.logger.info(f"Aggregated {var_name} data shape (monthly): {aggregated_data[var_name].shape}")

        elif self.config['aggregation'] == 'seasonal':
            seasons = {
                'DJF': [12, 1, 2],
                'MAM': [3, 4, 5],
                'JJA': [6, 7, 8],
                'SON': [9, 10, 11]
            }

            for var_name, var_data in snow_data_dict.items():
                if var_data.size == 0:
                    aggregated_data[var_name] = np.array([])
                    continue
                    
                # Determine aggregation method based on variable
                if var_name in ['snow_water_equivalent', 'snow_layer_thickness']:
                    # Use mean for stock variables
                    var_seasonal = [var_data[np.isin(dates.month, months) & (dates.year == year), :, :].mean(axis=0)
                                 for year in np.unique(dates.year)
                                 for season, months in seasons.items()]
                else:
                    # Use sum for flux variables
                    var_seasonal = [var_data[np.isin(dates.month, months) & (dates.year == year), :, :].sum(axis=0)
                                 for year in np.unique(dates.year)
                                 for season, months in seasons.items()]
                
                aggregated_data[var_name] = np.array(var_seasonal)
                self.logger.info(f"Aggregated {var_name} data shape (seasonal): {aggregated_data[var_name].shape}")

        elif self.config['aggregation'] == 'annual':
            for var_name, var_data in snow_data_dict.items():
                if var_data.size == 0:
                    aggregated_data[var_name] = np.array([])
                    continue
                    
                # Determine aggregation method based on variable
                if var_name in ['snow_water_equivalent', 'snow_layer_thickness']:
                    # Use mean for stock variables
                    var_annual = [var_data[dates.year == year, :, :].mean(axis=0) 
                               for year in np.unique(dates.year)]
                else:
                    # Use sum for flux variables
                    var_annual = [var_data[dates.year == year, :, :].sum(axis=0) 
                               for year in np.unique(dates.year)]
                
                aggregated_data[var_name] = np.array(var_annual)
                self.logger.info(f"Aggregated {var_name} data shape (annual): {aggregated_data[var_name].shape}")
        else:
            self.logger.warning(f"Unknown aggregation type: {self.config['aggregation']}. Using original data.")
            return snow_data_dict

        return aggregated_data

    def get_data(self, start_year=None, end_year=None) -> dict:
        data_start = time.time()
        self.logger.info("Extracting SNODAS data...")
        if start_year is None:
            start_year = int(self.config.get('start_year'))
        if end_year is None:
            end_year = int(self.config.get('end_year'))

        assert start_year is not None, "start_year is not provided."
        assert end_year is not None, "end_year is not provided."

        with h5py.File(self.snodas_path, 'r') as f:
            self.logger.info(f"SNODAS keys: {f.keys()}")
            
            # Get row/col indices if bounding box specified
            if self.config.get('bounding_box'):
                bbox = self.config['bounding_box']
                indices = self.get_rowcol_range_by_latlon(bbox[1], bbox[3], bbox[0], bbox[2])
                
                if not all(idx is not None for idx in indices):
                    self.logger.warning("Warning: Invalid coordinates")
                    return None
                    
                row_min, row_max, col_min, col_max = indices
            else:
                row_min = col_min = None
            
            # Check available years and variables
            available_years = []
            for year in range(start_year, end_year + 1):
                if f'250m/{year}' in f:
                    available_years.append(year)
            
            if not available_years:
                self.logger.warning(f"No data available for years {start_year}-{end_year}")
                return None
                
            self.logger.info(f"Available years: {available_years}")
            
            # Get available variables from first year
            first_year = str(available_years[0])
            available_vars = list(f[f'250m/{first_year}'].keys())
            self.logger.info(f"Available variables: {available_vars}")
            
            # Initialize dict to store data for each variable
            snow_data = {}
            for var_name in available_vars:
                snow_data[var_name] = []
            
            # Process each year
            for year in available_years:
                year_str = str(year)
                
                for var_name in available_vars:
                    try:
                        # Get scale factor from attributes
                        scale_factor = f[f'250m/{year_str}'][var_name].attrs.get('converters', 1.0)
                        
                        # Read data with bounds checking
                        if row_min is not None:
                            data = f[f'250m/{year_str}'][var_name][:, row_min:row_max+1, col_min:col_max+1]
                        else:
                            data = f[f'250m/{year_str}'][var_name][:]
                            
                        # Apply mask
                        if row_min is not None:
                            mask = self.base_mask[row_min:row_max+1, col_min:col_max+1]
                        else:
                            mask = self.base_mask
                            
                        time_mask = np.broadcast_to(mask, data.shape)
                        
                        # Scale and clean data
                        data = np.where(time_mask != 1, np.nan, data * scale_factor)
                        
                        # Append to our collection
                        snow_data[var_name].append(data)
                        self.logger.info(f"Extracted {var_name} for {year}, shape: {data.shape}")
                    except Exception as e:
                        self.logger.error(f"Error extracting {var_name} for {year}: {e}")
            
            # Concatenate data for each variable
            for var_name in available_vars:
                if snow_data[var_name]:
                    try:
                        snow_data[var_name] = np.concatenate(snow_data[var_name], axis=0)
                        self.logger.info(f"Final shape for {var_name}: {snow_data[var_name].shape}")
                    except Exception as e:
                        self.logger.error(f"Error concatenating data for {var_name}: {e}")
                        snow_data[var_name] = np.array([])
                else:
                    snow_data[var_name] = np.array([])
            
            # Apply temporal aggregation if specified
            if self.config.get('aggregation'):
                snow_data = self.aggregate_temporal_data(snow_data)
                
            total_time = time.time() - data_start
            self.logger.info(f"Total SNODAS data extraction took {total_time:.2f} seconds")
            
            return snow_data
    
    def get_spatial_average_over_time(self):
        start_time = time.time()
        snow_data = self.get_data()
        
        if not snow_data:
            self.logger.warning("No data to process")
            return {}
        
        spatial_averages = {}
        avg_start = time.time()
        
        for var_name, var_data in snow_data.items():
            if var_data.size > 0:
                spatial_averages[var_name] = np.nanmean(var_data, axis=(1, 2))
                self.logger.info(f"Shape of {var_name}: {spatial_averages[var_name].shape}, " 
                               f"Range: {np.nanmin(spatial_averages[var_name]):.2f} to "
                               f"{np.nanmax(spatial_averages[var_name]):.2f}")
                
        avg_time = time.time() - avg_start
        self.logger.info(f"Spatial averaging took {avg_time:.2f} seconds")
        self.logger.info(f"Total spatial average processing took {time.time() - start_time:.2f} seconds")
        
        return spatial_averages
        
    def calculate_snow_cover_duration(self):
        """
        Calculate the snow cover duration for each pixel in the dataset.
        
        Returns:
            Dictionary containing snow cover duration statistics
        """
        snow_data = self.get_data()
        if not snow_data or 'snow_water_equivalent' not in snow_data:
            self.logger.warning("No SWE data to calculate snow cover duration")
            return {}
            
        swe_data = snow_data['snow_water_equivalent']
        if swe_data.size == 0:
            return {}
            
        # Define SWE threshold for considering a pixel as snow-covered (e.g., 5mm)
        swe_threshold = 5.0
        
        # Calculate snow-covered days for each pixel
        snow_covered = np.where(swe_data > swe_threshold, 1, 0)
        
        # Calculate duration statistics
        snow_duration = np.sum(snow_covered, axis=0)  # Sum along time axis
        
        # Calculate statistics
        mean_duration = np.nanmean(snow_duration)
        max_duration = np.nanmax(snow_duration)
        min_duration = np.nanmin(snow_duration)
        std_duration = np.nanstd(snow_duration)
        
        # Store in dictionary
        result = {
            'duration_map': snow_duration,
            'mean_duration': mean_duration,
            'max_duration': max_duration,
            'min_duration': min_duration,
            'std_duration': std_duration
        }
        
        self.logger.info(f"Snow cover duration stats - Mean: {mean_duration:.1f}, Max: {max_duration:.1f}, Min: {min_duration:.1f} days")
        return result
        
    def detect_snow_timing(self):
        """
        Detect the timing of snow accumulation and melt across the domain.
        
        Returns:
            Dictionary with statistics about snow timing
        """
        snow_data = self.get_data()
        if not snow_data or 'snow_water_equivalent' not in snow_data:
            self.logger.warning("No SWE data to detect snow timing")
            return {}
            
        swe_data = snow_data['snow_water_equivalent']
        if swe_data.size == 0 or self.config.get('aggregation') != 'monthly':
            self.logger.warning("Snow timing detection requires monthly aggregation")
            return {}
            
        # Get spatial mean SWE for each time step
        spatial_mean_swe = np.nanmean(swe_data, axis=(1, 2))
        
        # Determine the month with maximum SWE in each year
        start_year = self.config.get('start_year')
        end_year = self.config.get('end_year')
        years = range(start_year, end_year + 1)
        peak_months = []
        
        for year_idx, year in enumerate(years):
            year_indices = range(year_idx*12, (year_idx+1)*12)
            if max(year_indices) < len(spatial_mean_swe):
                year_data = spatial_mean_swe[year_indices]
                peak_month = np.argmax(year_data) + 1  # +1 because months are 1-indexed
                peak_months.append(peak_month)
        
        result = {
            'peak_swe_months': peak_months,
            'mean_peak_month': np.mean(peak_months) if peak_months else None,
            'years': list(years)
        }
        
        self.logger.info(f"Snow peak timing - mean peak month: {result['mean_peak_month']:.1f}")
        return result
    
    def analyze_interannual_variability(self):
        """
        Analyze the inter-annual variability of snow variables.
        
        Returns:
            Dictionary with inter-annual variability statistics
        """
        spatial_averages = self.get_spatial_average_over_time()
        if not spatial_averages:
            return {}
            
        # If not annual aggregation, recalculate annual values
        if self.config.get('aggregation') != 'annual':
            # Get daily data and create annual averages
            original_agg = self.config.get('aggregation')
            self.config['aggregation'] = 'annual'
            annual_data = self.get_spatial_average_over_time()
            self.config['aggregation'] = original_agg
        else:
            annual_data = spatial_averages
            
        # Calculate coefficient of variation (CV) for each variable
        result = {}
        for var_name, var_data in annual_data.items():
            if len(var_data) > 1:  # Need at least 2 years for CV
                mean_value = np.nanmean(var_data)
                std_value = np.nanstd(var_data)
                cv = (std_value / mean_value) * 100 if mean_value > 0 else float('nan')
                
                anomalies = var_data - mean_value
                normalized_anomalies = anomalies / mean_value if mean_value > 0 else anomalies
                
                result[var_name] = {
                    'values': var_data.tolist(),
                    'mean': float(mean_value),
                    'std': float(std_value),
                    'cv_percent': float(cv),
                    'anomalies': anomalies.tolist(),
                    'normalized_anomalies': normalized_anomalies.tolist()
                }
                
        self.logger.info(f"Calculated inter-annual variability for {len(result)} variables")
        return result
    
    def calculate_extreme_statistics(self):
        """
        Calculate statistics about extreme snow events.
        
        Returns:
            Dictionary with extreme snow statistics
        """
        snow_data = self.get_data()
        if not snow_data:
            return {}
            
        result = {}
        
        # Analyze SWE extremes if available
        if 'snow_water_equivalent' in snow_data and snow_data['snow_water_equivalent'].size > 0:
            swe = snow_data['snow_water_equivalent']
            spatial_mean_swe = np.nanmean(swe, axis=(1, 2))
            
            # Find the highest SWE value and its time index
            max_swe_idx = np.nanargmax(spatial_mean_swe)
            max_swe_value = spatial_mean_swe[max_swe_idx]
            
            # Calculate percentiles for SWE distribution
            p90 = np.nanpercentile(spatial_mean_swe, 90)
            p95 = np.nanpercentile(spatial_mean_swe, 95)
            p99 = np.nanpercentile(spatial_mean_swe, 99)
            
            # Count extreme events (days above 90th percentile)
            extreme_days = np.sum(spatial_mean_swe > p90)
            
            result['swe_extremes'] = {
                'max_value': float(max_swe_value),
                'max_index': int(max_swe_idx),
                'p90': float(p90),
                'p95': float(p95),
                'p99': float(p99),
                'extreme_days': int(extreme_days)
            }
            
        # Analyze melt rate extremes if available
        if 'melt_rate' in snow_data and snow_data['melt_rate'].size > 0:
            melt = snow_data['melt_rate']
            spatial_mean_melt = np.nanmean(melt, axis=(1, 2))
            
            # Find the highest melt rate and its time index
            max_melt_idx = np.nanargmax(spatial_mean_melt)
            max_melt_value = spatial_mean_melt[max_melt_idx]
            
            # Calculate percentiles for melt rate distribution
            p90 = np.nanpercentile(spatial_mean_melt, 90)
            p95 = np.nanpercentile(spatial_mean_melt, 95)
            p99 = np.nanpercentile(spatial_mean_melt, 99)
            
            # Count extreme events (days above 90th percentile)
            extreme_days = np.sum(spatial_mean_melt > p90)
            
            result['melt_extremes'] = {
                'max_value': float(max_melt_value),
                'max_index': int(max_melt_idx),
                'p90': float(p90),
                'p95': float(p95),
                'p99': float(p99),
                'extreme_days': int(extreme_days)
            }
            
        self.logger.info(f"Calculated extreme statistics for {len(result)} variables")
        return result


if __name__ == "__main__":
    # Example usage
    config = { 
        "RESOLUTION": 250,
        "aggregation": "annual",
        "start_year": 2010,
        "end_year": 2015,
        'bounding_box': [-85.444332, 43.658148, -85.239256, 44.164683],
    }
    
    snodas_dataset = SNODAS_Dataset(config)
    snow_data = snodas_dataset.get_spatial_average_over_time()
    
    print("Available variables:")
    for var_name, values in snow_data.items():
        print(f"{var_name}: {values.shape}")

