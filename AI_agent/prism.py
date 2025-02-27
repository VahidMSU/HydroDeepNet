import numpy as np
import h5py
import pandas as pd
import time
try:
    from config import AgentConfig
except ImportError:
    from AI_agent.config import AgentConfig
    
class PRISM_Dataset:
    def __init__(self, config):
        self.config = config
        self.database_path = AgentConfig.HydroGeoDataset_ML_250_path
        self.prism_path = AgentConfig.PRISM_PATH
        self.base_mask = self._get_mask()

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
                print("Warning: No exact matches found, expanding search area...")
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
            
            print("Error: Could not find valid points for the given coordinates")
            return None, None, None, None

    def aggregate_temporal_data(self, pr, tmax, tmin) -> np.ndarray:
        min_temporal = min(pr.shape[0], tmax.shape[0], tmin.shape[0])
        pr = pr[:min_temporal, :, :]
        tmax = tmax[:min_temporal, :, :]
        tmin = tmin[:min_temporal, :, :]
        print(f"#################Aggregation method: {self.config['aggregation']}#################")
        
        total_days = pr.shape[0]
        start_date = f"{self.config['start_year']}-01-01"
        dates = pd.date_range(start=start_date, periods=total_days)
        
        if self.config['aggregation'] == 'monthly':
            pr_monthly = [pr[(dates.year == year) & (dates.month == month), :, :].sum(axis=0)
                        for year in np.unique(dates.year)
                        for month in range(1, 13) if np.sum((dates.year == year) & (dates.month == month)) > 0]
            
            tmax_monthly = [tmax[(dates.year == year) & (dates.month == month), :, :].mean(axis=0)
                            for year in np.unique(dates.year)
                            for month in range(1, 13) if np.sum((dates.year == year) & (dates.month == month)) > 0]
            
            tmin_monthly = [tmin[(dates.year == year) & (dates.month == month), :, :].mean(axis=0)
                            for year in np.unique(dates.year)
                            for month in range(1, 13) if np.sum((dates.year == year) & (dates.month == month)) > 0]
            
            pr = np.array(pr_monthly)
            tmax = np.array(tmax_monthly)
            tmin = np.array(tmin_monthly)
            print(f"Aggregated data shape (monthly): {pr.shape}, {tmax.shape}, {tmin.shape}")

        elif self.config['aggregation'] == 'seasonal':
            seasons = {
                'DJF': [12, 1, 2],
                'MAM': [3, 4, 5],
                'JJA': [6, 7, 8],
                'SON': [9, 10, 11]
            }

            pr_seasonal = [pr[np.isin(dates.month, months) & (dates.year == year), :, :].sum(axis=0)
                        for year in np.unique(dates.year)
                        for season, months in seasons.items()]
            
            tmax_seasonal = [tmax[np.isin(dates.month, months) & (dates.year == year), :, :].mean(axis=0)
                            for year in np.unique(dates.year)
                            for season, months in seasons.items()]
            
            tmin_seasonal = [tmin[np.isin(dates.month, months) & (dates.year == year), :, :].mean(axis=0)
                            for year in np.unique(dates.year)
                            for season, months in seasons.items()]

            pr = np.array(pr_seasonal)
            tmax = np.array(tmax_seasonal)
            tmin = np.array(tmin_seasonal)
            print(f"Aggregated data shape (seasonal): {pr.shape}, {tmax.shape}, {tmin.shape}")

        elif self.config['aggregation'] == 'annual':
            pr_annual = [pr[dates.year == year, :, :].sum(axis=0) for year in np.unique(dates.year)]
            tmax_annual = [tmax[dates.year == year, :, :].mean(axis=0) for year in np.unique(dates.year)]
            tmin_annual = [tmin[dates.year == year, :, :].mean(axis=0) for year in np.unique(dates.year)]

            pr = np.array(pr_annual)
            tmax = np.array(tmax_annual)
            tmin = np.array(tmin_annual)
            print(f"Aggregated data shape (annual): {pr.shape}, {tmax.shape}, {tmin.shape}")

        return pr, tmax, tmin

    def get_data(self, start_year=None, end_year=None) -> np.ndarray:
        data_start = time.time()
        print("Extracting PRISM data...")
        if start_year is None:
            start_year = int(self.config.get('start_year'))
        if end_year is None:
            end_year = int(self.config.get('end_year'))

        assert start_year is not None, "start_year is not provided."
        assert end_year is not None, "end_year is not provided."

        with h5py.File(self.prism_path, 'r') as f:
            print(f"PRISM keys: {f.keys()}")
            print(f"Range of available years: {f['ppt'].keys()}")
            
            # Get row/col indices once
            if self.config.get('bounding_box'):
                bbox = self.config['bounding_box']
                indices = self.get_rowcol_range_by_latlon(bbox[1], bbox[3], bbox[0], bbox[2])
                
                if not all(idx is not None for idx in indices):
                    print("Warning: Invalid coordinates")
                    return None, None, None
                    
                row_min, row_max, col_min, col_max = indices
            else:
                row_min = col_min = None

            # Pre-calculate available years
            available_years = np.arange(self.config['start_year'], self.config['end_year'] + 1) 
            
        
            try:
                io_start = time.time()
                
                # Read all years at once for each variable
                ppts = []
                tmaxs = []
                tmins = []
                
                # Create dataset references first
                datasets = {
                    'ppt': [f[f'ppt/{year}/data'] for year in available_years],
                    'tmax': [f[f'tmax/{year}/data'] for year in available_years],
                    'tmin': [f[f'tmin/{year}/data'] for year in available_years]
                }
                
                # Read data for all years
                for year_idx, year in enumerate(available_years):
                    # Read data with bounds checking
                    
                    
                    if row_min is not None:
                        ppt = datasets['ppt'][year_idx][:, row_min:row_max+1, col_min:col_max+1]
                        tmax = datasets['tmax'][year_idx][:, row_min:row_max+1, col_min:col_max+1]
                        tmin = datasets['tmin'][year_idx][:, row_min:row_max+1, col_min:col_max+1]
       
                    # Standardize time dimension
                    min_days = min(ppt.shape[0], tmax.shape[0], tmin.shape[0], 355)
                    ppt = ppt[:min_days]
                    tmax = tmax[:min_days]
                    tmin = tmin[:min_days]
                    
                    # Apply masks
                    if row_min is not None:
                        mask = self.base_mask[row_min:row_max+1, col_min:col_max+1]
                    else:
                        mask = self.base_mask
                        
                    time_mask = np.broadcast_to(mask, ppt.shape)
                    
                    ppt = np.where(time_mask != 1, -999, ppt)
                    tmax = np.where(time_mask != 1, -999, tmax)
                    tmin = np.where(time_mask != 1, -999, tmin)
                    
                    ppts.append(ppt)
                    tmaxs.append(tmax)
                    tmins.append(tmin)

                io_time = time.time() - io_start
                print(f"File I/O operations took {io_time:.2f} seconds")

                # Concatenate all years
                ppts = np.concatenate(ppts, axis=0)
                tmaxs = np.concatenate(tmaxs, axis=0)
                tmins = np.concatenate(tmins, axis=0)
                
                # Clean up any NaN values
                ppts = np.where(np.isnan(ppts), -999, ppts)
                tmaxs = np.where(np.isnan(tmaxs), -999, tmaxs)
                tmins = np.where(np.isnan(tmins), -999, tmins)

                ppts = np.where(ppts == -999, np.nan, ppts)
                tmaxs = np.where(tmaxs == -999, np.nan, tmaxs)
                tmins = np.where(tmins == -999, np.nan, tmins)


                    
                print(f"Final PRISM data shape: {ppts.shape}, {tmaxs.shape}, {tmins.shape}")
                
                if self.config.get('aggregation'):
                    agg_start = time.time()
                    ppts, tmaxs, tmins = self.aggregate_temporal_data(ppts, tmaxs, tmins)
                    agg_time = time.time() - agg_start
                    print(f"Data aggregation took {agg_time:.2f} seconds")

                total_time = time.time() - data_start
                print(f"Total PRISM data extraction took {total_time:.2f} seconds")
                return ppts, tmaxs, tmins
                
            except Exception as e:
                print(f"Error processing data: {e}")
                return None, None, None

    def get_average_data(self):
        ppts, tmaxs, tmins = self.get_data()
        ppts = np.nansum(ppts, axis=0)
        tmaxs = np.nanmean(tmaxs, axis=0)
        tmins = np.nanmean(tmins, axis=0)
        return ppts, tmaxs, tmins
    
    def get_spatial_average_over_time(self):
        
        start_time = time.time()
        ppts, tmaxs, tmins = self.get_data()
        
        avg_start = time.time()
        ppts = np.nanmean(ppts, axis=(1, 2))
        tmaxs = np.nanmean(tmaxs, axis=(1, 2))
        tmins = np.nanmean(tmins, axis=(1, 2))      
        avg_time = time.time() - avg_start
        print(f"shape of ppts: {ppts.shape}, shape of tmaxs: {tmaxs.shape}, shape of tmins: {tmins.shape}") 
        print(f"range of spatial average of total ppts: {np.min(ppts)}, {np.max(ppts)}")
        print(f"range of spatial average annual tmaxs: {np.min(tmaxs)}, {np.max(tmaxs)}")
        print(f"range of spatial average annual tmins: {np.min(tmins)}, {np.max(tmins)}")
        print(f"Spatial averaging took {avg_time:.2f} seconds")
        print(f"Total spatial average processing took {time.time() - start_time:.2f} seconds")
        
        return ppts, tmaxs, tmins
    

if __name__ == "__main__":

    config = { 
        "RESOLUTION": 250,
        "huc8": None,
        "video": False,
        "aggregation": "annual",
        "start_year": 2000,
        "end_year": 2003,
        'bounding_box': [-85.444332, 43.658148, -85.239256, 44.164683],
    }
    
    prism_dataset = PRISM_Dataset(config)
    pr_prism, tmax_prism, tmin_prism = prism_dataset.get_spatial_average_over_time()