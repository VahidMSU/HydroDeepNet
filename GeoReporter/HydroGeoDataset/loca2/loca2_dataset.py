from datetime import datetime
from logging import config
import h5py
import numpy as np
import os
import pandas as pd
from scipy.spatial import cKDTree

from matplotlib import animation, pyplot as plt

try:
    from config import AgentConfig
    from utils.Logger import LoggerSetup
except ImportError:
    from GeoReporter.config import AgentConfig
    from GeoReporter.utils.Logger import LoggerSetup
    


def list_of_cc_models(required_models="MPI-ESM1-2-HR", required_scenarios="historical"):

    """
    
    Read and return the list of climate models, scenarios, and ensembles in LOCA2 
    
    """
    # Climate model, scenario, and ensemble configuration for LOCA2
    list_of_climate_data = '/data/SWATGenXApp/GenXAppData/HydroGeoDataset/list_of_all_models.txt'


    dict_of_cc_models = {'cc_model': [], 'scenario': [], 'ensemble': []}
    with open(list_of_climate_data, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            parts = line.split(' ')
            if len(parts) == 4:
                idx, cc_model, scenario, ensemble = parts
                dict_of_cc_models['cc_model'].append(cc_model)
                dict_of_cc_models['scenario'].append(scenario)
                dict_of_cc_models['ensemble'].append(ensemble)
                #print(cc_model, scenario, ensemble)
            #else:

                #print(f"Skipping line: {line}")
        ## create a dataframe

        df = pd.DataFrame(dict_of_cc_models)[:99]



    return df

class DataImporter:
    def __init__(self, config, device=None):
        """
        Summary:
        Class for importing and processing various types of data for hydrogeological modeling.

        Explanation:
        This class provides methods for importing and processing different types of data including static, transient, and PFAS data. It handles tasks such as extracting features and preparing data for deep learning.

        """
        
        self.config = config if isinstance(config, dict) else config.__dict__
        self.device = device
        self.config['RESOLUTION'] = 250 if 'RESOLUTION' not in config else config['RESOLUTION']
        self.config['database_path'] = AgentConfig.HydroGeoDataset_ML_250_path
        self.config['aggregation'] = None if 'aggregation' not in config else config['aggregation']
        self.config['bounding_box'] = None if 'bounding_box' not in config else config['bounding_box']
        self.logger = LoggerSetup("/data/SWATGenXApp/codes/GeoReporter/logs", rewrite=True).setup_logger("HydroGeoDataset")

    

    @staticmethod
    def get_loca2_time_index_of_year(start_year, end_year):
        """Get the start and end indices of the given years in the LOCA2 dataset."""
        LOCA2_start_date = datetime(1950, 1, 1)
        LOCA2_end_date = datetime(2014, 12, 31)

        # Calculate the start index for the given start_year
        extract_year_start = datetime(start_year, 1, 1)
        index_year_start = (extract_year_start - LOCA2_start_date).days + 1

        # Calculate the end index for the given end_year
        extract_year_end = datetime(end_year, 12, 31)
        index_year_end = (extract_year_end - LOCA2_start_date).days + 1

        # Print the results
        #print(f"Start index of {start_year}: {index_year_start}")
        #print(f"End index of {end_year}: {index_year_end}")

        # Return the indices
        return index_year_start, index_year_end+1





    def video_data(self, data, name, number_of_frames=None) -> None:
        """
        Create a video from a 3D data array where each 2D array represents a frame.

        Parameters:
        data (numpy.ndarray): 3D array with the shape (frames, height, width).
        name (str): Name of the output video file.

        Returns:
        None
        """
        self.logger.info(f"Creating video of {name} data.")
        # Replace -999 with NaN for better visualization
        data = np.where(data == -999, np.nan, data)

        # Set up the figure and axis
        fig, ax = plt.subplots()

        # Calculate global min and max values for the entire time series (all frames)
        vmin = np.nanpercentile(data, 2.5)  # 2.5th percentile value across all frames
        vmax = np.nanpercentile(data, 97.5)  # 97.5th percentile value across all frames

        # Create the first image with the full time series color limits (vmin, vmax)
        im = ax.imshow(data[0], animated=True, cmap='viridis', vmin=vmin, vmax=vmax)

        # Add the color bar (created only once, with limits based on full time series)
        cbar = fig.colorbar(im, ax=ax)
        ## data unit
        dataunit = 'mm/day' if 'ppt' in name or 'pr' in name else 'C'
        cbar.set_label(f'95th Percentile Range ({dataunit})')

        def update_frame(i):
            """Update the frame for the animation."""
            # Update the data of the image for the new frame without recreating colorbar
            im.set_array(data[i])
            title = name.split('_')[1] + " " + name.split("_")[0].upper()
            ax.set_title(f"{title}\nStep {i+1}: Mean: {np.nanmean(data[i]):.2f} 97.25th: {np.nanpercentile(data[i], 97.25):.2f} 2.5th: {np.nanpercentile(data[i], 2.5):.2f}")
            return [im]

        # Create animation
        ani = animation.FuncAnimation(fig, update_frame, frames=range(data.shape[0]), interval=50, blit=True, repeat_delay=1000)

        # Ensure the output directory exists
        os.makedirs('input_videos', exist_ok=True)

        # Determine the output file name

        output_filename = f'input_videos/{name}.gif'
        if self.config.get('huc8'):
            output_filename = f'input_videos/{name}_{self.config["huc8"]}.gif'

        # Save the animation
        ani.save(output_filename, writer='pillow')

        # Close the figure
        plt.close(fig)

        self.logger.info(f"Video of {name} data saved as {output_filename}.")


    def LOCA2(self, start_year, end_year, cc_model, scenario, ensemble, cc_time_step, row=None, col=None, time_period=None) -> np.ndarray:
        self.config['start_year'] = start_year
        self.config['end_year'] = end_year
        
        # Handle the time period based on scenario
        if scenario == 'historical':
            # For historical scenario, use the standard time range
            time_range = '1950_2014'
        else:
            # For future scenarios, use the provided time period or determine it based on year range
            if time_period:
                time_range = time_period
            else:
                # Determine appropriate time period based on requested years
                if start_year >= 2015 and end_year <= 2044:
                    time_range = '2015_2044'
                elif start_year >= 2045 and end_year <= 2074:
                    time_range = '2045_2074'
                elif start_year >= 2075 and end_year <= 2100:
                    time_range = '2075_2100'
                else:
                    # Default case
                    year_diff_2044 = abs(start_year - 2044) + abs(end_year - 2015)
                    year_diff_2074 = abs(start_year - 2074) + abs(end_year - 2045)
                    year_diff_2100 = abs(start_year - 2100) + abs(end_year - 2075)
                    min_diff = min(year_diff_2044, year_diff_2074, year_diff_2100)
                    
                    if min_diff == year_diff_2044:
                        time_range = '2015_2044'
                    elif min_diff == year_diff_2074:
                        time_range = '2045_2074'
                    else:
                        time_range = '2075_2100'
                    
                    self.logger.warning(f"Year range {start_year}-{end_year} spans multiple time periods, using best match: {time_range}")
        
        # Calculate the day indices relative to the start of the given time period
        if scenario == 'historical':
            period_start_year = 1950
        else:
            period_start_year = int(time_range.split('_')[0])
            
        start_index = (datetime(start_year, 1, 1) - datetime(period_start_year, 1, 1)).days
        end_index = (datetime(end_year, 12, 31) - datetime(period_start_year, 1, 1)).days + 1
        
        self.logger.info(f"Using time range: {time_range}, extracting days {start_index} to {end_index}")
        
        path = '/data/SWATGenXApp/GenXAppData/HydroGeoDataset/LOCA2_MLP.h5'
        with h5py.File(path, 'r') as f:
            self.logger.info(f"Attempting to load climate data for {cc_model}, {scenario}, {ensemble}, {cc_time_step}, {time_range}.")
            
            # Build the base path
            base_path = f'e_n_cent/{cc_model}/{scenario}/{ensemble}/{cc_time_step}/{time_range}'
            
            # Verify the path exists
            if base_path not in f:
                available_paths = []
                # Check what's available at each level
                parts = base_path.split('/')
                for i in range(1, len(parts)+1):
                    check_path = '/'.join(parts[:i])
                    if check_path in f:
                        available = list(f[check_path].keys())
                        available_paths.append(f"{check_path} contains: {available}")
                    else:
                        available_paths.append(f"{check_path} does not exist")
                        break
                
                error_msg = f"Path {base_path} not found in the H5 file.\nAvailable paths: {available_paths}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Load the data
            try:
                pr = f[f'{base_path}/pr'][start_index:end_index]
                tmax = f[f'{base_path}/tasmax'][start_index:end_index]
                tmin = f[f'{base_path}/tasmin'][start_index:end_index]
            except Exception as e:
                # Try alternative names for variables
                var_alternatives = {
                    'pr': ['pr', 'precip', 'precipitation'],
                    'tasmax': ['tasmax', 'tmax', 'tx'],
                    'tasmin': ['tasmin', 'tmin', 'tn']
                }
                
                pr, tmax, tmin = None, None, None
                
                # Try each alternative variable name
                for var_name, alternatives in var_alternatives.items():
                    for alt in alternatives:
                        try:
                            if var_name == 'pr' and pr is None:
                                pr = f[f'{base_path}/{alt}'][start_index:end_index]
                            elif var_name == 'tasmax' and tmax is None:
                                tmax = f[f'{base_path}/{alt}'][start_index:end_index]
                            elif var_name == 'tasmin' and tmin is None:
                                tmin = f[f'{base_path}/{alt}'][start_index:end_index]
                        except:
                            continue
                
                # If we still don't have all variables, raise an error
                if pr is None or tmax is None or tmin is None:
                    raise ValueError(f"Could not find all required variables in {base_path}")

            # Load lat/lon data
            loca2_lats = f['lat'][:]  # 1D array
            loca2_lons = f['lon'][:]  # 1D array
            
            if self.config.get('bounding_box'):
                # Extract bounding box
                desired_min_lat = self.config['bounding_box'][1]
                desired_max_lat = self.config['bounding_box'][3]
                desired_min_lon = self.config['bounding_box'][0]
                desired_max_lon = self.config['bounding_box'][2]
                
                # Find indices of points within the bounding box
                lat_indices = np.where((loca2_lats >= desired_min_lat) & (loca2_lats <= desired_max_lat))[0]
                lon_indices = np.where((loca2_lons >= desired_min_lon) & (loca2_lons <= desired_max_lon))[0]
                
                if len(lat_indices) > 0 and len(lon_indices) > 0:
                    # Extract min/max indices
                    lat_min, lat_max = min(lat_indices), max(lat_indices) + 1
                    lon_min, lon_max = min(lon_indices), max(lon_indices) + 1
                    
                    # Clip the data arrays using these indices
                    pr = pr[:, lat_min:lat_max, lon_min:lon_max]
                    tmax = tmax[:, lat_min:lat_max, lon_min:lon_max]
                    tmin = tmin[:, lat_min:lat_max, lon_min:lon_max]
                    
                    self.logger.info(f"Data clipped to bounding box. New shapes: {pr.shape}, {tmax.shape}, {tmin.shape}")
                else:
                    self.logger.warning("No data found within the specified bounding box.")
            
            # Flip the climate data to correct the orientation (if needed)
            pr = np.flip(pr, axis=1).copy()
            tmax = np.flip(tmax, axis=1).copy()
            tmin = np.flip(tmin, axis=1).copy()
            self.logger.info("Flipping completed.")

            self.logger.info(f"Final data shapes: {pr.shape}, {tmax.shape}, {tmin.shape}")
            
            # Convert NaN and out-of-bounds values to -999
            pr = np.where(np.isnan(pr), -999, pr)
            tmax = np.where(np.isnan(tmax), -999, tmax)
            tmin = np.where(np.isnan(tmin), -999, tmin)
            
            # Convert units
            pr = pr * 86400  # Convert from kg m-2 s-1 to mm/day
            tmax = tmax - 273.15  # Convert from K to °C
            tmin = tmin - 273.15  # Convert from K to °C

            # Create video visualizations if enabled
            if self.config.get('video', False):
                self.video_data(pr, 'pr_LOCA2')
                self.video_data(tmax, 'tmax_LOCA2')
                self.video_data(tmin, 'tmin_LOCA2')

            # Aggregate temporal data if requested
            if self.config.get('aggregation'):
                pr, tmax, tmin = self.aggregate_temporal_data(pr, tmax, tmin)
                self.logger.info(f"Aggregated data shape: {pr.shape}, {tmax.shape}, {tmin.shape}")

            return pr, tmax, tmin


    def aggregate_temporal_data(self, pr, tmax, tmin) -> np.ndarray:
        """ Aggregate the temporal data based on the specified aggregation method. """
        min_temporal = min(pr.shape[0], tmax.shape[0], tmin.shape[0])
        pr = pr[:min_temporal, :, :]
        tmax = tmax[:min_temporal, :, :]
        tmin = tmin[:min_temporal, :, :]
        print(f"#################Aggregation method: {self.config['aggregation']}#################")
        
        total_days = pr.shape[0]
        # Create a date range assuming data starts on January 1st of the start year
        start_date = f"{self.config['start_year']}-01-01"
        dates = pd.date_range(start=start_date, periods=total_days)
        
        if self.config['aggregation'] == 'monthly':
            # Group by both year and month using the dates array
            pr_monthly = [pr[(dates.year == year) & (dates.month == month), :, :].mean(axis=0)
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
            self.logger.info(f"Aggregated data shape (monthly): {pr.shape}, {tmax.shape}, {tmin.shape}")

        elif self.config['aggregation'] == 'seasonal':
            # Define the seasons as months: DJF, MAM, JJA, SON
            seasons = {
                'DJF': [12, 1, 2],
                'MAM': [3, 4, 5],
                'JJA': [6, 7, 8],
                'SON': [9, 10, 11]
            }

            pr_seasonal = [pr[np.isin(dates.month, months) & (dates.year == year), :, :].mean(axis=0)
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
            self.logger.info(f"Aggregated data shape (seasonal): {pr.shape}, {tmax.shape}, {tmin.shape}")

        elif self.config['aggregation'] == 'annual':
            # Group by year using the dates array
            pr_annual = [pr[dates.year == year, :, :].mean(axis=0) for year in np.unique(dates.year)]
            tmax_annual = [tmax[dates.year == year, :, :].mean(axis=0) for year in np.unique(dates.year)]
            tmin_annual = [tmin[dates.year == year, :, :].mean(axis=0) for year in np.unique(dates.year)]

            pr = np.array(pr_annual)
            tmax = np.array(tmax_annual)
            tmin = np.array(tmin_annual)
            self.logger.info(f"Aggregated data shape (annual): {pr.shape}, {tmax.shape}, {tmin.shape}")

        return pr, tmax, tmin


if __name__ == "__main__":
####################### example of loading LOCA2 data ###################################
    print(f"list_of_cc_models: {list_of_cc_models()}")
    df = list_of_cc_models()

    print(f"colums: {df.columns}")  
    config = {
            "RESOLUTION": 250,
            "huc8": None,
            "video": False,
            "aggregation": "monthly",
            'bounding_box': [-85.444332, 43.658148, -85.239256, 44.164683], # min_longitude, min_latitude, max_longitude, max_latitude
        }

    importer = DataImporter(config)
    ### NOTE: the list of all models and their ensemble is in /data/LOCA2/list_of_all_models.txt
    start_year = 2000
    end_year = 2012
    cc_model = "ACCESS-CM2"
    scenario = "historical"
    ensemble = "r2i1p1f1"

    ppt_loca2, tmax_loca2, tmin_loca2 = importer.LOCA2(start_year=start_year, end_year=end_year, cc_model= cc_model, scenario=scenario, ensemble=ensemble, cc_time_step='daily')
