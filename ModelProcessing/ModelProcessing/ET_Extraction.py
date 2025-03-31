import h5py
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import os
from multiprocessing import Pool
import itertools
import shutil

class ExtractMODISET:
    def __init__(self, NAME, LEVEL, VPUID, MODEL_NAME, start_year, end_year, resolution):

        self.NAME = NAME
        self.LEVEL = LEVEL
        self.MODEL_NAME = MODEL_NAME
        self.start_year = start_year
        self.end_year = end_year
        self.resolution = resolution
        self.VPUID = VPUID
        self.lsus1_shp = f"/data/SWATGenXApp/GenXAppData/{self.LEVEL}/{self.NAME}/{self.MODEL_NAME}/Watershed/Shapes/lsus1.shp"
        self.database_path = f"/data/MyDataBase/HydroGeoDataset_ML_{resolution}.h5"
        self.h5_group_name = "MODIS_ET"
        self.original_model_path = f"/data/SWATGenXApp/GenXAppData/{self.LEVEL}/{self.NAME}/MODIS_ET/"
        self.model_processing_path = f"/data/MyDataBase/SWATplus_by_VPUID/{self.VPUID}/{self.LEVEL}/{self.NAME}/MODIS_ET/"
        shutil.rmtree(self.model_processing_path, ignore_errors=True)
        shutil.rmtree(self.original_model_path, ignore_errors=True)
        
        
    # Create argument list for parallel processing
    def generate_list_of_years_months(self, min_row_number, max_row_number, min_col_number, max_col_number):
        args_list = []
        for year, month in itertools.product(range(self.start_year, self.end_year + 1), range(1, 13)):
            dataset_name_pattern = f"MODIS_ET_{year}-{month:02d}"
            args_list.append((dataset_name_pattern, min_row_number, max_row_number, min_col_number, max_col_number, year, month))
        return args_list
    
    def extract_data(self, args):
        dataset_name_pattern, min_row_number, max_row_number, min_col_number, max_col_number, year, month = args
        dictionary = {"year": year, "month": month, "mean_ET": None}
        with h5py.File(self.database_path, "r") as f:
            for dataset_name in self.datasets:
                if dataset_name.startswith(dataset_name_pattern):
                    print(f"Extracting data for {dataset_name}")
                    # Extract data within the specified latitude and longitude range
                    img = f[f"{self.h5_group_name}/{dataset_name}"][min_row_number:max_row_number, min_col_number:max_col_number]
                    
                    # Handle NaN values
                    if np.isnan(np.mean(img)):
                        print(f"Year: {year}, Month: {month}, Mean ET: {np.mean(img)}, Shape: {img.shape}")
                    # Update dictionary with extracted data
                    dictionary["mean_ET"] = np.mean(img)
        return dictionary

    def get_rowcol_range_by_latlon(self, desired_min_lat, desired_max_lat, desired_min_lon, desired_max_lon):
        with h5py.File(self.database_path, 'r') as f:
            # Read latitude and longitude arrays
            lat_ = f[f"lat_{self.resolution}m"][:]
            lon_ = f[f"lon_{self.resolution}m"][:]

            # Replace missing values (-999) with NaN for better handling
            lat_ = np.where(lat_ == -999, np.nan, lat_)
            lon_ = np.where(lon_ == -999, np.nan, lon_)

            # Create masks for latitude and longitude ranges
            lat_mask = (lat_ >= desired_min_lat) & (lat_ <= desired_max_lat)
            lon_mask = (lon_ >= desired_min_lon) & (lon_ <= desired_max_lon)

            # Combine the masks to identify the valid rows and columns
            combined_mask = lat_mask & lon_mask
            
            # Check if any valid points are found
            if np.any(combined_mask):
                # Get row and column indices where the combined mask is True
                row_indices, col_indices = np.where(combined_mask)
                min_row_number = np.min(row_indices)
                max_row_number = np.max(row_indices)
                min_col_number = np.min(col_indices)
                max_col_number = np.max(col_indices)
                return min_row_number, max_row_number, min_col_number, max_col_number
            else:
                return None, None, None, None
            

    def ExtractMODISDataForCatchment(self, bounding_box):

        # Calculate the expected number of months/years
        expected_number_of_months = (self.end_year - self.start_year + 1) * 12

        min_lon, min_lat, max_lon, max_lat = bounding_box

        # Get row and column indices for the bounding box
        min_row_number, max_row_number, min_col_number, max_col_number = self.get_rowcol_range_by_latlon(min_lat, max_lat, min_lon, max_lon)
        args_list = self.generate_list_of_years_months(min_row_number, max_row_number, min_col_number, max_col_number)
        if min_row_number is None or min_row_number == 0 or (max_row_number - min_row_number) < 2 or (max_col_number - min_col_number) < 2:
            return pd.DataFrame()
        
        # Open the HDF5 file to get the list of datasets
        with h5py.File(self.database_path, "r") as f:
            self.datasets = list(f[self.h5_group_name].keys())
        

        
        # Run parallel extraction
        with Pool(10) as pool:
            results = pool.map(self.extract_data, args_list)
        
        # Create DataFrame from results
        df = pd.DataFrame(results)
        
        # Verify the number of months/years
        if len(df) != expected_number_of_months:
            print(f"Warning: Expected {expected_number_of_months} but got {len(df)}")
        
        return df

    def get_et_watersheds(self):
        
        gdf = gpd.read_file(self.lsus1_shp).to_crs(epsg=4326) 
        all_data = []
        for LSUID, geometry in zip(gdf.LSUID, gdf.geometry):
            # get the bounding box of the geometry
            bounds = geometry.bounds

            df = self.ExtractMODISDataForCatchment(bounds)
            df['LSUID'] = LSUID
            if len(df) > 0:
                all_data.append(df)

        if all_data:
            df = pd.concat(all_data)
            df['LSUID'] = df['LSUID'].astype(int)
            ## number of digits must fit the maximum number of LSUID
            ndigits = len(str(df['LSUID'].max()))
            df['LSUID'] = df['LSUID'].apply(lambda x: f"rtu{x:0{ndigits}d}")
            df = df.sort_values(by=["year", "month"])
        else:
            df = pd.DataFrame()

        return df
    def run(self):
        
        df = self.get_et_watersheds()
        
        if not df.empty:

            os.makedirs(self.original_model_path, exist_ok=True)
            
            os.makedirs(self.model_processing_path, exist_ok=True)

            df.to_csv(f"{self.original_model_path}/MODIS_ET.csv", index=False) 
            df.to_csv(f"{self.model_processing_path}/MODIS_ET.csv", index=False) 

            print(f"{self.original_model_path}/MODIS_ET.csv")
            print(f"{self.model_processing_path}/MODIS_ET.csv")


def parallel_processing(args):
    ExtractMODISET(*args).run()


if __name__ == "__main__":
    """
    This script is used to extract MODIS ET data for each watershed in the SWAT model.
    The output is a CSV file containing the MODIS ET data for each watershed.
    The output will save in the respective model VPUD/LEVEL/NAME/MODIS_ET/MODIS_ET.csv
    """
    # Directory path for your SWATplus models
    NAMES_DIR = "/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/"
    
    # List all directories or files in NAMES_DIR

    NAME = "04126740"
    MODEL_NAME = "SWAT_MODEL"
    VPUID = "0000"
    LEVEL = "huc12"
    resolution = "250"
    start_year = 2001
    end_year = 2023
    
    NAMES = os.listdir(NAMES_DIR)
    # Remove unwanted file(s)
    from multiprocessing import Process
    from functools import partial
    NAMES.remove("log.txt")
    processes = []
    for NAME in NAMES:
        #if NAME == "04126740":
        # Run the main function
        args = (NAME, LEVEL, VPUID, MODEL_NAME, start_year, end_year, resolution)
        #ExtractMODISET(NAME, LEVEL, VPUID, MODEL_NAME, start_year, end_year, resolution).run()
        #parallel_processing(args)
        parial_extract = partial(parallel_processing, args)
        p = Process(target=parial_extract)
        p.start()
        processes.append(p)
        if len(processes) == 60:
            for p in processes:
                p.join()
            processes = []
    for p in processes:
        p.join()






