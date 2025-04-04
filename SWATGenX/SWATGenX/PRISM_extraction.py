import os
import datetime

import xarray as xr
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from functools import partial
# Replace Process with ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor
# Comment out if you don't need the fallback imports
from SWATGenX.utils import get_all_VPUIDs, return_list_of_huc12s

def extract_PRISM_parallel(SWATGenXPaths, VPUID, LEVEL, NAME, list_of_huc12s=None):
    """
    Main function to extract PRISM data for a given VPUID using parallel processing.
    """
    extractor = PRISMExtractor(SWATGenXPaths, VPUID, LEVEL, NAME, list_of_huc12s)
    extractor.run()


class PRISMExtractor:
    
    """
    Extracts PRISM precipitation and temperature data for SWAT+ simulations.
    """

    def __init__(self, SWATGenXPaths, VPUID, LEVEL, NAME, list_of_huc12s=None):
        self.VPUID = VPUID
        self.LEVEL = LEVEL
        self.NAME = NAME
        self.list_of_huc12s = list_of_huc12s
        self.SWATGenXPaths = SWATGenXPaths
        self._prism_dir = SWATGenXPaths.PRISM_path
        self._outlet_dir = SWATGenXPaths.swatgenx_outlet_path
        self.SWAT_MODEL_PRISM_path = f"{self._outlet_dir}/{self.VPUID}/{self.LEVEL}/{self.NAME}/PRISM"
        # Define max number of threads for our pool
        self.max_workers = 4

    def _get_elevation(self, row, col):
        """Helper to extract DEM-based elevation using rasterio."""
        with rasterio.open(self.SWATGenXPaths.PRISM_dem_path) as src:
            elev_data = src.read(1)
        return elev_data[row, col]
    

    def clip_PRISM_by_VPUID(self):
        """
        Clips PRISM data by the WBDHU8 extent of the given VPUID.
        If already clipped, returns the existing shapefile.
        """
        clipped_grid_path = f"{self._prism_dir}/VPUID/{self.VPUID}/PRISM_grid.shp"
        os.makedirs(f"{self._prism_dir}/VPUID/{self.VPUID}", exist_ok=True)

        if os.path.exists(clipped_grid_path):
            print(f"Clipped PRISM data for {self.VPUID} exists.")
            return gpd.read_file(clipped_grid_path)

        print(f"Clipping PRISM data for {self.VPUID}...")
        wbdhu8 = pd.read_pickle(
            f"{self.SWATGenXPaths.extracted_nhd_swatplus_path}/{self.VPUID}/WBDHU8.pkl"
        ).to_crs("EPSG:4326")

        # Clip the PRISM mesh shapefile
        extent = wbdhu8.total_bounds
        prism_mesh = gpd.read_file(self.SWATGenXPaths.PRISM_mesh_path).to_crs("EPSG:4326")
        clipped = prism_mesh.cx[extent[0]:extent[2], extent[1]:extent[3]]

        # Overlay with WBDHU12
        wbdhu12 = pd.read_pickle(
            f"{self.SWATGenXPaths.extracted_nhd_swatplus_path}/{self.VPUID}/WBDHU12.pkl"
        ).to_crs("EPSG:4326")
        clipped_prism = gpd.overlay(clipped, wbdhu12[['huc12', 'geometry']], how='intersection')
        clipped_prism.to_file(clipped_grid_path)

        print(f"Clipping PRISM data for {self.VPUID} is done.")
        return clipped_prism

    def write_pcp_file(self, row, col, df, years, date_range, datasets, nbyr):
        # Gather all yearly data into one array
        filename = f"r{row}_c{col}.pcp"
        values = []
        for y in years:
            values = np.append(values, datasets[y]['data'][:, row, col])

        date_df = pd.DataFrame({
            'date': date_range,
            'ppt': values
        })
        date_df['YEAR'] = date_df['date'].dt.year
        date_df['DAY'] = date_df['date'].dt.dayofyear

        lat = df[(df.row == row) & (df.col == col)].lat.values[0]
        lon = df[(df.row == row) & (df.col == col)].lon.values[0]
        elev = self._get_elevation(row, col)

        filename = os.path.join(self.SWAT_MODEL_PRISM_path, f"r{row}_c{col}.pcp")
        with open(filename, 'w') as f:
            f.write(f"PRISM 4km grid for VPUID {self.VPUID}, r{row}, c{col}\n")
            f.write("nbyr\ttstep\tlat\tlon\telev\n")
            ### lat and lon rounded to 2 decimal places
            lat = round(lat, 2)
            lon = round(lon, 2)
            f.write(f"{nbyr} 0 {lat} {lon} {elev}\n")
            for _, d in date_df.iterrows():
                f.write(f"{d.YEAR}\t{d.DAY:03}\t{d.ppt:.2f}\n")

    def generating_swatplus_pcp(self, df, datasets, years):
        """
        Generate SWAT+ precipitation input files for a given subset of PRISM grid rows/cols.
        """
        start_date = datetime.datetime(years[0], 1, 1)
        end_date = datetime.datetime(years[-1], 12, 31)
        date_range = pd.date_range(start_date, end_date)
        nbyr = years[-1] - years[0] + 1
        
        # Replace Process-based parallelism with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for row, col in zip(df.row, df.col):
                executor.submit(self.write_pcp_file, row, col, df, years, date_range, datasets, nbyr)

    def write_tmp_file(self, row, col, df, datasets_max, datasets_min, years, nbyr, date_range):
        filename = f"r{row}_c{col}.tmp"
        # Accumulate max and min arrays across all years
        arr_max, arr_min = [], []
        for y in years:
            arr_max = np.append(arr_max, datasets_max[y]['data'][:, row, col])
            arr_min = np.append(arr_min, datasets_min[y]['data'][:, row, col])

        date_df = pd.DataFrame({
            'date': date_range,
            'tmax': arr_max,
            'tmin': arr_min
        })
        date_df['YEAR'] = date_df['date'].dt.year
        date_df['DAY'] = date_df['date'].dt.dayofyear

        lat = df[(df.row == row) & (df.col == col)].lat.values[0]
        lon = df[(df.row == row) & (df.col == col)].lon.values[0]
        elev = self._get_elevation(row, col)

        filename = os.path.join(self.SWAT_MODEL_PRISM_path, f"r{row}_c{col}.tmp")
        with open(filename, 'w') as f:
            f.write(f"PRISM 4km grid for VPUID {self.VPUID}, r{row}, c{col}\n")
            f.write("nbyr\ttstep\tlat\tlon\telev\n")
            ### lat and lon rounded to 2 decimal places
            lat = round(lat, 2)
            lon = round(lon, 2)
            f.write(f"{nbyr} 0 {lat} {lon} {elev}\n")
            for _, d in date_df.iterrows():
                f.write(f"{d.YEAR}\t{d.DAY:03}\t{d.tmax:.2f}\t{d.tmin:.2f}\n")

    def generating_swatplus_tmp(self, df, datasets_max, datasets_min, years):
        """
        Generate SWAT+ temperature input files (tmax, tmin) for each grid cell.
        """
        start_date = datetime.datetime(years[0], 1, 1)
        end_date = datetime.datetime(years[-1], 12, 31)
        date_range = pd.date_range(start_date, end_date)
        nbyr = years[-1] - years[0] + 1

        # Replace Process-based parallelism with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for row, col in zip(df.row, df.col):
                executor.submit(self.write_tmp_file, row, col, df, datasets_max, datasets_min, years, nbyr, date_range)

    def run(self):
        """Orchestrates clipping and file generation for precipitation and temperature."""
        prism_vpuid_grid = self.clip_PRISM_by_VPUID()
        print(f"Extracting PRISM data for {self.VPUID}...")

        # Filter by huc12 if needed
        if self.list_of_huc12s is not None:
            os.makedirs(self.SWAT_MODEL_PRISM_path, exist_ok=True)
            prism_vpuid_grid = prism_vpuid_grid[prism_vpuid_grid['huc12'].isin(self.list_of_huc12s)]
            prism_mesh = gpd.read_file(self.SWATGenXPaths.PRISM_mesh_path)
            extracted_grid = prism_mesh[
                (prism_mesh['row'].isin(prism_vpuid_grid['row'])) &
                (prism_mesh['col'].isin(prism_vpuid_grid['col']))
            ]
            extracted_grid.to_crs("EPSG:4326").to_file(
                os.path.join(self.SWAT_MODEL_PRISM_path, 'PRISM_grid.shp')
            )
        else:
            extracted_grid = prism_vpuid_grid

        years = np.arange(2000, 2021)

        # Replace nested Process execution with direct function calls to avoid daemon issues
        # Load precipitation data
        print("Loading PRISM ppt data...")
        ppt_data = {}
        for y in years:
            path = f"{self._prism_dir}/CONUS/ppt/{y}.nc"
            if os.path.exists(path):
                ppt_data[y] = xr.open_dataset(path)
            else:
                print(f"Missing {path}")
        
        t0 = datetime.datetime.now()
        self.generating_swatplus_pcp(extracted_grid, ppt_data, years)
        print(f"PCP extraction took {(datetime.datetime.now() - t0).total_seconds()} sec")

        # Load temperature data
        print("Loading PRISM temperature data...")
        tmax_data, tmin_data = {}, {}
        for y in years:
            path_max = f"{self._prism_dir}/CONUS/tmax/{y}.nc"
            path_min = f"{self._prism_dir}/CONUS/tmin/{y}.nc"
            if os.path.exists(path_max) and os.path.exists(path_min):
                tmax_data[y] = xr.open_dataset(path_max)
                tmin_data[y] = xr.open_dataset(path_min)
            else:
                print(f"Missing {path_max} or {path_min}")
        
        t1 = datetime.datetime.now()
        self.generating_swatplus_tmp(extracted_grid, tmax_data, tmin_data, years)
        print(f"TMP extraction took {(datetime.datetime.now() - t1).total_seconds()} sec")

if __name__ == "__main__":
    NAME = "04135700"
    list_of_huc12s, vpuid = return_list_of_huc12s(NAME)
    extract_PRISM_parallel(vpuid, "huc12", NAME, list_of_huc12s=list_of_huc12s)
