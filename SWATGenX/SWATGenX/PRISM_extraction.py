import os
import datetime
import glob

import xarray as xr
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from functools import partial
from concurrent.futures import ThreadPoolExecutor
try:
    from SWATGenX.utils import get_all_VPUIDs, return_list_of_huc12s
except ImportError:
    from utils import get_all_VPUIDs, return_list_of_huc12s

def extract_PRISM_parallel(SWATGenXPaths, VPUID, LEVEL, NAME, list_of_huc12s=None, overwrite=False):
    """
    Main function to extract PRISM data for a given VPUID using parallel processing.
    """
    extractor = PRISMExtractor(SWATGenXPaths, VPUID, LEVEL, NAME, list_of_huc12s, overwrite)
    extractor.run()


class PRISMExtractor:

    """
    Extracts PRISM precipitation and temperature data for SWAT+ simulations.
    """

    def __init__(self, SWATGenXPaths, VPUID, LEVEL, NAME, list_of_huc12s=None, overwrite=False):
        self.VPUID = VPUID
        self.LEVEL = LEVEL
        self.NAME = NAME
        self.list_of_huc12s = list_of_huc12s
        self.SWATGenXPaths = SWATGenXPaths
        self._prism_dir = SWATGenXPaths.PRISM_path
        self._outlet_dir = SWATGenXPaths.swatgenx_outlet_path
        self.SWAT_MODEL_PRISM_path = f"{self._outlet_dir}/{self.VPUID}/{self.LEVEL}/{self.NAME}/PRISM"
        self.max_workers = 3
        self.overwrite = overwrite

    def should_process_file(self, file_path):
        """Check if a file should be processed based on overwrite setting"""
        if self.overwrite:
            return True
        return not os.path.exists(file_path)

    def cleanup_existing_files(self):
        """Remove existing climate files before processing"""
        if not self.overwrite:
            print("Skipping cleanup as overwrite is False")
            return

        try:
            # Remove individual station files
            for pattern in ['*.tmp', '*.pcp']:
                files = glob.glob(os.path.join(self.SWAT_MODEL_PRISM_path, pattern))
                for f in files:
                    try:
                        os.remove(f)
                        print(f"Removed existing file: {f}")
                    except Exception as e:
                        print(f"Could not remove file {f}: {e}")

            # Remove CLI files
            for cli_file in ['tmp.cli', 'pcp.cli']:
                cli_path = os.path.join(self.SWAT_MODEL_PRISM_path, cli_file)
                if os.path.exists(cli_path):
                    try:
                        os.remove(cli_path)
                        print(f"Removed existing CLI file: {cli_path}")
                    except Exception as e:
                        print(f"Could not remove CLI file {cli_path}: {e}")

        except Exception as e:
            print(f"Error during cleanup: {e}")

    def _is_leap_year(self, year):
        """Helper method to determine if a year is a leap year"""
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

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
        """Write precipitation file using pandas for consistent formatting"""
        try:
            filename = os.path.join(self.SWAT_MODEL_PRISM_path, f"r{row}_c{col}.pcp")

            if not self.should_process_file(filename):
                print(f"Skipping existing file: {filename}")
                return

            # Get location data
            lat = df[(df.row == row) & (df.col == col)].lat.values[0]
            lon = df[(df.row == row) & (df.col == col)].lon.values[0]
            elev = self._get_elevation(row, col)

            # Gather all yearly data
            values = []
            for y in years:
                values = np.append(values, datasets[y]['data'][:, row, col])

            # Create DataFrame with the data
            data_df = pd.DataFrame({
                'year': [d.year for d in date_range],
                'day': [d.dayofyear for d in date_range],
                'value': values
            })

            # Group by year to ensure we only keep complete years
            year_groups = data_df.groupby('year')
            complete_years_data = []

            for year, group in year_groups:
                expected_days = 366 if self._is_leap_year(year) else 365
                if len(group) == expected_days and not group['value'].isna().any():
                    complete_years_data.append(group)

            if not complete_years_data:
                print(f"No complete years for row {row}, col {col}")
                return

            # Combine all complete years
            df_final = pd.concat(complete_years_data, axis=0)

            # Write the header lines first
            with open(filename, 'w') as f:
                f.write(f"PRISM 4km grid for VPUID {self.VPUID}, r{row}, c{col}\n")
                f.write("nbyr nstep lat lon elev\n")
                f.write(f"{len(set(df_final['year']))}\t0\t{lat:.2f}\t{lon:.2f}\t{elev:.2f}\n")

            # Append the data using pandas
            df_final.to_csv(
                filename,
                mode='a',
                header=False,
                index=False,
                sep='\t',
                float_format='%.2f',
                columns=['year', 'day', 'value']
            )

        except Exception as e:
            print(f"Error writing precipitation file for row {row}, col {col}: {e}")

    def write_tmp_file(self, row, col, df, datasets_max, datasets_min, years, nbyr, date_range):
        """Write temperature file using pandas for consistent formatting"""
        try:
            filename = os.path.join(self.SWAT_MODEL_PRISM_path, f"r{row}_c{col}.tmp")

            if not self.should_process_file(filename):
                print(f"Skipping existing file: {filename}")
                return

            # Get location data
            lat = df[(df.row == row) & (df.col == col)].lat.values[0]
            lon = df[(df.row == row) & (df.col == col)].lon.values[0]
            elev = self._get_elevation(row, col)

            # Accumulate max and min arrays
            arr_max, arr_min = [], []
            for y in years:
                arr_max = np.append(arr_max, datasets_max[y]['data'][:, row, col])
                arr_min = np.append(arr_min, datasets_min[y]['data'][:, row, col])

            # Create DataFrame with the data
            data_df = pd.DataFrame({
                'year': [d.year for d in date_range],
                'day': [d.dayofyear for d in date_range],
                'tmax': arr_max,
                'tmin': arr_min
            })

            # Group by year to ensure we only keep complete years
            year_groups = data_df.groupby('year')
            complete_years_data = []

            for year, group in year_groups:
                expected_days = 366 if self._is_leap_year(year) else 365
                if (len(group) == expected_days and
                    not group['tmax'].isna().any() and
                    not group['tmin'].isna().any()):
                    complete_years_data.append(group)

            if not complete_years_data:
                print(f"No complete years for row {row}, col {col}")
                return

            # Combine all complete years
            df_final = pd.concat(complete_years_data, axis=0)

            # Write the header lines first
            with open(filename, 'w') as f:
                f.write(f"PRISM 4km grid for VPUID {self.VPUID}, r{row}, c{col}\n")
                f.write("nbyr nstep lat lon elev\n")
                f.write(f"{len(set(df_final['year']))}\t0\t{lat:.2f}\t{lon:.2f}\t{elev:.2f}\n")

            # Append the data using pandas
            df_final[['year', 'day', 'tmax', 'tmin']].to_csv(
                filename,
                mode='a',
                header=False,
                index=False,
                sep='\t',
                float_format='%.2f'
            )

        except Exception as e:
            print(f"Error writing temperature file for row {row}, col {col}: {e}")

    def write_pcp_cli(self, extracted_grid):
        """Write precipitation CLI file"""
        cli_path = os.path.join(self.SWAT_MODEL_PRISM_path, 'pcp.cli')

        if not self.should_process_file(cli_path):
            print(f"Skipping existing CLI file: {cli_path}")
            return

        written_rows_cols = set()
        cli_data = []

        for _, row_data in extracted_grid.iterrows():
            row, col = row_data['row'], row_data['col']
            if (row, col) not in written_rows_cols:
                written_rows_cols.add((row, col))
                cli_data.append(f"r{row}_c{col}.pcp")

        with open(cli_path, 'w') as f:
            f.write(f"PRISM 4km grid for VPUID {self.VPUID}\n")
            f.write("precipitation file\n")
            f.write('\n'.join(cli_data))

    def write_tmp_cli(self, extracted_grid):
        """Write temperature CLI file"""
        cli_path = os.path.join(self.SWAT_MODEL_PRISM_path, 'tmp.cli')

        if not self.should_process_file(cli_path):
            print(f"Skipping existing CLI file: {cli_path}")
            return

        written_rows_cols = set()
        cli_data = []

        for _, row_data in extracted_grid.iterrows():
            row, col = row_data['row'], row_data['col']
            if (row, col) not in written_rows_cols:
                written_rows_cols.add((row, col))
                cli_data.append(f"r{row}_c{col}.tmp")

        with open(cli_path, 'w') as f:
            f.write(f"PRISM 4km grid for VPUID {self.VPUID}\n")
            f.write("temperature file\n")
            f.write('\n'.join(cli_data))

    def generating_swatplus_pcp(self, df, datasets, years):
        """
        Generate SWAT+ precipitation input files for a given subset of PRISM grid rows/cols.
        """
        start_date = datetime.datetime(years[0], 1, 1)
        end_date = datetime.datetime(years[-1], 12, 31)
        date_range = pd.date_range(start_date, end_date)
        nbyr = years[-1] - years[0] + 1

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for row, col in zip(df.row, df.col):
                executor.submit(self.write_pcp_file, row, col, df, years, date_range, datasets, nbyr)

    def generating_swatplus_tmp(self, df, datasets_max, datasets_min, years):
        """
        Generate SWAT+ temperature input files (tmax, tmin) for each grid cell.
        """
        start_date = datetime.datetime(years[0], 1, 1)
        end_date = datetime.datetime(years[-1], 12, 31)
        date_range = pd.date_range(start_date, end_date)
        nbyr = years[-1] - years[0] + 1

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for row, col in zip(df.row, df.col):
                executor.submit(self.write_tmp_file, row, col, df, datasets_max, datasets_min, years, nbyr, date_range)

    def run(self):
        """Orchestrates clipping and file generation with improved error handling"""
        try:
            # Clean up existing files first
            self.cleanup_existing_files()

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

            # Load and validate precipitation data
            print("Loading PRISM ppt data...")
            ppt_data = {}
            for y in years:
                path = f"{self._prism_dir}/CONUS/ppt/{y}.nc"
                if os.path.exists(path):
                    ppt_data[y] = xr.open_dataset(path)
                else:
                    print(f"Missing precipitation data for year {y}")
                    continue

            if not ppt_data:
                raise ValueError("No precipitation data could be loaded")

            # Process precipitation data
            t0 = datetime.datetime.now()
            start_date = datetime.datetime(years[0], 1, 1)
            end_date = datetime.datetime(years[-1], 12, 31)
            date_range = pd.date_range(start_date, end_date)
            nbyr = years[-1] - years[0] + 1

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for row, col in zip(extracted_grid.row, extracted_grid.col):
                    executor.submit(
                        self.write_pcp_file,
                        row, col, extracted_grid, years, date_range, ppt_data, nbyr
                    )
            print(f"PCP extraction took {(datetime.datetime.now() - t0).total_seconds()} sec")

            # Load and validate temperature data
            print("Loading PRISM temperature data...")
            tmax_data, tmin_data = {}, {}
            for y in years:
                path_max = f"{self._prism_dir}/CONUS/tmax/{y}.nc"
                path_min = f"{self._prism_dir}/CONUS/tmin/{y}.nc"
                if os.path.exists(path_max) and os.path.exists(path_min):
                    tmax_data[y] = xr.open_dataset(path_max)
                    tmin_data[y] = xr.open_dataset(path_min)
                else:
                    print(f"Missing temperature data for year {y}")
                    continue

            if not tmax_data or not tmin_data:
                raise ValueError("No temperature data could be loaded")

            # Process temperature data
            t1 = datetime.datetime.now()
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for row, col in zip(extracted_grid.row, extracted_grid.col):
                    executor.submit(
                        self.write_tmp_file,
                        row, col, extracted_grid, tmax_data, tmin_data, years, nbyr, date_range
                    )
            print(f"TMP extraction took {(datetime.datetime.now() - t1).total_seconds()} sec")

            # Write CLI files after processing
            self.write_pcp_cli(extracted_grid)
            self.write_tmp_cli(extracted_grid)

        except Exception as e:
            print(f"Error in PRISM extraction: {e}")
            raise

if __name__ == "__main__":
    from SWATGenXConfigPars import SWATGenXPaths
    username = "admin"
    NAME = "04127200"
    list_of_huc12s, vpuid = return_list_of_huc12s(NAME)
    SWATGenXPaths = SWATGenXPaths(username=username, VPUID=vpuid, NAME=NAME, LEVEL="huc12")
    extract_PRISM_parallel(SWATGenXPaths, vpuid, "huc12", NAME, list_of_huc12s=list_of_huc12s, overwrite=True)
