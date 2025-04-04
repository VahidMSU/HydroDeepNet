### the purpose of this code is to extract the SWAT PRISM locations for NSRDB extraction
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import h5pyd
import h5py
import rasterio
import glob
# Replace Process with ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from functools import partial

try:
    from SWATGenX.SWATGenXLogging import LoggerSetup
    from SWATGenX.utils import get_all_VPUIDs
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
except Exception:
    from SWATGenXLogging import LoggerSetup
    from utils import get_all_VPUIDs
    from SWATGenXConfigPars import SWATGenXPaths

def find_VPUID(station_no):
    
    CONUS_streamflow_data = pd.read_csv(SWATGenXPaths.USGS_CONUS_stations_path, dtype={'site_no': str,'huc_cd': str})
    return CONUS_streamflow_data[
        CONUS_streamflow_data.site_no == station_no
    ].huc_cd.values[0][:4]


def nsrdb_contructor_wrapper(variable, SWATGenXPaths, VPUID, LEVEL, NAME, overwrite=False):
    output_path = f"{SWATGenXPaths.swatgenx_outlet_path}/{VPUID}/{LEVEL}/{NAME}/PRISM/"
    shaefile_path = f"{SWATGenXPaths.swatgenx_outlet_path}/{VPUID}/{LEVEL}/{NAME}/PRISM/PRISM_grid.shp"
    
    contructor = NSRDB_contructor(SWATGenXPaths, variable, output_path, shaefile_path, overwrite)
    contructor.run()


class NSRDB_contructor:
    def __init__(self, SWATGenXPaths, variable, output_path, shaefile_path, overwrite=False):
        self.output_path = output_path
        self.shapefile_path = shaefile_path
        self.SWATGenXPaths = SWATGenXPaths
        self.years = range(2000, 2021)
        self.variable = variable
        self.max_workers = 4  # Number of threads to use
        self.overwrite = overwrite
        
        self.data_all = None
        
        self.swat_dict = {
            'ghi': 'slr',
            'wind_speed': 'wnd',
            'relative_humidity': 'hmd',
        }
        self.logger = LoggerSetup(verbose=True, rewrite=True)
        self.logger = self.logger.setup_logger("NSRDB_contructor")
        self.logger.info(f"NSRDB_contructor: {self.variable} extraction for SWAT locations")
    
    def _is_leap_year(self, year):
        """Helper method to determine if a year is a leap year"""
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

    def extract_SWAT_PRISM_locations(self):
        prism_shape = gpd.read_file(self.shapefile_path)
        prism_shape['ROWCOL'] = prism_shape['row'].astype(str) + prism_shape['col'].astype(str)
        NSRD_PRISM = pd.read_pickle(self.SWATGenXPaths.NSRDB_PRISM_path)
        NSRD_PRISM['ROWCOL'] = NSRD_PRISM['row'].astype(str) + NSRD_PRISM['col'].astype(str)
        NSRDB_SWAT = pd.merge(NSRD_PRISM.drop(columns=['row','col','geometry']), prism_shape, on = "ROWCOL", how = "inner")
        NSRDB_SWAT['NSRDB_index'] = NSRDB_SWAT['NSRDB_index'].astype(int)
        self.logger.info(f"NSRDB_SWAT colums: {NSRDB_SWAT.columns.values}")

        self.nsrdb_indexes = np.unique(NSRDB_SWAT.sort_values(by ='NSRDB_index').NSRDB_index.values)
        self.logger.info(f"number of nsrdb indexes: {len(self.nsrdb_indexes)}")    
        self.NSRD_PRISM = NSRD_PRISM[NSRD_PRISM.NSRDB_index.isin(np.unique(NSRDB_SWAT.NSRDB_index.values))]
        self.logger.info(f"number of nsrdb indexes to be extracted: {len(self.NSRD_PRISM)}")

    def get_elev(self, row, col):
        with rasterio.open(self.SWATGenXPaths.PRISM_dem_path) as src:
            elev_data = src.read(1)
        return elev_data[row, col]

    def extract_from_file(self, f):
        """Extract and process data from NSRDB file with proper handling of time intervals"""
        try:
            # Read data and apply scale factor
            data = f[self.variable][:, self.nsrdb_indexes]
            scale = f[self.variable].attrs['psm_scale_factor']
            data = np.divide(data, scale)
            
            # Ensure data is properly shaped before processing
            if data.shape[0] % 48 != 0:
                self.logger.warning(f"Data length {data.shape[0]} is not divisible by 48 intervals")
                # Trim to nearest complete day
                data = data[:(data.shape[0] // 48) * 48]
            
            # Reshape to (days, intervals per day, locations)
            data = data.reshape(-1, 48, data.shape[1])
            
            # Process based on variable type
            if self.variable == 'ghi':
                # Convert from W/m^2 (30min) to MJ/m^2/day
                data = data * 1800  # Convert 30-min values to Joules (multiply by seconds)
                daily_data = data.sum(axis=1)  # Sum all intervals in a day
                daily_data = daily_data / 1e6  # Convert J to MJ
            elif self.variable in ['wind_speed', 'relative_humidity']:
                # Take daily average
                daily_data = np.nanmean(data, axis=1)
                
            # Replace any invalid values with NaN
            daily_data[~np.isfinite(daily_data)] = np.nan
            
            return daily_data
            
        except Exception as e:
            self.logger.error(f"Error in extract_from_file: {e}")
            raise

    def fetch_nsrdb(self, year):
        file_path = f'{SWATGenXPaths.NSRDB_path}/nsrdb_{year}_full_filtered.h5'
        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            return None
        
        with h5py.File(file_path, mode='r') as f:
            daily_data = self.extract_from_file(f)
        return daily_data

    def should_process_file(self, file_path):
        """Check if a file should be processed based on overwrite setting"""
        if self.overwrite:
            return True
        return not os.path.exists(file_path)

    def write_file_for_index(self, nsrdb_idx, data_all, date_Range):
        """Helper function to write a single file for a specific index using pandas"""
        try:
            row = self.NSRD_PRISM[self.NSRD_PRISM.NSRDB_index == nsrdb_idx].row.values[0]
            col = self.NSRD_PRISM[self.NSRD_PRISM.NSRDB_index == nsrdb_idx].col.values[0]
            file_path = os.path.join(self.output_path, f"r{row}_c{col}.{self.swat_dict[self.variable]}")
            
            if not self.should_process_file(file_path):
                self.logger.info(f"Skipping existing file: {file_path}")
                return

            # Get location data
            lat = self.NSRD_PRISM[self.NSRD_PRISM.NSRDB_index == nsrdb_idx].latitude.values[0]
            lon = self.NSRD_PRISM[self.NSRD_PRISM.NSRDB_index == nsrdb_idx].longitude.values[0]
            elev = self.get_elev(row, col)
            j = np.where(self.nsrdb_indexes == nsrdb_idx)[0][0]

            # Create a DataFrame with the data
            df = pd.DataFrame({
                'year': [d.year for d in date_Range],
                'day': [d.dayofyear for d in date_Range],
                'value': data_all[:, j]
            })

            # Group by year to ensure we only keep complete years
            year_groups = df.groupby('year')
            complete_years_data = []
            
            for year, group in year_groups:
                expected_days = 366 if self._is_leap_year(year) else 365
                if len(group) == expected_days and not group['value'].isna().any():
                    complete_years_data.append(group)

            if not complete_years_data:
                self.logger.error(f"No complete years for index {nsrdb_idx}")
                return

            # Combine all complete years
            df_final = pd.concat(complete_years_data, axis=0)
            
            # Write the header lines first
            with open(file_path, 'w') as f:
                f.write(f"NSRDB(/nrel/nsrdb/v3/nsrdb_{self.years[0]}-{self.years[-1]}).INDEX:{nsrdb_idx}\n")
                f.write("nbyr nstep lat, lon elev\n")
                f.write(f"{len(set(df_final['year']))}\t0\t{lat:.2f}\t{lon:.2f}\t{elev:.2f}\n")

            # Append the data using pandas
            df_final.to_csv(
                file_path, 
                mode='a',
                header=False, 
                index=False, 
                sep='\t',
                float_format='%.2f',
                columns=['year', 'day', 'value'],
                lineterminator='\n'
            )

        except Exception as e:
            self.logger.error(f"Error in write_file_for_index: {e}")
            raise

    def write_to_file(self, data_all):
        """Write data for all NSRDB indexes using pandas"""
        try:
            # Create date range for full period
            date_range = pd.date_range(
                start=f"{self.years[0]}-01-01",
                periods=len(data_all),
                freq='D'
            )

            # Process each NSRDB index
            for nsrdb_idx in self.nsrdb_indexes:
                row = self.NSRD_PRISM[self.NSRD_PRISM.NSRDB_index == nsrdb_idx].row.values[0]
                col = self.NSRD_PRISM[self.NSRD_PRISM.NSRDB_index == nsrdb_idx].col.values[0]
                file_path = os.path.join(self.output_path, f"r{row}_c{col}.{self.swat_dict[self.variable]}")
                
                if not self.should_process_file(file_path):
                    self.logger.info(f"Skipping existing file: {file_path}")
                    continue

                # Get location data
                lat = self.NSRD_PRISM[self.NSRD_PRISM.NSRDB_index == nsrdb_idx].latitude.values[0]
                lon = self.NSRD_PRISM[self.NSRD_PRISM.NSRDB_index == nsrdb_idx].longitude.values[0]
                elev = self.get_elev(row, col)
                j = np.where(self.nsrdb_indexes == nsrdb_idx)[0][0]

                # Create DataFrame with the data
                df = pd.DataFrame({
                    'year': [d.year for d in date_range],
                    'day': [d.dayofyear for d in date_range],
                    'value': data_all[:, j]
                })

                # Group by year to ensure we only keep complete years
                year_groups = df.groupby('year')
                complete_years_data = []
                
                for year, group in year_groups:
                    expected_days = 366 if self._is_leap_year(year) else 365
                    if len(group) == expected_days and not group['value'].isna().any():
                        complete_years_data.append(group)

                if not complete_years_data:
                    self.logger.error(f"No complete years for index {nsrdb_idx}")
                    continue

                # Combine all complete years
                df_final = pd.concat(complete_years_data, axis=0)
                
                # Write the header lines first
                with open(file_path, 'w') as f:
                    f.write(f"NSRDB(/nrel/nsrdb/v3/nsrdb_{self.years[0]}-{self.years[-1]}).INDEX:{nsrdb_idx}\n")
                    f.write("nbyr nstep lat, lon elev\n")
                    f.write(f"{len(set(df_final['year']))}\t0\t{lat:.2f}\t{lon:.2f}\t{elev:.2f}\n")

                # Append the data using pandas
                df_final.to_csv(
                    file_path,
                    mode='a',
                    header=False,
                    index=False,
                    sep='\t',
                    float_format='%.2f',
                    columns=['year', 'day', 'value'],
                    lineterminator='\n'
                )
                
        except Exception as e:
            self.logger.error(f"Error in write_to_file: {e}")
            raise

    def write_cli_file(self):
        """Write the CLI file using pandas"""
        cli_name = os.path.join(self.output_path, f"{self.swat_dict[self.variable]}.cli")
        
        if not self.should_process_file(cli_name):
            self.logger.info(f"Skipping existing CLI file: {cli_name}")
            return

        # Create DataFrame for CLI file
        cli_data = []
        written_rows_cols = set()

        for _, row_data in self.NSRD_PRISM.iterrows():
            row, col = row_data['row'], row_data['col']
            if (row, col) not in written_rows_cols:
                written_rows_cols.add((row, col))
                cli_data.append(f"r{row}_c{col}.{self.swat_dict[self.variable]}")

        # Write CLI file
        with open(cli_name, 'w') as f:
            f.write("NSRDB(/nrel/nsrdb/v3/nsrdb_2000-2020).INDEX: obtained by h5pyd\n")
            f.write(f"{self.variable} file\n")
            f.write('\n'.join(cli_data))

    def cleanup_existing_files(self):
        """Remove existing climate files before processing"""
        try:
            # Remove individual station files
            for pattern in ['*.slr', '*.hmd', '*.wnd']:
                files = glob.glob(os.path.join(self.output_path, pattern))
                for f in files:
                    try:
                        os.remove(f)
                        self.logger.info(f"Removed existing file: {f}")
                    except Exception as e:
                        self.logger.warning(f"Could not remove file {f}: {e}")

            # Remove CLI files
            for cli_file in ['slr.cli', 'hmd.cli', 'wnd.cli']:
                cli_path = os.path.join(self.output_path, cli_file)
                if os.path.exists(cli_path):
                    try:
                        os.remove(cli_path)
                        self.logger.info(f"Removed existing CLI file: {cli_path}")
                    except Exception as e:
                        self.logger.warning(f"Could not remove CLI file {cli_path}: {e}")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def run(self):
        """Main execution method with improved data handling"""
        try:
            # Clean up existing files first
            self.cleanup_existing_files()
            
            self.extract_SWAT_PRISM_locations()
            yearly_data = []
            
            # Extract data for all years and store in a list
            for year in self.years:
                daily_var = self.fetch_nsrdb(year)
                if daily_var is not None:
                    yearly_data.append(daily_var)
                else:
                    self.logger.error(f"Failed to fetch data for year {year}")

            if not yearly_data:
                self.logger.error("No data was extracted")
                return

            # Calculate expected size for each year
            expected_days_per_year = [366 if self._is_leap_year(year) else 365 for year in self.years]
            expected_total_days = sum(expected_days_per_year)

            # Validate and concatenate data
            all_data = []
            current_idx = 0
            for year_idx, year_data in enumerate(yearly_data):
                expected_size = expected_days_per_year[year_idx]
                if len(year_data) != expected_size:
                    self.logger.warning(f"Year {self.years[year_idx]} has {len(year_data)} days, expected {expected_size}")
                    # Pad or trim data if necessary
                    if len(year_data) < expected_size:
                        pad_width = ((0, expected_size - len(year_data)), (0, 0))
                        year_data = np.pad(year_data, pad_width, mode='constant', constant_values=np.nan)
                    else:
                        year_data = year_data[:expected_size]
                all_data.append(year_data)

            # Concatenate all years
            all_data = np.concatenate(all_data, axis=0)
            if len(all_data) != expected_total_days:
                self.logger.error(f"Data size mismatch: got {len(all_data)} days, expected {expected_total_days}")
                return

            self.logger.info(f"Variable {self.variable} extracted: shape {all_data.shape}")
            
            # Create date range matching the data exactly
            date_range = pd.date_range(
                start=f"{self.years[0]}-01-01",
                periods=len(all_data),
                freq='D'
            )
            
            self.write_to_file(all_data)
            self.write_cli_file()
            
        except Exception as e:
            self.logger.error(f"Error in run method: {e}")
            raise

def NSRDB_extract(SWATGenXPaths, VPUID, LEVEL, NAME, overwrite=False):
    variables = ['ghi','wind_speed','relative_humidity']
    wrapped_extract_variable = partial(nsrdb_contructor_wrapper, 
                                     SWATGenXPaths=SWATGenXPaths, 
                                     VPUID=VPUID, 
                                     LEVEL=LEVEL, 
                                     NAME=NAME,
                                     overwrite=overwrite)
    with ThreadPoolExecutor(max_workers=48) as executor:
        executor.map(wrapped_extract_variable, variables)
    print("All NSRDB variables have been extracted for SWAT locations")

if __name__ == "__main__":
    
    NAME = "04127200"
    from SWATGenXConfigPars import SWATGenXPaths
    LEVEL = 'huc12'
    VPUID = find_VPUID(NAME)
    username = "admin"    
    SWATGenXPaths = SWATGenXPaths(username=username, LEVEL=LEVEL, VPUID=VPUID, station_name=NAME)
    NSRDB_extract(SWATGenXPaths, VPUID, LEVEL, NAME, overwrite=True)
