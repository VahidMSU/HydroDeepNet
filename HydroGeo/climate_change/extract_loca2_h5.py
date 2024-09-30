import os
import xarray as xr
import h5py
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds
import time

class ClimateDataProcessor:
    def __init__(self, cc_base_path, out_h5_path, reference_raster, current_dir, requested_model, monthly, cc_model_list):
        self.cc_base_path = cc_base_path
        self.out_h5_path = out_h5_path
        self.reference_raster = reference_raster
        self.current_dir = current_dir
        self.requested_model = requested_model
        self.monthly = monthly
        self.cc_model_list =  cc_model_list

    def save_h5_data(self, region, model, scenario, ensemble, variable_a, var_to_extract, units, time_range, time_step, file_path, lats, lons):
        open_type = 'a' if os.path.exists(self.out_h5_path) else 'w'
        with h5py.File(self.out_h5_path, open_type) as f:
            group_path = f"{region}/{model}/{scenario}/{ensemble}/{time_step}/{time_range}"
            group = f[group_path] if group_path in f else f.create_group(group_path)
            dataset_name = variable_a
            if dataset_name not in group:
                dset = group.create_dataset(dataset_name, data=var_to_extract)
                dset.attrs["description"] = file_path
                dset.attrs["unit"] = units
                dset.attrs["time_range"] = time_range
                print(f"Data saved in group: {group_path}")

            if "lat" not in f:
                f.create_dataset("lat", data=lats)
                print("Latitude data saved in root.")
            if "lon" not in f:
                f.create_dataset("lon", data=lons)
                print("Longitude data saved in root.")

    @staticmethod
    def return_file_name(file):
        file = os.path.basename(file)
        variable = file.split(".")[0]
        model = file.split(".")[1]
        scenario = file.split(".")[2]
        ensemble = file.split(".")[3]
        start_year = file.split(".")[4].split("-")[0]
        end_year = file.split(".")[4].split("-")[1]
        version = file.split(".")[5]
        if "monthly" in file:
            time_step = file.split(".")[6]
            region = file.split(".")[7]
        else:
            time_step = "daily"
            region = file.split(".")[6]
        return variable, model, scenario, ensemble, start_year, end_year, version, region, time_step

    def process_climate_data(self, min_lat, max_lat, min_lon, max_lon, cc_file, model, scenario, ensemble, region, variable_a, start_year, end_year, time_step, file_path):
        try:
            ds = xr.open_dataset(cc_file)
        except Exception as e:
            raise ValueError(
                f"Error: Could not open the NetCDF file {cc_file}. Error: {e}"
            ) from e

        units = ds[variable_a].attrs.get('units', 'unknown')
        time_range = f"{start_year}_{end_year}"
        print(f"Processing data range: {time_range} for variable: {variable_a}")
        ds.coords['lon'] = (ds['lon'] + 180) % 360 - 180
        region_slice = ds.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))

        if region_slice['lat'].size == 0 or region_slice['lon'].size == 0:

            raise ValueError("Error: No data found for the specified region.")
        var_to_extract = region_slice[variable_a]

        if var_to_extract.size == 0:
            raise ValueError("Error: No data found for the specified region.")

        lats = region_slice['lat'].values
        lons = region_slice['lon'].values

        self.save_h5_data(region, model, scenario, ensemble, variable_a, var_to_extract, units, time_range, time_step, file_path, lats, lons)

    def read_all_models_scenarios_ensemble(self):
        data = []
        with open(self.cc_model_list, 'r') as file:
            for line in file:
                if parts := line.split():
                    model = parts[1]
                    scen = parts[2]
                    ensembles = parts[3:]
                    data.extend([model, scen, ens] for ens in ensembles)
                    if "99" in parts[0]:
                        break

        df = pd.DataFrame(data, columns=['model', 'scenario', 'ensemble'])
        if self.requested_model:
            df = df[df.model == self.requested_model]
        return list(zip(df['model'], df['scenario'], df['ensemble']))

    def get_lat_lon_reference_raster(self):
        with rasterio.open(self.reference_raster) as src:
            bounds = src.bounds
            src_crs = src.crs
        return transform_bounds(src_crs, 'EPSG:4326', bounds.left, bounds.bottom, bounds.right, bounds.top)

    def process_all_data(self, variables):  # sourcery skip: low-code-quality
        self.min_lon, self.min_lat, self.max_lon, self.max_lat = self.get_lat_lon_reference_raster()
        print(f" ###############\n min_lat: {self.min_lat}\n max_lat: {self.max_lat}\n min_lon: {self.min_lon}\n max_lon: {self.max_lon}\n ###############")

        result = self.read_all_models_scenarios_ensemble()
        for model, scenario, ensembles in result:
            for ensemble in ensembles.split(","):
                for variable in variables:
                    print(f"Processing: model={model}, scenario={scenario}, ensemble={ensemble}, variable={variable}")
                    cc_path = f"{self.cc_base_path}/{model}/{self.region}/{self.resolution}/{ensemble}/{scenario}/{variable}/"
                    if not os.path.exists(cc_path):
                        print(f"Path does not exist: {cc_path}")
                        time.sleep(10)
                        continue

                    if self.monthly:
                        files = [f for f in os.listdir(cc_path) if f.endswith('.nc') and "month" in f]
                    else:
                        files = [f for f in os.listdir(cc_path) if f.endswith('.nc') and "month" not in f]

                    if not files:
                        print(f"############## No files found in: {cc_path}")
                        time.sleep(10)
                        continue

                    for file in files:
                        print(f"File: {file}")
                        file_path = os.path.join(cc_path, file)
                        variable, model, scenario, ensemble, start_year, end_year, version, region, time_step = self.return_file_name(file)
                        with open(os.path.join(self.current_dir, "log.txt"), 'a') as f:
                            f.write(f"{file}\n")
                            f.write(f"###############\n time_step: {time_step}\n variable: {variable}\n model: {model}\n scenario: {scenario}\n ensemble: {ensemble}\n start_year: {start_year}\n end_year: {end_year}\n version: {version}\n region: {region}\n###############\n")
                            print(f"###############\n time_step: {time_step}\n variable: {variable}\n model: {model}\n scenario: {scenario}\n ensemble: {ensemble}\n start_year: {start_year}\n end_year: {end_year}\n version: {version}\n region: {region}\n###############")
                            variable_a = f'{variable}_tavg' if "month" in file else f'{variable}'
                            cc_file = f"{cc_path}{file}"

                            self.process_climate_data(self.min_lat, self.max_lat, self.min_lon, self.max_lon, cc_file, model, scenario, ensemble, region, variable_a, start_year, end_year, time_step, file_path)


if __name__ == "__main__":

    """
    This script processes the LOCA2 data and saves the data in a single h5 file.
    
    we will do the following tasks on nc files:
    1. Read the nc files
    2. Extract the data for the given region
    3. Save the extracted data in a h5 file
    4. Save the latitude and longitude data in the root of the h5 file
    
    """





    current_dir = os.path.dirname(os.path.realpath(__file__))

    cc_base_path = "/data/LOCA2/CONUS_regions_split/"
    reference_raster = "/data/reference_rasters/DEM_250m.tif"
    cc_model_list = "/data/MyDataBase/climate_change/list_of_all_models.txt"
    out_h5_path = "/data/MyDataBase/LOCA2_MLP.h5"

    requested_model = None
    region          = 'e_n_cent'
    resolution      = '0p0625deg'
    variables       = ['tasmax', 'tasmin', 'pr']
    log_file_path = os.path.join(current_dir, "log.txt")
    log_file_path_error = os.path.join(current_dir, "log_error.txt")

    monthly= False

    #if os.path.exists(out_h5_path):
    #    os.remove(out_h5_path)

    #if os.path.exists(log_file_path):
    #    os.remove(log_file_path)

    #if os.path.exists(log_file_path_error):
    #    os.remove(log_file_path_error)

    processor = ClimateDataProcessor(cc_base_path, out_h5_path, reference_raster, current_dir, requested_model, monthly, cc_model_list)
    processor.region = region
    processor.resolution = resolution
    processor.process_all_data(variables)
