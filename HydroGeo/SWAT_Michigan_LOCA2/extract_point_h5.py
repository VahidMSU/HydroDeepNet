import h5py
import numpy as np
import pandas as pd
import os
import shutil
import geopandas as gpd
from functools import partial
from multiprocessing import Process

def read_all_models_scenarios_ensemble():
    data = []
    with open("E:/MyDataBase/climate_change/list_of_all_models.txt", 'r') as file:
        for line in file:
            if parts := line.split():
                model = parts[1]
                scen = parts[2]
                ensembles = parts[3:]
                data.extend([model, scen, ens] for ens in ensembles)
                if "99" in parts[0]:
                    break

    df = pd.DataFrame(data, columns=['model', 'scenario', 'ensemble'])
    #df = df[df.model == requested_model]
    return list(zip(df['model'], df['scenario'], df['ensemble']))

class LOCA2PointDataExtractor:
    def __init__(self, h5_path, lat, lon, elev, name, output_dir):
        self.h5_path = h5_path
        self.lat = lat
        self.lon = lon
        self.elev = elev
        self.name = name
        self.output_dir = output_dir

    def get_nearest_index(self, array, value):
        return (np.abs(array - value)).argmin()
    def verify_index(self, lats, lons, lat_idx, lon_idx):
        """
        Verifies that the lat and lon at the given indices are close to the target lat and lon.
        """
        lat_value = lats[lat_idx]
        lon_value = lons[lon_idx]
        lat_diff = abs(lat_value - self.lat)
        lon_diff = abs(lon_value - self.lon)

        if lat_diff > 0.01 or lon_diff > 0.01:
            print(f"Significant difference detected: Target lat/lon ({self.lat}, {self.lon}), "
                            f"Nearest lat/lon ({lat_value}, {lon_value})."
                            )
        return lat_value, lon_value

    def extract_data(self, region, model, scenario, ensemble, time_step, time_ranges, variable):


        if variable == "pcp":
            variable = "pr"

            with h5py.File(self.h5_path, 'r') as f:
                lats = f['lat'][:]
                lons = f['lon'][:]

                lat_idx = self.get_nearest_index(lats, self.lat)
                lon_idx = self.get_nearest_index(lons, self.lon)
                lat_value, lon_value = self.verify_index(lats, lons, lat_idx, lon_idx)

                #print(f"Nearest lat index: {lat_idx}, value: {lats[lat_idx]}")
                #print(f"Nearest lon index: {lon_idx}, value: {lons[lon_idx]}")

                all_time_series = []
                for time_range in time_ranges:
                    if time_step == "monthly":
                        group_path = f"/{region}/{model}/{scenario}/{ensemble}/{time_step}/{time_range}/{variable}_tavg"
                    else:
                        group_path = f"/{region}/{model}/{scenario}/{ensemble}/{time_step}/{time_range}/{variable}"
                    if group_path not in f:
                        with open("log.txt", 'a') as f:
                            f.write(f"Group {group_path} does not exist in the HDF5 file.\n")
                            return

                    dataset = f[group_path]
                    data = dataset[:, lat_idx, lon_idx]

                    time_series = pd.date_range(start=f"1/1/{time_range.split('_')[0]}", periods=len(data), freq='D')
                    all_time_series.append(pd.DataFrame(data, index=time_series, columns=[variable]))

                    description = dataset.attrs.get("description", "No description")
                    units = dataset.attrs.get("unit", "unknown")

                    all_time_series[-1]['year'] = all_time_series[-1].index.year
                    all_time_series[-1]['day'] = all_time_series[-1].index.dayofyear
                    all_time_series[-1]['name'] = f"{model}_{scenario}_{ensemble}"
                    all_time_series[-1]['region'] = region
                if not all_time_series:
                    return
                all_time_series = pd.concat(all_time_series)
                nbyr = len(all_time_series['year'].unique())
                tstep = "0"
                lat = self.lat
                lon = self.lon
                elev = self.elev

                if variable == "tas":
                    all_time_series[variable] = all_time_series[variable] - 273.15

                if variable == "pr":
                    all_time_series[variable] = all_time_series[variable] * 86400

                all_time_series[['year', 'day', variable]]

                with open(os.path.join(self.output_dir, f"{self.name}.pcp"), 'w') as f:
                    f.write(f"{variable} {model} {scenario} {ensemble} {region}\n")
                    f.write(f"nbyr tstep lat lon elev\n")
                    f.write(f"{nbyr} {tstep} {lat:.2f} {lon:.2f} {elev}\n")

                    all_time_series.apply(lambda row: f.write(f"{row['year']} {row['day']} {row[variable]:.2f}\n"), axis=1)

        elif variable == 'tmp':

            with h5py.File(self.h5_path, 'r') as f:
                lats = f['lat'][:]
                lons = f['lon'][:]

                lat_idx = self.get_nearest_index(lats, self.lat)
                lon_idx = self.get_nearest_index(lons, self.lon)

                print(f"Nearest lat index: {lat_idx}, value: {lats[lat_idx]}")
                print(f"Nearest lon index: {lon_idx}, value: {lons[lon_idx]}")

                all_time_series = []
                for time_range in time_ranges:
                    if time_step == "monthly":
                        group_path = f"/{region}/{model}/{scenario}/{ensemble}/{time_step}/{time_range}/{variable}_tavg"
                    else:
                        group_path = f"/{region}/{model}/{scenario}/{ensemble}/{time_step}/{time_range}/tasmax"
                        group_path2 = f"/{region}/{model}/{scenario}/{ensemble}/{time_step}/{time_range}/tasmin"

                    if group_path not in f.keys():
                        with open("log.txt", 'a') as log_file:
                            log_file.write(f"Group {group_path} does not exist in the HDF5 file.\n")
                        continue
                    if group_path2 not in f.keys():
                        with open("log.txt", 'a') as log_file:
                            log_file.write(f"Group {group_path2} does not exist in the HDF5 file.\n")
                        continue

                    try:
                        dataset = f[group_path]
                    except KeyError:
                        with open("log.txt", 'a') as log_file:
                            log_file.write(f"Dataset {group_path} does not exist in the HDF5 file.\n")
                        continue

                    dataset = f[group_path]
                    dataset2 = f[group_path2]

                    data = dataset[:, lat_idx, lon_idx]
                    data2 = dataset2[:, lat_idx, lon_idx]

                    time_series = pd.date_range(start=f"1/1/{time_range.split('_')[0]}", periods=len(data), freq='D')

                    data = pd.DataFrame(data, index=time_series, columns=['tasmax'])
                    data2 = pd.DataFrame(data2, index=time_series, columns=['tasmin'])
                    data = pd.merge(data, data2, left_index=True, right_index=True)

                    all_time_series.append(pd.DataFrame(data, index=time_series, columns=['tasmax', 'tasmin']))

                    description = dataset.attrs.get("description", "No description")
                    units = dataset.attrs.get("unit", "unknown")

                    all_time_series[-1]['year'] = all_time_series[-1].index.year
                    all_time_series[-1]['day'] = all_time_series[-1].index.dayofyear
                    all_time_series[-1]['name'] = f"{model}_{scenario}_{ensemble}"
                    all_time_series[-1]['region'] = region
                if not all_time_series:
                    return
                all_time_series = pd.concat(all_time_series)
                nbyr = len(all_time_series['year'].unique())
                tstep = "0"
                lat = self.lat
                lon = self.lon
                elev = self.elev

                if variable == "tmp":
                    all_time_series['tasmax'] = all_time_series['tasmax'] - 273.15
                    all_time_series['tasmin'] = all_time_series['tasmin'] - 273.15

                all_time_series[['year', 'day', 'tasmax', 'tasmin']]

                with open(os.path.join(self.output_dir, f"{self.name}.tmp"), 'w') as f:
                    f.write(f"{variable} {model} {scenario} {ensemble} {time_range} {region}\n")
                    f.write(f"nbyr tstep lat lon elev\n")
                    f.write(f"{nbyr} {tstep} {lat} {lon} {elev}\n")

                    all_time_series.apply(lambda row: f.write(f"{row['year']} {row['day']} {row['tasmax']:.2f} {row['tasmin']:.2f}\n"), axis=1)

def submit_job(gdf, cc_model, scenario, ensemble, output_dir):
    linux = False
    if linux:
        h5_path = "/data/HydroMetData/LOCA/LOCA2_extracted.h5"
    else:
        h5_path = "//35.9.219.75/Data/MyDataBase/LOCA2_MLP.h5"
    if scenario == "historical":
        time_ranges = ["1950_2014"]
    else:
        time_ranges = ["2015_2044","2045_2074","2075_2100"]

    region = "e_n_cent"
    time_step = "daily"
    for lat, lon, elev, name in zip(gdf.geometry.y, gdf.geometry.x, gdf.elev, gdf.name):
        #variable = "tmp"
        for variable in ["tmp", "pcp"]:

            extractor = LOCA2PointDataExtractor(h5_path, lat, lon, elev, name, output_dir)
            data = extractor.extract_data(region, cc_model, scenario, ensemble, time_step, time_ranges, variable)



def extract_model_scenario_ensemble(BASE_PATH, TARGET_PATH, NAME, cc_model, scenario, ensemble):
    ### read models scenarios and ensembles
    path  = "E:/MyDataBase/climate_change/list_of_all_models.txt"
    VPUID = f"0{NAME[:3]}"

    model_path = os.path.join(BASE_PATH, f"SWAT_input/huc12/{NAME}/PRISM")
    mesh_path = os.path.join(model_path, "PRISM_grid.shp")
    cc_path = os.path.join("E:/MyDataBase/climate_change/cirrus.ucsd.edu/~pierce/LOCA2/CONUS_regions_split")
    output_dir = os.path.join(TARGET_PATH, f"SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/climate_change_models/{cc_model}_{scenario}_{ensemble}")
    ### if the output_dir already exists and the number of files ending with pcp and tmp are the same, then continue

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)


    SWAT_PRISM = fr"E:/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/PRISM/PRISM_grid.shp"
    gdf = gpd.read_file(SWAT_PRISM).to_crs("EPSG:4326")
    submit_job(gdf, cc_model, scenario, ensemble, output_dir)

def check_if_empty(output_dir):
    if os.path.exists(output_dir):
        pcp_files = [f for f in os.listdir(output_dir) if f.endswith("pcp")]
        tmp_files = [f for f in os.listdir(output_dir) if f.endswith("tmp")]
        return len(pcp_files) == len(tmp_files)
    else:
        return False

def check_number_of_files(output_dir):
    if not os.path.exists(output_dir):
        return 0, 0
    pcp_files = [f for f in os.listdir(output_dir) if '.pcp' in f]
    tmp_files = [f for f in os.listdir(output_dir) if '.tmp' in f]
    return len(pcp_files), len(tmp_files)

if __name__ == "__main__":

    BASE_PATH = "D:/MyDataBase"
    TARGET_PATH = "E:/MyDataBase"

    model_scenario_ensemble = read_all_models_scenarios_ensemble()

    processes = []
    NAMES = os.listdir(os.path.join(BASE_PATH, "SWAT_input/huc12"))
    NAMES.remove('log.txt')
    Parallel = True
    i = 0
    if Parallel:
        for NAME in NAMES:
            VPUID = f"0{NAME[:3]}"
            for cc_model, scenario, ensembles in model_scenario_ensemble:
                for ensemble in ensembles.split(","):
                    output_dir_ = os.path.join(TARGET_PATH, f"SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/climate_change_models/{cc_model}_{scenario}_{ensemble}")
                    base_dir_ =  f"E:/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/SWAT_gwflow_MODEL/Scenarios/Scenario_verification_stage_0"
                    os.makedirs(output_dir_, exist_ok=True)
                    if check_number_of_files(output_dir_) != check_number_of_files(base_dir_):
                        i+=1
                        print(f"{i}  Extracting {NAME} {cc_model}_{scenario}_{ensemble} for {NAME}",  check_number_of_files(output_dir_), check_number_of_files(base_dir_))

                        wrapped = partial(extract_model_scenario_ensemble, BASE_PATH, TARGET_PATH, NAME, cc_model, scenario, ensemble)
                        process = Process(target=wrapped)
                        process.start()
                        processes.append(process)
                        if len(processes) == 100:
                            for process in processes:
                                process.join()
                            processes = []
        for process in processes:
            process.join()
    else:
        # E:\MyDataBase\SWATplus_by_VPUID\0405\huc12\40500010102\climate_change_models\ACCESS-CM2_historical_r1i1p1f1
        NAME = "40500010102"
        VPUID = f"0{NAME[:3]}"
        TARGET_PATH = "E:/MyDataBase"
        scenario = "historical"
        cc_model = "ACCESS-CM2"
        ensemble = "r1i1p1f1"

        output_dir_ = os.path.join(TARGET_PATH, f"SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/climate_change_models/{cc_model}_{scenario}_{ensemble}")
        if check_number_of_files(output_dir_) != check_number_of_files(base_dir_):
            extract_model_scenario_ensemble(BASE_PATH, TARGET_PATH, NAME, cc_model, scenario, ensemble)