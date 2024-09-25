import mpi4py
from mpi4py import MPI
import h5py
import numpy as np
import pandas as pd
import os
from multiprocessing import Process
import time
import geopandas as gpd
from multiprocessing import Process, Queue
from functools import partial
import os
import shutil
#### removing the models 


def remove_incomplete():
    VPUIDs =  os.listdir("/data/MyDataBase/SWATplus_by_VPUID")
    for VPUID in VPUIDs:
        if VPUID =="0405":
            continue
        NAMES = os.listdir(f"/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12")
        NAMES.remove("log.txt")
        for NAME in NAMES:
            verification_path = f"/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/climate_change_models/"
            base_scenario_path = f"/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/SWAT_gwflow_MODEL/Scenarios/Scenario_verification_stage_0"

            files = os.listdir(base_scenario_path)
            ## get files ending with pcp and tmp
            base_pcp_files = [file for file in files if file.endswith("pcp")]
            base_tmp_files = [file for file in files if file.endswith("tmp")]

            cc_models = os.listdir(verification_path)
            for cc_model in cc_models:
                path = os.path.join(verification_path, cc_model)
                files = os.listdir(path)
                cc_pcp_files = [file for file in files if file.endswith("pcp")] 
                cc_tmp_files = [file for file in files if file.endswith("tmp")]
                if len(cc_pcp_files) != len(base_pcp_files) or len(cc_tmp_files) != len(base_tmp_files):
                    print(f"Deleting {cc_model}")
                    shutil.rmtree(path) 

class PointDataExtractor:
    def __init__(self, LOCA2, swat_lat, swat_lon, elev, name, output_dir, loca2_lats, loca2_lons):
        self.LOCA2 = LOCA2
        self.swat_lat = swat_lat
        self.swat_lon = swat_lon
        self.elev = elev
        self.name = name
        self.output_dir = output_dir
        self.loca2_lats = loca2_lats
        self.loca2_lons = loca2_lons

    def get_nearest_index(self, array, value):
        return (np.abs(array - value)).argmin()
    def verify_index(self, lats, lons, lat_idx, lon_idx):
        """
        Verifies that the lat and lon at the given indices are close to the target lat and lon.
        """
        lat_value = lats[lat_idx]
        lon_value = lons[lon_idx]
        lat_diff = abs(lat_value - self.swat_lat)
        lon_diff = abs(lon_value - self.swat_lon)

        if lat_diff > 0.1 or lon_diff > 0.1:
            print(f"Significant difference detected: Target lat/lon ({self.swat_lat}, {self.swat_lon}), "
                            f"Nearest lat/lon ({lat_value}, {lon_value})."
                            )
        return lat_value, lon_value

    def write_tmp_file(self, region, model, scenario, ensemble, time_step, time_ranges, lat_idx, lon_idx):

        all_time_series = []
        for time_range in time_ranges:
            loca2_tasmax_path = f"/{region}/{model}/{scenario}/{ensemble}/{time_step}/{time_range}/tasmax"
            loca2_tasmin_path = f"/{region}/{model}/{scenario}/{ensemble}/{time_step}/{time_range}/tasmin"

            if loca2_tasmax_path not in self.LOCA2.keys():
                with open("log.txt", 'a') as log_file:
                    log_file.write(f"Group {loca2_tasmax_path} does not exist in the HDF5 file.\n")
                continue
            if loca2_tasmin_path not in self.LOCA2.keys():
                with open("log.txt", 'a') as log_file:
                    log_file.write(f"Group {loca2_tasmin_path} does not exist in the HDF5 file.\n")
                continue

            loca2_tasmax = self.LOCA2[loca2_tasmax_path]
            loca2_tasmin = self.LOCA2[loca2_tasmin_path]

            tmax = loca2_tasmax[:, lat_idx, lon_idx]
            tmin = loca2_tasmin[:, lat_idx, lon_idx]
            length = len(tmax)
            time_series = pd.date_range(start=f"1/1/{time_range.split('_')[0]}", periods=length, freq='D')

            tmax = pd.DataFrame(tmax, index=time_series, columns=['tasmax'])
            tmin = pd.DataFrame(tmin, index=time_series, columns=['tasmin'])
            tmp_data = pd.merge(tmax, tmin, left_index=True, right_index=True)

            all_time_series.append(pd.DataFrame(tmp_data, index=time_series, columns=['tasmax', 'tasmin']))


            all_time_series[-1]['year'] = all_time_series[-1].index.year
            all_time_series[-1]['day'] = all_time_series[-1].index.dayofyear


        if not all_time_series:
           # print(f"No data found for {model} {scenario} {ensemble} {time_range} {region}")
            return
        all_time_series = pd.concat(all_time_series)
        nbyr = len(all_time_series['year'].unique())

        all_time_series['tasmax'] = all_time_series['tasmax'] - 273.15
        ## round to 2 decimal places
        all_time_series['tasmax'] = all_time_series['tasmax'].round(2)
        all_time_series['tasmin'] = all_time_series['tasmin'] - 273.15
        all_time_series['tasmin'] = all_time_series['tasmin'].round(2)
        all_time_series[['year', 'day', 'tasmax', 'tasmin']]

        with open(os.path.join(self.output_dir, f"{self.name}.tmp"), 'w') as tmp_file:
            tmp_file.write(f"{model} {scenario} {ensemble} {time_range} {region}\n")
            tmp_file.write(f"nbyr tstep lat lon elev\n")
            tmp_file.write(f"{nbyr}\t0\t{self.swat_lat:.2f}\t{self.swat_lon:.2f}\t{self.elev}\n")

        all_time_series.to_csv(os.path.join(self.output_dir, f"{self.name}.tmp"), sep='\t', header=False, index=False, columns=['year', 'day', 'tasmax', 'tasmin'], lineterminator="\n", mode='a')

    def write_pcp_file(self, region, model, scenario, ensemble, time_step, time_ranges, lat_idx, lon_idx):
        loca2_variable = "pr"

        all_time_series = []
        for time_range in time_ranges:
            loca2_pr = f"/{region}/{model}/{scenario}/{ensemble}/{time_step}/{time_range}/{loca2_variable}"
            if loca2_pr not in self.LOCA2.keys():
                with open("log.txt", 'a') as log_file:
                    log_file.write(f"Group {loca2_pr} does not exist in the HDF5 file.\n")
                    return

            dataset = self.LOCA2[loca2_pr]

            data = dataset[:, lat_idx, lon_idx]

            time_series = pd.date_range(start=f"1/1/{time_range.split('_')[0]}", periods=len(data), freq='D')
            all_time_series.append(pd.DataFrame(data, index=time_series, columns=[loca2_variable]))

            all_time_series[-1]['year'] = all_time_series[-1].index.year
            all_time_series[-1]['day'] = all_time_series[-1].index.dayofyear

        if not all_time_series:
            return

        all_time_series = pd.concat(all_time_series)
        nbyr = len(all_time_series['year'].unique())

        all_time_series[loca2_variable] = all_time_series[loca2_variable] * 86400  # convert from kg/m^2/s to mm/day
        ## round to 2 decimal places
        all_time_series[loca2_variable] = all_time_series[loca2_variable].round(2)
        all_time_series[['year', 'day', loca2_variable]]

        with open(os.path.join(self.output_dir, f"{self.name}.pcp"), 'w') as pcp_file:
            pcp_file.write(f"{loca2_variable} {model} {scenario} {ensemble} {region}\n")
            pcp_file.write(f"nbyr tstep lat lon elev\n")
            pcp_file.write(f"{nbyr}\t0\t{self.swat_lat:.2f}\t{self.swat_lon:.2f}\t{self.elev}\n")

        all_time_series.to_csv(os.path.join(self.output_dir, f"{self.name}.pcp"), sep='\t', header=False, index=False, columns=['year', 'day', loca2_variable], lineterminator="\n", mode='a')


    def extract_data(self, region, model, scenario, ensemble, time_step, time_ranges, variables):

        lat_idx = self.get_nearest_index(self.loca2_lats, self.swat_lat)
        lon_idx = self.get_nearest_index(self.loca2_lons, self.swat_lon)
        #lat_value, lon_value = self.verify_index(self.loca2_lats, self.loca2_lons, lat_idx, lon_idx)

        for variable in variables:
            if variable == "pcp":
                self.write_pcp_file(region, model, scenario, ensemble, time_step, time_ranges, lat_idx, lon_idx)
            elif variable == 'tmp':
                self.write_tmp_file(region, model, scenario, ensemble, time_step, time_ranges, lat_idx, lon_idx)


class LOCA2Extractor_scheduler:
    def __init__(self, h5_path, BASE_PATH, TARGET_PATH):
        self.LOCA2 = h5py.File(h5_path, 'r')
        self.BASE_PATH = BASE_PATH
        self.TARGET_PATH = TARGET_PATH
        self.loca2_lats = self.LOCA2['lat'][:]
        self.loca2_lons = self.LOCA2['lon'][:]
        self.current_dir = os.path.dirname(os.path.realpath(__file__))
        self.list_of_models = os.path.join(self.current_dir, "input/list_of_all_models.txt")

    def get_time_ranges(self, scenario):
        if scenario == "historical":
            return ["1950_2014"]
        else:
            return ["2015_2044", "2045_2074", "2075_2100"]

    def submit_job(self, gdf, cc_model, scenario, ensemble, output_dir):
        time_ranges = self.get_time_ranges(scenario)
        region = "e_n_cent"
        time_step = "daily"
        start_time = time.time()
        for lat, lon, elev, name in zip(gdf.geometry.y, gdf.geometry.x, gdf.elev, gdf.name):
            extractor = PointDataExtractor(self.LOCA2, lat, lon, elev, name, output_dir, self.loca2_lats, self.loca2_lons)
            extractor.extract_data(region, cc_model, scenario, ensemble, time_step, time_ranges, variables=['tmp', 'pcp'])
        end_time = time.time()
        print(f"Time taken for {output_dir} is {int(end_time - start_time)} seconds")

    def extract_model_scenario_ensemble(self, VPUID, NAME, cc_model, scenario, ensemble, LEVEL = "huc12"):
        output_dir = f"{self.TARGET_PATH}/{VPUID}/{LEVEL}/{NAME}/climate_change_models/{cc_model}_{scenario}_{ensemble}"

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        os.makedirs(output_dir, exist_ok=True)

        SWAT_PRISM = fr"{TARGET_PATH}/{VPUID}/{LEVEL}/{NAME}/PRISM/PRISM_grid.shp"
        gdf = gpd.read_file(SWAT_PRISM).to_crs("EPSG:4326")
        self.submit_job(gdf, cc_model, scenario, ensemble, output_dir)

    def read_all_models_scenarios_ensemble(self):
        data = []
        with open(self.list_of_models, 'r') as file:
            for line in file:
                if parts := line.split():
                    model = parts[1]
                    scen = parts[2]
                    ensembles = parts[3:]
                    data.extend([model, scen, ens] for ens in ensembles)
                    if "99" in parts[0]:
                        break

        df = pd.DataFrame(data, columns=['model', 'scenario', 'ensemble'])
        return list(zip(df['model'], df['scenario'], df['ensemble']))

    def check_number_of_files(self, output_dir):
        if not os.path.exists(output_dir):
            return 0, 0
        pcp_files = [f for f in os.listdir(output_dir) if '.pcp' in f]
        tmp_files = [f for f in os.listdir(output_dir) if '.tmp' in f]
        return len(pcp_files), len(tmp_files)

    def prepare_data_for_workers(self, model_scenario_ensemble, LEVEL = "huc12"):
        tasks = []
        VPUIDS = os.listdir(self.TARGET_PATH)
        for VPUID in VPUIDS:
            if VPUID in ["0405"]:    #### only for 0405 to save space
                continue
            NAMES = os.listdir(f"{self.TARGET_PATH}/{VPUID}/{LEVEL}")
            NAMES.remove("log.txt")
            for NAME in NAMES:
                for cc_model, scenario, ensembles in model_scenario_ensemble:
                    for ensemble in ensembles.split(","):
                        output_dir_ = f"{self.TARGET_PATH}/{VPUID}/{LEVEL}/{NAME}/climate_change_models/{cc_model}_{scenario}_{ensemble}"
                        base_scenario =  f"{self.TARGET_PATH}/{VPUID}/{LEVEL}/{NAME}/SWAT_gwflow_MODEL/Scenarios/Scenario_verification_stage_0"
                        os.makedirs(output_dir_, exist_ok=True)
                        if self.check_number_of_files(output_dir_) != self.check_number_of_files(base_scenario):
                            tasks.append((VPUID, NAME, cc_model, scenario, ensemble))
        return tasks

    def worker(self, rank, size, tasks):
        task_chunk = tasks[rank::size]
        for VPUID, NAME, cc_model, scenario, ensemble in task_chunk:
            print(f"Process {rank} extracting {cc_model}_{scenario}_{ensemble} for {NAME}")
            self.extract_model_scenario_ensemble(VPUID, NAME, cc_model, scenario, ensemble)


if __name__ == "__main__":

    # Note: This is based on MPI and to run this code, use: mpiexec -n 5 python 3_1_extract_point_h5.py
    BASE_PATH = "/data/MyDataBase/SWATplus_by_VPUID"
    TARGET_PATH = "/data/MyDataBase/SWATplus_by_VPUID"
    h5_path = "/data/MyDataBase/LOCA2_MLP.h5"
    start_time = time.time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    LOCA2Extractor_scheduler = LOCA2Extractor_scheduler(h5_path, BASE_PATH, TARGET_PATH)

    if rank == 0:
        model_scenario_ensemble = LOCA2Extractor_scheduler.read_all_models_scenarios_ensemble()
        tasks = LOCA2Extractor_scheduler.prepare_data_for_workers(model_scenario_ensemble)
        print(f"Prepared data for extraction: {tasks}")
    else:
        tasks = None

    tasks = comm.bcast(tasks, root=0)
    LOCA2Extractor_scheduler.worker(rank, size, tasks)

    comm.Barrier()
    if rank == 0:
        LOCA2Extractor_scheduler.LOCA2.close()
    end_time = time.time()
    print(f"Total time taken is {int(end_time - start_time)} seconds")
