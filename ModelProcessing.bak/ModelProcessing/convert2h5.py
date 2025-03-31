import itertools
import numpy as np
import pandas as pd
import os
import h5py
import logging
from ModelProcessing.unit_global import get_channel_unit, get_hrus_unit
from multiprocessing import Process, Semaphore, Queue, current_process

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

class Gwflow2h5:
    def __init__(self, VPUID, LEVEL, NAME, MODEL_NAME, SCENARIO, file_name):
        self.VPUID = VPUID
        self.LEVEL = LEVEL
        self.NAME = NAME
        self.MODEL_NAME = MODEL_NAME
        self.SCENARIO = SCENARIO
        self.file_name = file_name
        self.scenario_path = os.path.join(SWATGenXPaths.swatgenx_outlet_path, f"{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}/{SCENARIO}")
        self.output_h5_path = os.path.join(SWATGenXPaths.swatgenx_outlet_path,  VPUID, LEVEL, NAME, MODEL_NAME, SCENARIO, "SWAT_OUTPUT.h5")
        self.gwflow_var_path = os.path.join(SWATGenXPaths.swatgenx_outlet_path, f"{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}/{SCENARIO}/{file_name}")
        self.gwflow_input = os.path.join(SWATGenXPaths.swatgenx_outlet_path, f"{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}/{SCENARIO}/gwflow.input")
        self.year = 0
        self.row = 0
        self.title = ""
        self.data = None

    def extract_row_col(self):
        try:
            with open(self.gwflow_input, 'r') as f:
                lines = f.readlines()
                self.rows, self.cols = int(lines[3].split()[0]), int(lines[3].split()[1])
                self.data = np.zeros((0, self.rows, self.cols))  # Initialize with 0 along the first dimension
        except Exception as e:
            logging.error(f"Error extracting row/col: {e}")

    def write_gwflow_on_h5(self, actual_year):
        """ Write gwflow recharge data to the h5 file """
        try:
            with h5py.File(self.output_h5_path, 'a') as f:
                dataset_address = f"{self.file_name.split('_')[0]}/{self.file_name.split('_')[1]}_{self.file_name.split('_')[2]}/{actual_year}"
                if dataset_address in f:
                    del f[dataset_address]
                f.create_dataset(dataset_address, data=self.data[self.year-1, ...])
                f[dataset_address].attrs['unit'] = self.title.split('(')[1].split(')')[0]
                f[dataset_address].attrs['description'] = self.title.split('(')[0]
        except Exception as e:
            logging.error(f"Error writing gwflow to h5: {e}")


    def read_gwflow_variable(self):
        """ Read gwflow recharge data from the SWAT output file """
        try:
            with open(self.gwflow_var_path, 'r') as f:
                logging.info(f"Reading {self.gwflow_var_path}")
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if "Annual" in line:
                        self.title = line.replace(" ", "")
                    elif "year" in line or 'Soil' in line or 'Saturation' in line or 'Tile' in line or 'Lake' in line or 'Groundwater' in line:
                        actual_year = int(line.split(':')[1])
                        logging.info(f"###### actual_year: {actual_year} ######")
                        if self.year > 0:  # write previous year's data to h5 before moving to the next
                            self.write_gwflow_on_h5(actual_year-1)
                        self.year += 1
                        self.row = 0
                        if self.year > self.data.shape[0]:  # Expand the array to accommodate the new year
                            self.data = np.resize(self.data, (self.year, self.rows, self.cols))
                    elif line and len(line.split()) > 1:
                        try:
                            self.data[self.year-1, self.row, :] = np.fromstring(line, sep=' ')
                            self.row += 1
                        except IndexError as e:
                            logging.error(f"IndexError: {e}")
                            logging.error(f"self.year: {self.year}, self.row: {self.row}, line: {line}")
                            raise
                if self.year > 0:  # Ensure the last year's data is written
                    self.write_gwflow_on_h5(actual_year)
        except Exception as e:
            logging.error(f"Error reading gwflow variable: {e}")

def remove_gwflow_dataset(SWAT_H5_path):
    try:
        with h5py.File(SWAT_H5_path, 'a') as f:
            if 'gwflow' in f.keys():
                del f['gwflow']
    except Exception as e:
        logging.error(f"Error removing gwflow dataset: {e}")

def write_hru_metadata(VPUID, LEVEL, NAME, MODEL_NAME, SCENARIO):
    def create_hru_datasets(f, df, column_name, dataset_name):
        f.create_dataset(f"metadata/hru/{dataset_name}", data=df[column_name].values, compression='gzip')

    """ Write HRU metadata to the h5 file"""
    try:
        scenario_path = os.path.join(SWATGenXPaths.swatgenx_outlet_path, f"{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}/{SCENARIO}/")
        hru_data = os.path.join(scenario_path, "hru-data.hru")
        output_h5_path = os.path.join(scenario_path, "SWAT_OUTPUT.h5")

        df = pd.read_csv(hru_data, sep='\s+', skiprows=1, engine='python')
        for col in df.columns:
            df[col] = df[col].astype('str')

        with h5py.File(output_h5_path, 'a') as f:
            create_hru_datasets(f, df, 'name', 'name')
            create_hru_datasets(f, df, 'topo', 'topo')
            create_hru_datasets(f, df, 'hydro', 'hydro')
            create_hru_datasets(f, df, 'soil', 'soil')
            create_hru_datasets(f, df, 'lu_mgt', 'lu_mgt')
            create_hru_datasets(f, df, 'soil_plant_init', 'soil_plant_init')
            create_hru_datasets(f, df, 'surf_stor', 'surf_stor')
            create_hru_datasets(f, df, 'snow', 'snow')
            create_hru_datasets(f, df, 'field', 'field')
    except Exception as e:
        logging.error(f"Error writing HRU metadata: {e}")

def read_SWAT_data(swat_output_file):
    try:
        with open(swat_output_file, 'r') as file:
            header = file.readlines()[1].strip().split()  # Second line is the actual header
        header = [f'null{i+1}' if col == 'null' else col for i, col in enumerate(header)]
        df = pd.read_csv(swat_output_file, sep='\s+', skiprows=3, names=header, low_memory=False)
        df.drop(columns=[col for col in df.columns if 'null' in col], inplace=True)
        components = len(df['name'].unique())
        names = df['name'].unique()
        months = df['mon'].unique()
        years = df['yr'].unique()
        jdays = df['jday'].unique()
        df['id'] = pd.factorize(df['name'])[0]
        ts_length = len(df) // components
        return df.sort_values(by=['name', 'yr', 'mon']), components, ts_length, months, years, names
    except Exception as e:
        logging.error(f"Error reading SWAT data: {e}")

def convert2h5( VPUID, LEVEL, NAME, MODEL_NAME, SCENARIO, TIME_STEPS, FILE_NAMES):
    try:
        scenario_path = os.path.join(SWATGenXPaths.swatgenx_outlet_path, f"{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}/{SCENARIO}/")
        output_h5_path = os.path.join(scenario_path, "SWAT_OUTPUT.h5")

        if os.path.exists(output_h5_path):
            os.remove(output_h5_path)

        for FILE_NAME, TIME_STEP in itertools.product(FILE_NAMES, TIME_STEPS):
            logging.info(f"###### Working on {FILE_NAME} at {TIME_STEP} time step ######")
            swat_output_file = os.path.join(scenario_path, f"{FILE_NAME}_{TIME_STEP}.txt")

            if not os.path.exists(swat_output_file):
                continue

            df, components, ts_length, months, years, names = read_SWAT_data(swat_output_file)

            open_method = 'a' if os.path.exists(output_h5_path) else 'w'
            with h5py.File(output_h5_path, open_method) as f:
                for parameter in df.columns:
                    if parameter in ['jday', 'mon', 'day', 'yr', 'unit', 'gis_id']:
                        continue
                    parameter_hrus = df[parameter].values.reshape(components, ts_length).T
                    assert np.all(df['id'].values.reshape(components, ts_length).T == np.arange(components)[None, :])
                    logging.info(f"Writing {parameter} to h5 file as {FILE_NAME}/{TIME_STEP}/{parameter} with shape {parameter_hrus.shape}")
                    f.create_dataset(f"{FILE_NAME}/{TIME_STEP}/{parameter}", data=parameter_hrus, compression='gzip')
                    f[f"{FILE_NAME}/{TIME_STEP}/{parameter}"].attrs['unit'] = get_channel_unit(parameter) if FILE_NAME == 'channel_sd' else get_hrus_unit(parameter)

                if "metadata/months" not in f:
                    f.create_dataset("metadata/months", data=months)
                if "metadata/years" not in f:
                    f.create_dataset("metadata/years", data=years)
    except Exception as e:
        logging.error(f"Error converting to h5: {e}")

def process_scenario(VPUID, LEVEL, NAME, MODEL_NAME, SCENARIO, semaphore, queue):
    TIME_STEPS = ["yr", "mon"]
    FILE_NAMES = ["channel_sd", 'hru_wb']

    try:
        semaphore.acquire()
        convert2h5(VPUID, LEVEL, NAME, MODEL_NAME, SCENARIO, TIME_STEPS, FILE_NAMES)
        write_hru_metadata(VPUID, LEVEL, NAME, MODEL_NAME, SCENARIO)

        FILE_NAMES = [
            'gwflow_flux_gwet', 'gwflow_flux_gwsoil', 'gwflow_flux_gwsw',
            'gwflow_flux_lake', 'gwflow_flux_lateral',
            'gwflow_flux_pumping_ag', 'gwflow_flux_recharge',
            'gwflow_flux_satex', 'gwflow_flux_tile',
        ]

        scenario_path = os.path.join(SWATGenXPaths.swatgenx_outlet_path, f"{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}/{SCENARIO}")
        SWAT_H5_path = os.path.join(scenario_path, "SWAT_OUTPUT.h5")

        remove_gwflow_dataset(SWAT_H5_path)
        for FILE_NAME in FILE_NAMES:
            logging.info(f"###### Working on {FILE_NAME} ######")
            processor = Gwflow2h5(VPUID, LEVEL, NAME, MODEL_NAME, SCENARIO, FILE_NAME)
            processor.extract_row_col()
            processor.read_gwflow_variable()

    except Exception as e:
        logging.error(f"Error processing scenario: {e}")
    finally:
        semaphore.release()
        queue.put(current_process().name)

from ModelProcessing.SWATGenXConfigPars import SWATGenXPaths

def write_SWAT_OUTPUT_h5(CC_MODEL, selected_NAME=None, number_of_processes=50):
    LEVEL = "huc12"
    MODEL_NAME = "climate_change_models"
    DIC = SWATGenXPaths.swatgenx_outlet_path
    max_processes = 50
    semaphore = Semaphore(max_processes)
    queue = Queue()

    processes = []
    VPUIDs = os.listdir("{SWATGenXPaths.swatgenx_outlet_path}")
    for VPUID in VPUIDs:
        NAMES = os.listdir(f"{SWATGenXPaths.swatgenx_outlet_path}/{VPUID}/huc12")
        NAMES.remove("log.txt")
        for NAME in NAMES:
            if selected_NAME and NAME != selected_NAME:
                continue
            cc_model_path = f"{SWATGenXPaths.swatgenx_outlet_path}/{VPUID}/huc12/{NAME}/climate_change_models"

            SCENARIOS = [cc_name for cc_name in os.listdir(cc_model_path) if not cc_name.endswith('.jpeg') and CC_MODEL in cc_name]
            for SCENARIO in SCENARIOS:
                destination_path = os.path.join(SWATGenXPaths.swatgenx_outlet_path, f"{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}/{SCENARIO}")
                if "SWAT_OUTPUT.h5" in os.listdir(destination_path):
                    os.remove(os.path.join(destination_path, "SWAT_OUTPUT.h5"))
                
                logging.info(f"###### Working on {VPUID}/{NAME}/{SCENARIO} ######")
                p = Process(target=process_scenario, args=(SWATGenXPaths.swatgenx_outlet_path, VPUID, LEVEL, NAME, MODEL_NAME, SCENARIO, semaphore, queue))
                p.start()
                processes.append(p)

    for p in processes:
        p.join()

    while not queue.empty():
        logging.info(f"Completed: {queue.get()}")


if __name__ == "__main__":
    parallel = False
    name = "40500020207"
    cc_name = "NorESM2-LM_historical_r2i1p1f1"
    #"40500020207\climate_change_models\NorESM2-LM_historical_r2i1p1f1"
    write_SWAT_OUTPUT_h5(cc_name, name)