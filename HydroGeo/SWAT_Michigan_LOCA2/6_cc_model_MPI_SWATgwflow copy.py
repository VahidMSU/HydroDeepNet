import os
import shutil
import subprocess
import logging
import datetime
import sys
from multiprocessing import Process, Queue
from convert2h5 import write_SWAT_OUTPUT_h5
from unit_global import get_channel_unit, get_hrus_unit
from extract_point_h5 import extract_model_scenario_ensemble, check_number_of_files
import pandas as pd

sys.path.append('/data/MyDataBase/SWATGenXAppData/codes/ModelProcessing/')
from ModelProcessing.cc_evaluation import SwatModelEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[
    logging.FileHandler("swat_model_runner.log"),
    logging.StreamHandler()
])

class SWATModelRunner:
    def __init__(self, scenario_path, cc_model_path, log_path, name, vpuid, config):
        self.config = config
        self.BASE_PATH = self.config.get("BASE_PATH", "E:/MyDataBase/SWATplus_by_VPUID")
        self.scenario_path = scenario_path
        self.cc_model_path = cc_model_path
        self.log_path = log_path
        self.current_dir = os.getcwd()
        self.general_log_path = os.path.join(self.current_dir, "cc_mpi_log.txt")
        self.name = name
        self.level = self.config.get("level", "huc12")
        self.start_year = self.config.get("start_year", 1997)
        self.end_year = self.config.get("end_year", 2015)
        self.vpuid = vpuid
        self.model_name = self.config.get("model_name", "SWAT_gwflow_MODEL")
        self.no_value = self.config.get("no_value", 1e6)
        self.nyskip = self.config.get("nyskip", 3)
        self.stage = self.config.get("stage", "random")   # historical or future

    def adjust_time_sim(self, cc_model):
        with open(os.path.join(self.cc_model_path, cc_model, "time.sim"), 'w') as f:
            line = f"time.sim: by vahid rafiei on 2024-05-27 22:26\nday_start  yrc_start  day_end  yrc_end  step\n0  {self.start_year}  0  {self.end_year}  0\n"
            f.write(line)

    def adjust_print_cc(self, cc_model):
        with open(os.path.join(self.cc_model_path, cc_model, "print.prt"), 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if "hru_wb" in line:
                    lines[i] = "hru_wb    n    n    n    n\n"
                if "channel_sd" in line:
                    lines[i] = "channel_sd    y    y    y    n\n"
        with open(os.path.join(self.cc_model_path, cc_model, "print.prt"), 'w') as f:
            f.writelines(lines)

    def copy_original_print_print_cc(self, cc_model):
        shutil.copy2(os.path.join(self.scenario_path, "print.prt"), os.path.join(self.cc_model_path, cc_model, "print.prt"))
        shutil.copy2(os.path.join(self.scenario_path, "time.sim"), os.path.join(self.cc_model_path, cc_model, "print.sim"))

    def clean_up(self, cc_name):
        cc_model_path_new = os.path.join(self.cc_model_path, cc_name)
        logging.info(f"Cleaning up {cc_model_path_new}")
        files = os.listdir(cc_model_path_new)
        for file in files:
            if not (file.endswith('.pcp') or file.endswith('.tmp')):
                os.remove(os.path.join(cc_model_path_new, file))

    def run_swatplus_cc_model(self, cc_name):
        cc_model_path_new = os.path.join(self.cc_model_path, cc_name)
        pcp_files = [f for f in os.listdir(cc_model_path_new) if f.endswith('.pcp')]
        tmp_files = [f for f in os.listdir(cc_model_path_new) if f.endswith('.tmp')]
        if len(pcp_files) != len(tmp_files):
            with open(self.log_path, 'a') as f:
                f.write(f"cc model {cc_name} is removed due to having not the same number of .pcp and .tmp files\n")
            shutil.rmtree(cc_model_path_new)
            return

        if "swatplus.exe" not in os.listdir(cc_model_path_new):
            for file in os.listdir(self.scenario_path):
                if not (file.endswith('.txt') or file.endswith('.csv') or file.endswith('.pcp') or file.endswith('.tmp') or file.endswith('.jpeg') or file.endswith('.png') or file.endswith('.h5')):
                    shutil.copy2(os.path.join(self.scenario_path, file), cc_model_path_new)
                    logging.info(f"Copying {file} to {cc_model_path_new}")
        logging.info(f"Running {cc_name} for {cc_model_path_new}")

        self.copy_original_print_print_cc(cc_name)
        self.adjust_time_sim(cc_name)
        self.adjust_print_cc(cc_name)

        with open(self.general_log_path, 'a') as f:
            f.write(f"Running {cc_name} for {cc_model_path_new}\n")

        subprocess.run(["swatplus.exe"], cwd=cc_model_path_new)

        streamflow_data_path = os.path.join(self.BASE_PATH, self.vpuid, self.level, self.name, 'streamflow_data/')
        fig_files_paths = os.path.join(self.BASE_PATH, self.vpuid, self.level, self.name)
        evaluator = SwatModelEvaluator("E:/MyDataBase", self.vpuid, self.level, self.name, self.model_name, self.start_year, self.end_year, self.nyskip, self.no_value, 'historical')
        objective_function_values = evaluator.model_evaluation(cc_model_path_new, streamflow_data_path, fig_files_paths, cc_name)
    def clean_up_cc_model(self, cc_name):
        cc_model_path_new = os.path.join(self.cc_model_path, cc_name)
        files = os.listdir(cc_model_path_new)
        for file in files:
            if not (file.endswith('.pcp') or file.endswith('.tmp')):
                os.remove(os.path.join(cc_model_path_new, file))

    def run_all_cc_models(self, cc_name):
        self.clean_up(cc_name)
        self.run_swatplus_cc_model(cc_name)
        if self.config.get("convert2h5", False):
            write_SWAT_OUTPUT_h5(cc_name, self.name, number_of_processes=1)
        if self.config.get("clean_up_cc_model", False):
            self.clean_up_cc_model(cc_name)

class SWAT_cc_controller:
    def __init__(self, parallel=False, config=None):
        self.BASE_PATH = config.get("BASE_PATH", "E:/MyDataBase/SWATplus_by_VPUID")
        self.NAME = config.get("NAME", None)
        self.VPUID = config.get("VPUID", None)
        self.CC_SCENARIO = config.get("CC_SCENARIO", "historical")
        self.cc_name = config.get("cc_name", None)
        self.num_processes = config.get("num_processes", 1)
        self.parallel = parallel
        self.config = config

    def process(self):
        if self.parallel:
            logging.info("##################Running in parallel################")
            self.SWAT_cc_parallel_workers()
        else:
            scenario_path = f"{self.BASE_PATH}/{self.VPUID}/huc12/{self.NAME}/SWAT_gwflow_MODEL/Scenarios/{self.config['ver_scenario']}"
            cc_model_path = f"{self.BASE_PATH}/{self.VPUID}/huc12/{self.NAME}/climate_change_models"
            log_path = f"{self.BASE_PATH}/{self.VPUID}/huc12/{self.NAME}/log.txt"
            task = (self.cc_name, scenario_path, cc_model_path, log_path, self.NAME, self.VPUID)
            if self.check_historical(self.VPUID, self.NAME, self.cc_name):
                self.worker(task)

    def worker(self, task_queue):
        while not task_queue.empty():
            task = task_queue.get()
            if task is None:
                break
            cc_name, scenario_path, cc_model_path, log_path, name, vpuid = task
            logging.info(f"Running {cc_name} for {name}")
            cc_model = cc_name.split('_')[0]
            scenario = cc_name.split('_')[1]
            ensemble = cc_name.split('_')[2]
            cc_name = f"{cc_model}_{scenario}_{ensemble}"
            output_dir_ = f"{self.BASE_PATH}/{vpuid}/huc12/{name}/climate_change_models/{cc_model}_{scenario}_{ensemble}/"
            base_dir_ = f"{self.BASE_PATH}/{vpuid}/huc12/{name}/SWAT_gwflow_MODEL/Scenarios/{self.config['ver_scenario']}"

            #if check_number_of_files(output_dir_) != check_number_of_files(base_dir_):
            #    logging.info(f"Extracting climate data {cc_model} {scenario} {ensemble} for {name}.....")
            extract_model_scenario_ensemble("D:/MyDataBase", "E:/MyDataBase", name, cc_model, scenario, ensemble)

            #runner = SWATModelRunner(scenario_path, cc_model_path, log_path, name, vpuid, self.config)
            #runner.run_all_cc_models(cc_name)

    def SWAT_cc_parallel_workers(self):
        tasks = self.prepare_data_for_workers()
        task_queue = Queue()
        for task in tasks:
            task_queue.put(task)

        processes = []
        for _ in range(self.num_processes):
            p = Process(target=self.worker, args=(task_queue,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    def check_historical(self, vpuid, name, cc_name):
        logging.info(f"################### Checking historical for {vpuid} {name} {cc_name} ###################")
        path = f"{self.BASE_PATH}/{vpuid}/huc12/{name}/{self.config['stage']}_performance_scores.txt"
        if os.path.exists(path):
            return self._extracted_from_check_historical_(path, vpuid, name, cc_name)
        logging.info(f"VPUID: {vpuid}, NAME: {name} does not have a performance_scores.txt file.")
        return False

    # TODO Rename this here and in `check_historical`
    def _extracted_from_check_historical_(self, path, vpuid, name, cc_name):
        df = pd.read_csv(path, sep="\t", skiprows=1)
        scenarios = df['SCENARIO'].nunique()
        self.write_scores_table(df, path, vpuid, name)
        if cc_name not in df['SCENARIO'].values:
            return False
        logging.info(f"VPUID: {vpuid}, NAME: {name} has a historical simulation.")
        return True

    def write_scores_table(self, df, path, vpuid, name):
        ### rewrite the file
        first_line = f"modified on {datetime.datetime.now()}\n"
        ### sort by date and remove older duplicates
        df['Time_of_writing'] = pd.to_datetime(df['Time_of_writing'])
        df = df.sort_values(by='Time_of_writing')
        ### remove duplicates
        df = df.drop_duplicates(subset=["station", "time_step", "MODEL_NAME", "SCENARIO"], keep='last')
        with open(path, 'w') as f:
            f.write(first_line)
            # Specify line_terminator to avoid extra newlines
            df.to_csv(f, sep='\t', index=False, lineterminator='\n')
        logging.info(f"VPUID: {vpuid}, NAME: {name} has less than 180 rows. Deleting file.")

    def prepare_data_for_workers(self):
        all_cc_names = []
        all_scenario_paths = []
        all_cc_model_paths = []
        all_log_paths = []
        all_names = []
        all_vpuids = []
        vpuids = os.listdir(self.BASE_PATH)
        current_dir = os.getcwd()
        general_log_path = os.path.join(current_dir, "cc_mpi_log.txt")
        if os.path.exists(general_log_path):
            os.remove(general_log_path)

        for vpuid in vpuids:
            logging.info(f"################### Searching within {vpuid} ###################")
            names = os.listdir(f"{self.BASE_PATH}/{vpuid}/huc12")
            names.remove("log.txt")
            for name in names:
                if name != "40500010102":
                    continue
                logging.info(f"################### Searching within {name} ###################")
                scenario_path = f"{self.BASE_PATH}/{vpuid}/huc12/{name}/SWAT_gwflow_MODEL/Scenarios/{self.config['ver_scenario']}"
                if not os.path.exists(scenario_path):
                    with open(general_log_path, 'a') as f:
                        f.write(f"Path does not exist: {scenario_path}\n")
                    continue

                cc_model_path = f"{self.BASE_PATH}/{vpuid}/huc12/{name}/climate_change_models"
                log_path = f"{self.BASE_PATH}/{vpuid}/huc12/{name}/log.txt"


                cc_names = os.listdir(cc_model_path)
                cc_names = [cc_name for cc_name in cc_names if not cc_name.endswith('.jpeg') and self.CC_SCENARIO in cc_name]
                for cc_name in cc_names:
                    if self.check_historical(vpuid, name, cc_name):
                        logging.info(f"VPUID: {vpuid}, NAME: {name} has a historical simulation.")
                        continue

                    all_cc_names.append(cc_name)
                    all_scenario_paths.append(scenario_path)
                    all_cc_model_paths.append(cc_model_path)
                    all_log_paths.append(log_path)
                    all_names.append(name)
                    all_vpuids.append(vpuid)

        return list(zip(all_cc_names, all_scenario_paths, all_cc_model_paths, all_log_paths, all_names, all_vpuids))

# sourcery skip: swap-if-else-branches, use-named-expression
if __name__ == "__main__":
    parallel = True

    # rules:
    # 0- we will extract pcp and tmp if they are not extracted or the number of extracted files is not the same as the base model
    # 1- we will remove the cc_model directory if the number of pcp and tmp files are not the same and will not proceed
    # 2- we will run the model

    if not parallel:
        config = {
            "NAME": "40500010102",
            "VPUID": "0405",
            "cc_name": "ACCESS-CM2_historical_r1i1p1f1",
            "BASE_PATH": "E:/MyDataBase/SWATplus_by_VPUID",
            "start_year": 1997,
            "end_year": 2015,
            "model_name": "SWAT_gwflow_MODEL",
            "num_processes": 1,
            "stage": "historical",
            "convert2h5": False,
            'clean_up_cc_model': False,
            'ver_scenario': 'Scenario_verification_stage_0'
        }
        SWAT_cc_controller(parallel=False, config=config).process()
    else:
        config = {
            "CC_SCENARIO": "historical",
            "BASE_PATH": "E:/MyDataBase/SWATplus_by_VPUID",
            "start_year": 1997,
            "end_year": 2015,
            "model_name": "SWAT_gwflow_MODEL",
            "num_processes": 80,
            "stage": "historical",
            "convert2h5": False,
             'clean_up_cc_model': True,
             'ver_scenario': 'Scenario_verification_stage_0'
        }
        SWAT_cc_controller(parallel=True, config=config).process()
