import os
import shutil
import subprocess
import logging
import datetime
from ModelProcessing.evaluation import SwatModelEvaluator
from multiprocessing import Process, Queue
from ModelProcessing.convert2h5 import write_SWAT_OUTPUT_h5
import pandas as pd
import time 

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[
		logging.FileHandler("swat_model_runner.log"),
		logging.StreamHandler()
])

class SWATModelRunner:
		def __init__(self, scenario_path, cc_model_path, log_path, name, vpuid, config):
				self.config = config
				self.BASE_PATH = self.config.get("BASE_PATH", "/data/MyDataBase/SWATplus_by_VPUID")
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
				self.filter_vpuid = self.config.get("filter_vpuid", None)

		def adjust_time_sim(self, cc_model):
				with open(os.path.join(self.cc_model_path, cc_model, "time.sim"), 'w') as f:
						line = f"time.sim: by vahid rafiei on 2024-05-27 22:26\nday_start  yrc_start  day_end  yrc_end  step\n0  {self.start_year}  0  {self.end_year}  0\n"
						f.write(line)

		def adjust_print_cc(self, cc_model):
				with open(os.path.join(self.cc_model_path, cc_model, "print.prt"), 'r') as f:
						lines = f.readlines()
						for i, line in enumerate(lines):
								if "hru_wb" in line:
										lines[i] = self.config.get("hru_wb", "hru_wb    n    y    y    y\n")
								if "channel_sd" in line:
										lines[i] = self.config.get("channel_sd", "channel_sd    n    y    y    y\n")
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
								message = f"cc model {cc_name} is removed due to having not the same number of .pcp and .tmp files"
								logging.info(message)
						shutil.rmtree(cc_model_path_new)
						return

				logging.info(f"##Copying SWAT+ input to:\n{cc_model_path_new}")
				for file in os.listdir(self.scenario_path):
						if not (file.endswith('.txt') or file.endswith('.csv') or file.endswith('.pcp') or file.endswith('.tmp') or file.endswith('.jpeg') or file.endswith('.png') or file.endswith('.h5') or file.endswith('.exe')):
								if "swatplus" in file:
										continue
								shutil.copy2(os.path.join(self.scenario_path, file), cc_model_path_new)

				self.copy_original_print_print_cc(cc_name)
				self.adjust_time_sim(cc_name)
				self.adjust_print_cc(cc_name)

				with open(self.general_log_path, 'a') as f:
						f.write(f"Running {cc_name} for {cc_model_path_new}\n")
				## supress output
				start_time = time.time()
				logging.info(f"Running: {cc_name}, {self.name}")
				subprocess.run(["/data/MyDataBase/bin/swatplus"], cwd=cc_model_path_new, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
				end_time = time.time()
				duration = (end_time - start_time)/60
				logging.info(f"Elapsed time for {cc_name}:  {duration:.2f} minutes")

				streamflow_data_path = os.path.join(self.BASE_PATH, self.vpuid, self.level, self.name, 'streamflow_data/')
				fig_files_paths = os.path.join(self.BASE_PATH, self.vpuid, self.level, self.name)
				if self.config.get("stage", "historical") == "historical":
					evaluator = SwatModelEvaluator("/data/MyDataBase", self.vpuid, self.level, self.name, self.model_name, self.start_year, self.end_year, self.nyskip, self.no_value, 'historical')
					objective_function_values = evaluator.model_evaluation(cc_model_path_new, streamflow_data_path, fig_files_paths, cc_name)
		def clean_up_cc_model(self, cc_name):
				cc_model_path_new = os.path.join(self.cc_model_path, cc_name)
				files = os.listdir(cc_model_path_new)
				for file in files:
						if not (file.endswith('.pcp') or file.endswith('.tmp')):
								if file != "SWAT_OUTPUT.h5":
									os.remove(os.path.join(cc_model_path_new, file))
		
		def run_all_cc_models(self, cc_name):
				self.clean_up(cc_name)
				self.run_swatplus_cc_model(cc_name)
				if self.config.get("convert2h5", False):
						write_SWAT_OUTPUT_h5(cc_name, self.name, number_of_processes=1)
				if self.config.get("clean_up_cc_model", False):
						self.clean_up_cc_model(cc_name)

class SWAT_cc_controller:
		def __init__(self, config=None):
				self.BASE_PATH = config.get("BASE_PATH", "/data/MyDataBase/SWATplus_by_VPUID")
				self.NAME = config.get("NAME", None)
				self.VPUID = config.get("VPUID", None)
				self.CC_SCENARIO = config.get("CC_SCENARIO", "historical")
				self.cc_name = config.get("cc_name", None)
				self.num_processes = config.get("num_processes", 1)
				self.config = config
				self.filter_vpuid = config.get("filter_vpuid", None)
				self.rewrite = config.get("rewrite", False)
				self.stage = config.get("stage", "historical")

		def process(self):
					self.task_manager()

		def worker(self, task_queue):
				while not task_queue.empty():
						task = task_queue.get()
						if task is None:
								break
						cc_name, scenario_path, cc_model_path, log_path, name, vpuid = task
						logging.info(f"############ Running {cc_name} for {name} #############")
						cc_model = cc_name.split('_')[0]
						scenario = cc_name.split('_')[1]
						ensemble = cc_name.split('_')[2]
						cc_name = f"{cc_model}_{scenario}_{ensemble}"

						runner = SWATModelRunner(scenario_path, cc_model_path, log_path, name, vpuid, self.config)
						runner.run_all_cc_models(cc_name)

		def task_manager(self):
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
				# sourcery skip: swap-if-else-branches
				scores_path = f"{self.BASE_PATH}/{vpuid}/huc12/{name}/{self.config['stage']}_performance_scores.txt"
				if not os.path.exists(scores_path):
						logging.info(f"VPUID: {vpuid}, NAME: {name} does not have a performance_scores.txt file.")
						return False
				else:
						return self.if_scenario_exist(scores_path, cc_name)
		
		def if_scenario_exist(self, path, cc_name):
				logging.info(f"Checking if {cc_name} exists in {path}")
				df = pd.read_csv(path, sep="\t", skiprows=1)
				self.write_scores_table(df, path)
				return cc_name in df['SCENARIO'].values
		
		def get_best_ver_nse(self, path):
				df = pd.read_csv(path, sep="\t", skiprows=1)
				return  df["NSE"].max()

		def write_scores_table(self, df, path):
				### rewrite the file
				first_line = f"modified on {datetime.datetime.now()} by Vahid Rafiei\n"
				### sort by date and remove older duplicates
				df['Time_of_writing'] = pd.to_datetime(df['Time_of_writing'])
				df = df.sort_values(by='Time_of_writing')
				### remove duplicates
				df = df.drop_duplicates(subset=["station", "time_step", "MODEL_NAME", "SCENARIO"], keep='last')
				with open(path, 'w') as f:
						f.write(first_line)
						# Specify line_terminator to avoid extra newlines
						df.to_csv(f, sep='\t', index=False, lineterminator='\n')
		
		def best_ver_num(self, vpuid, name):
				df = pd.read_csv(f"{self.BASE_PATH}/{vpuid}/huc12/{name}/verification_performance_scores.txt", skiprows=1, sep="\t")
				df["NSE"] = df["NSE"].astype(float)
				max_nse_scenario = df.sort_values(by="NSE", ascending=False)["SCENARIO"].values[0]
				return max_nse_scenario.split('_')[-1]
		def prepare_data_for_workers(self):  # sourcery skip: low-code-quality
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
					if self.filter_vpuid and vpuid not in self.filter_vpuid:
						continue
					logging.info(f"################### Searching within {vpuid} ###################")
					names = os.listdir(f"{self.BASE_PATH}/{vpuid}/huc12")
					names.remove("log.txt")
					for name in names:
							if len(name)<10:
								logging.info(f"VPUID: {vpuid}, NAME: {name} is the new versions of SWAT+ and will be skipped until the next update.")
								continue
							logging.info(f"################### Searching within {name} ###################")
							
							num = self.best_ver_num(vpuid, name)
							best_nse = self.get_best_ver_nse(f"{self.BASE_PATH}/{vpuid}/huc12/{name}/verification_performance_scores.txt")
							if best_nse < -1:
								import time
								logging.info(f"VPUID: {vpuid}, NAME: {name} has a bad NSE score.")
								time.sleep(10)
								continue

							scenario_path = f"{self.BASE_PATH}/{vpuid}/huc12/{name}/SWAT_gwflow_MODEL/Scenarios/Scenario_verification_stage_{num}"
							if not os.path.exists(scenario_path):
									logging.info(f"VPUID: {vpuid}, NAME: {name} does not have a verification stage.")
									continue

							cc_model_path = f"{self.BASE_PATH}/{vpuid}/huc12/{name}/climate_change_models"
							log_path = f"{self.BASE_PATH}/{vpuid}/huc12/{name}/log.txt"


							cc_names = os.listdir(cc_model_path)
							cc_names = [cc_name for cc_name in cc_names if not cc_name.endswith('.jpeg') and self.CC_SCENARIO in cc_name]
							for cc_name in cc_names:
									_temp_path = os.path.join(cc_model_path, cc_name,"SWAT_OUTPUT.h5")
									if self.stage == "historical" and self.check_historical(vpuid, name, cc_name):
										if self.rewrite:
											logging.info(f"VPUID: {vpuid}, NAME: {name} will be rewritten...")
										else:
											logging.info(f"VPUID: {vpuid}, NAME: {name} has a historical simulation.")
											continue

									all_cc_names.append(cc_name)
									all_scenario_paths.append(scenario_path)
									all_cc_model_paths.append(cc_model_path)
									all_log_paths.append(log_path)
									all_names.append(name)
									all_vpuids.append(vpuid)

				return list(zip(all_cc_names, all_scenario_paths, all_cc_model_paths, all_log_paths, all_names, all_vpuids))
		