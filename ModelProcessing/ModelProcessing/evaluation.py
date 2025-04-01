import os
import shutil
import subprocess
import time
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.dates import DateFormatter, MonthLocator, YearLocator
from uuid import uuid4 as random_scenario_name_generator
import sys
from multiprocessing import Pool
from ModelProcessing.logging_utils import get_logger, log_to_file

try:
    from ModelProcessing.utils import log_errors, write_model_parameters
    from ModelProcessing.utils import filling_observations
except Exception:
    from utils import log_errors, filling_observations, write_model_parameters

# Create a module-level logger
logger = get_logger('ModelProcessing.evaluation')

def simulate_and_evaluate_swat_model_wrapper(params,username, BASE_PATH, VPUID, LEVEL, NAME, MODEL_NAME, START_YEAR, END_YEAR, nyskip, no_value, stage, problem, param_files, operation_types, TxtInOut, SCENARIO):
    evaluator = SwatModelEvaluator(username, BASE_PATH, VPUID, LEVEL, NAME, MODEL_NAME, START_YEAR, END_YEAR, nyskip, no_value, stage, TxtInOut=TxtInOut, SCENARIO=SCENARIO)
    return evaluator.simulate_and_evaluate_swat_model(params, problem, param_files, operation_types)

class SwatModelEvaluator:
    """This class is used to evaluate the SWAT model
    
    Processes: 
    1. Streamflow evaluation in daily and monthly time steps
    2. Groundwater head evaluation in average annual 
    3. ET evaluation in monthly
    4. Overall evaluation of streamflow, groundwater head, and ET
    5. Writing performance scores 
    """
    
    def __init__(self, username, BASE_PATH, VPUID, LEVEL, NAME, MODEL_NAME, START_YEAR, END_YEAR, nyskip, no_value, stage, TxtInOut=None, SCENARIO=None):
        self.execution_file = '/data/SWATGenXApp/codes/bin/swatplus'
        self.key = str(random_scenario_name_generator())
        self.SCENARIO = SCENARIO if SCENARIO is not None else "Scenario_" + self.key
        self.BASE_PATH = BASE_PATH
        self.VPUID = VPUID
        self.LEVEL = LEVEL
        self.NAME = NAME
        self.MODEL_NAME = MODEL_NAME
        self.START_YEAR = START_YEAR
        self.END_YEAR = END_YEAR
        self.scenario_TxtInOut = f'{self.BASE_PATH}/SWATplus_by_VPUID/{self.VPUID}/{self.LEVEL}/{self.NAME}/{self.MODEL_NAME}/Scenarios/{self.SCENARIO}/'
        self.original_TxtInOut = TxtInOut if TxtInOut is not None else f'{self.BASE_PATH}/SWATplus_by_VPUID/{self.VPUID}/{self.LEVEL}/{self.NAME}/{self.MODEL_NAME}/Scenarios/Default/TxtInOut/'
        self.hru_new_target = f'{self.BASE_PATH}/SWATplus_by_VPUID/{self.VPUID}/{self.LEVEL}/{self.NAME}/{self.MODEL_NAME}/Scenarios/Default/TxtInOut/hru.con'
        self.model_log_path = os.path.join(self.BASE_PATH, f"SWATplus_by_VPUID/{self.VPUID}/{self.LEVEL}/{self.NAME}/log.txt")
        self.general_log_path = os.path.join(self.BASE_PATH, f'SWATplus_by_VPUID/{self.VPUID}/{self.LEVEL}/log.txt')
        self.fig_files_paths = os.path.join(self.BASE_PATH, f'SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/figures_{MODEL_NAME}')
        self.streamflow_data_path = os.path.join(self.BASE_PATH, f"SWATplus_by_VPUID/{VPUID}/{LEVEL}/{self.NAME}/streamflow_data/")
        self.model_log_path = os.path.join(BASE_PATH, f"SWATplus_by_VPUID/{VPUID}/{LEVEL}/{str(NAME)}/log.txt")
        self.basin_yield_path = f"/data/SWATGenXApp/Users/{username}/SWATplus_by_VPUID/{self.VPUID}/huc12/{self.NAME}/{self.MODEL_NAME}/Scenarios/{self.SCENARIO}/basin_crop_yld_yr.txt"
        self.nyskip = nyskip
        self.no_value = no_value
        self.stage = "random" if stage is None else stage
        self.cms_to_cfs = 35.3147
        
        # Create logger specific to this evaluation instance
        self.logger = get_logger(f'ModelProcessing.evaluation.{self.MODEL_NAME}.{self.NAME}')
        self.logger.info(f"Initialized SWAT model evaluator for {self.MODEL_NAME}:{self.NAME}:{self.VPUID}")

    @staticmethod
    def nse(observed, simulated):
        """Nash-Sutcliffe Efficiency"""
        observed_mean = observed.mean()
        nse = 1 - sum((observed - simulated) ** 2) / sum((observed - observed_mean) ** 2)
        return nse

    @staticmethod
    def mape(observed, simulated):
        """Mean Absolute Percentage Error"""
        mape = sum(abs((observed - simulated) / observed)) / len(observed)
        return mape

    @staticmethod
    def pbias(observed, simulated):
        """Percent Bias"""
        pbias = sum((observed - simulated) / observed) / len(observed) * 100
        return pbias

    @staticmethod
    def rmse(observed, simulated):
        """Root Mean Square Error"""
        rmse = (sum((observed - simulated) ** 2) / len(observed)) ** 0.5
        return rmse

    @staticmethod
    def kge(observed, simulated):
        """Kling-Gupta Efficiency"""
        observed_mean = observed.mean()
        simulated_mean = simulated.mean()
        kge = 1 - ((np.corrcoef(observed, simulated)[0, 1] - 1) ** 2 + (observed.std() / simulated.std() - 1) ** 2 + (observed_mean / simulated_mean - 1) ** 2) ** 0.5
        return kge

    @staticmethod
    def calculate_metrics(observed, simulated):
        """Calculate NSE, MAPE, PBIAS, RMSE"""
        nse = SwatModelEvaluator.nse(observed, simulated)
        mape = SwatModelEvaluator.mape(observed, simulated)
        pbias = SwatModelEvaluator.pbias(observed, simulated)
        rmse = SwatModelEvaluator.rmse(observed, simulated)
        kge = SwatModelEvaluator.kge(observed, simulated)

        return nse, mape, pbias, rmse, kge

    def prepare_scenario_files(self):
        self.logger.debug(f"Preparing scenario files in {self.scenario_TxtInOut}")
        shutil.copytree(self.original_TxtInOut, self.scenario_TxtInOut, dirs_exist_ok=True)
        shutil.copy(self.execution_file, self.scenario_TxtInOut)

    def define_timeout(self):
        # Read the list of HRUs
        list_of_hrus = pd.read_csv(self.hru_new_target, skiprows=1)
        number_of_hrus = len(list_of_hrus)
        # Define the regression parameters
        slope = 1.56e-3
        intercept = 2.49
        # Calculate the execution time in minutes using the regression equation
        execution_time_minutes = slope * number_of_hrus + intercept
        # Convert execution time to seconds
        execution_time_seconds = execution_time_minutes * 60
        if self.stage == 'calibration':
            timeout = max(execution_time_seconds, 3 * 3600)
        else:
            timeout = max(execution_time_seconds, 8 * 3600)

        self.logger.debug(f"Timeout set to {timeout/3600:.2f} hours for {number_of_hrus} HRUs")
        return timeout

    def save_model_parameters(self, params, problem):
        params_dict = dict(zip(problem['names'], params))
        parameter_path = f"{self.BASE_PATH}/SWATplus_by_VPUID/{self.VPUID}/{self.LEVEL}/{self.NAME}/CentralParameters.txt"
        os.makedirs(os.path.dirname(parameter_path), exist_ok=True)

        if not os.path.exists(parameter_path):
            with open(parameter_path, 'w') as file:
                # First line is the header
                file.write(f'key {" ".join(problem["names"])}\n')
                file.write(f'{self.key} {" ".join([str(params_dict[name]) for name in problem["names"]])}\n')
            self.logger.debug(f"Created new parameters file: {parameter_path}")
        else:
            # If exists, append the new parameters
            with open(parameter_path, 'a') as file:
                # Key and parameters
                file.write(f'{self.key} {" ".join([str(params_dict[name]) for name in problem["names"]])}\n')
            self.logger.debug(f"Appended parameters to existing file: {parameter_path}")

    def simulate_and_evaluate_swat_model(self, params, problem, param_files, operation_types):
        self.logger.info(f"Starting model simulation for {self.MODEL_NAME}:{self.NAME}:{self.VPUID}")

        self.prepare_scenario_files()
        self.logger.info(f'Writing model parameters for {self.MODEL_NAME}:{self.NAME}:{self.VPUID}')
        self.save_model_parameters(params, problem)
        write_model_parameters(
            param_files, params, problem, operation_types,
            self.scenario_TxtInOut, self.original_TxtInOut
        )

        start_model_time = time.time()
        timeout_threshold = self.define_timeout()

        try:
            # Run the model in silent mode
            self.logger.debug(f"Executing SWAT model in {self.scenario_TxtInOut}")
            subprocess.run(
                [self.execution_file],
                cwd=self.scenario_TxtInOut, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL,
                timeout=timeout_threshold
            )

            end_model_time = time.time()
            duration = round((end_model_time - start_model_time) / (60 * 60), 2)
            message = f'{self.MODEL_NAME}:{self.NAME}:{self.VPUID} execution terminated within {duration} hours'
            self.logger.info(message)

        except subprocess.TimeoutExpired:
            message = f'{self.MODEL_NAME}:{self.NAME}:{self.VPUID} wall-time met: {round(timeout_threshold / (60 * 60), 2)} hours'
            self.logger.warning(message)

        # Log to the file system
        log_to_file(message, self.model_log_path)
        log_to_file(message, self.general_log_path)

        return self.model_evaluation()

    def trash_empty_files(self, file):
        if not bool(os.path.exists(file) and os.path.getsize(file) != 0):
            message = f'{self.MODEL_NAME}:{self.NAME}:{self.VPUID} {file} is empty, removing the file'
            log_to_file(message, self.model_log_path)
            self.logger.warning(message)

            try:
                os.remove(file)
                return True
            except Exception as e:
                self.logger.error(f"Error removing file {file}: {e}")
                return True
        return False

    def model_evaluation(self):
        self.logger.info(f"Starting model evaluation for {self.MODEL_NAME}:{self.NAME}:{self.VPUID}")
        os.chdir(self.scenario_TxtInOut)
        file_path = "channel_sd_day.txt"

        if not os.path.exists(file_path):
            message = f'{self.MODEL_NAME}:{self.NAME}:{self.VPUID} No streamflow data exists, returning {self.no_value}'
            self.logger.error(message)
            log_to_file(message, self.model_log_path)
            return self.no_value

        if os.path.getsize(file_path) == 0:
            message = f'Empty SWAT simulation file {file_path} exists, returning {self.no_value}'
            self.logger.error(message)
            log_to_file(message, self.model_log_path)
            return self.no_value

        try:
            usecols = ['day', 'mon', 'yr', 'gis_id', 'flo_out']
            chunks = list(
                pd.read_csv(
                    file_path,
                    skiprows=[0, 2],
                    usecols=usecols,
                    chunksize=10 ** 6,
                    sep=' ',
                    skipinitialspace=True,
                )
            )
            channel_sd_day_data = pd.concat(chunks, axis=0)

            if len(channel_sd_day_data) == 0:
                message = f'{self.MODEL_NAME}:{self.NAME}:{self.VPUID} channel_sd_day_data is empty, returning {self.no_value}'
                self.logger.error(message)
                log_to_file(message, self.model_log_path)
                return self.no_value

            files = glob.glob(f'{self.streamflow_data_path}*.csv')
            if len(files) == 0:
                message = f'{self.MODEL_NAME}:{self.NAME}:{self.VPUID} No streamflow data files exist, returning {self.no_value}'
                self.logger.error(message)
                log_to_file(message, self.model_log_path)
                return self.no_value

            files = [file for file in files if not self.trash_empty_files(file)]

            if channel_sd_day_data.empty:
                message = f'{self.MODEL_NAME}:{self.NAME}:{self.VPUID} SWAT simulation output is empty, returning {self.no_value}'
                self.logger.error(message)
                log_to_file(message, self.model_log_path)
                return self.no_value
            else:
                overall_streamflow_performance = 0
                for file in files:
                    if os.path.exists(file):
                        self.logger.debug(f"Processing streamflow file: {os.path.basename(file)}")
                        objective_ = self.cal_streamflow_obj_val(file, channel_sd_day_data)
                        if objective_ == self.no_value or np.isnan(objective_):
                            self.logger.warning(f"Invalid objective value for {os.path.basename(file)}, skipping")
                            continue    
                        overall_streamflow_performance += objective_
                        self.logger.debug(f"Objective value for {os.path.basename(file)}: {objective_}")

                ##############################################################
                # Overall Objective function
                ##############################################################

                objective_function_values = -1 * (overall_streamflow_performance)
                if np.isnan(objective_function_values):
                    objective_function_values = self.no_value    

                self.logger.info(f'{self.MODEL_NAME}:{self.NAME}:{self.VPUID} objective_function_values: {objective_function_values}')
                return objective_function_values

        except Exception as e:
            message = f'{self.MODEL_NAME}:{self.NAME}:{self.VPUID} Error in model_evaluation: {str(e)}'
            self.logger.error(message, exc_info=True)
            log_to_file(message, self.model_log_path)
            return self.no_value

    def cal_streamflow_obj_val(self, file, channel_sd_day_data):
        """Calculate streamflow objective value for a station by comparing observed and simulated data
        
        Args:
            file: Path to observed streamflow data CSV file
            channel_sd_day_data: DataFrame containing simulated streamflow data
            
        Returns:
            Objective function value based on NSE scores or self.no_value if calculation fails
        """
        self.logger.debug(f"Calculating streamflow objective value for {os.path.basename(file)}")
        
        try:
            # 1. Read and preprocess observed data
            observed, gap_length, gap_percentage, observed_monthly = self.read_observed_data(file)
            self.logger.debug(f"Observed data length: {len(observed)}, gap length: {gap_length}, gap percentage: {gap_percentage:.2f}")
            
            # Basic validation checks
            if observed.empty or "date" not in observed.columns:
                message = f'Empty obs data for station {os.path.basename(file)} returning "0" for station'
                self.logger.warning(message)
                log_to_file(message, self.model_log_path)
                return 0
            
            elif gap_percentage > 0.1:
                message = f'{os.path.basename(file)} has {gap_percentage:.2f}% gaps, over 10% gap, returning "0"'
                self.logger.warning(message)
                log_to_file(message, self.model_log_path)
                return 0
            
            elif gap_length != 0 and gap_percentage < 0.1:
                message = f'{file} has {gap_percentage:.2f}% gaps, time series imputation will fill the gaps'
                self.logger.info(message)
                log_to_file(message, self.model_log_path)
                observed['streamflow'] = np.where(observed.streamflow == -1, np.nanmean(observed.streamflow), observed.streamflow)
                observed_monthly['streamflow'] = np.where(observed_monthly.streamflow == -1, np.nanmean(observed_monthly.streamflow), observed_monthly.streamflow)
            
            # 2. Extract station ID from filename
            try:
                station_id = int(os.path.basename(file).split('.')[0].split('_')[0])
                self.logger.debug(f"Processing station ID: {station_id}")
            except (ValueError, IndexError) as e:
                message = f'Error parsing station ID from filename {file}: {str(e)}'
                self.logger.error(message)
                log_to_file(message, self.model_log_path)
                return self.no_value
                
            # 3. Filter simulated data for the relevant station
            simulated_raw = channel_sd_day_data[channel_sd_day_data.gis_id == station_id].copy()
            
            if simulated_raw.empty:
                message = f'No simulated data found for station {station_id} in {self.NAME}'
                self.logger.error(message)
                log_to_file(message, self.model_log_path)
                return self.no_value
            
            # Check if all simulated flow values are zero
            if np.all(simulated_raw['flo_out'] == 0):
                message = f'ERROR: All simulated flow values are ZERO for station {station_id}. This indicates a serious model error.'
                self.logger.error(message)
                log_to_file(message, self.model_log_path)
                # This is a critical error that should not be handled silently
                raise RuntimeError(message)
                
            # 4. Convert simulated data date columns to datetime
            self.logger.debug("Converting simulated data dates to datetime format")
            # Create proper datetime column from yr, mon, day
            simulated_raw['date'] = pd.to_datetime(
                simulated_raw.yr.astype(str) + '-' + 
                simulated_raw.mon.astype(str).str.zfill(2) + '-' + 
                simulated_raw.day.astype(str).str.zfill(2)
            )
            
            # Convert streamflow from cms to cfs
            simulated_raw['flo_out'] = simulated_raw['flo_out'] * self.cms_to_cfs
            
            # 5. Determine date range overlap between observed and simulated
            obs_date_range = (observed['date'].min(), observed['date'].max())
            sim_date_range = (simulated_raw['date'].min(), simulated_raw['date'].max())
            
            self.logger.debug(f"Observed data range: {obs_date_range[0]} to {obs_date_range[1]}")
            self.logger.debug(f"Simulated data range: {sim_date_range[0]} to {sim_date_range[1]}")
            
            # Check for overlap
            if obs_date_range[1] < sim_date_range[0] or sim_date_range[1] < obs_date_range[0]:
                message = f'No overlap between observed and simulated date ranges for station {station_id}'
                self.logger.error(message)
                log_to_file(message, self.model_log_path)
                return self.no_value
                
            # 6. Create a continuous date range for the full period
            start_date = max(obs_date_range[0], sim_date_range[0])
            end_date = min(obs_date_range[1], sim_date_range[1])
            full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            self.logger.debug(f"Common date range: {start_date} to {end_date} ({len(full_date_range)} days)")
            
            # 7. Create consistent dataframes with the full date range
            # For daily data
            date_df = pd.DataFrame({'date': full_date_range})
            
            # Merge observed data with full date range
            observed_aligned = pd.merge(
                date_df,
                observed[['date', 'streamflow']],
                on='date',
                how='left'
            )
            
            # Merge simulated data with full date range
            simulated_aligned = pd.merge(
                date_df,
                simulated_raw[['date', 'flo_out']],
                on='date',
                how='left'
            )
            
            # 8. Verify data alignment and check for missing values
            missing_obs = observed_aligned['streamflow'].isna().sum()
            missing_sim = simulated_aligned['flo_out'].isna().sum()
            
            self.logger.debug(f"Missing values - Observed: {missing_obs}/{len(full_date_range)} ({missing_obs/len(full_date_range):.1%}), "
                              f"Simulated: {missing_sim}/{len(full_date_range)} ({missing_sim/len(full_date_range):.1%})")
            
            # Check if all simulated values in the aligned data are zero
            if np.all(simulated_aligned['flo_out'].dropna() == 0):
                message = f'ERROR: All aligned simulated flow values are ZERO for station {station_id}. This indicates a serious model error.'
                self.logger.error(message)
                log_to_file(message, self.model_log_path)
                # This is a critical error that should not be handled silently
                raise RuntimeError(message)
            
            # Filter to only include days where both observed and simulated data exist
            valid_data = pd.merge(
                observed_aligned[['date', 'streamflow']].dropna(),
                simulated_aligned[['date', 'flo_out']].dropna(),
                on='date',
                how='inner'
            )
            
            if len(valid_data) < 30:  # Require at least 30 days of overlapping data
                message = f'Insufficient overlapping data for station {station_id}: {len(valid_data)} days'
                self.logger.error(message)
                log_to_file(message, self.model_log_path)
                return self.no_value
                
            self.logger.info(f"Found {len(valid_data)} days of valid overlapping data for evaluation")
            
            # 9. Set date as index for simulated data for monthly resampling
            simulated_aligned_indexed = simulated_aligned.set_index('date')
            
            # 10. Generate monthly data by resampling
            simulated_monthly = simulated_aligned_indexed.resample('M').sum()
            simulated_monthly['yr'] = simulated_monthly.index.year
            simulated_monthly['mon'] = simulated_monthly.index.month
            
            # 11. Calculate scores with aligned data
            self.logger.debug(f"Calculating daily streamflow scores for {os.path.basename(file)}")
            daily_streamflow_nse_score = self.daily_streamflow_scores(
                valid_data[['date', 'streamflow']].reset_index(drop=True), 
                valid_data[['date', 'flo_out']].set_index('date'),
                os.path.basename(file).split('.')[0]
            )
            
            self.logger.info(f"Daily streamflow NSE score: {daily_streamflow_nse_score}")
            
            # If daily score calculation failed, exit
            if daily_streamflow_nse_score == self.no_value:
                return self.no_value
                
            # 12. Filter observed monthly data to match the simulated period
            observed_monthly['date'] = pd.to_datetime(
                observed_monthly.yr.astype(str) + '-' + 
                observed_monthly.mon.astype(str).str.zfill(2) + '-01'
            )
            
            observed_monthly_filtered = observed_monthly[
                (observed_monthly.date >= start_date) & 
                (observed_monthly.date <= end_date)
            ].reset_index(drop=True)
            
            # 13. Calculate monthly scores with aligned data
            self.logger.debug(f"Calculating monthly streamflow scores for {os.path.basename(file)}")
            monthly_streamflow_nse_score = self.monthly_streamflow_scores(
                observed_monthly_filtered,
                simulated_monthly.reset_index(),
                os.path.basename(file).split('.')[0]
            )
            
            overall_streamflow_nse_score = daily_streamflow_nse_score + monthly_streamflow_nse_score
            self.logger.info(f"Daily NSE: {daily_streamflow_nse_score:.3f}, Monthly NSE: {monthly_streamflow_nse_score:.3f}, Total: {overall_streamflow_nse_score:.3f}")
            
            return overall_streamflow_nse_score
        
        except RuntimeError as e:
            # Let RuntimeError propagate up to stop the program
            raise
        except Exception as e:
            message = f'Error in processing data for {self.NAME}: {str(e)}'
            self.logger.error(message, exc_info=True)
            log_to_file(message, self.model_log_path)
            return self.no_value

    def daily_streamflow_scores(self, observed, simulated, title):
        """Calculate daily streamflow scores"""
        self.logger.debug(f"Calculating daily streamflow scores for {title}")
        
        try:
            # Create a common dataframe with aligned dates
            common_dates = pd.merge(
                observed[['date', 'streamflow']],
                simulated.reset_index()[['date', 'flo_out']],
                on='date',
                how='inner'
            )
            
            # Ensure we have matching data
            if len(common_dates) == 0:
                self.logger.warning(f"No matching dates between observed and simulated data for {title}")
                return self.no_value
            
            self.logger.debug(f"Found {len(common_dates)} matching dates for daily evaluation")
            
            # Calculate metrics on the aligned data
            daily_nse_value, daily_mape_value, pbias_value, rmse_value, kge_value = self.calculate_metrics(
                common_dates.streamflow.values, 
                common_dates.flo_out.values
            )
            
            self.write_performance_scores(title, "Daily", daily_nse_value, daily_mape_value, pbias_value, rmse_value, kge_value)
            self.plot_streamflow(common_dates.streamflow.values, common_dates.flo_out.values, title, "daily", daily_nse_value)
            
            self.logger.info(f"{self.NAME} {title} Daily Streamflow: NSE: {daily_nse_value:.2f}, MAPE: {daily_mape_value:.2f}, PBIAS: {pbias_value:.2f}, RMSE: {rmse_value:.2f}, KGE: {kge_value:.2f}")
            
            return daily_nse_value
        except Exception as e:
            error_msg = f"Error calculating daily streamflow scores for {title}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            log_to_file(error_msg, self.model_log_path)
            return self.no_value

    def monthly_streamflow_scores(self, observed_monthly, simulated_monthly, title):
        """Calculate monthly streamflow scores with a more robust approach"""
        self.logger.debug(f"Calculating monthly streamflow scores for {title}")
        
        try:
            # Create a common dataframe with aligned months and years
            common_data = pd.merge(
                observed_monthly[['yr', 'mon', 'streamflow']],
                simulated_monthly.reset_index()[['yr', 'mon', 'flo_out']],
                on=['yr', 'mon'],
                how='inner'
            )
            
            # Ensure we have matching data
            if len(common_data) == 0:
                self.logger.warning(f"No matching months between observed and simulated data for {title}")
                return self.no_value
            
            self.logger.debug(f"Found {len(common_data)} matching months for monthly evaluation")
            
            # Calculate metrics on the aligned data
            monthly_nse_value, monthly_mape_value, pbias_value, rmse_value, kge_value = self.calculate_metrics(
                common_data.streamflow.values, 
                common_data.flo_out.values
            )
            
            # Write performance scores and plot streamflow
            self.write_performance_scores(title, "Monthly", monthly_nse_value, monthly_mape_value, pbias_value, rmse_value, kge_value)
            self.plot_streamflow_monthly(common_data.streamflow.values, common_data.flo_out.values, title, "monthly", monthly_nse_value)
            
            self.logger.info(f"{self.NAME} {title} Monthly Streamflow: NSE: {monthly_nse_value:.2f}, MAPE: {monthly_mape_value:.2f}, PBIAS: {pbias_value:.2f}, RMSE: {rmse_value:.2f}, KGE: {kge_value:.2f}")
            
            return monthly_nse_value
        except Exception as e:
            error_msg = f"Error calculating monthly streamflow scores for {title}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            log_to_file(error_msg, self.model_log_path)
            return self.no_value

    def write_performance_scores(self, title, time_step, NSE, MPE, PBIAS, RMSE, KGE):
        """Write performance scores to the central performance log file"""
        self.logger.debug(f"Writing performance scores for {title} ({time_step})")
        
        try:

            # Format current time
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Define the performance scores path
            performance_scores_path = os.path.join(
                self.BASE_PATH, 
                f'SWATplus_by_VPUID/{self.VPUID}/{self.LEVEL}/{self.NAME}/CentralPerformance.txt'
            )
            
            # Format the performance line
            perf_line = f'{self.key}\t{current_time}\t{title}\t{time_step}\t{self.stage}\t{self.MODEL_NAME}\t{NSE:.3f}\t{MPE:.3f}\t{PBIAS:.3f}\t{RMSE:.3f}\t{KGE:.3f}\n'
            
            # Write to file - create if it doesn't exist
            if not os.path.exists(performance_scores_path):
                self.logger.debug(f"Creating new performance log file: {performance_scores_path}")
                with open(performance_scores_path, 'w') as file:
                    file.write(f'key\tTime_of_writing\tstation\ttime_step\tstage\tMODEL_NAME\tNSE\tMPE\tPBIAS\tRMSE\tKGE\n')
                    file.write(perf_line)
            else:
                # Append to existing file
                self.logger.debug(f"Appending to existing performance log file: {performance_scores_path}")
                with open(performance_scores_path, 'a') as file:
                    file.write(perf_line)
                    
            self.logger.info(f"Performance scores for {title} ({time_step}): NSE={NSE:.3f}, PBIAS={PBIAS:.3f}, KGE={KGE:.3f}")
            
        except Exception as e:
            error_msg = f"Error writing performance scores for {title}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            log_to_file(error_msg, self.model_log_path)

    def read_observed_data(self, file):
        """Read observed streamflow data from CSV file"""
        self.logger.debug(f"Reading observed data from {os.path.basename(file)}")
        
        try:
            # Read daily data
            obs = pd.read_csv(file, parse_dates=['date'])[["date", "streamflow"]]
            obs['date'] = pd.to_datetime(obs['date'])
            
            if "date" not in obs.columns:
                self.logger.error(f"date column missing in observed data file {os.path.basename(file)}")
                return pd.DataFrame(), 0, 0, pd.DataFrame()
                
            # Replace missing values (-1) with NaN
            obs['streamflow'] = np.where(obs['streamflow'] == -1, np.nan, obs['streamflow'])
            
            # Create monthly data
            observed_monthly = pd.read_csv(file)
            observed_monthly['date'] = pd.to_datetime(observed_monthly['date'])
            
            # Resample to monthly
            observed_monthly = observed_monthly.set_index('date')
            observed_monthly = observed_monthly.resample('M').sum()
            observed_monthly['yr'] = observed_monthly.index.year
            observed_monthly['mon'] = observed_monthly.index.month
            observed_monthly = observed_monthly.reset_index(drop=True)
            observed_monthly = observed_monthly[['yr', 'mon', 'streamflow']]
            
            # Replace missing values in monthly data
            observed_monthly['streamflow'] = np.where(observed_monthly['streamflow'] == -1, np.nan, observed_monthly['streamflow'])
            
            # Filter to evaluation period
            observed = obs[(obs.date >= f'{self.START_YEAR + self.nyskip}-01-01') & 
                           (obs.date <= f'{self.END_YEAR}-12-31')].reset_index(drop=True)
            
            observed_monthly = observed_monthly[(observed_monthly.yr >= (self.START_YEAR + self.nyskip)) & 
                                               (observed_monthly.yr <= self.END_YEAR)].reset_index(drop=True)
            
            self.logger.debug(f"Observed data range: {observed.date.min()} to {observed.date.max()}")
            
            # Calculate gap statistics
            missing_dates = observed[observed.streamflow.isna() == True]
            total_length = len(observed)
            gap_length = len(missing_dates)
            gap_percentage = gap_length / total_length if total_length > 0 else 0
            
            self.logger.debug(f"Observed data length: {total_length}, gaps: {gap_length} ({gap_percentage:.2%})")
            
            return observed, gap_length, gap_percentage, observed_monthly
            
        except Exception as e:
            error_msg = f"Error reading observed data from {os.path.basename(file)}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            log_to_file(error_msg, self.model_log_path)
            return pd.DataFrame(), 0, 0, pd.DataFrame()

    def plot_streamflow(self, observed, simulated, title, time_step, name_score):
        """Plot the observed and simulated streamflow
        
        Args:
            observed (np.array): Array of observed streamflow values
            simulated (np.array): Array of simulated streamflow values
            title (str): Title for the plot
            time_step (str): Time step for the plot (daily or monthly)
            name_score (float): NSE score to include in the plot
            
        Returns:
            None: Saves the plot to file
        """
        self.logger.debug(f"Plotting {time_step} streamflow for {title}")
        
        try:
            # Create the appropriate date range based on the time step
            data_range = pd.date_range(
                start=f'{self.START_YEAR + self.nyskip}-01-01', 
                end=f'{self.END_YEAR}-12-31', 
                freq='D'
            )
            
            # Handle length mismatch between date range and data
            if len(data_range) != len(observed):
                self.logger.warning(f"Date range length ({len(data_range)}) doesn't match data length ({len(observed)})")
                # Use the shorter length to avoid index errors
                min_length = min(len(data_range), len(observed), len(simulated))
                data_range = data_range[:min_length]
                observed = observed[:min_length]
                simulated = simulated[:min_length]
            
            nse_score, mape_score, pbias_score, rmse_score, kge_score = self.calculate_metrics(observed, simulated)
            
            # Plot the observed and simulated streamflow
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data_range, observed, label='Observed', color='blue', linewidth=1)
            ax.plot(data_range, simulated, label='Simulated', color='red', linewidth=1)
            ax.set_title(f'{title} {self.stage} Streamflow')
            ax.set_xlabel('Date')
            ax.set_ylabel('Streamflow (cfs)')
            
            # Set up the x-axis for displaying years as major ticks and months as minor ticks
            ax.xaxis.set_major_locator(YearLocator())
            ax.xaxis.set_major_formatter(DateFormatter('%Y'))
            ax.xaxis.set_minor_locator(MonthLocator())
            
            # Use grid only for the months (minor ticks)
            ax.grid(True, which='minor', linestyle='--', linewidth=0.5)
            ax.legend()
            
            # Add performance metrics to the plot
            plt.annotate(
                f'NSE: {nse_score:.2f}\nMAPE: {mape_score:.2f}\nPBIAS: {pbias_score:.2f}', 
                xy=(0.05, 0.85), 
                xycoords='axes fraction', 
                fontsize=12
            )
            
            # Ensure directory exists
            output_dir = f'{self.fig_files_paths}/SF/{self.stage}/{time_step}'
            os.makedirs(output_dir, exist_ok=True)
            
            # Save the plot
            filename = f'{output_dir}/{name_score:.2f}_{title}_{int(time.time())}.png'
            plt.savefig(filename, dpi=300)
            plt.close()
            
            self.logger.debug(f"Saved streamflow plot to {filename}")
            
        except Exception as e:
            error_msg = f"Error plotting streamflow for {title}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            log_to_file(error_msg, self.model_log_path)

    def plot_streamflow_monthly(self, observed_monthly, simulated_monthly, title, time_step, name_score):
        """Plot the observed and simulated monthly streamflow
        
        Args:
            observed_monthly (np.array): Array of observed monthly streamflow values
            simulated_monthly (np.array): Array of simulated monthly streamflow values
            title (str): Title for the plot
            time_step (str): Time step for the plot (should be 'monthly')
            name_score (float): NSE score to include in the plot
            
        Returns:
            None: Saves the plot to file
        """
        self.logger.debug(f"Plotting monthly streamflow for {title}")
        
        try:
            # Create a figure
            plt.figure(figsize=(12, 6))
            plt.grid(linestyle='--', linewidth=0.5)
            
            # Create a time range for x-axis (using month numbers)
            x_values = np.arange(len(observed_monthly))
            
            # Plot the observed and simulated data
            plt.plot(x_values, observed_monthly, label='Observed', color='blue', linewidth=1)
            plt.plot(x_values, simulated_monthly, label='Simulated', color='red', linewidth=1)
            
            # Set labels and title
            plt.title(f'{title} {self.stage} Monthly Streamflow')
            plt.xlabel('Month')
            plt.ylabel('Streamflow (cfs)')
            plt.legend()
            
            # Add performance metrics to the plot
            nse_score, mape_score, pbias_score, rmse_score, kge_score = self.calculate_metrics(
                observed_monthly, simulated_monthly
            )
            plt.annotate(
                f'NSE: {nse_score:.2f}\nMAPE: {mape_score:.2f}\nPBIAS: {pbias_score:.2f}', 
                xy=(0.05, 0.85), 
                xycoords='axes fraction', 
                fontsize=12
            )
            
            # Ensure directory exists
            output_dir = f'{self.fig_files_paths}/SF/{self.stage}/{time_step}'
            os.makedirs(output_dir, exist_ok=True)
            
            # Save the plot
            filename = f'{output_dir}/{name_score:.2f}_{title}_{int(time.time())}.png'
            plt.savefig(filename, dpi=300)
            plt.close()
            
            self.logger.debug(f"Saved monthly streamflow plot to {filename}")
            
        except Exception as e:
            error_msg = f"Error plotting monthly streamflow for {title}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            log_to_file(error_msg, self.model_log_path)

