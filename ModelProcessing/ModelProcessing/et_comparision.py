
import matplotlib.pyplot as plt
import pandas as pd 
import os 
import numpy as np
import uuid
import time





class Compare_MODIS_et_SWAT:
		def __init__(self, NAME, LEVEL, VPUID, TxtInOut_path, key=None, stage='test', MODEL_NAME='SWAT_gwflow_MODEL'):
				self.key = uuid.uuid4().hex if key is None else key
				self.rerun = False
				self.NAME = NAME
				self.VPUID = VPUID
				self.LEVEL = LEVEL
				self.MODEL_NAME = MODEL_NAME
				self.BASE_PATH = f"/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/{LEVEL}"
				self.MODID_ET_path = f"{self.BASE_PATH}/{NAME}/MODIS_ET/MODIS_ET.csv"
				self.fig_path = os.path.join(self.BASE_PATH, f'{NAME}/figures_{MODEL_NAME}')
				self.result_path = f"{self.BASE_PATH}/{NAME}/MODIS_ET/comparision_results"
				self.extracted_MODIS_ET_path = f"{self.result_path}/MODISvsSWAT_{self.key}.csv"
				self.TxtInOut_path = TxtInOut_path
				self.SWAT_LS_path = f"{self.TxtInOut_path}/lsunit_wb_mon.txt"
				self.print_prt_path = f"{self.TxtInOut_path}/print.prt"
				self.gwflow_input = f'{self.TxtInOut_path}/gwflow.input'
				self.time_sim = f'{self.TxtInOut_path}/time.sim'
				self.start_year = None
				self.end_year = None
				self.stage = stage	
				
				

		@staticmethod
		def nse(obs, sim):
				return 1 - (np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2))
		@staticmethod
		def rmse(obs, sim):
				return np.sqrt(np.mean((obs - sim) ** 2))
		@staticmethod
		def pbias(obs, sim):
				return 100 * np.sum(sim - obs) / np.sum(obs)
		@staticmethod
		def mape(obs, sim):
				#### replace 0 with 1
				obs = np.where(obs < 1, 1, obs)	
				return np.mean(np.abs((obs - sim) / obs)) * 100
		@staticmethod
		def kge(obs, sim):
				r = np.corrcoef(obs, sim)[0, 1]
				alpha = np.std(sim) / np.std(obs)
				beta = np.mean(sim) / np.mean(obs)
				return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
				
		def plot_overal_comparision(self, merged_df, nse_value, rmse_value):
				"""
				Plot SWAT ET vs MODIS ET along with performance metrics NSE and RMSE.
				"""

				# Create the scatter plot
				plt.figure(figsize=(10, 6))
				plt.scatter(merged_df['et'], merged_df['mean_ET'], alpha=0.7, edgecolors='k', label='Data Points')
				
				# Add grid and labels
				plt.grid(True)
				plt.xlabel("SWAT ET (mm/month)", fontsize=12)
				plt.ylabel("MODIS ET (mm/month)", fontsize=12)
				plt.title("Comparison of SWAT ET vs MODIS ET", fontsize=14)
				
				# Annotate performance metrics on the plot
				textstr = f'NSE: {nse_value:.2f}\nRMSE: {rmse_value:.2f}'
				plt.text(0.95, 0.05, textstr, transform=plt.gca().transAxes,
								fontsize=12, verticalalignment='bottom', horizontalalignment='right',
								bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

				# Add a 1:1 reference line
				min_et = min(merged_df['et'].min(), merged_df['mean_ET'].min())
				max_et = max(merged_df['et'].max(), merged_df['mean_ET'].max())
				plt.plot([min_et, max_et], [min_et, max_et], color='red', linestyle='--', label="1:1 Line")
				
				# Add legend
				plt.legend(loc="upper left", fontsize=10)

				# Save the plot as a high-resolution image
				plt.tight_layout()
				os.makedirs(f"{self.fig_path}/ET", exist_ok=True)
				plt.savefig(f"{self.fig_path}/ET/{nse_value:.2f}_{int(time.time())}.png", dpi=150)
				plt.close()



		def calculate_metrics(self, merged_df):
				"""
				Plot SWAT ET vs MODIS ET along with performance metrics NSE and RMSE.

				Parameters:
				merged_df (pd.DataFrame): DataFrame containing 'et' (SWAT ET) and 'mean_ET' (MODIS ET).
				"""
				# ensure the data types are float
				merged_df['et'] = merged_df['et'].astype(float)
				# Calculate performance metrics (NSE and RMSE)
				nse_value = self.nse(merged_df['et'], merged_df['mean_ET'])
				rmse_value = self.rmse(merged_df['et'], merged_df['mean_ET'])
				kge_value = self.kge(merged_df['et'], merged_df['mean_ET'])	
				pbias_value = self.pbias(merged_df['et'], merged_df['mean_ET'])	
				mape_value = self.mape(merged_df['et'], merged_df['mean_ET'])
				print(f"{self.NAME} ET Performance: NSE: {nse_value:.2f} RMSE: {rmse_value:.2f} KGE: {kge_value:.2f} PBIAS: {pbias_value:.2f} MAPE: {mape_value:.2f}")

				return nse_value, mape_value, pbias_value, rmse_value, kge_value
		
		def configure_setup(self):
				os.makedirs(self.result_path, exist_ok=True)
				if not os.path.exists(self.print_prt_path):
						print(f"print.prt does not exist in {self.TxtInOut_path}")
						return


		def process_ET(self):
				try:
						
					self.configure_setup()
					merged_df = self.process_swat_et_output()
					nse_score, mape_score, pbias_score, rmse_score, kge_score = self.calculate_metrics(merged_df)
					
					self.plot_overal_comparision(merged_df, nse_score, rmse_score)
					merged_df.to_csv(self.extracted_MODIS_ET_path, index=False)
					return nse_score, mape_score, pbias_score, rmse_score, kge_score
				except Exception as e:
						print(f"Error in {self.NAME}: {e}")
						return None, None, None, None, None


		def process_swat_et_output(self):
				# read the SWAT ET data
				df = pd.read_csv(self.SWAT_LS_path, sep='\s+', skiprows=[0,2], engine='python', header=0)[['name', 'yr', 'mon','et']]
				## read the MODIS ET data
				df_et = pd.read_csv(self.MODID_ET_path)
				## now merge the two dataframes
				merged_df = pd.merge(df, df_et, left_on=['yr','mon','name'], right_on=['year','month','LSUID'], how='inner')
				## plot the data
				## calucate average annual monthly ET
				#merged_df['mean_ET'] = merged_df.groupby(['year','mon'])['et'].transform('mean')
				#overal_nse_score, overal_rmse_score = self.calculate_metrics(merged_df)

				#head_nse_score, head_mape_score, head_pbias_score, head_rmse_score, head_kge_score, swl_nse_score, swl_mape_score, swl_pbias_score, swl_rmse_score, swl_kge_score
				return merged_df

		

if __name__ == "__main__":
		""" 
		This script is used to compare the SWAT ET with MODIS ET for each catchment in the SWAT model.
		If the SWAT ET is not available, the script will run the SWAT model to generate the ET data.
		The output is a CSV file containing the SWAT ET and MODIS ET for each catchment.
		The output will save in the respective model VPUD/LEVEL/NAME/MODIS_ET/MODISvsSWAT.csv
		"""


		BASE_PATH = "/data/MyDataBase/SWATplus_by_VPUID/0000/huc12"
		NAMES = os.listdir(BASE_PATH)
		NAMES.remove("log.txt")
		
		for NAME in NAMES:
			if NAME == "04126740":
				MODEL_NAME = "SWAT_gwflow_MODEL"
				VPUID = "0000"
				LEVEL = "huc12"
				TxtInOut_path = f"{BASE_PATH}/{NAME}/{MODEL_NAME}/Scenarios/Scenario_verification_stage_0"
				Compare_MODIS_et_SWAT(NAME, LEVEL, VPUID, TxtInOut_path).process_ET()

	