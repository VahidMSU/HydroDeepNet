import pandas as pd 
import numpy as np  
import sys
import matplotlib.pyplot as plt
import os 
import geopandas as gpd
import h5py
import matplotlib.pyplot as plt
import uuid
import time
class GroundwaterModelAnalysis:
		def __init__(self, NAME, MODEL_NAME, VPUID, LEVEL, TxtInOut_path, resolution="250", key=None):
				self.resolution = resolution
				self.NAME_PATH = f"/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/"
				self.MODEL_PATH = f"{self.NAME_PATH}/{MODEL_NAME}/"
				self.TxtInOut_path = TxtInOut_path
				self.h5_path = f"/data/MyDataBase/HydroGeoDataset_ML_{self.resolution}.h5"
				self.gwflow_state_head= f"{self.TxtInOut_path}/gwflow_state_head"
				self.gwflow_centroids = f"{self.MODEL_PATH}/gwflow_gis/grids_points.shp"
				self.fig_path = f"{self.NAME_PATH}/figures_{MODEL_NAME}"
				self.key = uuid.uuid4().hex if key is None else key
				self.NAME = NAME
				


		def check_files(self):
				os.makedirs(self.fig_path, exist_ok=True)
				
				if os.path.exists(self.gwflow_state_head):
						return True
				
		def process(self):
				self.extract_observation_heads(self.gwflow_centroids)
				self.get_simulated_heads()
								
		@staticmethod
		def nse(obs, sim):
				return 1 - (np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2))
		@staticmethod
		def rmse(obs, sim):
				return np.sqrt(np.mean((obs - sim) ** 2))
		@staticmethod
		def kge(obs, sim):
				r = np.corrcoef(obs, sim)[0, 1]
				alpha = np.std(sim) / np.std(obs)
				beta = np.mean(sim) / np.mean(obs)
				return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
		@staticmethod
		def pbias(obs, sim):
				return 100 * np.sum(sim - obs) / np.sum(obs)
		@staticmethod
		def mape(obs, sim):
				return np.mean(np.abs((obs - sim) / obs)) * 100
		
		#nse, mape, pbias, rmse, kge
		def get_rowcol_range_by_latlon(self, desired_min_lat, desired_max_lat, desired_min_lon, desired_max_lon):
				with h5py.File(self.h5_path, 'r') as f:
						# Read latitude and longitude arrays
						lat_ = f["lat_250m"][:]
						lon_ = f["lon_250m"][:]

						# Replace missing values (-999) with NaN for better handling
						lat_ = np.where(lat_ == -999, np.nan, lat_)
						lon_ = np.where(lon_ == -999, np.nan, lon_)

						# Create masks for latitude and longitude ranges
						lat_mask = (lat_ >= desired_min_lat) & (lat_ <= desired_max_lat)
						lon_mask = (lon_ >= desired_min_lon) & (lon_ <= desired_max_lon)

						# Combine the masks to identify the valid rows and columns
						combined_mask = lat_mask & lon_mask

						# Check if any valid points are found
						if np.any(combined_mask):
								# Get row and column indices where the combined mask is True
								row_indices, col_indices = np.where(combined_mask)
						else:
								print("No valid points found for the given latitude and longitude range.")
								return None, None, None, None

						min_row_number = np.min(row_indices)
						max_row_number = np.max(row_indices)
						min_col_number = np.min(col_indices)
						max_col_number = np.max(col_indices)

						return min_row_number, max_row_number, min_col_number, max_col_number

		def extract_observation_heads(self, inpu_path):
				gdf = gpd.read_file(self.gwflow_centroids).to_crs(epsg=4326)
				#print(gdf.head())
				nrows = np.unique(gdf['Row']).shape[0]
				ncols = np.unique(gdf['Col']).shape[0]

				bounding = gdf.total_bounds
				h5_group_name = f"obs_SWL_{self.resolution}m"
				min_lon, min_lat, max_lon, max_lat = bounding

				min_row_number, max_row_number, min_col_number, max_col_number = self.get_rowcol_range_by_latlon(min_lat, max_lat, min_lon, max_lon)

				# Open the HDF5 file
				with h5py.File(self.h5_path, "r") as f:
						rows = max_row_number - min_row_number 
						cols = max_col_number - min_col_number 

						diff_rows = nrows - rows
						diff_cols = ncols - cols
						if diff_rows > 0:
								min_row_number -= diff_rows
						if diff_cols > 0:
								min_col_number -= diff_cols

						swl = f[h5_group_name][min_row_number:max_row_number + diff_rows, min_col_number:max_col_number + diff_cols]
						DEM = f["DEM_250m"][min_row_number:max_row_number + diff_rows, min_col_number:max_col_number + diff_cols]

						swl = np.where(swl == -999, np.nan, swl) 
						swl = swl * 0.3048  # Convert feet to meters
						self.DEM = np.where(swl == -999, np.nan, DEM)

						self.observed_head = DEM - swl 


		def get_simulated_heads(self):
				
				with open(self.gwflow_state_head, 'r') as f:
						lines = f.readlines()
						dictionary = {"year": [], "month": [], "gw_heads": []}

						for i, line in enumerate(lines):
								if "Groundwater Head for" in line:
										year = int(line.split(":")[1].split()[0])
										month = int(line.split(":")[1].split()[1])
										dictionary["year"].append(year)
										dictionary["month"].append(month)
										rows = []   
										for line in lines[i+1:]:
												if len(line.strip()) == 0:
														break
												rows.append(line.strip().split())
										simulated_heads_ = np.stack(rows, axis=0).astype(float)
										dictionary["gw_heads"].append(simulated_heads_)

				simulated_heads = np.mean(dictionary["gw_heads"], axis=0)
				####padding
				nrows, ncols = self.observed_head.shape
				sim_nrows, sim_ncols = simulated_heads.shape
				diff_rows = nrows - sim_nrows
				diff_cols = ncols - sim_ncols
				if diff_rows > 0:
						simulated_heads = np.pad(simulated_heads, ((0, diff_rows), (0, 0)), mode='constant', constant_values=np.nan)
						
				if diff_cols > 0:
						simulated_heads = np.pad(simulated_heads, ((0, 0), (0, diff_cols)), mode='constant', constant_values=np.nan)

				self.simulated_heads = np.where(self.observed_head == np.nan, np.nan, simulated_heads)
				
				assert self.simulated_heads.shape == self.observed_head.shape, f"Expected {self.observed_head.shape} but got {self.simulated_heads.shape}"
				self.observed_head = np.where(simulated_heads == 0, np.nan, self.observed_head)

		def plot_heads(self, data_heads, name="observed_heads"):

				#print(f"Shape of img: {observed_head.shape}")
				fig = plt.figure()
				plt.imshow(np.where(data_heads== -999, np.nan, data_heads))
				plt.colorbar()
				os.makedirs(f"{self.fig_path}/{name}", exist_ok=True)
				plt.savefig(f"{self.fig_path}/{name}/{int(time.time())}.png")
				plt.close()


		def calculate_metrics(self):
				valid_mask = ~np.isnan(self.observed_head) & ~np.isnan(self.simulated_heads)
				observed_valid = self.observed_head[valid_mask]
				simulated_valid = self.simulated_heads[valid_mask]

				head_nse_score = self.nse(observed_valid, simulated_valid)
				head_rmse_score = self.rmse(observed_valid, simulated_valid)
				head_kge_score = self.kge(observed_valid, simulated_valid)	
				head_pbias_score = self.pbias(observed_valid, simulated_valid)
				head_mape_score = self.mape(observed_valid, simulated_valid)

				DEM_valid = self.DEM[valid_mask]
				SWL = DEM_valid - observed_valid
				simulated_SWL = DEM_valid - simulated_valid

				swl_nse_score = self.nse(SWL, simulated_SWL)
				swl_rmse_score = self.rmse(SWL, simulated_SWL)
				swl_kge_score = self.kge(SWL, simulated_SWL)
				swl_pbias_score = self.pbias(SWL, simulated_SWL)
				swl_mape_score = self.mape(SWL, simulated_SWL)

				print(f"{self.NAME} GW heads: NSE: {head_nse_score:.2f}, RMSE: {head_rmse_score:.2f}, KGE: {head_kge_score:.2f}, PBIAS: {head_pbias_score:.2f}, MAPE: {head_mape_score:.2f}")
				print(f"{self.NAME} GW   SWL: NSE: {swl_nse_score:.2f}, RMSE: {swl_rmse_score:.2f}, KGE: {swl_kge_score:.2f}, PBIAS: {swl_pbias_score:.2f}, MAPE: {swl_mape_score:.2f}")
				return head_nse_score, head_mape_score, head_pbias_score, head_rmse_score, head_kge_score, swl_nse_score, swl_mape_score, swl_pbias_score, swl_rmse_score, swl_kge_score

		def compare_heads_swl(self):
				valid_mask = ~np.isnan(self.observed_head) & ~np.isnan(self.simulated_heads)
				observed_valid = self.observed_head[valid_mask]
				simulated_valid = self.simulated_heads[valid_mask]

				nse_score = self.nse(observed_valid, simulated_valid)
				rmse_score = self.rmse(observed_valid, simulated_valid)

				fig = plt.figure()
				plt.scatter(self.observed_head.flatten(), self.simulated_heads.flatten(), s=1)
				plt.title(f"NSE: {nse_score:.2f}, RMSE: {rmse_score:.2f}")
				plt.xlabel("Observed heads")
				plt.ylabel("Simulated heads")
				plt.grid(linestyle='--', linewidth=0.5)
				os.makedirs(f"{self.fig_path}/GW", exist_ok=True)
				plt.savefig(f"{self.fig_path}/GW/{nse_score:.2f}_{int(time.time())}.png", dpi=150)
				plt.close()

				DEM_valid = self.DEM[valid_mask]
				SWL = DEM_valid - observed_valid
				simulated_SWL = DEM_valid - simulated_valid

				nse_score = self.nse(SWL, simulated_SWL)
				rmse_score = self.rmse(SWL, simulated_SWL)
				fig = plt.figure()
				plt.scatter(SWL, simulated_SWL, s=1)
				plt.title(f"NSE: {nse_score:.2f}, RMSE: {rmse_score:.2f}")
				plt.xlabel("Observed SWL")
				plt.ylabel("Simulated SWL")
				plt.grid(linestyle='--', linewidth=0.5)
				os.makedirs(f"{self.fig_path}/SWL", exist_ok=True)
				plt.savefig(f"{self.fig_path}/SWL/{nse_score:.2f}_{int(time.time())}.png", dpi=150)
				plt.close()


if __name__ == "__main__":

		NAME = "04126740"
		MODEL_NAME = "SWAT_gwflow_MODEL"
		VPUID = "0000"
		LEVEL = "huc12"
		MODEL_PATH = f"/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}/"
		TxtInOut_path = f"{MODEL_PATH}/Scenarios/Scenario_60b06818-d35f-4054-a42e-10198a63ddad"

		print(f"model path: {MODEL_PATH}")
		analysis = GroundwaterModelAnalysis(NAME, MODEL_NAME, VPUID, LEVEL, TxtInOut_path)
		if analysis.check_files():
				analysis.process()
				head_nse_score, head_mape_score, head_pbias_score, head_rmse_score, head_kge_score, swl_nse_score, swl_mape_score, swl_pbias_score, swl_rmse_score, swl_kge_score = analysis.calculate_metrics()
				analysis.compare_heads_swl()
				analysis.plot_heads(analysis.observed_head, name="observed_heads")
				analysis.plot_heads(analysis.simulated_heads, name="simulated_heads")
		else:
				print(f"File not found for {NAME}")
