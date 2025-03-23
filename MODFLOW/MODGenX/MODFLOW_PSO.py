import sys
import os
import uuid
import shutil
import matplotlib.pyplot as plt
import flopy
from sklearn.metrics import mean_absolute_error, r2_score
from functools import partial
from PSO_calibration import PSOOptimizer, save_final_results
import numpy as np
import pandas as pd
import shutil
import matplotlib.colors as colors
import os
import zipfile

def read_control_file(cal_parms):
	param_files = {}
	operation_types = {}
	problem = {'num_vars': 0, 'names': [], 'bounds': []}

	for _, row in cal_parms.iterrows():
		name = row['name']
		file_name = row['file_name']
		min_val = float(row['min'])  # Explicitly cast to float
		max_val = float(row['max'])
		operation = row['operation']

		if file_name not in param_files:
			param_files[file_name] = []
		param_files[file_name].append(name)

		operation_types[name] = operation
		problem['names'].append(name)
		problem['bounds'].append((min_val, max_val))
	problem['num_vars'] = len(problem['names'])
	return param_files, operation_types, problem



def update_zone_arrays(zone_array, zones_parameters, factors, name):
	for i in range(len(zone_array)):
		for j, factor in enumerate(factors):
			zone_key = f"{name}_zone_{j+1}"
			zone_array[i] += zones_parameters[name][zone_key] * factor

def update_parameters(mf, params, loaded_zones_dict):
	""" here we update the parameters of the model based on the input parameters
	"""
	def creating_separate_zones(arrays, array_names):
		""" here we create separate zones for each parameter
		How: """
		result_arrays = {}
		for array, name in zip(arrays, array_names):
			unique_values = np.unique(array)
			zone_arrays = {}
			for i, val in enumerate(unique_values):
				mask = array == val
				new_zone = np.zeros_like(array)
				new_zone[mask] = array[mask]
				zone_arrays[f'{name}_zone_{i+1}'] = new_zone
			result_arrays[name] = zone_arrays
		return result_arrays

	hk_factors = [params[f'hk_{i+1}_zone_{j+1}'] for i in range(2) for j in range(3)]

	vk_factors = [params[f'vk_{i+1}_zone_{j+1}'] for i in range(2) for j in range(3)]
	thickness_factors = [params[f'thickness_{i+1}_zone_{j+1}'] for i in range(2) for j in range(3)]
	recharge_factor = params['recharge']
	initial_head_factor = [params[f'initial_head_zone_{j+1}'] for j in range(3)]


	zones = {key: loaded_zones_dict[key] for key in ["SWL_zones", "k_vert_1_zones", "k_vert_2_zones","k_horiz_1_zones", "k_horiz_2_zones", "thickness_1_zones", "thickness_2_zones"]}

	zones_parameters = creating_separate_zones(
		list(zones.values()),
		['SWL', 'k_vert_1', 'k_vert_2', 'k_horiz_1', 'k_horiz_2', 'thickness_1', 'thickness_2']
	)

	hk_array, vk_array, botm_array = mf.upw.hk.array.copy(), mf.upw.vka.array.copy(), mf.dis.botm.array.copy()

	zone_names = ['k_vert_1', 'k_vert_2', 'k_horiz_1', 'k_horiz_2', 'thickness_1', 'thickness_2']
	arrays = [vk_array[:2,:,:], vk_array[2:,:,:], hk_array[:2,:,:], hk_array[2:,:,:], botm_array[:2,:,:], botm_array[2:,:,:]]
	factors_list = [vk_factors[:3], vk_factors[3:], hk_factors[:3], hk_factors[3:], thickness_factors[:3], thickness_factors[3:]]

	for name, arr, factors in zip(zone_names, arrays, factors_list):
		update_zone_arrays(arr, zones_parameters, factors, name)

	mf.upw.hk, mf.upw.vka, mf.dis.botm = hk_array, vk_array, botm_array

	mf.rch.rech = recharge_factor * mf.rch.rech.array.copy()[0,...]

	strt_array = mf.bas6.strt.array.copy()
	for i in range(3):
		strt_array[0,:,:] += initial_head_factor[i] * zones_parameters['SWL'][f'SWL_zone_{i+1}']

	mf.bas6.strt = strt_array

	riv_cond_factor = params['rv_cond']
	# Access river package data
	riv_data = mf.riv.stress_period_data.data

	factor = 1
	riv_data[0]['stage'] = factor*riv_data[0]['stage']
	riv_data[0]['cond'] += riv_cond_factor*riv_data[0]['cond']
	riv_data[0]['rbot'] = factor*riv_data[0]['rbot']
	mf.riv.stress_period_data = {0: riv_data[0]}  # Assuming 0 is the stress period you are interested in

	drn_cond_factor = params['drn_cond']
	drn_data = mf.drn.stress_period_data.data
	# For the drain package
	drn_data[0]['cond'] += drn_cond_factor* drn_data[0]['cond']
	mf.drn.stress_period_data = {0: drn_data[0]}  # Assuming 0 is the stress period you are interested in

	return mf


def read_plot_evaluate(mf, new_model_ws, modflow_model_name, VPUID, NAME, LEVEL, MODEL_NAME, scenario_name, BASE_PATH, no_value,figure_directory):
	head_path = os.path.join(new_model_ws, f'{modflow_model_name}.hds')
	try:
		if os.path.getsize(head_path) == 0:
			print(f"Head file is empty: {head_path}")
			return no_value
		else:
			# Load simulation output head file
			headobj = flopy.utils.binaryfile.HeadFile(head_path)

			sim_head = headobj.get_data(totim=headobj.get_times()[-1])
			### get no value of the data
			
			first_head = sim_head[0,:,:]
			#print(f"debug:\n unique values in first layer {np.unique(first_head)}")
			# replace 9999 with nan
			first_head[first_head == 9999] = np.nan

			# Create a figure with two subplots
			fig, (ax1, ax2) = plt.subplots(1, 2)
			# specify figure size
			fig.set_size_inches(12, 6)
			# Plot the first figure on the left subplot
			ax1.imshow(first_head)
			ax1.set_title('Dis Head')

			with open(os.path.join(mf.model_ws, f"{modflow_model_name}.hob.out")) as file:
				lines = file.readlines()

				sim = np.array([float(line.split()[0]) for line in lines[1:]])
				obs = np.array([float(line.split()[1]) for line in lines[1:]])

				lst_file = os.path.join(new_model_ws, f"{modflow_model_name}.list")
				lst = flopy.utils.MfListBudget(lst_file)

				CMB = lst.get_cumulative()

				print("Cumulative Mass Balance Error:", CMB['IN-OUT'][-1])

				# Remove outliers
				sim, obs = sim[(sim > np.percentile(sim, 1)) & (sim < np.percentile(sim, 99)) & (obs > np.percentile(obs, 1)) & (obs < np.percentile(obs, 99))], obs[(sim > np.percentile(sim, 1)) & (sim < np.percentile(sim, 99)) & (obs > np.percentile(obs, 1)) & (obs < np.percentile(obs, 99))]

				# Calculate the performance metrics
				rmse = np.sqrt(np.mean((sim - obs)**2))
				nse = 1 - (np.sum((obs - sim)**2) / np.sum((obs - np.mean(obs))**2))
				mae = mean_absolute_error(obs, sim)
				r2 = r2_score(obs, sim)

				# Plot the second figure on the right subplot
				ax2.scatter(sim, obs, s=3, color='black', marker='o')
				ax2.set_xlabel('Simulated')
				ax2.set_ylabel('Observed')
				ax2.set_title('Simulated vs Observed')

				# Add annotations to the second subplot
				ax2.annotate(f"RMSE: {rmse:.2f}\nNSE: {nse:.2f}\nMAE: {mae:.2f}\nR2: {r2:.2f}\nCMB: {CMB['IN-OUT'][-1]: .2f}", xy=(0.7, 0.1), xycoords='axes fraction')

				# Save the combined figure
				figure_directory = os.path.join(f'/data/SWATGenXApp/GenXApp/{username}', f'SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/calibration_figures_{MODEL_NAME}/')
				os.makedirs(figure_directory, exist_ok=True)
				plt.savefig(os.path.join(figure_directory, f'combined_figure_{scenario_name}.jpeg'), dpi=300)

				plt.close()

				return -nse

	except FileNotFoundError:
		print("File not found. Passing default error values.")
		return no_value

def run_model_simulation(mf, new_model_ws, new_exe_name):
	mf.model_ws = str(new_model_ws)
	mf.exe_name = str(new_exe_name)
	mf.write_input()
	success, buff = mf.run_model()
	return success, buff



def load_selective_zones(selected_filenames, directory):
	def load_zones_np(filename):
		return np.load(f'{filename}.npy')

	return {
		fname: load_zones_np(os.path.join(str(directory), fname))
		for fname in selected_filenames
	}


def generate_new_model_ws_and_load(params, base_directory, MODEL_NAME, scenario_name, original_model_ws, modflow_model_name, original_exe_name, problem):
	new_model_ws = os.path.join(base_directory, MODEL_NAME,  "Scenarios",  scenario_name)
	new_exe_name = os.path.join("/data/SWATGenXApp/codes/bin/", "modflow-nwt")
	# make the directory if it does not exist
	os.makedirs(new_model_ws, exist_ok=True)
	# copy the files from the original directory to the new directory
	content = os.listdir(original_model_ws)
	## find anything with name "Michigan" and copy it to the new directory
	for file in content:
		if modflow_model_name in file:
			src_file_path = os.path.join(original_model_ws, file)  # Corrected source path
			dst_file_path = os.path.join(new_model_ws, file)  # Destination path
			try:
				shutil.copy2(src_file_path, dst_file_path)
			except Exception as e:
				print(f"Error: {e}")


	shutil.copy(original_exe_name, new_exe_name)

	params_dict = dict(zip(problem['names'], params))

	mf = flopy.modflow.Modflow.load(modflow_model_name, exe_name = new_exe_name, model_ws = new_model_ws, version='mfnwt')
	return mf, new_model_ws, new_exe_name, params_dict



def simulate_and_evaluate_modflow_model(params, problem, MODEL_NAME, NAME, LEVEL, BASE_PATH, no_value, VPUID):


	filenames = ["SWL_zones", "k_horiz_1_zones", "k_horiz_2_zones", "k_vert_1_zones", "k_vert_2_zones", "thickness_1_zones", "thickness_2_zones"]


	modflow_model_name = "Michigan"



	base_directory = os.path.join(f'/data/SWATGenXApp/GenXApp/{username}', f'SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/')
	original_exe_name = os.path.join("/data/SWATGenXApp/codes/bin/", "modflow-nwt")
	original_model_ws = os.path.join(base_directory , MODEL_NAME)
	figure_directory = os.path.join(base_directory, f'calibration_figures_{MODEL_NAME}')
	loaded_zones_dict = load_selective_zones(filenames, original_model_ws)
	scenario_name = str(uuid.uuid4())

	mf, new_model_ws, new_exe_name, params_dict = generate_new_model_ws_and_load(params, base_directory, MODEL_NAME, scenario_name,original_model_ws, modflow_model_name, original_exe_name, problem)

	mf_updated = update_parameters(mf, params_dict,  loaded_zones_dict)

	success, buff = run_model_simulation(mf_updated, new_model_ws, new_exe_name)

	objective_value = read_plot_evaluate(mf_updated, new_model_ws, modflow_model_name,VPUID, NAME, LEVEL, MODEL_NAME, scenario_name, BASE_PATH, no_value, figure_directory)

	print(f"Task completed with objective_value: {objective_value}")

	return objective_value

def plot_zones(filenames, model_directory):
	plot_filenames = [os.path.join(model_directory, f"{filename}.npy") for filename in filenames]
	# Define the number of discrete colors you want in your colorbar
	n_colors = 3
	# load the files
	for file in plot_filenames:
		data = np.load(file)
		cmap = plt.get_cmap('viridis', n_colors)  # Use any colormap that suits your data
		# Optional: Normalize data to the range [0, n_colors-1] for discrete color mapping
		norm = colors.BoundaryNorm(boundaries=np.linspace(data.min(), data.max(), n_colors + 1), ncolors=n_colors)
		fig, ax = plt.subplots()
		cax = ax.imshow(data, cmap=cmap, norm=norm)
		cbar = fig.colorbar(cax, ticks=np.linspace(data.min(), data.max(), n_colors), spacing='proportional')
		cbar.ax.set_yticklabels(['{:.2f}'.format(i) for i in np.linspace(data.min(), data.max(), n_colors)])  # Optional: format tick labels
		plt.title(f"{file.split('/')[-1]}")
		plt.xlabel("column")
		plt.ylabel("row")
		#plt.close()
		plt.close()


def zip_copy_unzip(source_dir, dest_dir):
	# Create a zip file for the source directory
	shutil.make_archive(source_dir, 'zip', source_dir)

	# Define the paths for the zip files
	source_zip = f"{source_dir}.zip"
	dest_zip = os.path.join(dest_dir, os.path.basename(source_dir) + '.zip')

	# Ensure the destination directory exists
	os.makedirs(dest_dir, exist_ok=True)

	# Copy the zip file to the destination directory
	shutil.copy2(source_zip, dest_zip)

	# Unzip the file in the destination directory
	with zipfile.ZipFile(dest_zip, 'r') as zip_ref:
		zip_ref.extractall(dest_dir)
	# Remove the zip files
	os.remove(source_zip)
	os.remove(dest_zip)

# Use the function

if __name__ == "__main__":

	## run a single evaluation
	NAMES = os.listdir('/data/SWATGenXApp/Users/{username}/SWATplus_by_VPUID/0000/huc12/')
	#NAME = "40500012304"
	#VPUID = "0405"
	LEVEL = "huc12"
	RESOLUTION = 250
	MODEL_NAME = "MODFLOW_{RESOLUTION}m"
	BASE_PATH = "D:/MyDataBase"
	OUTPUT_path = "/data/SWATGenXApp/GenXAppData/"
	for NAME in NAMES:
		VPUID = f"0{NAME[:3]}"
		NAMES.remove('log.txt')
		model_base = f"SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/"
		MODFLOW_path = os.path.join(BASE_PATH,f"SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}")

		source_dir = os.path.join(f'/data/SWATGenXApp/GenXAppData/{username}/', f"SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/MODFLOW_{RESOLUTION}m")
		dest_dir = os.path.join(f'/data/SWATGenXApp/GenXAppData/{username}/', f"SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/MODFLOW_{RESOLUTION}m")
		zip_copy_unzip(source_dir, dest_dir)
		initial_points_path      = os.path.join(BASE_PATH, model_base,  f'initial_points_{MODEL_NAME}.csv')
		initial_values_path      = os.path.join(BASE_PATH, model_base,  f'initial_values_{MODEL_NAME}.csv')
		best_simulation_filename = os.path.join(BASE_PATH, model_base,  f'best_solution_{MODEL_NAME}.txt')

		model_log_path           = os.path.join(BASE_PATH, model_base,   'log.txt')
		general_log_path         = os.path.join(f'/data/SWATGenXApp/GenXAppData/{username}/', f"SWATplus_by_VPUID/{VPUID}/{LEVEL}/log.txt")

		figure_directory = os.path.join(BASE_PATH, model_base, f'calibration_figures_{MODEL_NAME}')
		# remove figures directory if it exists
		shutil.rmtree(figure_directory, ignore_errors=True)
		os.makedirs(figure_directory)

		filenames = ["SWL_zones", "k_horiz_1_zones", "k_horiz_2_zones", "k_vert_1_zones", "k_vert_2_zones", "thickness_1_zones", "thickness_2_zones"]

		model_directory = os.path.join(BASE_PATH, model_base, MODEL_NAME)

		# plot filenames
		plot_zones(filenames, model_directory)

		cal_parms = pd.read_csv("/data/SWATGenXApp/GenXAppData/bin/cal_parms_MODFLOW.cal", sep="\s+", skiprows =1)

		#now define params
		param_files, operation_types, problem = read_control_file(cal_parms)

		params = [np.random.uniform(low, high) for low, high in problem['bounds']]

		wrapped_model_evaluation = partial (simulate_and_evaluate_modflow_model, problem = problem, LEVEL = LEVEL,
											MODEL_NAME = MODEL_NAME, BASE_PATH = BASE_PATH, no_value = 1e6, VPUID = VPUID,
											NAME = NAME)

		opt = PSOOptimizer(

						wrapped_model_evaluation = wrapped_model_evaluation,
						problem = problem, cal_parms = cal_parms, VPUID = VPUID,
						BASE_PATH = BASE_PATH, LEVEL = LEVEL,
						NAME = NAME, MODEL_NAME = MODEL_NAME,
						model_log_path = model_log_path, general_log_path = general_log_path,
						max_it = 15 , n_particles = 50,
						best_simulation_filename = best_simulation_filename,
						termination_tolerance = 10, epsilon = 0.001,
						C1F=0.5, C1I=1, C2I=0.5, C2F=1, Vmax=0.1,
						InertiaMin=0.4, InertiaMax=1

						)

		opt.tell()  # Optionally, pass initial values and points

		best_position, best_score = opt.ask()
		## save the best position to the file
		save_final_results(best_score,best_position,cal_parms, best_simulation_filename, model_log_path)
