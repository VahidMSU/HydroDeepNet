import os
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import glob
try:
	from ModelProcessing.visualization import plot_domain
except Exception:
	from visualization import plot_domain
import rasterio
from rasterio.features import rasterize
import logging
import os
from multiprocessing import Process
from ModelProcessing.SWATGenXConfigPars import SWATGenXPaths
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

def create_recharge_image_for_name(gwflow_target_path, LEVEL,VPUID, NAME, RESOLUTION,gis_folder,rech_out_folder, start_year, end_year, nyskip):

	""" this function create annual recharge figures
	processes:
			1- removing previous figures
			2- checking the gwflow_flux_recharge exists:
			3- reading row and columns from gwflow.input
			4- reading grid arrays for extracting active domain
			5- reading gwflow flux recharge
			6- removing out of active zone values
			7- plotting gwflow recharge figures
	"""
	logging.info(f'Processing recharge for {NAME} from {start_year} to {end_year}')
	number_of_years = end_year - start_year + 1 + nyskip
	if not os.path.exists(rech_out_folder):
		os.makedirs(rech_out_folder, exist_ok = True)
	files=glob.glob(os.path.join(rech_out_folder , '*jpeg'))
	for file in files:
		os.remove(file)

	files=glob.glob(os.path.join(rech_out_folder, '*shp'))
	for file in files:
		os.remove(file)
	print("##############################",gwflow_target_path)
	path = os.path.join(gwflow_target_path,  "gwflow_flux_recharge")
	
	if not os.path.exists(path):
			message = f'gwflow_flux_recharge does not exist. {path}'
			logging.error(message)
			return message

	with open(path,'r') as file:
			lines=file.readlines()
			if len(lines)<500:
				return 'the model probably crashed during verification: incomplete gwflow_flux_recharge'

	with open(os.path.join(gwflow_target_path ,"gwflow.input"), 'r') as file:
		lines = file.readlines()
		nrows, ncols = map(int, lines[3].split())
	# Initialize the recharge array
	domain = np.zeros([nrows, ncols])

	# Read the grid arrays file
	with open(os.path.join(gwflow_target_path,'gwflow_grid_arrays'), 'r') as file:
		m = 0  # row index
		for line in file:
			if "inactive" in line:
				continue
			for i, col_value in enumerate(line.split()):
				domain[m, i] = int(col_value)
			m += 1
			if m >= nrows:  # Stop if you've read enough rows
				break

	fig_output_path = os.path.join(rech_out_folder, 'active_domain.jpeg')

	plot_domain(domain,fig_output_path)

	recharge = np.zeros([number_of_years, nrows, ncols])

	# Read the recharge data
	year_index = 0  # Index for the year
	row_index = 0  # Index for the row within a year

	with open(os.path.join( gwflow_target_path,"gwflow_flux_recharge"), 'r') as file:
		for line in file:
			if "Recharge" in line or 'Annual' in line or line.strip() == '':
				if row_index != 0:  # If it's not the first line of a year
					year_index += 1  # Move to the next year
					row_index = 0  # Reset row index for the new year
				continue
			# Split the line into columns and assign to recharge array
			try:
				recharge[year_index, row_index, :] = np.array(line.split(), dtype = float)
				row_index += 1
			except Exception as e:
				logging.error(f'failed reading year index in gwflow flux recharge: {e}')


	for i, year in enumerate(range(start_year, end_year)):  # Example years
		# Apply domain mask to recharge data
		masked_recharge = np.where(domain == 1, recharge[i], np.nan)

		# Calculate 95th percentile of the non-NaN values
		percentile_95 = np.nanpercentile(masked_recharge, 95)

		# Plot
		plt.figure(figsize=(10, 10))
		plt.imshow(masked_recharge, vmax=percentile_95)
		plt.colorbar().set_label(" Recharge (m3/day)")
		plt.title(f"Annual Average Daily Recharge {year}: {RESOLUTION}m resolution")
		plt.xlabel("Column")
		plt.ylabel("Row")

		# Save plot
		os.makedirs(rech_out_folder, exist_ok = True)
		plt.savefig(os.path.join(rech_out_folder, f'recharge_{year}.jpeg'), dpi=400)
		# plt.show()
		plt.close()

		create_recharge_shapefile(LEVEL, VPUID, NAME, RESOLUTION, masked_recharge, year, rech_out_folder)

		logging.info(f'Finished processing recharge for {NAME} {year}')



def create_recharge_shapefile(LEVEL,VPUID, NAME,RESOLUTION, recharge, year,rech_out_folder):
	input_grids_path = f'/data2/MyDataBase/SWATGenXAppData/SWAT_input/{LEVEL}/{NAME}/MODFLOW_{RESOLUTION}m/Grids_MODFLOW/Grids_MODFLOW.shp'
	output_recharge_path = os.path.join(rech_out_folder, f'recharge_{year}.shp')
	SWAT_dem_path = f'/data2/MyDataBase/SWATGenXAppData/SWAT_input/{LEVEL}/{NAME}/DEM_{RESOLUTION}m.tif'
	output_recharge_raster_path = os.path.join(rech_out_folder, f'recharge_{year}.tif')

	# Read the grid shapefile
	grids = gpd.read_file(input_grids_path)

	# Assign recharge values to each grid cell
	grids['Recharge'] = [recharge[row, col] for row, col in zip(grids['Row'], grids['Col'])]

	# Save the new shapefile
	grids.to_file(output_recharge_path)

	# Read DEM for spatial reference
	with rasterio.open(SWAT_dem_path) as dem:
		dem_meta = dem.meta.copy()

	shapes = ((geom, value) for geom, value in zip(grids.geometry, grids['Recharge']))

	# Create a raster image based on the shapefile
	with rasterio.open(output_recharge_raster_path, 'w', **dem_meta) as out_raster:
		out_raster.write_band(1, rasterize(shapes, out_shape=out_raster.shape, fill=np.nan, transform=out_raster.transform))


def process_scenario(scenario_details):
	gwflow_target_path, LEVEL, VPUID, NAME, RESOLUTION, gis_folder, rech_out_folder, start_year, end_year, nyskip = scenario_details
	create_recharge_image_for_name(gwflow_target_path, LEVEL, VPUID, NAME, RESOLUTION, gis_folder, rech_out_folder, start_year, end_year, nyskip)

def recharge_generator_helper(NAME):
	NAME_ = NAME
	VPUID = "0000"
	NAMES = os.listdir(f"{SWATGenXPaths.swatgenx_outlet_path}/{VPUID}/huc12")
	LEVEL = "huc12"
	MODEL_NAME = "SWAT_gwflow_MODEL"
	RESOLUTION = 250
	start_year = 1997
	end_year = 2020
	nyskip = 3
	gis_folder = f"{SWATGenXPaths.swatgenx_outlet_path}"

	processes = []

	for NAME in NAMES:
		if NAME != NAME_:
				continue
		for i in range(5):
			SCENARIO = f"verification_stage_{i}"
			gwflow_target_path = f"{SWATGenXPaths.swatgenx_outlet_path}/{VPUID}/huc12/{NAME}/{MODEL_NAME}/Scenarios/{SCENARIO}"
			rech_out_folder = f"{SWATGenXPaths.swatgenx_outlet_path}/{VPUID}/huc12/{NAME}/recharg_output_{MODEL_NAME}/{SCENARIO}"

			scenario_details = (
					gwflow_target_path, LEVEL, VPUID, NAME, RESOLUTION, gis_folder, rech_out_folder, start_year, end_year, nyskip
			)

			p = Process(target=process_scenario, args=(scenario_details,))
			processes.append(p)
			p.start()
				

	for p in processes:
			p.join()

	logging.info('All scenarios processed')



if __name__ == "__main__":
	NAME  = "04166000"
	recharge_generator_helper(NAME)