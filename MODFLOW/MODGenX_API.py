import os
from MODGenX.MODGenXCore import MODGenXCore
from multiprocessing import Process, Queue
from functools import partial
import shutil
import warnings
import geopandas as gpd
import rasterio
"""
/***************************************************************************
    SWATGenX
                            -------------------
        begin                : 2023-05-15
        copyright            : (C) 2024 by Vahid Rafiei
        email                : rafieiva@msu.edu
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
#warnings.filterwarnings("ignore", category=DeprecationWarning)

if __name__ == '__main__':

	print('start')
	BASE_PATH = "/data2/MyDataBase/SWATGenXAppData/"
	LEVEL = 'huc12'

	NAMES = os.listdir(fr'/data2/MyDataBase/SWATGenXAppData/SWAT_input/{LEVEL}/')
	try:
		NAMES.remove('log.txt')
	except:
		pass
	RESOLUTION = 250
	modelnames = ['MODFLOW_30m']

	print('start')
	models = []
	test = True
	parallel = False

	if test:
		NAME = '04106000'  ### testing, keep it or condition it
		MODEL_NAME = 'MODFLOW_250m'
		SWAT_MODEL_NAME = 'SWAT_MODEL'
		ML = False
		modflow_model = MODGenXCore(NAME, BASE_PATH, LEVEL, RESOLUTION, MODEL_NAME, ML, SWAT_MODEL_NAME)
		modflow_model.create_modflow_model()

	elif parallel:
			print('parallel')
			queue = Queue()
			processes = []
			max_processes = 80
			SWAT_MODEL_NAME = 'SWAT_MODEL_30m'

			for MODEL_NAME in modelnames:
				ML = '_ML_' in MODEL_NAME
				for NAME in NAMES:
					if len(NAME)>10:
						continue

					path_to_metrics = os.path.join(BASE_PATH, f"SWAT_input/{LEVEL}/{NAME}/{MODEL_NAME}/metrics.csv")

					if os.path.exists(path_to_metrics):
						print(f'{NAME} already exists')
						if not os.path.exists(path_to_metrics) and os.path.exists(f"/data2/MyDataBase/SWATGenXAppData/SWAT_input/{LEVEL}/{NAME}/{MODEL_NAME}/Grids_MODFLOW.geojson"):
							print(f"{NAME} failed probably due to lack of good data")

							continue
						continue

					subbasin_path = f"/data2/MyDataBase/SWATGenXAppData/SWAT_input/huc12/{NAME}/{SWAT_MODEL_NAME}/Watershed/Shapes/SWAT_plus_subbasins.shp"
					if not os.path.exists(subbasin_path):
						print(f"{NAME} does not have subbasin shapefile")
						continue


					subbasin = gpd.read_file(subbasin_path).to_crs(epsg=26990)
					centroid = subbasin.geometry.centroid[0]
					reference_raster = f"/data2/MyDataBase/SWATGenXAppData/all_rasters/DEM_{RESOLUTION}m.tif"

					## get the extent
					with rasterio.open(reference_raster) as src:
						extent = src.bounds

					## check if the centroid is within the extent
					if not (extent[0] < centroid.x < extent[2] and extent[1] < centroid.y < extent[3]):
						## remove the folder
						path = f"Z:/MyDataBase/SWATplus_by_VPUID/0000/{LEVEL}/{NAME}/"
						second_path = f"/data2/MyDataBase/SWATGenXAppData/SWAT_input/{LEVEL}/{NAME}/"
						print(f"{NAME} is out of extent and will be removed from calibration list")
						shutil.rmtree(second_path)
						if os.path.exists(path):
							shutil.rmtree(path)
						continue

					modflow_model = MODGenXCore(NAME, BASE_PATH, LEVEL, RESOLUTION, MODEL_NAME, ML, SWAT_MODEL_NAME)
					wrapped_model = partial(modflow_model.create_modflow_model)
					queue.put(wrapped_model)

			while not queue.empty():
				if len(processes) < max_processes:
					p = Process(target=queue.get())
					p.start()
					processes.append(p)
				else:
					for p in processes:
						p.join()
					processes = []
			for p in processes:
				p.join()
	else:
		# Simple loop for running models
		for MODEL_NAME in modelnames:
			ML = '_ML_' in MODEL_NAME
			for NAME in NAMES:
				model_output_figure_path = os.path.join(BASE_PATH, f"SWAT_input/{LEVEL}/{NAME}/{MODEL_NAME}/SWL_simulated_figure.jpeg")
				if not os.path.exists(model_output_figure_path):
					modflow_model = MODGenXCore(NAME, BASE_PATH, LEVEL, RESOLUTION, MODEL_NAME, ML, SWAT_MODEL_NAME)
					modflow_model.create_modflow_model()
		print('end')