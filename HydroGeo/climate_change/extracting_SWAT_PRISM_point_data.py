import sys
import os
from multiprocessing import Process, Manager, Queue

try:
	from extraction_utils import loca2_wrapper
except Exception:
	from climate_change.extraction_utils import loca2_wrapper

import geopandas as gpd
import time
import h5py
import numpy as np


class DataExtractor:
	"""
		Class for extracting climate change data from specified locations.

		Args:
			cc_path (str): The base directory where the data is stored.
			mesh_path (str): The path to the mesh file containing locations to extract data from.
			output_dir (str): The directory where the extracted data will be saved.

		Returns:
			None

		Raises:
			No specific exceptions are raised.

		Examples:
			extractor = DataExtractor(cc_path='data_folder', mesh_path='mesh.shp', output_dir='output_data')
			extractor.start_extraction()
		"""

	def __init__(self, config):

		self.locations_to_extract = gpd.read_file(config["mesh_path"]).to_crs("EPSG:4326")
		self.cc_path = config["cc_path"]
		self.region = config["region"]                              # "e_n_cent"
		self.scenarios = config['scenarios']                        # ["ssp585", "ssp245", "ssp370", "historical"]
		self.model = config['model']                                # "CNRM-CM6-1"
		self.resolution = config['resolution']                      # "0p0625deg"
		self.ensemble = config['ensemble']                          # "r1i1p1f2"
		self.parameter_types = config['parameter_types']            # ["pcp", "tmp"]
		self.output_dir = config['output_dir']                      # output_dir
		self.processes = []
		self.parallel_processing = True

	def start_extraction(self):

		print("Starting extraction")
		for scenario in self.scenarios:
			print(self.locations_to_extract.head())
			for index, row in self.locations_to_extract.iterrows():
				lon = row['geometry'].centroid.x
				lat = row['geometry'].centroid.y
				name = row['name']
				print(f"Extracting data for {scenario} at {lat}, {lon}")
				elev = row['elev']
				try:
					PRISM_index = f"r{row.row}_c{row.col}"
				except Exception:
					PRISM_index = f"{name}"
					elev = row.elev

				for parameter_type in self.parameter_types:
					dict_request = {'cc_path': self.cc_path,
									'lat': lat,
									'lon': lon,
									'elev': elev,
									'model': self.model,
									'scenario': scenario,
									'ensemble': self.ensemble,
									'region': self.region,
									'resolution': self.resolution,
									'PRISM_index': f"{PRISM_index}",
									'parameter_type': parameter_type,
									'output_dir': self.output_dir,
					}

					if self.parallel_processing:
						process = Process(target=loca2_wrapper, args=(dict_request,))
						self.processes.append(process)
						process.start()
						print(f"Started process for {scenario} {parameter_type} {PRISM_index}")
					else:
						print(f"Extracting data for {scenario} {parameter_type} {PRISM_index}")
						loca2_wrapper(dict_request)

				if len(self.processes) == 10 and self.parallel_processing:
					process.join()
					self.processes = []

		for process in self.processes:
			process.join()

if __name__ == "__main__":
	base_path = "/data/"

	mesh_path = "/data/SWATplus_models/04102700/PRISM/PRISM_grid.shp"
	cc_path = os.path.join(base_path,"climate_change/cirrus.ucsd.edu/~pierce/LOCA2/CONUS_regions_split")
	output_dir = os.path.join(base_path,"climate_change/processed_data")

	config = {
		"cc_path": cc_path,
		"mesh_path": mesh_path,
		"output_dir": output_dir,
		"region": "e_n_cent",
		"scenarios": ["ssp585"],#, "ssp245", "ssp370", "historical"],
		"model": "CNRM-CM6-1",
		"resolution": "0p0625deg",
		"ensemble": "r1i1p1f2",
		"parameter_types": ["pcp", "tmp"]
	}

	extractor = DataExtractor(config)
	extractor.start_extraction()
