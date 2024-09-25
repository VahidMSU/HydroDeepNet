import os
import numpy as np
import h5py

class Gwflow2h5:
	def __init__(self, base_directory, vpuid, level, name, model_name, scenario, file_name):
		self.base_directory = base_directory
		self.vpuid = vpuid
		self.level = level
		self.name = name
		self.model_name = model_name
		self.scenario = scenario
		self.file_name = file_name
		self.output_h5_path = os.path.join(base_directory, 'codes/SWAT_H5/', "SWAT_OUTPUT.h5")
		self.gwflow_var_path = os.path.join(base_directory, f"SWATplus_by_VPUID/{vpuid}/{level}/{name}/{model_name}/Scenarios/{scenario}/{file_name}")
		self.gwflow_input = os.path.join(base_directory, f"SWATplus_by_VPUID/{vpuid}/{level}/{name}/{model_name}/Scenarios/{scenario}/gwflow.input")
		self.year = 0
		self.row = 0
		self.title = ""
	def extract_row_col(self):
		with open(self.gwflow_input, 'r') as f:
			lines = f.readlines()
			self.rows, self.cols = int(lines[3].split()[0]), int(lines[3].split()[1])
			self.data = np.zeros((17, int(self.rows), int(self.cols)))

	def write_gwflow_on_h5(self, actual_year=None):
		""" Write gwflow recharge data to the h5 file """
		with h5py.File(self.output_h5_path, 'a') as f:

			dataset_address =  f"{self.file_name.split('_')[0]}/{self.file_name.split('_')[1]}_{self.file_name.split('_')[2]}//{actual_year}"
			f.create_group(dataset_address)
			variable_name = self.file_name.split('_')[-1]

			f[dataset_address].create_dataset("data", data=self.data[self.year-1,...])
			f[dataset_address].attrs['unit'] = self.title.split('(')[1].split(')')[0]
			f[dataset_address].attrs['description'] = self.title.split('(')[0]
			
	def read_gwflow_variable(self):
		""" Read gwflow recharge data from the SWAT output file """
		with open(self.gwflow_var_path, 'r') as f:
			lines = f.readlines()
			for line in lines:
				line = line.strip()
				if "Annual" in line:
					self.title = line.replace(" ", "")
				elif "year" in line or 'Soil' in line or 'Saturation' in line or 'Tile' in line or 'Lake' in line or 'Groundwater' in line:
					actual_year = int(line.split(':')[1])
					if self.year > 0:  # write previous year's data to h5 before moving to the next
						self.write_gwflow_on_h5(actual_year-1)
					self.year += 1
					self.row = 0
				elif line and len(line.split())>1:
					self.data[self.year-1, self.row, :] = np.fromstring(line, sep=' ')
					self.row += 1
			if self.year > 0:  # Ensure the last year's data is written
				self.write_gwflow_on_h5(actual_year)




def remove_gwflow_dataset(SWAT_H5_path):
	with h5py.File(SWAT_H5_path, 'a') as f:
		if 'gwflow' in f.keys():
			del f['gwflow']
if __name__ == "__main__":
	DIC = "D:/MyDataBase"
	VPUID = "0405"
	LEVEL = "huc12"
	NAME = "40500010102"
	MODEL_NAME = "SWAT_gwflow_MODEL"
	SCENARIO = "Scenario_verification_stage_1"
	FILE_NAMES =  [
		'gwflow_flux_gwet', 'gwflow_flux_gwsoil', 'gwflow_flux_gwsw', 
		'gwflow_flux_lake', 'gwflow_flux_lateral', 
		'gwflow_flux_pumping_ag', 'gwflow_flux_recharge',
		'gwflow_flux_satex', 'gwflow_flux_tile', 
		]
	SWAT_H5_path = os.path.join(DIC, 'codes/SWAT_H5/', "SWAT_OUTPUT.h5")
	remove_gwflow_dataset(SWAT_H5_path)
	for FILE_NAME in FILE_NAMES:
		print(f"###### Working on {FILE_NAME} ######")
		processor = Gwflow2h5(DIC, VPUID, LEVEL, NAME, MODEL_NAME, SCENARIO, FILE_NAME)
		processor.extract_row_col()
		processor.read_gwflow_variable()