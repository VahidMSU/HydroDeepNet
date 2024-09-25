import contextlib
import os
import rasterio
import h5py
import numpy as np

def get_raster_dimensions(path):
	with rasterio.open(path) as src:
		width = src.width
		height = src.height
	return width, height

class DataIntegration:
	def __init__(self, rasters_path, database_path):
		self.rasters_path = rasters_path
		self.database_path = database_path
		self.remove_existing_output()
		self.initiate_database()

	def initiate_database(self):
		with h5py.File(self.database_path, 'w') as f:
			f.create_group('rasters')
			f.create_group('labels')
			f.create_group('metadata')

	def remove_existing_output(self):
		if os.path.exists(self.database_path):
			os.remove(self.database_path)

	def process_rasters(self):  # sourcery skip: low-code-quality
		rasters = os.listdir(self.rasters_path)
		selected = [
			raster for raster in rasters if f'{RESOLUTION}' in raster and 'predictions' not in raster and raster.endswith('.tif')
		]

		print(selected)
		rasters_to_copy = []
		with contextlib.suppress(Exception):
			os.remove(self.database_path)
		reference_raster = os.path.join(self.rasters_path, f'HUC8_{RESOLUTION}m.tif')

		with rasterio.open(reference_raster) as src:
			mask = self._extracted_from_process_rasters_16(src)

		if os.path.exists(self.database_path):
			os.remove(self.database_path)

		ref_width, ref_height = get_raster_dimensions(reference_raster)
		print('width, height, raster')
		print(ref_width, ref_height)
		for selected_raster in selected:
			with rasterio.open(os.path.join(self.rasters_path, selected_raster)) as src:
				print(src.width, src.height, selected_raster)

				if src.width == ref_width and src.height == ref_height:
					rasters_to_copy.append(selected_raster)
					data_array = src.read(1)
					nodata = src.nodatavals[0]

					# Ensure data type can handle -99
					if data_array.dtype in [np.uint8, np.int8, np.uint16, np.int16]:
						data_array = data_array.astype(np.int32)

					data_array[data_array == nodata] = -999
					data_array[mask] = -999

					if "recharge" in selected_raster:
						data_array[data_array < 0] = -999
						data_array[mask] = -999

					if "DEM" in selected_raster:
						data_type = 'float32'
					elif "landforms" in selected_raster:
						data_type = 'int16'
					elif "_lat_" in selected_raster:
						data_type = 'float32'
					elif "_lon_" in selected_raster:
						data_type = 'float32'
					elif "_x_" in selected_raster or "_y_" in selected_raster:
						continue
					elif selected_raster == f"x_{RESOLUTION}m.tif":
						data_type = 'float32'
					elif selected_raster == f"y_{RESOLUTION}m.tif":
						data_type = 'float32'
					elif selected_raster == f"BaseRaster_{RESOLUTION}m.tif":
						data_type = 'int16'
					elif selected_raster == f"COUNTY_{RESOLUTION}m.tif":
						data_type = 'int16'
					elif selected_raster == f"HUC8_{RESOLUTION}m.tif":
						data_type = 'int64'
					elif selected_raster == f"HUC12_{RESOLUTION}m.tif":
						data_type = 'int64'
					elif 'gSSURGO' in selected_raster:
						data_type = 'float32'
					else:
						data_type = data_array.dtype

					if data_type == 'int8':
						data_type = 'int16'

					with h5py.File(self.database_path, 'a') as f:
						dset = f.create_dataset(selected_raster.split('.')[0], (src.height, src.width), dtype=data_type, data=data_array)
						dset.flush()
						dset.attrs['transform'] = src.transform
						dset.attrs['crs'] = src.crs.to_string()
						dset.attrs['nodata'] = -999
						dset.attrs['width'] = src.width
						dset.attrs['height'] = src.height
						dset.attrs['count'] = src.count
						dset.attrs['driver'] = src.driver
				else:
					print('not selected', selected_raster)

	def _extracted_from_process_rasters_16(self, src):
		reference_transform = src.transform
		reference_crs = src.crs
		reference_nodata = src.nodatavals[0]
		reference_width = src.width
		reference_height = src.height
		return src.read(1) == reference_nodata

if __name__ == '__main__':
	RESOLUTIONS = [250]

	for RESOLUTION in RESOLUTIONS:
		if RESOLUTION in [50, 100]:
			rasters_path = '/data/MyDataBase/SWATGenXAppData/all_rasters/upscaled'
		else:
			rasters_path = '/data/MyDataBase/SWATGenXAppData/all_rasters'
		database_path = f'Z:/HydroGeoDataset_ML_{RESOLUTION}.h5'
		if os.path.exists(database_path):
			os.remove(database_path)
		data_integration = DataIntegration(rasters_path, database_path)
		data_integration.process_rasters()
