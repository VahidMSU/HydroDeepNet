import pandas as pd
import geopandas as gpd
from rasterio.features import rasterize
from multiprocessing import Process
from functools import partial
import rasterio
import pyproj
def create_raster(column, raster_path, new_width, new_height, new_transform, catchments_EROMMA):
	shapes = list(zip(catchments_EROMMA.geometry, catchments_EROMMA[column]))
	with rasterio.open(raster_path, 'w', driver='GTiff', height=new_height, width=new_width, count=1, dtype='float32', crs=pyproj.CRS("EPSG:26990"), transform=new_transform) as dst:
		dst.write(rasterize(shapes, out_shape=(new_height, new_width), transform=new_transform, fill=-999), 1)
if __name__ == '__main__':
	path =  "/data/MyDataBase/SWATGenXAppData/NHDPlusData/Michigan/NHDPlusEROMMA.pkl"
	EROMMA = pd.read_pickle(path).drop(columns='geometry')
	catchments = gpd.GeoDataFrame(pd.read_pickle("/data/MyDataBase/SWATGenXAppData/NHDPlusData/Michigan/NHDPlusCatchment.pkl"))
	catchments = catchments.to_crs("EPSG:26990")
	catchments_EROMMA = catchments.merge(EROMMA, on='NHDPlusID', how='inner')
	catchments_EROMMA = catchments_EROMMA[['geometry', 'QAMA', 'VAMA', 'QIncrAMA',
	'QBMA', 'VBMA', 'QIncrBMA', 'QCMA', 'VCMA', 'QIncrCMA', 'QDMA', 'VDMA',
	'QIncrDMA', 'QEMA', 'VEMA', 'QIncrEMA', 'QFMA', 'QIncrFMA', 'ArQNavMA',
	'PETMA', 'QLossMA', 'QGAdjMA', 'QGNavMA', 'GageAdjMA', 'AvgQAdjMA',
	'GageIDMA', 'GageQMA']]

	# Define the bounds of your raster
	minx, miny, maxx, maxy = catchments_EROMMA.geometry.total_bounds
	width = int((maxx - minx) // 250)
	height = int((maxy - miny) // 250)
	new_width = width + 100
	new_height = height + 100
	# Calculate the new bounds
	new_minx = minx - (100 * 250)
	new_miny = miny - (100 * 250)
	new_maxx = maxx + (100 * 250)
	new_maxy = maxy + (100 * 250)
	new_transform = rasterio.transform.from_origin(new_minx, new_maxy, 250, 250)
	# Define the transform
	transform = rasterio.transform.from_origin(minx, maxy, 250, 250)
	processes = []
	print(catchments_EROMMA.columns[1:])
	print('starting.....')
	for column in catchments_EROMMA.columns[1:]:
		print(column)
		raster_path = f"/data/MyDataBase/SWATGenXAppData/NHDPlusData/Michigan/{column}.tif"
		wrapped_process = partial(create_raster, column = column, raster_path = raster_path, new_width = new_width, new_height = new_height, new_transform = new_transform, catchments_EROMMA = catchments_EROMMA)
		process = Process(target=wrapped_process)
		process.start()
		print(f"Started {column}")
		processes.append(process)
	for process in processes:
		process.join()
	print("All done")