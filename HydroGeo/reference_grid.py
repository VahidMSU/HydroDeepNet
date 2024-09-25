import rasterio

def get_raster_dimensions(path):
	with rasterio.open(path) as src:
		width = src.width
		height = src.height
	return width, height

RESOLUTION = 30
path = fr"/data/MyDataBase/SWATGenXAppData/all_rasters/DEM_{RESOLUTION}m.tif"
width, height = get_raster_dimensions(path)
print(width, height)