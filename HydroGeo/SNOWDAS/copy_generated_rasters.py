import os
import shutil
path =  "/data/MyDataBase/SWATGenXAppData/snow"
dest = "/data/MyDataBase/SWATGenXAppData/all_rasters"

rasters =  os.listdir(path)

for raster in rasters:
	if raster.endswith('.tif') and ("30m" in raster or "250m" in raster):
		print(f'{raster} moved to {dest}')
		shutil.copy2(os.path.join(path, raster), dest)
print('done')