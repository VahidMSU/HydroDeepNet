## the aim of this script is to create a raster with a higher resolution than the original one
import arcpy
import os

reference_raster = "/data/MyDataBase/SWATGenXAppData/all_rasters/DEM_30m.tif"
arcpy.env.workspace = "/data/MyDataBase/SWATGenXAppData/all_rasters"
arcpy.env.overwriteOutput = False
target_resolutions = [50,100]

os.listdir(arcpy.env.workspace)	
## select rasters with 30m resolution
rasters = [raster for raster in os.listdir(arcpy.env.workspace) if "30m" in raster and raster.endswith('.tif')]
print(rasters)
os.makedirs("/data/MyDataBase/SWATGenXAppData/all_rasters/upscaled", exist_ok=True)
for raster in rasters:
	for res in target_resolutions:
		print(f"Processing {raster} to {res}m")
		new_raster_name = raster.replace("30m", f"{res}m")
		try:
			if "kriging" in raster and 'obs' not in raster or "DEM" in raster or "melt" in raster or "snow" in raster or "temperature" in raster:
				print(f'Upscaling {raster} with BILINEAR interpolation')
				arcpy.Resample_management(raster, f"upscaled/{new_raster_name.split('.')[0]}.tif", res, "BILINEAR")
			elif "obs" in raster:
				### do not use any interpolatoion and only save with new resolution
				print(f'Upscaling {raster} without interpolation')
				arcpy.Resample_management(raster, f"upscaled/{new_raster_name.split('.')[0]}.tif", res)
			elif "landforms" in raster or "geomorphons" in raster:  ## use majority
				print(f'Upscaling {raster} with MAJORITY interpolation')	
				arcpy.Resample_management(raster, f"upscaled/{new_raster_name.split('.')[0]}.tif", res, "MAJORITY")
			else:
				print(f'Upscaling {raster} with NEAREST interpolation')
				arcpy.Resample_management(raster, f"upscaled/{new_raster_name.split('.')[0]}.tif", res, "NEAREST")
			
		except Exception as e:
			print(f"Upscaling {raster} failed: {e}")
			continue
	print('done')