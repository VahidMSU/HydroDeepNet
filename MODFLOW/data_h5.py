import GDAL
from GDAL.sa import *
import h5py
import os
import numpy as np

# Define paths
DIC = "Z:/"
ml_h5_path = "Z:/out"
datasets = os.listdir(ml_h5_path)

# Get all the rasters ending with .h5 and containing _50 in the name
rasters = [file.split()[1].split('.')[0].split("_50")[0] for file in datasets if file.endswith(".h5") and "_50" in file]
print(rasters)
# Example: ['AQ_THK_1_50m', 'AQ_THK_2_50m', 'H_COND_1_50m', 'H_COND_2_50m', 'SWL_50m', 'TRANSMSV_1_50m', 'TRANSMSV_2_50m', 'V_COND_1_50m', 'V_COND_2_50m']

resolutions = [250]
spaces = ['AQ_THK_1', 'AQ_THK_2', 'H_COND_1', 'H_COND_2', 'SWL', 'TRANSMSV_1', 'TRANSMSV_2', 'V_COND_1', 'V_COND_2']

# Set environment variables for ArcPy
GDAL.env.overwriteOutput = True

for resolution in resolutions:
    for space in spaces:
        output_path = os.path.join("/data/SWATGenXApp/GenXAppData/all_rasters", f"predictions_ML_{space}_{resolution}.tif")
        input_data_path = os.path.join(ml_h5_path, f"FFR_Predicted obs_{space}_{resolution}m.h5")
        input_xy_path = os.path.join(DIC, f"HydroGeoDataset_ML_{resolution}.h5")
        

        print(f"input_data_path: {input_data_path}")
        
        # Read data from HDF5 file
        with h5py.File(input_data_path, 'r') as f:
            data = f[f'FFR_Predicted obs_{space}_{resolution}m'][:]
        # replace nan qith -999
        data = np.nan_to_num(data, nan=-999)
        # Create the raster
        reference_raster = f"/data/SWATGenXApp/GenXAppData/all_rasters/DEM_{resolution}m.tif"
        
        # Set ArcPy environment variables based on reference raster
        GDAL.env.outputCoordinateSystem = GDAL.Describe(reference_raster).spatialReference
        GDAL.env.snapRaster = reference_raster
        GDAL.env.extent = GDAL.Describe(reference_raster).extent

        # Extract cell size from reference raster
        cell_size = GDAL.Describe(reference_raster).meanCellWidth
        
        # Convert numpy array to raster
        lower_left_corner = GDAL.Point(GDAL.Describe(reference_raster).extent.XMin, GDAL.Describe(reference_raster).extent.YMin)
        raster = GDAL.NumPyArrayToRaster(data, lower_left_corner, cell_size, cell_size, -999)
        
        # Save the raster
        
        raster.save(output_path)
        print(f"Created raster: {output_path}")
