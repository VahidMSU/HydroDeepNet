
import arcpy
import os
import numpy as np
import itertools
import os
from datetime import datetime, timedelta
import arcpy
import os
import numpy as np
from datetime import datetime, timedelta
from mpi4py import MPI

def create_dummy_raster(reference_raster, output_path, var, year, month, day):
    arcpy.env.workspace = output_path
    arcpy.env.extent = reference_raster
    arcpy.env.snapRaster = reference_raster
    arcpy.env.cellSize = reference_raster
    arcpy.env.outputCoordinateSystem = reference_raster
    arcpy.env.overwriteOutput = True

    # Create a new raster with the same properties as the reference raster
    ref_desc = arcpy.Describe(reference_raster)
    cell_size = ref_desc.meanCellWidth

    # Get the number of rows and columns from the reference raster
    rows = int((ref_desc.extent.YMax - ref_desc.extent.YMin) / cell_size)
    cols = int((ref_desc.extent.XMax - ref_desc.extent.XMin) / cell_size)

    # Create an empty numpy array with the shape of the reference raster and fill it with -999
    array = np.full((rows, cols), -999, dtype=np.float32)

    # Convert the numpy array to a raster object
    dummy_raster = arcpy.NumPyArrayToRaster(array, lower_left_corner=arcpy.Point(ref_desc.extent.XMin, ref_desc.extent.YMin), x_cell_size=cell_size, y_cell_size=cell_size, value_to_nodata=-999)

    # Save the dummy raster
    dummy_raster_path = os.path.join(output_path, f"PRISM_{var}_stable_4kmD2_{year}{month}{day}_bil.tif")
    dummy_raster.save(dummy_raster_path)

    # Copy the raster to .bil format
    arcpy.Copy_management(
        dummy_raster_path,
        f"{output_path}/PRISM_{var}_stable_4kmD2_{year}{month}{day}_bil.bil"
    )
    arcpy.Delete_management(dummy_raster_path)

def clip_PRISM(var, year, month, day, reference_raster, output_path):
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f"{output_path}{var}", exist_ok=True)
    os.makedirs(f"{output_path}{var}/{year}", exist_ok=True)
    PRISM_path =  f"/data/MyDataBase/SWATGenXAppData/PRISM/unzipped_daily/{var}/{year}/PRISM_{var}_stable_4kmD2_{year}{month}{day}_bil/PRISM_{var}_stable_4kmD2_{year}{month}{day}_bil.bil"
    ### if PRISM raster does not exist, create a raster with the size of the reference raster with no data value of -999
    if not os.path.exists(PRISM_path):
        create_dummy_raster(reference_raster, output_path, var, year, month, day)
    ### mask PRISM raster by reference raster with no change in cell size of the PRISM raster
    arcpy.env.workspace = f"/data/MyDataBase/SWATGenXAppData/PRISM/unzipped_daily/{var}/{year}/PRISM_{var}_stable_4kmD2_{year}{month}{day}_bil"
    arcpy.env.extent = reference_raster
    arcpy.env.overwriteOutput = True
    arcpy.gp.ExtractByMask_sa(f"PRISM_{var}_stable_4kmD2_{year}{month}{day}_bil.bil", reference_raster, f"{output_path}PRISM_{var}_stable_4kmD2_{year}{month}{day}_bil.tif")
    ### now project to the reference raster without changing the cell size
    arcpy.env.workspace = output_path
    arcpy.env.extent = reference_raster
    arcpy.env.overwriteOutput = True
    arcpy.ProjectRaster_management(f"PRISM_{var}_stable_4kmD2_{year}{month}{day}_bil.tif", f"{output_path}PRISM_{var}_stable_4kmD2_{year}{month}{day}_bil_projected.tif", reference_raster)
    ### resample to the reference raster and make sure width and height are the same
    arcpy.env.workspace = output_path
    arcpy.env.extent = reference_raster
    arcpy.env.cellSize = reference_raster
    arcpy.env.overwriteOutput = True
    arcpy.Resample_management(f"PRISM_{var}_stable_4kmD2_{year}{month}{day}_bil_projected.tif", f"{output_path}PRISM_{var}_stable_4kmD2_{year}{month}{day}_bil_resampled.tif", "250 250", "NEAREST")
    ### now clip the resampled raster to the reference raster
    arcpy.env.workspace = output_path
    arcpy.env.extent = reference_raster
    arcpy.env.overwriteOutput = True
    arcpy.env.snapRaster = reference_raster
    arcpy.gp.ExtractByMask_sa(f"PRISM_{var}_stable_4kmD2_{year}{month}{day}_bil_resampled.tif", reference_raster, f"{output_path}PRISM_{var}_stable_4kmD2_{year}{month}{day}_bil_resampled_clipped.tif")
    ## now move it to {output_path}{var}{year}
    arcpy.Copy_management( f"{output_path}PRISM_{var}_stable_4kmD2_{year}{month}{day}_bil_resampled_clipped.tif", f"{output_path}{var}/{year}/PRISM_{var}_stable_250m_{year}{month}{day}_ML.tif")
    ## now remove all other rasters except the original
    arcpy.env.workspace = output_path
    arcpy.Delete_management(f"PRISM_{var}_stable_4kmD2_{year}{month}{day}_bil.tif")
    arcpy.Delete_management(f"PRISM_{var}_stable_4kmD2_{year}{month}{day}_bil_projected.tif")
    arcpy.Delete_management(f"PRISM_{var}_stable_4kmD2_{year}{month}{day}_bil_resampled.tif")
    arcpy.Delete_management(f"PRISM_{var}_stable_4kmD2_{year}{month}{day}_bil_resampled_clipped.tif")


def process_day(var, year, month, day, reference_raster, output_path):
    if not os.path.exists(f"{output_path}{var}/{year}/PRISM_{var}_stable_250m_{year}{month:02d}{day:02d}_ML.tif"):
        clip_PRISM(var, year, f"{month:02d}", f"{day:02d}", reference_raster, output_path)

def generate_tasks(reference_raster, output_path):
    tasks = []
    years = range(1990, 2023)  # Adjust the end year as needed
    for var in ["ppt", "tmax", "tmin"]:
        for year in years:
            for month in range(1, 13):
                start_date = datetime(year, month, 1)
                while start_date.month == month:
                    day = start_date.day
                    tasks.append((var, year, month, day, reference_raster, output_path))
                    start_date += timedelta(days=1)
    return tasks



if __name__ == "__main__":
    reference_raster = "/data/MyDataBase/SWATGenXAppData/all_rasters/DEM_250m.tif"
    output_path = "E:/PRISM/"
    single = False
    ## NOTE: run this code with MPI:  mpiexec -n 100 python 1_read_and_clip.py
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if single:
        var = "ppt"
        year = 1990
        month = 2
        day = 28
        clip_PRISM(var, year, f"{month:02d}", f"{day:02d}", reference_raster, output_path)
    else:
        tasks = generate_tasks(reference_raster, output_path)
        chunk_size = len(tasks) // size
        start = rank * chunk_size
        end = start + chunk_size if rank != size - 1 else len(tasks)

        for task in tasks[start:end]:
            process_day(*task)
