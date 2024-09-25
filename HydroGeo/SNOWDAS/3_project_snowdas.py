import arcpy
import os
import itertools
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Process, Queue, cpu_count

class SnowDataProcessor:
    def __init__(self, base_path, reference_raster):
        self.base_path = base_path
        self.reference_raster = reference_raster
        self.spatial_ref_name = arcpy.Describe(reference_raster).spatialReference.name
        arcpy.env.overwriteOutput = True

    def delete_files(self, path):
        if not os.path.exists(path):
            print(f"Path {path} does not exist")
            return

        files = os.listdir(path)
        files_path = [os.path.join(path, f) for f in files]
        for i in ["EPSG", "250m", "_resample", "_EPSG26990"]:
            filtered_files = [file for file in files_path if file.endswith(".tif") and i in file]
            for file in filtered_files:
                arcpy.Delete_management(file)
                print(f"Deleted {file}")

    def cleanup(self):
        paths_to_process = []
        for year, month, day in itertools.product(range(2004, 2024), range(1, 13), range(1, 32)):
            path = os.path.join(self.base_path, str(year), str(month), str(day))
            paths_to_process.append(path)

        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = {executor.submit(self.delete_files, path) for path in paths_to_process}

            for future in as_completed(futures):
                future.result()  # To raise exceptions if any

    def start_analysis(self, file):
        ## check if the file exists
        output_file = self.project_raster(file)
        return self.resample_raster(output_file)

    def project_raster(self, file):
        arcpy.env.overwriteOutput = False
        ### if exists, return the file
        if os.path.exists(file.replace(".tif", "_EPSG26990.tif")):
            return file.replace(".tif", "_EPSG26990.tif")
        arcpy.env.extent = arcpy.Describe(self.reference_raster).extent
        output_file = file.replace(".tif", "_EPSG26990.tif")
        arcpy.ProjectRaster_management(file, output_file, arcpy.SpatialReference(26990), cell_size="1000")
        return output_file

    def resample_raster(self, output_file):
        # Resample the raster to 250m resolution
        arcpy.env.overwriteOutput = False
        if os.path.exists(output_file.replace("_EPSG26990.tif", "_resample.tif")):
            return output_file.replace("_EPSG26990.tif", "_resample.tif")
        arcpy.env.snapRaster = self.reference_raster
        arcpy.env.cellSize = self.reference_raster
        arcpy.env.extent = arcpy.Describe(self.reference_raster).extent
        arcpy.env.outputCoordinateSystem = arcpy.SpatialReference(self.spatial_ref_name)
        resampled_file = output_file.replace("_EPSG26990.tif", "_resample.tif")
        arcpy.Resample_management(output_file, resampled_file, "250", "NEAREST")
        return resampled_file

    def worker(self, task_queue, result_queue):
        arcpy.env.workspace = os.getcwd()  # Set workspace
        while True:
            path = task_queue.get()
            if path is None:
                break
            result = self.process_day(path)
            result_queue.put(result)

    def process_day(self, path):
        if not os.path.exists(path):
            print(f"Path {path} does not exist")
            return []

        files = os.listdir(path)
        # Get the files ending with tif
        tif_files = [f for f in files if f.endswith(".tif")]
        # Create paths
        tif_files = [os.path.join(path, f) for f in tif_files]

        projected_files = []
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.start_analysis, file): file for file in tif_files}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    projected_files.append(result)
                except Exception as e:
                    print(f"Error processing {futures[future]}: {e}")

        return projected_files

    def process_month(self, year, month, task_queue):
        for day in range(1, 32):
            path = os.path.join(self.base_path, str(year), str(month), str(day))
            task_queue.put(path)

    def main(self):
        task_queue = Queue()
        result_queue = Queue()
        num_workers = 12  # Use the number of CPU cores available

        processes = []
        for _ in range(num_workers):
            p = Process(target=self.worker, args=(task_queue, result_queue))
            p.start()
            processes.append(p)

        all_projected_files = []
        for year, month in itertools.product(range(2004, 2024), range(1, 13)):
            print(f"Processing year: {year}, month: {month}")
            self.process_month(year, month, task_queue)

        # Signal the workers to exit
        for _ in range(num_workers):
            task_queue.put(None)

        for p in processes:
            p.join()

        while not result_queue.empty():
            all_projected_files.extend(result_queue.get())

        print(f"Projected files: {all_projected_files}")

if __name__ == "__main__":
    processor = SnowDataProcessor("/data/MyDataBase/SWATGenXAppData/snow/snow/michigan", "/data/MyDataBase/SWATGenXAppData/all_rasters/DEM_250m.tif")
    #processor.cleanup()
    processor.main()
