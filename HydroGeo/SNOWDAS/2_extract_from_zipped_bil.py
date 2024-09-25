import arcpy
from arcpy.sa import Raster, Con
import tarfile
import gzip
import shutil
import os
import numpy as np
import glob
from tqdm import tqdm
from multiprocessing import Process, Lock, cpu_count
from get_extent import get_lat_lon_reference_raster

import numpy as np
import rasterio

def read_geotiff_metadata(meta_data_path):
    # sourcery skip: low-code-quality
    """
    Read metadata from a GeoTIFF metadata file.

    Parameters:
    meta_data_path (str): Path to the metadata file.

    Returns:
    dict: A dictionary containing the metadata information.
    """
    geoTIFF_metadata = {}
    with open(meta_data_path, 'r') as f:
        meta = f.readlines()
        for line in meta:
            if 'Description' in line:
                geoTIFF_metadata['variable_name'] = line.split(':')[1].strip()
            if 'Number of columns' in line:
                geoTIFF_metadata['num_columns'] = int(line.split(':')[1].strip())
            elif 'Number of rows' in line:
                geoTIFF_metadata['num_rows'] = int(line.split(':')[1].strip())
            elif 'No data value' in line:
                geoTIFF_metadata['no_data_value'] = float(line.split(':')[1].strip())
            elif 'Minimum x-axis coordinate' in line:
                geoTIFF_metadata['x_min'] = float(line.split(':')[1].strip())
            elif 'Maximum y-axis coordinate' in line:
                geoTIFF_metadata['y_max'] = float(line.split(':')[1].strip())
            elif 'X-axis resolution' in line:
                geoTIFF_metadata['x_res'] = float(line.split(':')[1].strip())
            elif 'Y-axis resolution' in line:
                geoTIFF_metadata['y_res'] = -float(line.split(':')[1].strip())
            elif 'X-axis offset' in line:
                geoTIFF_metadata['x_skew'] = float(line.split(':')[1].strip())
            elif 'Y-axis offset' in line:
                geoTIFF_metadata['y_skew'] = float(line.split(':')[1].strip())
            elif 'Start year' in line:
                geoTIFF_metadata['year'] = int(line.split(':')[1].strip())
            elif 'Start month' in line:
                geoTIFF_metadata['month'] = int(line.split(':')[1].strip())
            elif 'Start day' in line:
                geoTIFF_metadata['day'] = int(line.split(':')[1].strip())
            elif 'Minimum data value' in line:
                geoTIFF_metadata['min'] = float(line.split(':')[1].strip())
            elif 'Maximum data value' in line:
                geoTIFF_metadata['max'] = float(line.split(':')[1].strip())
    return geoTIFF_metadata


class SnowDataProcessor:
    """
    A class to process snow data files, including extracting, converting, and clipping raster data.
    """

    def __init__(self, base_directory, subset_name, reference_raster, temp_directory, n_processes=None):
        """
        Initialize the SnowDataProcessor with given parameters.

        Parameters:
        base_directory (str): The base directory containing the data files.
        subset_name (str): The name of the subset for processing.
        reference_raster (str): Path to the reference raster file.
        temp_directory (str): Directory to use for temporary workspaces.
        n_processes (int): Number of processes to use for parallel processing.
        """
        self.base_directory = base_directory
        self.subset_name = subset_name
        self.target_dir = os.path.join(base_directory, subset_name)
        self.reference_raster = reference_raster
        self.temp_directory = temp_directory
        self.n_processes = n_processes or cpu_count()
        self.get_rectangle(reference_raster)
        self.lock = Lock()
        self.clean_up()


    def clean_up(self):
        """
        Clean up the temporary directory.
        """
        if os.path.exists(self.temp_directory):
            shutil.rmtree(self.temp_directory)
        os.makedirs(self.temp_directory, exist_ok=True)
        if os.path.exists(self.target_dir):
            shutil.rmtree(self.target_dir)
        os.makedirs(self.target_dir, exist_ok=True)


    def extract_gz_files(self, workspace):
        """
        Extract all .gz files in the workspace.

        Parameters:
        workspace (str): Path to the workspace directory.
        """
        file_paths = glob.glob(f'{workspace}/*.gz')
        for file_path in file_paths:
            if os.path.exists(file_path):
                new_file_path = os.path.join(workspace, os.path.basename(file_path[:-3]))
                with gzip.open(file_path, 'rb') as f_in:
                    with open(new_file_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(file_path)

    def rename_dat_to_bil(self, file, workspace):
        """
        Rename a .dat file to a .bil file.

        Parameters:
        file (str): The name of the .dat file.
        workspace (str): Path to the workspace directory.
        """
        new_file_path = os.path.join(workspace, f"{file[:-4]}.bil")
        os.rename(os.path.join(workspace, file), new_file_path)

    def remove_files(self, files_to_remove, workspace):
        """
        Remove specified files from the workspace.

        Parameters:
        files_to_remove (list): List of file names to remove.
        workspace (str): Path to the workspace directory.
        """
        for file in files_to_remove:
            file_path = os.path.join(workspace, file)
            if os.path.exists(file_path):
                arcpy.Delete_management(file_path)


    def get_rectangle(self, reference_raster):
        min_lat, min_lon, max_lat, max_lon = get_lat_lon_reference_raster(self.reference_raster)
        x_min, y_min, x_max, y_max = min_lat, min_lon, max_lat, max_lon
        self.rectangle = f"{x_min} {y_min} {x_max} {y_max}"
    def process_snowdas_file(self, tar_file_path):
        """
        Process a SnowDAS tar file: extract, convert, clip, and project raster data.

        Parameters:
        tar_file_path (str): Path to the tar file to process.
        """


        workspace = os.path.join(self.temp_directory, os.path.basename(tar_file_path).replace('.tar', ''))

        os.makedirs(workspace, exist_ok=True)

        try:
            with tarfile.open(tar_file_path, "r:") as tar:
                tar.extractall(path=workspace)

            self.extract_gz_files(workspace)

            files_to_remove = []
            for file in os.listdir(workspace):
                if file.endswith(".dat"):
                    print(f"Processing file: {file}")
                    file_path = os.path.join(workspace, file)
                    if not os.path.exists(file_path):
                        continue

                    with self.lock:
                        self.rename_dat_to_bil(file, workspace)

                    bil_file_path = f'{file_path[:-4]}.bil'
                    meta_data_path = f'{file_path[:-4]}.hdr'

                    with open(meta_data_path, 'w') as f:
                        f.write('byteorder M\nlayout bil\nnbands 1\nnbits 16\nncols 6935\nnrows 3351\nulxmap -124.729583333331703\nulymap 52.871249516804028\nxdim 0.0083333333\nydim 0.00833333333\n')

                    meta_data_path = f'{file_path[:-4]}.txt'
                    geoTIFF_metadata = read_geotiff_metadata(meta_data_path)

                    day = geoTIFF_metadata['day']
                    month = geoTIFF_metadata['month']
                    year = geoTIFF_metadata['year']

                    variable_name = geoTIFF_metadata['variable_name'].split(',')[0].replace('-', '_').replace(' ', '_').replace('\n', '')

                    out_geotif_path = os.path.join(self.target_dir, f'{year}', f'{month}', f'{day}', f'{variable_name}.tif')
                    os.makedirs(os.path.dirname(out_geotif_path), exist_ok=True)
                    arcpy.env.overwriteOutput = True
                    arcpy.env.outputCoordinateSystem = arcpy.SpatialReference(4326)

                    clipped_raster_path = out_geotif_path

                    arcpy.Clip_management(bil_file_path, self.rectangle, clipped_raster_path, nodata_value=geoTIFF_metadata['no_data_value'])


        except Exception as e:
            print(f"Error processing file {tar_file_path}: {e}")
    def worker(self, tar_files):
        """

        Worker function to process files from the given list.

        Parameters:
        tar_files (list): List of tar files to process.
        """
        for tar_file_path in tar_files:
            self.process_snowdas_file(tar_file_path)

    def process_all_tar_files(self):
        """
        Process all tar files in the base directory.
        """
        tar_file_paths = glob.glob(f'{self.base_directory}*.tar')
        print(f'Number of tar files: {len(tar_file_paths)}')

        chunk_size = 180
        processes = []

        for i in range(self.n_processes):
            start = i * chunk_size
            end = None if i == self.n_processes - 1 else (i + 1) * chunk_size
            tar_files_chunk = tar_file_paths[start:end]

            p = Process(target=self.worker, args=(tar_files_chunk,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

if __name__ == '__main__':
    base_directory = '/data/MyDataBase/SWATGenXAppData/snow/snow/'
    subset_name = 'michigan'
    reference_raster = "/data/MyDataBase/SWATGenXAppData/all_rasters/DEM_250m.tif"
    temp_directory = "/data/MyDataBase/SWATGenXAppData/temp_workspaces/"
    snow_processor = SnowDataProcessor(base_directory, subset_name, reference_raster, temp_directory)
    snow_processor.process_all_tar_files()
