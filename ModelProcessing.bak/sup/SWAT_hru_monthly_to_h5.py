import os
import itertools
import pandas as pd
import numpy as np
import logging
from multiprocessing import Pool
import rasterio
from osgeo import ogr
import h5py
import shutil
from filelock import FileLock
import rasterio
import h5py
import shutil
import time


def generate_mask(NAME):
    """Generate a mask raster with 30m resolution."""
    path = f"/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12/{NAME}/hru_rasterized/hru_30m.tif"
    mask_raster = f"/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12/{NAME}/hru_rasterized/mask_30m.tif"
    with rasterio.open(path, 'r') as src:
        data = src.read(1)
        data = np.where(data < 0.0, -999, data)

    with rasterio.open(path, 'r') as src:
        profile = src.profile
        profile.update(nodata=0)
        with rasterio.open(mask_raster, 'w', **profile) as dst:
            dst.write(np.where(data < 0.0, 0, 1), 1)
            logging.info(f"Mask raster saved at: {mask_raster}")


def run_task(args):
    """Run a task with locking to avoid file conflicts."""

    generator = HydroGenerator(args)
    generator.process_NAME_VER()


class HydroGenerator:
    def __init__(self, args):
        self.NAME = args["NAME"]
        self.ver = args["ver"]
        self.h5_path = f"/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12/{self.NAME}/SWAT_gwflow_MODEL/Scenarios/verification_stage_{self.ver}/SWATplus_output.h5"
        self.hru_wb_mon_txt = f"/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12/{self.NAME}/SWAT_gwflow_MODEL/Scenarios/verification_stage_{self.ver}/hru_wb_mon.txt"
        self.variable_name = args["variable_name"]
        self.hru_shape_path = f"/data2/MyDataBase/SWATGenXAppData/SWAT_input/huc12/{self.NAME}/SWAT_MODEL/Watershed/Shapes/hrus2.shp"
        self.watershed_boundary_path = f"/data2/MyDataBase/SWATGenXAppData/SWAT_input/huc12/{self.NAME}/SWAT_MODEL/Watershed/Shapes/watershed_boundary.shp"
        self.reference_raster_path = f"/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12/{self.NAME}/hru_rasterized/hru_30m.tif"
        self.mask_raster_path = f"/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12/{self.NAME}/hru_rasterized/mask_30m.tif"
        self.start_year = 2000
        self.end_year = 2020
        self.chunksize = 100000

    def read_hru_wb_mon_txt(self):
        if not os.path.exists(self.hru_wb_mon_txt):
            raise FileNotFoundError(f"File not found: {self.hru_wb_mon_txt}")

        with open(self.hru_wb_mon_txt, 'r') as f:
            columns = f.readlines()[1].split()

        dtype = {
            'gis_id': 'int32',
            self.variable_name: 'float32',
            'yr': 'int16',
            'mon': 'int8'
        }

        chunks = pd.read_csv(self.hru_wb_mon_txt,
                             sep='\s+',
                             names=columns,
                             skiprows=3,
                             chunksize=self.chunksize,
                             dtype=dtype,
                             low_memory=False)

        return pd.concat(chunks)[['gis_id', self.variable_name, 'yr', 'mon']]

    def generate_reference_raster(self):
        ## generate a check point for that other processes not to generate the same raster
        
        if os.path.exists(self.reference_raster_path) and os.path.exists(self.mask_raster_path):
            logging.info(f"Reference raster and mask already exist for {self.NAME}")
            return self.reference_raster_path, self.mask_raster_path

        driver = ogr.GetDriverByName("ESRI Shapefile")
        dataSource = driver.Open(self.watershed_boundary_path, 0)
        layer = dataSource.GetLayer()
        xmin, xmax, ymin, ymax = layer.GetExtent()

        os.makedirs(os.path.dirname(self.reference_raster_path), exist_ok=True)

        gdal_command_rasterize = [
            "gdal_rasterize",
            "-a", "HRUS",
            "-tr", "30", "30",
            f"-te {xmin} {ymin} {xmax} {ymax}",
            "-ot", "Int32",
            "-of", "GTiff",
            "-a_nodata", "-999",
            self.hru_shape_path,
            self.reference_raster_path
        ]

        os.system(" ".join(gdal_command_rasterize))

        generate_mask(self.NAME)

        return self.reference_raster_path, self.mask_raster_path

    def create_gis_id_mapping(self, src_array, df):
        gis_id_to_perc = np.full(np.max(src_array) + 1, np.nan, dtype=np.float32)
        for gis_id, value in zip(df['gis_id'], df[self.variable_name]):
            gis_id_to_perc[gis_id] = value
        return gis_id_to_perc

    def merge_with_mapping(self, src_array, gis_id_to_perc, mask_array, no_value):
        # Ensure src_array contains only valid positive indices
        sanitized_array = np.where(src_array > 0, src_array, 0)  # Replace negatives/zeros with 0
        result_array = np.where(mask_array == 1, gis_id_to_perc[sanitized_array], no_value)
        result_array[result_array == no_value] = np.nan

        return result_array


    def process_file_to_h5(self, h5_file, dataset_path, src_array_revised, mask_array, no_value):
        if dataset_path not in h5_file:
            height, width = src_array_revised.shape
            h5_file.create_dataset(
                dataset_path, 
                (height, width), 
                dtype='f4', 
                fillvalue=no_value, 
                compression="gzip",  # Use gzip compression
                compression_opts=9   # Highest compression level
            )

        h5_file[dataset_path][:, :] = np.where(mask_array == 1, src_array_revised, no_value)
    def process_chunk(self, src_array, mask_array, df_filtered, no_value, yr, mon):   
        # Create gis_id_mapping only for the filtered data
        gis_id_mapping = self.create_gis_id_mapping(src_array, df_filtered)

        # Use the filtered data to generate merged_array
        merged_array = self.merge_with_mapping(src_array, gis_id_mapping, mask_array, no_value)
        with FileLock(self.h5_path + ".lock"):   
        ## if the File is not locked then process the file othersie the code will wait until the file is unlocked
        ### open the h5 file and write the data
            with h5py.File(self.h5_path, 'a', libver='latest') as h5_file:
        # Write the data to the HDF5 file
                self.process_file_to_h5(h5_file, f"/hru_wb_30m/{yr}/{mon}/{self.variable_name}", merged_array, mask_array, no_value)

    def process_NAME_VER(self):
        """ Process the SWAT+ output for the given NAME and VER """

        ### check if the hru_wb_mon.txt file exists
        if not os.path.exists(self.hru_wb_mon_txt):
            logging.warning(f"File not found: {self.hru_wb_mon_txt}")
            return

        df = self.read_hru_wb_mon_txt()
        reference_raster_path, mask_raster_path = self.generate_reference_raster()

        ## check if the reference raster and mask raster exist
        if not os.path.exists(reference_raster_path) or not os.path.exists(mask_raster_path):
            logging.error(f"Reference raster or mask not found for {self.NAME}")
            return

        ### read the reference raster and mask raster
        with rasterio.open(reference_raster_path) as ref_src, rasterio.open(mask_raster_path) as mask_src:
            src_array = ref_src.read(1)
            mask_array = mask_src.read(1)
            no_value = ref_src.nodata or -999

        for yr, mon in itertools.product(range(self.start_year, self.end_year), range(1, 13)):
            df_filtered = df[(df['yr'] == yr) & (df['mon'] == mon)]
            print(f"Processing {self.NAME} verification stage {self.ver} for {self.variable_name} year {yr} month {mon} with range {df_filtered[self.variable_name].min():.2f} to {df_filtered[self.variable_name].max():.2f}")
            if df_filtered.empty:
                print(f"No data found for {yr}/{mon}")
                continue

            self.process_chunk(src_array, mask_array, df_filtered, no_value, yr, mon)

        print(f"Finished processing {self.NAME} verification stage {self.ver} for {self.variable_name} final file path {self.h5_path}")

def cleanup_swath5():

    """ 
    Remove the hru_rasterized folder and SWATplus_output.h5 files in the verification stage folders
    
    """
    
    NAMES = os.listdir("/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12")
    NAMES.remove("log.txt")

    for NAME in NAMES:
        path = f"/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12/{NAME}/hru_rasterized"
        if os.path.exists(path):
            shutil.rmtree(path) 
        for ver in range(0, 6):
            path = f"/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12/{NAME}/SWAT_gwflow_MODEL/Scenarios/verification_stage_{ver}/SWATplus_output.h5"
            if os.path.exists(path):
                os.remove(path)

if __name__ == "__main__":
    """" 
    
    Here we first remove if the is any SWATPlus output h5 in the verification stage folders 
    
    and then we add SWATplus hru monthly output to the SWATPlus_output.h5 file
    
    
    """

    #cleanup_swath5()
    
    num_workers = 100
    
    NAMES = [name for name in os.listdir("/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12") if name != "log.txt"]
    vers = [0, 1, 2, 3, 4, 5]

    test = False

    if test:
        NAMES = ['04136000']
        vers = [0]

    
    
    #  precip     snofall      snomlt    surq_gen        latq    wateryld        perc          et     ecanopy      eplant       esoil   surq_cont
    #, 
    variable_names = ["perc", "snofall", "precip", "surq_gen", "et" ,"wateryld", "snomlt"]

    for variable_name in variable_names:
        args_list = [
            {
            "NAME": NAME,
            "ver": ver, 
            "variable_name": variable_name
                }
            for NAME, ver in itertools.product(NAMES, vers)
        ]

        with Pool(num_workers) as pool:
            pool.map(run_task, args_list)

    print("Finished processing all tasks.")
