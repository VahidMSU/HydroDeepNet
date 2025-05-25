import os
import numpy as np
import geopandas as gpd
from shapely.geometry import box
import pandas as pd
import rasterio
import fiona
from rasterio.mask import mask
try:
    from sa import sa
except ImportError:
    from SWATGenX.sa import sa
# Attempt to import necessary modules, handling potential import errors



def generate_swatplus_rasters(SWATGenXPaths,
                              vpuid: str, level: str, name: str, model_name: str,
                               landuse_product: str, landuse_epoch: str,
                               ls_resolution: str, dem_resolution: str) -> None:
    """Generate SWAT+ raster files."""
    extractor = SWATplusRasterGenerator(SWATGenXPaths, vpuid, name, level, model_name,
                                         landuse_product, landuse_epoch,
                                         ls_resolution, dem_resolution)
    extractor.generate_rasters()

class SWATplusRasterGenerator:
    def __init__(self, SWATGenXPaths, vpuid: str, name: str, level: str, model_name: str,
                 landuse_product: str, landuse_epoch: str,
                 ls_resolution: str, dem_resolution: str):
        self.paths = SWATGenXPaths
        print(f"################## Generating raster files for {name} {level} {vpuid} ##################")

        # Define paths
        self.SOURCE = self.paths.construct_path(self.paths.swatgenx_outlet_path, vpuid, level, name, model_name)

        # Input paths
        self.original_landuse_path = self.paths.construct_path(self.paths.NLCD_path, f"VPUID/{vpuid}/{landuse_product}_{vpuid}_{landuse_epoch}_{ls_resolution}m.tif")
        self.original_soil_path = self.paths.construct_path(self.paths.gSSURGO_path, f"VPUID/{vpuid}/gSSURGO_{vpuid}_{ls_resolution}m.tif")
        self.original_dem_path = self.paths.construct_path(self.paths.DEM_path, "VPUID", vpuid)

        # Output paths
        self.swatplus_shapes_path = os.path.join(self.SOURCE, "Watershed/Shapes/")
        self.swatplus_landuse_path = os.path.join(self.SOURCE, "Watershed/Rasters/Landuse/")
        self.swatplus_dem_input = os.path.join(self.SOURCE, "Watershed/Rasters/DEM/")
        self.swatplus_soil_input = os.path.join(self.SOURCE, "Watershed/Rasters/Soil/")
        self.swatplus_landuse_output = os.path.join(self.swatplus_landuse_path, "landuse.tif")
        self.swatplus_soil_output = os.path.join(self.swatplus_soil_input, "soil.tif")
        self.swatplus_soil_temp = os.path.join(self.swatplus_soil_input, f"soil_{ls_resolution}m_temp.tif")
        self.swatplus_subbasins_input = os.path.join(self.swatplus_shapes_path, "SWAT_plus_subbasins.shp")
        self.watershed_boundary_path = os.path.join(self.swatplus_shapes_path, "watershed_boundary.shp")
        self.swatplus_dem_output = os.path.join(self.swatplus_dem_input, "dem.tif")

        # Store resolutions
        self.ls_resolution = ls_resolution
        self.dem_resolution = dem_resolution

    def generate_rasters(self) -> None:
        """Generate the required raster files."""
        if dem_names := [
            name
            for name in os.listdir(self.original_dem_path)
            if name.endswith(".tif") and f'{self.dem_resolution}m' in name
        ]:
            self.original_dem_path = os.path.join(self.original_dem_path, dem_names[0])
        else:
            raise FileNotFoundError(f"No DEM file found for resolution {self.dem_resolution}m.")

        # Create necessary directories
        os.makedirs(self.swatplus_landuse_path, exist_ok=True)
        os.makedirs(self.swatplus_dem_input, exist_ok=True)
        os.makedirs(self.swatplus_soil_input, exist_ok=True)

        # Create the watershed boundary
        self.create_watershed_boundary()

        # Perform spatial analysis
        spatial_analysis = sa()
        self.extract_and_save_rasters(spatial_analysis)

        # Process soil raster
        self.process_soil_raster()

        self.process_landuse_raster()

        print("################## Finished generating raster ##################")


    def replace_invalid_landuse_cells(self, landuse_array: np.ndarray) -> int:
        """Replace invalid landuse cells and return the most common value."""
        unique, counts = np.unique(landuse_array[landuse_array != 0], return_counts=True)
        most_common_value = unique[np.argmax(counts)]
        print(f"Most common value: {most_common_value}")

        # Replace invalid cells
        landuse_array = np.where(landuse_array == 0, most_common_value, landuse_array)
        return most_common_value, landuse_array

    def process_landuse_raster(self) -> None:
        """Process the landuse raster to replace invalid cells."""
        with rasterio.open(self.swatplus_landuse_output) as src:
            landuse_array = src.read(1)

        # Find the most common value in the landuse array
        most_common_value, landuse_array = self.replace_invalid_landuse_cells(landuse_array)

        # Write the updated landuse_array back to a raster
        self.write_updated_landuse_raster(landuse_array, most_common_value)

    def write_updated_landuse_raster(self, landuse_array: np.ndarray, most_common_value: int) -> None:
        """Write the updated landuse array back to a raster."""
        with rasterio.open(self.swatplus_landuse_output) as src:
            transform = src.transform
            crs = src.crs
            profile = src.profile
            profile.update(dtype=rasterio.uint8, compress='lzw')

        with rasterio.open(self.swatplus_landuse_output, 'w', **profile) as dst:
            dst.write(landuse_array, 1)

    def create_watershed_boundary(self) -> None:
        """Create and save the watershed boundary."""
        gdf = gpd.read_file(self.swatplus_subbasins_input)
        xmin, ymin, xmax, ymax = gdf.total_bounds
        watershed_boundary = box(xmin, ymin, xmax, ymax)
        watershed_boundary = gpd.GeoDataFrame(geometry=[watershed_boundary], crs=gdf.crs).buffer(250)
        watershed_boundary.to_file(self.watershed_boundary_path)

    def extract_and_save_rasters(self, spatial_analysis) -> None:
        """Extract and save DEM, landuse, and soil rasters."""
        # Read original DEM to get just the elevation band
        with rasterio.open(self.original_dem_path) as src:
            dem_data = src.read(1)  # Read only first band (elevation)
            profile = src.profile.copy()
            profile.update(count=1, nodata=-9999)  # Ensure output has only one band and proper nodata value
            
            # Create temporary single-band DEM
            temp_dem = os.path.join(self.swatplus_dem_input, "temp_dem.tif")
            with rasterio.open(temp_dem, 'w', **profile) as dst:
                dst.write(dem_data, 1)

        # Extract and save DEM using the single-band temp file
        dem_raster = spatial_analysis.ExtractByMask(temp_dem, self.watershed_boundary_path)
        dem_raster.save(self.swatplus_dem_output)

        # Validate the saved DEM
        with rasterio.open(self.swatplus_dem_output) as src:
            print(f"DEM validation - Bands: {src.count}, Shape: {src.shape}, CRS: {src.crs}")
            profile = src.profile
            profile.update(count=1, nodata=-9999)  # Ensure single band and proper nodata
            dem_data = src.read(1)
            
            # Save with validated settings
            with rasterio.open(self.swatplus_dem_output, 'w', **profile) as dst:
                dst.write(dem_data, 1)

        # Clean up temp file
        if os.path.exists(temp_dem):
            os.remove(temp_dem)

        # Extract and save Landuse
        landuse_raster = spatial_analysis.ExtractByMask(self.original_landuse_path, self.watershed_boundary_path)
        landuse_raster.save(self.swatplus_landuse_output)

        # Extract and save gSSURGO
        soil_raster = spatial_analysis.ExtractByMask(self.original_soil_path, self.watershed_boundary_path)
        soil_raster.save(self.swatplus_soil_output)

        # Snap rasters
        spatial_analysis.snap_rasters(self.swatplus_landuse_output, self.swatplus_soil_output)



    def process_soil_raster(self) -> None:
        """Process the soil raster to replace invalid cells."""
        with rasterio.open(self.swatplus_soil_output) as src:
            soil_array = src.read(1)
            nodata_value = src.nodata if src.nodata is not None else 2147483647
            soil_array[soil_array == nodata_value] = nodata_value

        # Find the most common value in the soil array
        most_common_value, soil_array = self.replace_invalid_soil_cells(soil_array)

        # Write the updated soil_array back to a raster
        self.write_updated_soil_raster(soil_array, most_common_value)

    def replace_invalid_soil_cells(self, soil_array: np.ndarray) -> int:
        """Replace invalid soil cells and return the most common value."""
        unique, counts = np.unique(soil_array[soil_array != 2147483647], return_counts=True)

        df = pd.read_csv(self.paths.construct_path(self.paths.swatplus_gssurgo_csv))
        mask_value = df.muid.values.astype(int)
        ### remove the uniques that are not in the mask
        mask = np.isin(unique, mask_value)
        unique = unique[mask]
        counts = counts[mask]

        ### if unique is not no
        assert len(unique) > 2, "No valid soil cells found in the raster."
        most_common_value = unique[np.argmax(counts)]

        print(f"Most common value: {most_common_value}")

        # Replace invalid cells
        soil_array = np.where(soil_array == 2147483647, most_common_value, soil_array)
        soil_array = np.where(soil_array == 0, most_common_value, soil_array)
        
        soil_array = np.where(np.isin(soil_array, mask_value), soil_array, most_common_value)

        return most_common_value, soil_array

    def write_updated_soil_raster(self, soil_array: np.ndarray, most_common_value: int) -> None:
        """Write the updated soil array back to a raster."""
        with rasterio.open(self.swatplus_soil_output) as src:
            transform = src.transform
            crs = src.crs
            profile = src.profile
            profile.update(dtype=rasterio.uint32, compress='lzw')

        with rasterio.open(self.swatplus_soil_temp, 'w', **profile) as dst:
            dst.write(soil_array, 1)

        # Clip the temp soil raster to watershed boundary and save it
        with fiona.open(self.watershed_boundary_path, "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]
            with rasterio.open(self.swatplus_soil_temp) as src:
                out_image, out_transform = mask(src, shapes, crop=True)
                out_meta = src.meta.copy()
                out_meta.update({"driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform})

            with rasterio.open(self.swatplus_soil_output, "w", **out_meta) as dest:
                dest.write(out_image)

        # Delete the temp soil raster
        if os.path.exists(self.swatplus_soil_temp):
            os.remove(self.swatplus_soil_temp)

if __name__ == "__main__":
    VPUID = "0410"
    LEVEL = "huc8"
    NAME = "04100013"
    landuse_product = "NLCD"
    landuse_epoch = "2021"
    ls_resolution = "250"
    dem_resolution = "30"
    MODEL_NAME = "SWAT_MODEL"
    generate_swatplus_rasters(VPUID, NAME, LEVEL, MODEL_NAME, landuse_product, landuse_epoch, ls_resolution, dem_resolution)
