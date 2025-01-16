import contextlib
import rasterio
import os
from rasterio.mask import mask
from shapely.geometry import mapping
import fiona
import geopandas as gpd
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.warp import transform_geom
from osgeo import gdal
from rasterio.windows import Window
from shapely.geometry import shape, mapping
from shapely.affinity import scale
from shapely.geometry import box
def align_rasters(landuse, dem, soil):
    """Align rasters to the smallest common shape and save them."""

    assert os.path.exists(landuse), f"Landuse raster not found: {landuse}"
    assert os.path.exists(dem), f"DEM raster not found: {dem}"
    assert os.path.exists(soil), f"Soil raster not found: {soil}"

    def get_valid_bounds(raster_path):
        with rasterio.open(raster_path, 'r') as src:
            data = src.read(1)
            mask = data != src.nodata
            rows, cols = mask.any(axis=1), mask.any(axis=0)
            row_start, row_end = rows.argmax(), len(rows) - rows[::-1].argmax()
            col_start, col_end = cols.argmax(), len(cols) - cols[::-1].argmax()
            return row_start, row_end, col_start, col_end

    # Get bounds for each raster
    landuse_bounds = get_valid_bounds(landuse)
    dem_bounds = get_valid_bounds(dem)
    soil_bounds = get_valid_bounds(soil)

    # Determine the smallest common shape
    min_row_start = max(landuse_bounds[0], dem_bounds[0], soil_bounds[0])
    min_row_end = min(landuse_bounds[1], dem_bounds[1], soil_bounds[1])
    min_col_start = max(landuse_bounds[2], dem_bounds[2], soil_bounds[2])
    min_col_end = min(landuse_bounds[3], dem_bounds[3], soil_bounds[3])

    # Clip all rasters to this size
    common_window = Window(
        col_off=min_col_start,
        row_off=min_row_start,
        width=min_col_end - min_col_start,
        height=min_row_end - min_row_start,
    )

    # Clip and save rasters
    output_dir = "/data/SWATGenXApp/codes/clipped_rasters"
    os.makedirs(output_dir, exist_ok=True)

    for raster_path, name in zip([landuse, dem, soil], ["landuse", "dem", "soil"]):
        with rasterio.open(raster_path, 'r') as src:
            profile = src.profile
            profile.update({
                'height': common_window.height,
                'width': common_window.width,
                'transform': src.window_transform(common_window)
            })
            clipped_path = os.path.join(output_dir, f"{name}_clipped.tif")
            with rasterio.open(clipped_path, 'w', **profile) as dst:
                dst.write(src.read(1, window=common_window), 1)
        print(f"Clipped {name} raster saved to {clipped_path}")

    ### now copy to the original location with the original name
    os.system(f"mv {output_dir}/landuse_clipped.tif {landuse}")
    os.system(f"mv {output_dir}/dem_clipped.tif {dem}")
    os.system(f"mv {output_dir}/soil_clipped.tif {soil}")


# Step 2: Extract metadata from the reference raster
def get_raster_metadata(raster_path):
    """Extract resolution and extent from a raster."""
    print(f"Extracting metadata from raster: {raster_path}")
    ds = gdal.Open(raster_path)
    if not ds:
        raise RuntimeError(f"Failed to open raster: {raster_path}")
    
    transform = ds.GetGeoTransform()
    xres, yres = transform[1], transform[5]
    xmin, ymax = transform[0], transform[3]
    xmax = xmin + (ds.RasterXSize * xres)
    ymin = ymax + (ds.RasterYSize * yres)
    
    ds = None  # Close dataset
    return xres, yres, xmin, ymin, xmax, ymax


# Step 3: Snap the landuse raster to the soil raster grid
def snap_raster(input_path, output_path, xres, yres, xmin, ymin, xmax, ymax):
    """Snap a raster to a reference grid."""
    print(f"Snapping raster: {input_path} to grid")
    options = gdal.WarpOptions(
        outputBounds=(xmin, ymin, xmax, ymax),
        xRes=xres,
        yRes=abs(yres),  # yRes is negative, take absolute
        resampleAlg="near"
    )
    result = gdal.Warp(output_path, input_path, options=options)
    if not result:
        raise RuntimeError(f"Failed to snap raster: {input_path}")
    
    result.FlushCache()
    print(f"Snapped raster saved to: {output_path}")

class sa:
    def __init__(self, in_raster=None, in_mask_data=None):
        self.env = self.Env()
        self.in_raster = in_raster
        self.in_mask_data = in_mask_data
        self.out_image = None
        self.out_transform = None
        self.out_meta = None
        self.workspace = None
        self.outputCoordinateSystem = None
        self.reference_raster = None

    class Env:
        def __init__(self):
            self.outputCoordinateSystem = None
            self.workspace = None

    def set_workspace(self, workspace_path):
        """Sets the workspace directory and makes it persistent."""
        if not os.path.exists(workspace_path):
            raise ValueError(f"Workspace path does not exist: {workspace_path}")
        self.env.workspace = workspace_path  # Changed from sa._workspace to self.env.workspace
        print(f"Workspace set to: {workspace_path}")

    def snap_rasters(self, input_path, reference_path):
        output_path = input_path.replace(".tif", "_snapped.tif")
        xres, yres, xmin, ymin, xmax, ymax = get_raster_metadata(reference_path)
        snap_raster(input_path, output_path, xres, yres, xmin, ymin, xmax, ymax)
        ## remove initial landuse
        os.remove(input_path)
        os.rename(output_path, input_path)
        print("Done!")


    def ListRasters(self):
        """Lists all raster datasets in the current workspace, including in a .gdb."""
        if not self.env.workspace:
            raise RuntimeError("Workspace is not set. Please set a workspace first.")

        if self.env.workspace.endswith('.gdb'):  # Check if it's a geodatabase
            print(f"Listing rasters in geodatabase: {self.env.workspace}")
            try:
                # Open the geodatabase and list all datasets
                with fiona.Env():
                    layers = fiona.listlayers(self.env.workspace)
                    rasters = []
                    for layer in layers:
                        # Try to open with rasterio to confirm it's a raster
                        with contextlib.suppress(rasterio.errors.RasterioIOError):
                            with rasterio.open(f"{self.env.workspace}/{layer}") as dataset:
                                rasters.append(layer)  # If it opens, it's a raster
                    return rasters
            except Exception as e:
                raise RuntimeError(f"Error accessing geodatabase: {e}") from e
        else:
            print(f"Listing rasters in workspace: {self.env.workspace}")
            try:
                return [raster for raster in os.listdir(self.env.workspace) if raster.endswith(('.tif', '.img'))]
            except Exception as e:
                raise RuntimeError(f"Error listing rasters: {e}") from e

    def ExtractByMask(self, in_raster, in_mask_data):
        """Extracts the cells of a raster that correspond to the areas defined by a mask, retaining all touched cells."""
        try:
            self.in_raster = in_raster
            self.in_mask_data = in_mask_data

            with rasterio.open(self.in_raster) as src:
                input_crs = src.crs

                # Check if in_mask_data is a file path (shapefile or other vector format)
                if isinstance(self.in_mask_data, str):
                    with fiona.open(self.in_mask_data, "r") as shapefile:
                        shapes = [feature["geometry"] for feature in shapefile]
                        mask_crs = shapefile.crs

                    # Reproject mask to the input raster CRS if needed
                    if mask_crs != input_crs:
                        shapes = [transform_geom(mask_crs, input_crs, shape) for shape in shapes]
                else:
                    shapes = [mapping(self.in_mask_data)]  # Direct mapping if already in correct CRS

                # Perform the masking operation
                self.out_image, self.out_transform = mask(src, shapes, crop=True)
                self.out_meta = src.meta.copy()
                self.out_meta.update({
                    "driver": "GTiff",
                    "height": self.out_image.shape[1],
                    "width": self.out_image.shape[2],
                    "transform": self.out_transform
                })
        except Exception as e:
            raise RuntimeError(f"Error during raster extraction: {e}") from e
        return self  # Enable chaining with save()


    def save(self, path):
        """Saves the extracted raster to the specified path."""
        if self.out_image is None or self.out_meta is None:
            raise RuntimeError("You must run the extract method before saving.")

        try:
            with rasterio.open(path, "w", **self.out_meta) as dest:
                dest.write(self.out_image)
        except Exception as e:
            raise RuntimeError(f"Error saving raster: {e}") from e
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject

    def Resample_management(self, in_raster, out_raster, reference_raster=None, cell_size=None, resampling_type="NEAREST"):
        """Resamples a raster dataset to a different resolution while preserving the original extent and handling CRS."""
        try:
            # Define resampling methods mapping
            resampling_map = {
                "NEAREST": Resampling.nearest,
                "BILINEAR": Resampling.bilinear,
                "CUBIC": Resampling.cubic,
                "MODE": Resampling.mode,  # Alias for majority resampling
                "MAJORITY": Resampling.mode  # Alias for majority resampling
            }

            # Check if resampling type is valid
            resampling_method = resampling_map.get(resampling_type.upper())
            if not resampling_method:
                raise ValueError(f"Unsupported resampling type: {resampling_type}")

            with rasterio.open(in_raster) as src:
                # Set the CRS and transform from the reference raster, if provided
                if reference_raster:
                    with rasterio.open(reference_raster) as ref_src:
                        dst_crs = ref_src.crs
                        ref_transform = ref_src.transform
                        ref_bounds = ref_src.bounds
                else:
                    dst_crs = src.crs  # Default to input raster CRS if no reference is given
                    ref_transform = src.transform
                    ref_bounds = src.bounds

                # Use cell size from reference raster if not explicitly provided
                if not cell_size:
                    cell_size = abs(ref_transform[0])  # Use the reference raster's cell size

                # Calculate the new transform and dimensions for the target resolution
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *ref_bounds, resolution=cell_size
                )

                # Update profile for the output raster
                profile = src.profile
                profile.update({
                    "driver": "GTiff",
                    "height": dst_height,
                    "width": dst_width,
                    "transform": dst_transform,
                    "crs": dst_crs,  # Ensure we apply the reference CRS if provided
                    "compress": "lzw",
                    "nodata": src.nodata or 0  # Ensure nodata value is propagated
                })

                # Debug: Check output profile details
                print(f"Resampling output profile: {profile}")

                # Perform the resampling and write to the output raster
                with rasterio.open(out_raster, "w", **profile) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=dst_transform,
                            dst_crs=dst_crs,
                            resampling=resampling_method
                        )

                print(f"Resampling completed successfully, output saved to: {out_raster}")

        except Exception as e:
            raise RuntimeError(f"Error during raster resampling: {e}") from e

    from rasterio.warp import calculate_default_transform, reproject, Resampling
    # ProjectRaster_management method
    def ProjectRaster_management(self, in_raster, out_raster, out_coor_system=None, reference_raster=None):
        """Projects a raster dataset into a new spatial reference, preserving the extent."""
        try:
            with rasterio.open(in_raster) as src:
                # Set the CRS from the reference raster if provided, otherwise use the output coordinate system
                if reference_raster:
                    with rasterio.open(reference_raster) as ref_src:
                        out_coor_system = ref_src.crs
                        print(f"Using CRS from reference raster: {out_coor_system}")
                elif not out_coor_system:
                    out_coor_system = self.env.outputCoordinateSystem or src.crs
                    print(f"Using output coordinate system from environment or input raster: {out_coor_system}")

                # Calculate the new transform and dimensions for the output CRS
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    src.crs, out_coor_system, src.width, src.height, *src.bounds
                )

                # Update the profile to match the new CRS and dimensions
                profile = src.profile
                profile.update({
                    "crs": out_coor_system,
                    "transform": dst_transform,
                    "width": dst_width,
                    "height": dst_height,
                    "driver": "GTiff",
                    "compress": "lzw"
                })

                # Debug: Display the updated profile
                print(f"Reprojecting raster with profile: {profile}")

                # Reproject the raster data to the new CRS and save the output
                with rasterio.open(out_raster, "w", **profile) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=dst_transform,
                            dst_crs=out_coor_system,
                            resampling=Resampling.nearest
                        )

                print(f"Projection completed successfully, output saved to: {out_raster}")

        except Exception as e:
            raise RuntimeError(f"Error projecting raster: {e}") from e

    def Clip_management(self, in_raster, in_mask_data, out_raster, nodata_value=None, clipping_geometry="NONE"):
        """Clips a raster to a polygon boundary or based on the extent if geometry is not provided."""

        try:
            # Open the input raster to get its CRS
            with rasterio.open(in_raster) as src:
                input_crs = src.crs  # Get the CRS of the input raster

                # Initialize reprojected_shapes
                reprojected_shapes = None

                # Load the mask data (shapefile) and its CRS or get the extent if clipping_geometry is set to "EXTENT"
                if clipping_geometry == "NONE":
                    # Load the mask data (shapefile) and its CRS
                    with fiona.open(in_mask_data, "r") as shapefile:
                        mask_crs = shapefile.crs  # Get the CRS of the mask (shapefile)
                        shapes = [feature["geometry"] for feature in shapefile]

                    # Reproject the mask geometry to the input raster's CRS if needed
                    if mask_crs != input_crs:
                        print(f"Reprojecting mask from {mask_crs} to {input_crs}")
                        reprojected_shapes = [transform_geom(mask_crs, input_crs, shape) for shape in shapes]
                    else:
                        reprojected_shapes = shapes

                elif clipping_geometry == "EXTENT":
                    # Use the extent (bounding box) of the mask data as the clipping geometry
                    with fiona.open(in_mask_data, "r") as shapefile:
                        bounds = shapefile.bounds

                        extent_geom = box(bounds[0], bounds[1], bounds[2], bounds[3])
                        reprojected_shapes = [mapping(extent_geom)]
                    print(f"Using the extent of {in_mask_data} for clipping.")

                # Perform the clipping (masking) operation
                out_image, out_transform = mask(src, reprojected_shapes, crop=True)
                out_meta = src.meta.copy()

                # Update metadata for the clipped raster
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                })

                # Check if we need to reproject to the outputCoordinateSystem
                if self.env.outputCoordinateSystem and self.env.outputCoordinateSystem != input_crs:
                    print(f"Reprojecting to {self.env.outputCoordinateSystem}")

                    # Calculate the new transform and dimensions for the target CRS
                    dst_transform, dst_width, dst_height = calculate_default_transform(
                        input_crs, self.env.outputCoordinateSystem, out_image.shape[2], out_image.shape[1], *src.bounds
                    )

                    # Update the metadata to match the new CRS
                    out_meta.update({
                        "crs": self.env.outputCoordinateSystem,
                        "transform": dst_transform,
                        "width": dst_width,
                        "height": dst_height,
                    })

                    # Create the output file and reproject the clipped image to the new CRS
                    with rasterio.open(out_raster, "w", **out_meta) as dest:
                        for i in range(1, src.count + 1):
                            reproject(
                                source=out_image[i - 1],
                                destination=rasterio.band(dest, i),
                                src_transform=out_transform,
                                src_crs=input_crs,
                                dst_transform=dst_transform,
                                dst_crs=self.env.outputCoordinateSystem,
                                resampling=Resampling.nearest
                            )
                else:
                    # If no reprojection is needed, simply save the clipped raster
                    with rasterio.open(out_raster, "w", **out_meta) as dest:
                        dest.write(out_image)

                print(f"Clipping completed successfully, output saved to: {out_raster}")

        except Exception as e:
            raise RuntimeError(f"Error clipping raster: {e}") from e
    def Describe(self, path):
        """Returns metadata for raster (.tif, .img) and shapefile (.shp) inputs."""
        if path.endswith(('.tif', '.img')):  # Support both .tif and .img raster files
            return self.describe_raster(path)
        elif path.endswith('.shp'):
            return self.describe_shapefile(path)
        else:
            raise ValueError("Unsupported file type. Only .tif, .img (raster) and .shp (shapefile) are supported.")

    def describe_raster(self, path):
        """Describes the metadata of a raster file."""
        try:
            with rasterio.open(path) as src:
                self.extract_raster_metadata(src)
        except Exception as e:
            raise RuntimeError(f"Error describing raster: {e}") from e
        return self

    def extract_raster_metadata(self, src):
        """Extracts and stores metadata from a raster dataset."""
        self.meta = src.meta
        self.crs = src.crs
        self.extent = src.bounds
        self.meanCellWidth = abs(src.transform[0])  # Pixel size in x-direction
        self.meanCellHeight = abs(src.transform[4])  # Pixel size in y-direction
        self.spatialReference = self.crs.to_string() if self.crs else None


    def describe_shapefile(self, path):
        """Describes the metadata of a shapefile."""
        try:
            gdf = gpd.read_file(path)
            self.crs = gdf.crs
            self.extent = gdf.total_bounds  # [xmin, ymin, xmax, ymax]
            self.spatialReference = self.crs.to_string() if self.crs else "No CRS available"
        except Exception as e:
            raise RuntimeError(f"Error describing shapefile: {e}") from e
        return self
