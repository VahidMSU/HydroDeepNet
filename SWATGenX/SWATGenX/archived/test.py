import os
import rasterio
from rasterio.windows import Window


def align_rasters(landuse, dem, soil):
    """Align rasters to the smallest common shape and save them."""

    assert os.path.exists(landuse), f"Landuse raster not found: {landuse}"
    assert os.path.exists(dem), f"DEM raster not found: {dem}"
    assert os.path.exists(soil), f"gSSURGO raster not found: {soil}"

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


landuse = "/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0415/huc12/04292000/SWAT_MODEL/Watershed/Rasters/Landuse/landuse.tif"
dem = "/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0415/huc12/04292000/SWAT_MODEL/Watershed/Rasters/DEM/dem.tif"
soil = "/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0415/huc12/04292000/SWAT_MODEL/Watershed/Rasters/gSSURGO/soil.tif"

align_rasters(landuse, dem, soil)