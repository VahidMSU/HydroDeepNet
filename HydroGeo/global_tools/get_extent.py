import rasterio
from rasterio.warp import transform_bounds

def get_lat_lon_reference_raster(reference_raster):
    with rasterio.open(reference_raster) as src:
        bounds = src.bounds
        src_crs = src.crs
    return transform_bounds(src_crs, 'EPSG:4326', bounds.left, bounds.bottom, bounds.right, bounds.top)


if __name__ == "__main__":
    reference_raster = "/data/reference_rasters/DEM_250m.tif"
    min_lat, min_lon, max_lat, max_lon = get_lat_lon_reference_raster(reference_raster)
    print(f"min_lat: {min_lat}\nmax_lat: {max_lat}\nmin_lon: {min_lon}\nmax_lon: {max_lon}")