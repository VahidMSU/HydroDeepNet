import os
from osgeo import gdal, osr, ogr

# Input and output paths
in_raster = "/data/SWATGenXApp/GenXAppData/all_rasters/kriging_output_SWL_250m.tif"
reference_raster = "/data/SWATGenXApp/Users/vahidr32/SWATplus_by_VPUID/0712/huc12/05536265/DEM_250m.tif"
output_path = "/data/SWATGenXApp/Users/vahidr32/SWATplus_by_VPUID/0712/huc12/05536265/kriging_output_SWL_250m_clipped.tif"

import os
from osgeo import gdal, osr, ogr
from osgeo import gdal

def print_raster_info(raster_path):
    ds = gdal.Open(raster_path)
    if ds is None:
        print(f"Error: Cannot open {raster_path}")
        return

    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    print(f"Raster: {raster_path}")
    print(f"Projection: {proj}")
    print(f"GeoTransform: {gt}")
    print(f"Size: {ds.RasterXSize} x {ds.RasterYSize}")
    print(f"Extent: {gt[0]}, {gt[3]}, {gt[0] + gt[1] * ds.RasterXSize}, {gt[3] + gt[5] * ds.RasterYSize}\n")
    
    ds = None

print_raster_info(in_raster)
print_raster_info(reference_raster)



import os
from osgeo import gdal, ogr, osr

def create_extent_shapefile(reference_raster, shapefile_path):
    # Open reference raster
    ref_ds = gdal.Open(reference_raster)
    if ref_ds is None:
        raise FileNotFoundError(f"Reference raster not found: {reference_raster}")

    # Get reference raster properties
    gt = ref_ds.GetGeoTransform()
    proj = ref_ds.GetProjection()
    x_min = gt[0]
    y_max = gt[3]
    x_res = gt[1]
    y_res = gt[5]
    x_max = x_min + (ref_ds.RasterXSize * x_res)
    y_min = y_max + (ref_ds.RasterYSize * y_res)

    # Create shapefile
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(shapefile_path):
        driver.DeleteDataSource(shapefile_path)

    ds = driver.CreateDataSource(shapefile_path)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj)  # Use the same projection as the raster

    layer = ds.CreateLayer("extent", srs, ogr.wkbPolygon)
    field_def = ogr.FieldDefn("id", ogr.OFTInteger)
    layer.CreateField(field_def)

    # Create polygon geometry
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(x_min, y_min)
    ring.AddPoint(x_min, y_max)
    ring.AddPoint(x_max, y_max)
    ring.AddPoint(x_max, y_min)
    ring.AddPoint(x_min, y_min)

    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)

    # Create feature
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetGeometry(poly)
    feature.SetField("id", 1)
    layer.CreateFeature(feature)

    # Cleanup
    feature = None
    ds = None
    ref_ds = None
    print(f"Shapefile saved to {shapefile_path}")

# Define shapefile path
shapefile_path = "/data/SWATGenXApp/Users/vahidr32/SWATplus_by_VPUID/0712/huc12/05536265/reference_extent.shp"

# Run function
create_extent_shapefile(reference_raster, shapefile_path)


def clip_raster(in_raster, shapefile_path, output_path):
    # Open the input raster
    in_ds = gdal.Open(in_raster)
    if in_ds is None:
        raise FileNotFoundError(f"Input raster not found: {in_raster}")

    # Use gdal.Warp to clip the raster
    options = gdal.WarpOptions(
        format="GTiff",
        cutlineDSName=shapefile_path,
        cropToCutline=True,
        dstNodata=-9999
    )
    gdal.Warp(output_path, in_ds, options=options)

    # Cleanup
    in_ds = None
    print(f"Clipped raster saved to {output_path}")

# Run the clip_raster function
clip_raster(in_raster, shapefile_path, output_path)