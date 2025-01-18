
## get crs
def reprojection(path, EPSG=32613):
    from osgeo import gdal, osr
    import os 
    raster = gdal.Open(path)
    output = path.replace(".tif", "_reprojected.tif")
    target_crs = osr.SpatialReference()
    target_crs.ImportFromEPSG(EPSG)
    warp_options = gdal.WarpOptions(dstSRS=target_crs, format="GTiff", resampleAlg="near", creationOptions=["COMPRESS=LZW", "BIGTIFF=YES"])
    gdal.Warp(output, raster, options=warp_options)
    ## remvoe original file
    os.remove(path)
    ## rename the reprojected file
    os.rename(output, path)


path = "/data/SWATGenXApp/GenXAppData/Soil/gSSURGO_CONUS/1206/gSSURGO_1206_30m.tif"
reprojection(path)