
import arcpy
for year in range(2009, 2023):
    path =  f"/data/MyDataBase/SWATGenXAppData/CDL landuse/{year}_30m_cdls_MIP_26990_projected.tif"
    reference_raster = "/data/MyDataBase/SWATGenXAppData/all_rasters/DEM_30m.tif"
    clipped_output = f"/data/MyDataBase/SWATGenXAppData/CDL landuse/{year}_30m_cdls_MIP_26990_projected_clipped.tif"
    assert arcpy.Exists(path), "The input raster does not exist"
    assert arcpy.Exists(reference_raster), "The reference raster does not exist"

    ## Extract by mask
    try:
        arcpy.env.workspace = "/data/MyDataBase/SWATGenXAppData/CDL landuse"
        arcpy.env.snapRaster = reference_raster
        arcpy.env.extent = reference_raster
        arcpy.env.cellSize = reference_raster

        arcpy.gp.ExtractByMask_sa(path, reference_raster, clipped_output)
    except Exception as e:
        print(e)

    ### now compare the width and height of the two rasters
    clipped_desc = arcpy.Describe(clipped_output)
    reference_desc = arcpy.Describe(reference_raster)
    print("Clipped raster width: ", clipped_desc.width, "Clipped raster height: ", clipped_desc.height)
    print("Reference raster width: ", reference_desc.width, "Reference raster height: ", reference_desc.height)
    assert clipped_desc.width == reference_desc.width, "Widths do not match"
    assert clipped_desc.height == reference_desc.height, "Heights do not match"

    ## now delete the original raster
    arcpy.Delete_management(path)
