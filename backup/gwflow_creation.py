import arcpy
import os
import shutil
import subprocess
from arcpy import env
from arcpy.sa import *
import os
import geopandas as gpd
import numpy as np
import os
import shutil
import rasterio
import geopandas as gpd
import numpy as np
from rasterio import features
from shapely.geometry import shape
import rasterio
import geopandas as gpd
import numpy as np
from rasterio.mask import mask
def generate_paths(BASE_PATH, LEVEL, NAME, RESOLUTION):
    GLOCAL_DATA_SOURCE = "/data/MyDataBase/SWATGenXAppData/gwflow/Tutorial/1 Datasets and Shapefiles/Global datasets" 
    EBK_RASTERS_path = os.path.join(BASE_PATH, f"SWATplus_by_VPUID/{LEVEL}/{NAME}/MODFLOW_{RESOLUTION}/rasters_input/")

    # Global datasets
    perm = os.path.join(GLOCAL_DATA_SOURCE, "GLHYMPS.shp")  # Not projected
    

    thickrast = os.path.join(EBK_RASTERS_path, f"{NAME}_kriging_output_AQ_THK_1_{RESOLUTION}m.tif.tif")  # EPSG:26990
    
    SOURCE_inout = os.path.join(BASE_PATH, f"SWATplus_by_VPUID/{LEVEL}/{NAME}/SWAT_MODEL/Scenarios/Default/TxtInOut")
    
    SWAT_SHAPES = os.path.join(BASE_PATH, f"SWATplus_by_VPUID/{LEVEL}/{NAME}/SWAT_MODEL/Watershed/Shapes/")
    
    # SWAT+ shape files
    demrast = os.path.join(BASE_PATH, f"SWATplus_by_VPUID/{LEVEL}/{NAME}/", 'DEM_250m.tif')  # EPSG:26990
    hrushp = os.path.join(SWAT_SHAPES, 'hrus2.shp')  # EPSG:26990
    rivshp = os.path.join(SWAT_SHAPES, "rivs1.shp")  # EPSG:26990
    subshp = os.path.join(SWAT_SHAPES, "subs1.shp")  # EPSG:26990
    #lakes = os.path.join(GLOCAL_DATA_SOURCE, "HydroLAKES_polys_v10.shp")  # Not projected
    lakes = os.path.join(SWAT_SHAPES, "SWAT_plus_lakes.shp")  # Not projected

    
    gwflow_target_path = os.path.join(BASE_PATH, f"SWATplus_by_VPUID/{LEVEL}/{NAME}/gwflow")
    print('Target path:', gwflow_target_path)
    # gwflow.data file (will be created at end of script)
    gwflow_data = os.path.join(gwflow_target_path, "gwflow.data")
    # Pathways for local files
    gis_folder = os.path.join(gwflow_target_path, "gwflow_gis")
    print('gis_folder',gis_folder)


    
    return perm, lakes, thickrast, SOURCE_inout, demrast, hrushp, rivshp, subshp, gwflow_target_path, gwflow_data, gis_folder

def prepare_target_directories(BASE_PATH, gis_folder, gwflow_target_path,SOURCE_inout):
    os.makedirs(gis_folder, exist_ok=True)
    os.makedirs(gwflow_target_path, exist_ok=True)
    shutil.copytree(SOURCE_inout, gwflow_target_path, dirs_exist_ok=True)    
    shutil.copy2(os.path.join(BASE_PATH, r"bin/swat+gwflow_Ryan_version.exe"), gwflow_target_path)
    shutil.copy2(os.path.join(BASE_PATH, r"bin/gwflow_input_files.exe"), gwflow_target_path)


def create_fishnet_or_use_existing_one(BASE_PATH, NAME, LEVEL, gis_folder):
    
    infc = os.path.join(BASE_PATH, f"SWATplus_by_VPUID/{LEVEL}/{NAME}/MODFLOW_250/Grids_MODFLOW/Grids_MODFLOW.shp")
    
    if os.path.exists(infc):
        print('using existing grid shapefile')
        df = gpd.read_file(infc)
        df ['Id'] = np.arange(1,len(df)+1)
        df[['Id','geometry']].to_file(os.path.join(gis_folder, "grid1.shp"))
        print("fishnet are generated based on existing gridshapefile")
        num_cols = df.Row.max()
        num_rows = df.Col.max()

        
    else:
        env.outputCoordinateSystem = arcpy.SpatialReference(CRS)
        env.workspace = gis_folder

        print("     create grid")
        fc = "watershed_poly.shp"
        desc = arcpy.Describe(fc)
        str_ll = str(desc.extent.lowerLeft)
        str_yaxis = str(desc.extent.XMin) + " " + str(desc.extent.YMax + 10)
        str_ur = str(desc.extent.upperRight)
        outfc = "fishnet.shp"
        arcpy.CreateFishnet_management(outfc,str_ll,str_yaxis,str(RESOLUTION),str(RESOLUTION),"0","0",str_ur,"NO_LABELS","#","POLYGON")
        infc = "fishnet.shp"
        print("fishnet are generated from scratch")

        fishnet_rast = os.path.join(env.workspace, "fishnet_rast.tif")

        arcpy.PolygonToRaster_conversion(fishnet_shape_path,"FID",fishnet_rast,"CELL_CENTER","NONE",RESOLUTION)
        arcpy.RasterToPolygon_conversion(fishnet_rast, "grid1.shp", "NO_SIMPLIFY","#")
        arcpy.DeleteField_management("grid1.shp",["gridcode"])
        
        # Retrieve the number of rows and columns in the grid raster (will be written out in gwflow.data)
        rowResult = arcpy.GetRasterProperties_management(fishnet_rast, "ROWCOUNT")
        num_rows = rowResult.getOutput(0)
        colResult = arcpy.GetRasterProperties_management(fishnet_rast, "COLUMNCOUNT")
        num_cols = colResult.getOutput(0)
    
    return num_cols+1, num_rows+1



def clear_directory(directory_path):
    # Check if the directory exists
    if not os.path.exists(directory_path):
        print("Directory does not exist:", directory_path)
        return

    # Iterate over each file and directory inside the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        try:
            # If it's a file or a link, delete it; if it's a directory, use shutil.rmtree to delete the directory and all its contents
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))



def create_gwflow(NAME, RESOLUTION, LEVEL, BASE_PATH, CRS):
    
    print('begin')
    perm, lakes, thickrast, SOURCE_inout, demrast, hrushp, rivshp, subshp, gwflow_target_path, gwflow_data, gis_folder = generate_paths(BASE_PATH, LEVEL, NAME, RESOLUTION)
    prepare_target_directories (BASE_PATH, gis_folder,gwflow_target_path,SOURCE_inout)
    
    wtdepth_start = 10.0       # initial water table depth (meters), for each grid cell
    wtdepth_start_str = "10.0" # initial water table depth (meters), in string format
    recharge_delay = "3.0"     # recharge delay (days), for each grid cell
    time_step = "1.0"          # time step (days)
    specific_yield = "0.20"    # aquifer specific yield
    porosity = "0.30"          # aquifer porosity
    river_cond = "0.005"       # river bed hydraulic conductivity (m/day)
    river_thick = "1.00"       # river bed thickness (m)
    start_year = 1990          # beginning year of simulation
    end_year = 2021            # end year of simulation
    dist_below = "5.0"         # vertical distance (m) of river bed below the DEM value"
    tile_depth = "1.22"        # tile drain depth (m) below ground surface
    tile_area = "50.0"         # tile drain area of groundwater inflow (m2)
    tile_cond = "5.00"         # tile drain hydraulic conductivity (m/day)
    lake_cond = "0.005"        # lake bed hydraulic conductivity (m/day)
    lake_thick = "2.00"        # lake bed thickness (m)
    exdp = 1.00                # depth (m) below ground surface that groundwater ET ceases

    # perform ArcMap routines ----------------------------------------------------------------------------------------------
    print("Performing ArcMap routines to set up gwflow cell data...")
    
    # create a watershed shape file (single polygon), using the subbasin shape file
    print("     create watershed boundary")
    # set coordinate system for all GIS files #################
    env.outputCoordinateSystem = arcpy.SpatialReference(CRS)
    env.workspace = gis_folder
    clear_directory(gis_folder)

    
    arcpy.Dissolve_management(subshp,"watershed_poly.shp")
    
    num_cols, num_rows = create_fishnet_or_use_existing_one(BASE_PATH, NAME, LEVEL, gis_folder)

    # Identify active cells; create grid2_active.shp
    print("     identify cells within watershed boundary")
    print("          clip grid to watershed area")

    
    grid_gdf = gpd.read_file(os.path.join(gis_folder, "grid1.shp"))
    watershed_gdf = gpd.read_file(os.path.join(gis_folder,"watershed_poly.shp"))
    watershed_gdf['active'] = 1
    active_cells_gdf = gpd.overlay(watershed_gdf[['geometry', 'active']], grid_gdf[['Id','geometry']], how='intersection')
    
    active_cells_gdf[['Id','active','geometry']].to_file(os.path.join(gis_folder,"active_cells.shp"))

    grid1 = gpd.read_file(os.path.join(gis_folder, "grid1.shp"))
    active_cells = gpd.read_file(os.path.join(gis_folder, "active_cells.shp"))
    grid2_active = grid1.merge(active_cells, on='Id', how= 'left')
    grid2_active = grid1.merge(active_cells[['Id','active']], on='Id', how= 'left')
    grid2_active = grid2_active.fillna(0)
    grid2_active.to_file(os.path.join(gis_folder,"grid2_active.shp"))
    print("          perform spatial join")
    # Determine aquifer thickness for each grid cell
    print("     determine aquifer thickness for each grid cell")
    print("          convert raster to polygon")
    # Ensure thickrast is a Raster object
    thickrast = Raster(thickrast)


    ######################################################### REVISION ###############################
    
    # Project raster to the spatial reference if necessary
    if thickrast.spatialReference.factoryCode != 26990:
        thickrast = arcpy.management.ProjectRaster(thickrast, os.path.join(env.workspace, "thickrast_projected.tif"), arcpy.SpatialReference(26990))

    # Convert the raster to integer if it is floating-point
    
    if not thickrast.isInteger:
        thickrast = Int(thickrast)

    # Save the converted raster
    converted_raster_path = os.path.join(env.workspace, "converted_thickrast_2.tif")
    thickrast.save(converted_raster_path)
    
    #################################################################################################

    # Paths
    shapefile_path = os.path.join(gis_folder, "grid2_active.shp")
    out_shapefile_path = os.path.join(gis_folder, "grid3_thick.shp")
    
    # Read shapefile
    gdf = gpd.read_file(shapefile_path)
    
    # Read raster
    with rasterio.open(converted_raster_path) as src:
        for i, row in gdf.iterrows():
            # Mask raster with each geometry, mean value extraction
            out_image, out_transform = mask(src, [row['geometry']], crop=True, nodata=0, all_touched=True)
            gdf.at[i, "thick_m"] = np.mean(out_image[out_image > 0]) * 0.3048  # feet to meters
        # Verify the geometry column
    print("columns:", gdf.columns)
    print("thick_m volumn:", gdf.thick_m.values)
    print("Geometry columns:", gdf.geometry.name)
    print("Number of geometry columns:", len(gdf.columns[gdf.dtypes == 'geometry']))
    # Save to file
    gdf.to_file(out_shapefile_path)

    # Determine ground surface elevation for each grid cell (using DEM)
    print("     determine ground surface elevation for each grid cell")
    outrast = Int(demrast)
    outrast.save("dem_int.tif")
    arcpy.Resample_management("dem_int.tif", "dem_resample.tif", str(RESOLUTION), "BILINEAR")
    # spatial join to grid cells
    print("          perform spatial join")
    arcpy.RasterToPolygon_conversion("dem_resample.tif", "dem_poly.shp", "NO_SIMPLIFY","#")
    arcpy.SpatialJoin_analysis("grid3_thick.shp", "dem_poly.shp", "grid4_dem.shp")
    arcpy.DeleteField_management("grid4_dem.shp",["Join_Count"])
    arcpy.DeleteField_management("grid4_dem.shp",["TARGET_FID"])
    arcpy.DeleteField_management("grid4_dem.shp",["Avg_Id"])
    
    # Determine the aquifer properties for each grid cell
    print("     determine aquifer properties for each grid cell")
    print("          clip global dataset to watershed area")
    arcpy.Clip_analysis(perm, "watershed_poly.shp", "glhymps_clipped.shp")
    print("          calculate K (m/day)")
    arcpy.AddField_management("glhymps_clipped.shp", "K_mday", "DOUBLE", 9, 5, "", "refcode", "NULLABLE")
    cur = arcpy.UpdateCursor("glhymps_clipped.shp")
    for row in cur:
        logK = row.getValue("logK_Ferr_") / 100.0
        perm_msec = 10.0 ** logK
        hc_msec = perm_msec * 1000.0 * 9.81 / 0.001
        hc_mday = hc_msec * 86400.0
        row.setValue("K_mday",hc_mday)
        cur.updateRow(row)
    del cur
    del row
    arcpy.DeleteField_management("glhymps_clipped.shp",["OBJECTID_1"])
    arcpy.DeleteField_management("glhymps_clipped.shp",["logK_Ice_x"])
    arcpy.DeleteField_management("glhymps_clipped.shp",["logK_Ferr_"])
    arcpy.DeleteField_management("glhymps_clipped.shp",["K_stdev_x1"])
    arcpy.DeleteField_management("glhymps_clipped.shp",["OBJECTID"])
    arcpy.DeleteField_management("glhymps_clipped.shp",["XX"])
    arcpy.DeleteField_management("glhymps_clipped.shp",["YY"])
    arcpy.DeleteField_management("glhymps_clipped.shp",["ZZ"])
    arcpy.DeleteField_management("glhymps_clipped.shp",["AA"])
    arcpy.DeleteField_management("glhymps_clipped.shp",["DD"])
    arcpy.DeleteField_management("glhymps_clipped.shp",["Shape_Leng"])
    arcpy.DeleteField_management("glhymps_clipped.shp",["GUM_K"])
    arcpy.DeleteField_management("glhymps_clipped.shp",["Prmfrst"])
    arcpy.DeleteField_management("glhymps_clipped.shp",["Shape_Le_1"])
    arcpy.DeleteField_management("glhymps_clipped.shp",["Shape_Area"])
    arcpy.DeleteField_management("glhymps_clipped.shp",["Transmissi"])
    arcpy.DeleteField_management("glhymps_clipped.shp",["COUNT"])
    arcpy.DeleteField_management("glhymps_clipped.shp",["AREA_1"])
    arcpy.DeleteField_management("glhymps_clipped.shp",["MEAN"])
    arcpy.DeleteField_management("glhymps_clipped.shp",["STD"])
    # provide a zone ID for each grid cell
    print("          assign a zone ID to each geologic type")
    arcpy.AddField_management("glhymps_clipped.shp", "zone", "LONG", 9, "", "", "refcode", "NULLABLE")
    cur = arcpy.UpdateCursor("glhymps_clipped.shp")
    zone_id = 1
    for row in cur:
        row.setValue("zone",zone_id)
        cur.updateRow(row)
        zone_id += 1
    del cur
    del row
    # Spatial join the K polygon shape file to the grid cell shape file
    print("          perform spatial join")
    arcpy.SpatialJoin_analysis("grid4_dem.shp", "glhymps_clipped.shp", "grid5_K.shp")
    arcpy.DeleteField_management("grid5_K.shp",["Join_Count"])
    arcpy.DeleteField_management("grid5_K.shp",["TARGET_FID"])
    arcpy.DeleteField_management("grid5_K.shp",["IDENTITY_"])
    # Search for any cells that have zone = 0; if so, then give a default value = 1
    cur = arcpy.UpdateCursor("grid5_K.shp")
    for row in cur:
        zone_id = row.getValue("zone")
        if zone_id == 0:
            row.setValue("zone",1)
            cur.updateRow(row)
    del cur
    del row
    
    # Determine boundary cells (cells along the perimeter of the watershed)
    print("     determine boundary cells")
    arcpy.PolygonToLine_management("watershed_poly.shp","watershed_boundary.shp","IGNORE_NEIGHBORS")
    arcpy.Intersect_analysis(["grid1.shp","watershed_boundary.shp"], "boundary_cells", "ALL", "", "")
    arcpy.DeleteField_management("boundary_cells.shp",["FID_grid1"])
    arcpy.DeleteField_management("boundary_cells.shp",["Id"])
    arcpy.DeleteField_management("boundary_cells.shp",["gridcode"])
    arcpy.DeleteField_management("boundary_cells.shp",["FID_waters"])
    arcpy.DeleteField_management("boundary_cells.shp",["ORIG_FID"])
    arcpy.AddField_management("boundary_cells.shp", "boundary", "LONG", 9, "", "", "refcode", "NULLABLE")
    cur = arcpy.UpdateCursor("boundary_cells.shp")
    for row in cur:
        boundary = 1
        row.setValue("boundary",boundary)
        cur.updateRow(row)
    del cur
    del row
    print("          perform spatial join")
    arcpy.SpatialJoin_analysis("grid5_K.shp", "boundary_cells.shp", "grid6_boundary.shp")
    arcpy.DeleteField_management("grid6_boundary.shp",["Count_"])
    arcpy.DeleteField_management("grid6_boundary.shp",["boundary_cells_FID"])
    
    # Determine presence of lakes
    print("     determine presence of lakes")
    arcpy.Clip_analysis(lakes, "watershed_poly.shp", "lakes_clipped.shp")
    arcpy.AddField_management("lakes_clipped.shp", "lake", "LONG", 9, "", "", "refcode", "NULLABLE")
    
    
    num_rows_lake = int(arcpy.GetCount_management("lakes_clipped.shp").getOutput(0))
    if num_rows_lake > 0:
        cur = arcpy.UpdateCursor("lakes_clipped.shp")
        for row in cur:
            lake_flag = 1
            row.setValue("lake",lake_flag)
            cur.updateRow(row)
        del cur
        del row
    arcpy.SpatialJoin_analysis("grid6_boundary.shp", "lakes_clipped.shp", "grid7_lake.shp")
    arcpy.DeleteField_management("grid7_lake.shp",["Join_Count"])
    arcpy.DeleteField_management("grid7_lake.shp",["TARGET_FID"])
    
    # Identify River Cells using river network shape file
    print("     identify river cells using river network")
    arcpy.Intersect_analysis(["grid7_lake.shp",rivshp], "cell_channel_inters", "ALL", "", "")
    arcpy.AddField_management("cell_channel_inters.shp", "riv_length", "DOUBLE", 9, 5, "", "refcode", "NULLABLE")
    arcpy.CalculateField_management("cell_channel_inters.shp", "riv_length","!SHAPE.LENGTH@METERS!", "PYTHON_9.3")
    
    # Connect grid cells to HRUs
    print("     connect grid cells to HRUs")
    arcpy.Intersect_analysis(["grid7_lake.shp",hrushp], "hru_cells", "ALL", "", "")
    arcpy.AddField_management("hru_cells.shp", "poly_area", "DOUBLE", 9, 5, "", "refcode", "NULLABLE")
    arcpy.CalculateField_management("hru_cells.shp", "poly_area","!shape.area@squaremeters!", "PYTHON_9.3")
    
    
    
    # Create gwflow.data file ----------------------------------------------------------------------------------------------
    print("Creating gwflow.data file...")
    gwfile = open(gwflow_data,"w") # open text file gwflow.data
    env.workspace = gis_folder # set path again (to make sure)
    
    # write out basic information and flags
    str1 = "This files contains ArcMap data that will be processed to create the gwflow input files"
    gwfile.write(str1+"\n")
    str1 = ""
    gwfile.write(str1+"\n")
    str1 = "Basic information"
    gwfile.write(str1+"\n")
    str1 = str(RESOLUTION)
    gwfile.write(str1+"\n")
    row_col = []                    # number of rows and columns in the grid
    row_col.append([num_rows,num_cols])
    row_vals = row_col[0]
    str1 = ' '.join(str(e) for e in row_vals)
    gwfile.write(str1+"\n")
    str1 = "3			Water table initiation flag"
    gwfile.write(str1+"\n")
    str1 = wtdepth_start_str
    gwfile.write(str1+"\n")
    str1 = "1			Boundary conditions (1=constant head; 2=no flow)"
    gwfile.write(str1+"\n")
    str1 = "1	                Groundwater-->soil transfer is simulated (0 = no; 1 = yes)"
    gwfile.write(str1+"\n")
    str1 = "1			Saturation excess flow is simulated (0 = no; 1 = yes)"
    gwfile.write(str1+"\n")
    str1 = "1			Groundwater ET is simulated (0 = no; 1 = yes)"
    gwfile.write(str1+"\n")
    str1 = "1			Tile drain flow is simulated (0 = no; 1 = yes)"
    gwfile.write(str1+"\n")
    str1 = "1			Groundwater-Lake interaction is simulated (0 = no; 1 = yes)"
    gwfile.write(str1+"\n")
    str1 = "0			Groundwater pumping is specified by user (0 = no; 1 = yes)"
    gwfile.write(str1+"\n")
    str1 = "1	                Recharge is passed from HRUs (0 = no; 1 = yes)"
    gwfile.write(str1+"\n")
    str1 = recharge_delay
    gwfile.write(str1+"\n")
    str1 = "0			Groundwater nutrient transport (0 = no; 1 = yes)"
    gwfile.write(str1+"\n")
    str1 = time_step
    gwfile.write(str1+"\n")
    str1 = "1 1 1			Flags for groundwater and nutrient mass balance files (daily; annual; average annual)"
    gwfile.write(str1+"\n")
    str1 = ""
    gwfile.write(str1+"\n")
    
    print("         write aquifer property data")
    # write out aquifer property data (glhymps_clipped.shp)
    str1 = "Aquifer Properties, by zone"
    gwfile.write(str1+"\n")
    
    cur = arcpy.UpdateCursor("glhymps_clipped.shp")
    row_counter = 0
    for row in cur:
        row_counter += 1
    del cur
    del row
    str1 = str(row_counter)
    gwfile.write(str1+"\n")
    str1 = "Zone	K (m/d)	Sy	n"
    gwfile.write(str1+"\n")
    cur = arcpy.UpdateCursor("glhymps_clipped.shp")
    row_counter = 1
    for row in cur:
        K_value = row.getValue("K_mday")
        Sy_value = specific_yield
        n_value = porosity
        row_vals = [row_counter,K_value,Sy_value,n_value]
        str1 = ' '.join(str(e) for e in row_vals)
        gwfile.write(str1+"\n")
        row_counter += 1
    del cur
    del row
    
    # write out stream bed property data
    str1 = ""
    gwfile.write(str1+"\n")
    str1 = "Streambed Properties, by zone"
    gwfile.write(str1+"\n")
    str1 = "1	Number of zones"
    gwfile.write(str1+"\n")
    str1 = "Zone	K (m/d)	Thickness (m)"
    gwfile.write(str1+"\n")
    str1 = "1"
    gwfile.write(str1+"\n")
    str1 = river_cond
    gwfile.write(str1+"\n")
    str1 = river_thick
    gwfile.write(str1+"\n")
    
    print("         groundwater head output control")
    # write out groundwater head output control
    str1 = ""
    gwfile.write(str1+"\n")
    str1 = "Groundwater Head Output Control (days for which head will be output for each grid cell)"
    gwfile.write(str1+"\n")
    num_years = end_year - start_year
    num_years = num_years + 1
    row_vals = [num_years,"Number of output times"]
    str1 = ' '.join(str(e) for e in row_vals)
    gwfile.write(str1+"\n")
    for year in list(range(start_year,end_year+1,1)):
        row_vals = [year,270]
        str1 = ' '.join(str(e) for e in row_vals)
        gwfile.write(str1+"\n")
    
    print("         groundwater head observation locations")
    # write out groundwater head observation locations (wells_cells.shp)
    str1 = ""
    gwfile.write(str1+"\n")
    str1 = "Groundwater Head Observation Locations (cells for which head will be output for each time step)"
    gwfile.write(str1+"\n")
    str1 = "0"
    gwfile.write(str1+"\n")
    
    # cell for detailed daily sources/sink output
    str1 = ""
    gwfile.write(str1+"\n")
    str1 = "Cell for detailed daily sources/sink output"
    gwfile.write(str1+"\n")
    str1 = "Row     Column"
    gwfile.write(str1+"\n")
    str1 = "1	1"
    gwfile.write(str1+"\n")
    
    # river cell information
    str1 = ""
    gwfile.write(str1+"\n")
    str1 = "River Cell Information"
    gwfile.write(str1+"\n")
    str1 = dist_below
    gwfile.write(str1+"\n")
    
    # hydrograph separation
    str1 = ""
    gwfile.write(str1+"\n")
    str1 = "Hydrograph separation"
    gwfile.write(str1+"\n")
    str1 = "1			Number of channels"
    gwfile.write(str1+"\n")
    str1 = "1			Channel ID"
    gwfile.write(str1+"\n")
    
    # tile drain parameters
    str1 = ""
    gwfile.write(str1+"\n")
    str1 = "Tile Drain Parameters"
    gwfile.write(str1+"\n")
    str1 = tile_depth
    gwfile.write(str1+"\n")
    str1 = tile_area
    gwfile.write(str1+"\n")
    str1 = tile_cond
    gwfile.write(str1+"\n")
    str1 = "0			Tile cell groups (flag: 0=no; 1=yes)"
    gwfile.write(str1+"\n")
    
    # lake parameters
    str1 = ""
    gwfile.write(str1+"\n")
    str1 = "Lake Parameters"
    gwfile.write(str1+"\n")
    str1 = lake_thick
    gwfile.write(str1+"\n")
    str1 = lake_cond
    gwfile.write(str1+"\n")
    str1 = "2.00			NO3-N concentration (mg/L) of lake water"
    gwfile.write(str1+"\n")
    str1 = "0.01			P concentration (mg/L) of lake water"
    gwfile.write(str1+"\n")
    
    # groundwater nutrient transport
    str1 = ""
    gwfile.write(str1+"\n")
    str1 = "Groundwater nutrient transport"
    gwfile.write(str1+"\n")
    str1 = "-0.0001			First-order rate constant for denitrification (1/day)"
    gwfile.write(str1+"\n")
    str1 = "5.00			Dispersion coefficient (m2/day)"
    gwfile.write(str1+"\n")
    str1 = "1.00			Sorption retardation coefficient for Nitrate"
    gwfile.write(str1+"\n")
    str1 = "2.00                    Sorption retardation coefficient for Phosphorus"
    gwfile.write(str1+"\n")
    str1 = "1                       Number of transport time steps per flow time step"
    gwfile.write(str1+"\n")
    
    
    print("         cell information")
    # cell information (sorted by cell ID)
    str1 = ""
    gwfile.write(str1+"\n")
    str1 = "Cell Information (sorted by Cell ID)"
    gwfile.write(str1+"\n")
    str1 = "Cell_ID	Active Thick Elev Zone Bound Head EXDP ET_fix Tile InitNO3 InitP Lake Lake_bed Lake_stage"
    gwfile.write(str1+"\n")
    cell_info = []
    cur = arcpy.UpdateCursor("grid7_lake.shp")
    
    for row in cur:
        
        cell_id = row.getValue("Id")
        active = row.getValue("active")
        
        thick = row.getValue("thick_m")
        elev = row.getValue("gridcode")
        zone = row.getValue("zone")
        bound = row.getValue("boundary")
        head = elev - wtdepth_start
        
        tile = 0
        
        lake = row.getValue("lake")
        lake_stage = elev
        lake_bed = elev - 2 ### AVERAGE DEPTH IS CONSIDER 2 METERS
        et_fix = 0.00
        initno3 = 3.00
        initp =  0.05

        
        cell_info.append([cell_id,active,thick,elev,zone,bound,head,exdp,et_fix,tile,initno3,initp,lake,lake_bed,lake_stage])
        
        
    del cur
    del row
    
    cell_info.sort(key = lambda z : z[0], reverse = False) # sort by cell id
    
    counter = 0
    for x in cell_info:
        row_vals = cell_info[counter]
        str1 = ' '.join(str(e) for e in row_vals)
        gwfile.write(str1+"\n") 
        counter += 1

    print("         channel-cell connection information")
    # channel-cell connection information
    str1 = ""
    gwfile.write(str1+"\n")
    str1 = "Channel-Cell Connection Information"
    gwfile.write(str1+"\n")
    # count the number of river cells, then print out
    cur = arcpy.UpdateCursor("cell_channel_inters.shp")
    row_counter = 0
    
    for row in cur:
        row_counter += 1
    del cur
    del row
    
    row_vals = [row_counter,"Number of cells that intersect channels"]
    str1 = ' '.join(str(e) for e in row_vals)
    gwfile.write(str1+"\n")
    # write out header information
    str1 = "CellID	  Zone    Elev    Length"
    gwfile.write(str1+"\n")
    # read in river cell information; sort by cell id; print out to file
    row_info = []


#    gdf = gpd.read_file(os.path.join(gis_folder,"cell_channel_inters.shp"))
    
    # Sort by channel number
#    gdf.sort_values(by='Channel', inplace=True)
    # Find the missing channel numbers
#    all_channels = set(range(1, gdf.Channel.max()+1))  # Expected range
#    actual_channels = set(gdf['Channel'])
#    missing_channels = sorted(all_channels - actual_channels)
    
    # Adjust the channel numbers
#    for missing in missing_channels:
#        gdf['Channel'] = gdf['Channel'].apply(lambda x: x-1 if x > missing else x)
    
#    gdf.to_file(os.path.join(gis_folder,"cell_channel_inters.shp"))
    
    cur = arcpy.UpdateCursor("cell_channel_inters.shp")
    
    for row in cur:
        chan_id = row.getValue("Channel")
        cell_id = row.getValue("Id")
        zone = row.getValue("zone")
        elev = row.getValue("gridcode")
        length = row.getValue("riv_length")
        row_info.append([cell_id,elev,chan_id,length,zone])  
        
    del cur
    del row
    row_info.sort(key = lambda z : z[0], reverse = False)
    counter = 0
    for x in row_info:
        row_vals = row_info[counter]
        str1 = ' '.join(str(e) for e in row_vals)
        gwfile.write(str1+"\n") 
        counter += 1
    
    print("         hru-cell connection information")
    # hru-cells connection information
    str1 = ""
    gwfile.write(str1+"\n")
    str1 = "HRU-Cells Connection Information"
    gwfile.write(str1+"\n")
    # count the number of connections, then print out
    cur = arcpy.UpdateCursor("hru_cells.shp")
    row_counter = 0
    for row in cur:
        row_counter += 1
    del cur
    del row
    row_vals = [row_counter,"Number of connections"]
    str1 = ' '.join(str(e) for e in row_vals)
    gwfile.write(str1+"\n")
    # write out header information
    str1 = "Cell_ID	  Area_m2   HRU   poly_area_m2"
    gwfile.write(str1+"\n")
    
    # read in hru-cells connection information
    row_info = []
    cur = arcpy.UpdateCursor("hru_cells.shp")

    
    
    for row in cur:
        cell_id = row.getValue("Id")
        area_m2 = row.getValue("Area") * 10000
        hru_id = row.getValue("HRUS") # this is text
        hru_int = int(hru_id)
        poly_area = row.getValue("poly_area")
        row_info.append([cell_id,area_m2,hru_int,poly_area])       
    del cur
    del row
    
    # sort by cell id, then write out to file
    row_info.sort(key = lambda z : z[0], reverse = False)
    counter = 0
    for x in row_info:
        row_vals = row_info[counter]
        str1 = ' '.join(str(e) for e in row_vals)
        gwfile.write(str1+"\n") 
        counter += 1
        
    # now sort by HRU id, and write out to file
    str1 = "Cell_ID	  Area_m2   HRU   poly_area_m2"
    gwfile.write(str1+"\n")
    row_info.sort(key = lambda z : z[2], reverse = False)
    counter = 0
    for x in row_info:
        row_vals = row_info[counter]
        str1 = ' '.join(str(e) for e in row_vals)
        gwfile.write(str1+"\n") 
        counter += 1
    
    
    # close the file
    gwfile.close()





