import arcpy
import subprocess
import geopandas as gpd
import numpy as np
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import rasterio
from rasterio.features import rasterize

def check_gwflow_integration(base_directory, level, model_name, MODFLOW_MODEL_NAME):
    names = os.listdir(os.path.join(base_directory, f"SWAT_input/{level}"))
    if 'log.txt' in names:  # Remove 'log.txt' if it exists
        names.remove('log.txt')

    success = []
    failed = []
    not_created = []

    for name in names:
        model_base = os.path.join(base_directory, f'SWAT_input/{level}/{name}/{model_name}')

        gwflow_path = os.path.join(model_base, 'Scenarios/Default/TxtInOut/gwflow_flux_recharge')
        modflow_path = os.path.join(base_directory, f'SWAT_input/{level}/{name}/{MODFLOW_MODEL_NAME}/rasters_input')

        if os.path.exists(gwflow_path):
            size = os.path.getsize(gwflow_path)
            if size > 32:
                print('gwflow_flux_recharge', name, 'Size:', size, 'success')
                success.append(name)
            else:
                print('gwflow_flux_recharge', name, 'Size:', size, 'failed')
                failed.append(name)
        else:
            print(f'{name}: gwflow not created')
            not_created.append(name)

        if not os.path.exists(modflow_path):
            print('modflow raster folder is not available:', modflow_path)

    return success, failed, not_created


def creating_gwflow(BASE_PATH,LEVEL, NAME, RESOLUTION,start_year, end_year,MODEL_NAME, MODFLOW_MODEL_NAME,SWAT_MODEL_NAME, CRS = 26990):
    print('begin')


    inout_path = Setting_env_and_returning_object_path (BASE_PATH, NAME, LEVEL, RESOLUTION,start_year, end_year, MODEL_NAME,MODFLOW_MODEL_NAME, SWAT_MODEL_NAME)

    outpath_object     = inout_path['outpath_object']
    gis_folder         = inout_path['gis_folder']
    grids_polygon      = inout_path['grids_polygon']
    gwflow_target_path = inout_path['gwflow_target_path']
    grids_point_path   = inout_path['grids_point_path']
    grids_polygon_path = inout_path['grids_polygon_path']
    rech_out_folder    = inout_path['rech_out_folder']

    Zone, ZONE_K_TABLE, outpath_object['Zone'] = creating_zones(outpath_object['HHC'], outpath_object['aq_zone'],  gis_folder)

    ## adding grids point and grid polygone path
    outpath_object['grids_point'] = grids_point_path
    outpath_object['grids_polygon'] = grids_polygon_path

    #####  extracting formatted channel-cell, hru-cell and cell-hru tables from the above objects
    CEll_HRU_TABLE, HRU_CELL_TABLE, CELL_CHANNEL_TABLE = getting_CELL_CHANNEL_HRU_TABLES (outpath_object, RESOLUTION,
                                                                                        gwflow_target_path)
    #### extracting grid tables from the above object and processes
    grids_point , GRIDS_TABLE, GRIDS_TABLE_path, num_rows, num_cols  = getting_grid_table(outpath_object,
                                                                                        gis_folder, gwflow_target_path)
    print('rows:',num_rows,'cols:', num_cols)  ### printing some output

    print('HRUs that are connected to cells (file gwflow.hrucell)',len(CEll_HRU_TABLE.HRUS.unique()))
    print('HRUs that are connected to cells (file gwflow.cellhru)',len(CEll_HRU_TABLE.Cell_ID.unique()))


    writing_SWATplus_GW_input(BASE_PATH, LEVEL, NAME, RESOLUTION, CEll_HRU_TABLE,
                            HRU_CELL_TABLE, CELL_CHANNEL_TABLE, GRIDS_TABLE,
                            num_rows, num_cols, Zone, ZONE_K_TABLE, grids_point,
                            gwflow_target_path, gis_folder, MODEL_NAME)

#    completion_status = execution(gwflow_target_path)  ### setting environment for the program executions

#    try:
#        recharge_df = process_and_plot_recharge(gwflow_target_path, num_rows, num_cols)
#    except Exception as e:
#        completion_status = f"{NAME}: No Recharged saved"


#    create_recharge_image_for_name(gwflow_target_path, LEVEL, NAME, RESOLUTION,
#                                   gis_folder,rech_out_folder,
#                                   start_year, end_year )

    completion_status ='SWATgwflow are prepared'

    return completion_status, GRIDS_TABLE

def clear_directory(directory_path):
    # Check if the directory exists
    if not os.path.exists(directory_path):
        print("Directory does not exist:", directory_path)
        return
    # Iterate over each file and directory inside the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def extract_raster_values_to_points(point_shapefile, raster_path, output_shapefile, gis_folder, column_name):
    arcpy.env.workspace = gis_folder
    arcpy.env.overwriteOutput = True

    # Perform the extraction
    temp_output = "in_memory/temp_output"
    arcpy.sa.ExtractValuesToPoints(point_shapefile, raster_path, temp_output)
    arcpy.AddField_management(temp_output, column_name, "FLOAT")
    arcpy.CalculateField_management(temp_output, column_name, "!RASTERVALU!", "PYTHON3")
    arcpy.CopyFeatures_management(temp_output, output_shapefile)
    arcpy.DeleteField_management(in_table=output_shapefile, drop_field="RASTERVALU")
    print(f"Extracted values saved to: {output_shapefile}")
    return os.path.join(gis_folder, output_shapefile)

def getting_grid_table(outpath_object, gis_folder, gwflow_target_path,feet_to_meter=0.3048):


    Lake_bed_depth = 5.000
    Lake_stage_height= 5.000

    subbasin_path       = outpath_object['Subbasin']
    hru_path            = outpath_object['HRUS']
    channel_path        = outpath_object['Channel']
    elev_path           = outpath_object['Elev']
    zone_path           = outpath_object['Zone']

    try:
        lakes_path          = outpath_object['LakeId']
    except Exception as e:
        lakes_path = None
        print(f'proceeed without lake: {e}')

    Thick_path          = outpath_object['Thick']
    SWL_path            = outpath_object['SWL']
    Bound_path          = outpath_object['Bound']
    Active_path         = outpath_object['Active']
    grids_point_path    = outpath_object['grids_point']
    grids_polygon_path  = outpath_object['grids_polygon']

    grids_point = gpd.read_file(grids_point_path)
    print('check:','Number of grid cells loaded', len(grids_point))

    Thick  = gpd.read_file(Thick_path)
    SWL    = gpd.read_file(SWL_path)
    DEM    = gpd.read_file(elev_path)
    Zone   = gpd.read_file(zone_path)
    # Convert feet to meter in DEM and handle NaN values
    Thick ['Thick'] = feet_to_meter*Thick ['Thick']
    # The minimum thickness is 5 meters
    Thick ['Thick'] =np.where( Thick ['Thick'] <10, 10, Thick ['Thick'] )
    # Convert feet to meter in SWL and handle NaN values
    SWL ['SWL']     = feet_to_meter*SWL ['SWL']
    # The maximum SWL is the thickness
    #SWL['SWL'] = np.where(SWL['SWL'] > Thick['Thick'], Thick['Thick']+1, SWL['SWL'])
    # negative SWL is set to 0
    SWL['SWL'] = np.where(SWL['SWL'] < 0 , 0 , SWL['SWL'])


    if lakes_path:
        Lakes = gpd.read_file(lakes_path).fillna(0)
    else:
        Lakes = pd.DataFrame({'LakeId': [0], 'Cell_ID': [0]})
        Lakes = gpd.GeoDataFrame(Lakes)

    HRUs     = gpd.read_file(hru_path)
    Channels = gpd.read_file(channel_path)


    chandeg = pd.read_csv(os.path.join(gwflow_target_path,'chandeg.con'),skiprows=1, sep='\s+' )
    Channels = Channels.merge(chandeg[['lcha','gis_id']], right_on='gis_id', left_on='Channel').drop(columns='Channel').rename(columns={'lcha':'Channel'})


    Subbasin = gpd.read_file(subbasin_path)
    Bound    = gpd.read_file(Bound_path).drop_duplicates(subset='Cell_ID').fillna(0)
    Active   = gpd.read_file(Active_path).drop_duplicates(subset='Cell_ID').fillna(0)


    ### preparing Grid table
    grids_point = grids_point.merge(Bound[['Bound','Cell_ID']], on='Cell_ID', how='left')
    grids_point = grids_point.merge(Active[['Active','Cell_ID']], on='Cell_ID', how='left')
    grids_point = grids_point.merge(DEM[['Elev','Cell_ID']], on='Cell_ID', how='left')
    grids_point = grids_point.merge(Thick[['Thick','Cell_ID']], on='Cell_ID', how='left')
    grids_point = grids_point.merge(Zone[['Zone','Cell_ID']], on='Cell_ID', how='left')
    grids_point = grids_point.merge(Lakes[['LakeId','Cell_ID']], on='Cell_ID', how='left').rename(columns={"LakeId":"Lake"})
    grids_point = grids_point.merge(SWL[['SWL','Cell_ID']], on='Cell_ID', how='left')
    grids_point = grids_point.fillna(0)

    #### defining lakes
    lake_index = grids_point[grids_point['Lake']!=0].index
    grids_point ['Lake_bed'] = 0
    grids_point ['Lake_stage'] = 0

    grids_point.loc[lake_index, 'Lake_bed'] = (
        grids_point.loc[lake_index, 'Elev'] - Lake_bed_depth
    )
    grids_point.loc[lake_index, 'Lake_stage'] = (
        grids_point.loc[lake_index, 'Elev'] + Lake_stage_height
        )

    #### defining heads
    grids_point ['Head']       = grids_point['Elev'] - grids_point['SWL']

    #grids_point.loc[lake_index, 'Head'] = grids_point.loc[lake_index, 'Elev'] + 1 ### setting the gw head of lakes at their DEM level
    ## check how many negative head we generated
    print('check:', 'Number of negative heads:', len(grids_point[grids_point['Head'] < 0]))

    #grids_point ['Head']       = np.where(grids_point ['Head'] < 0, grids_point['Elev'], grids_point ['Head'])
    # check the number of heads 10m above the elevation
    print('check:', 'Number of heads 10m above the elevation:', len(grids_point[grids_point['Head'] > grids_point['Elev']+10]))
    # if head is too far above the elevation, set it to the elevation
    #  grids_point = grids_point[grids_point.Elev>0].reset_index(drop=True)

    ### nutrient & other parameters
    grids_point ['InitP']      = 0
    grids_point ['InitNO3']    = 0
    grids_point ['Tile']       = 0
    grids_point ['ET_fix']     = 0
    grids_point ['EXDP']       = 0

    num_rows = len(grids_point.Row.unique())
    num_cols = len(grids_point.Col.unique())

    GRIDS_TABLE = grids_point
    print('check:','Total number of grids cell', len(GRIDS_TABLE))
    GRIDS_TABLE_path = os.path.join(gis_folder,'grids_table.txt')
    GRIDS_TABLE.to_csv(GRIDS_TABLE_path)

    int_columns= ['Cell_ID',"Active",'Bound','Lake','Zone']

    float_columns= ['Thick',"Elev",'Head','Lake_bed','Lake_stage','EXDP',"ET_fix",'Tile','InitNO3','InitP']

    GRIDS_TABLE.loc[:, int_columns] = GRIDS_TABLE[int_columns].astype(int)

    GRIDS_TABLE.loc[:, float_columns] = GRIDS_TABLE[float_columns].astype(float).round(2)

    GRIDS_TABLE['Active'] = np.where(GRIDS_TABLE.Bound==2, 2, GRIDS_TABLE.Active)

    print('number of zones:',GRIDS_TABLE.Zone.unique())
    GRIDS_TABLE['Zone'] = np.where(GRIDS_TABLE.Zone==0, 1, GRIDS_TABLE.Zone)
    print('number of zones:',GRIDS_TABLE.Zone.unique())

    GRIDS_TABLE = GRIDS_TABLE[[

            "Cell_ID", "Active", "Thick", "Elev",
                    "Zone","Bound", "Head", "EXDP", "ET_fix",
                    "Tile","InitNO3", "InitP","Lake",
                        "Lake_bed","Lake_stage"
                ]]

    return grids_point ,GRIDS_TABLE,  GRIDS_TABLE_path, num_rows, num_cols


def creating_zones(path_to_HHC, path_to_aq_zone, gis_folder, feet_to_meter=0.306):

    HHC = gpd.read_file(path_to_HHC)
    aq_zone = gpd.read_file(path_to_aq_zone)    #### NOTE: we import the zone for aquifer characteristics, but I did not use it. We need an initial estimate of Sy for these zone before integrating them into the model
    # Convert feet to meter in HHC and handle NaN values
    HHC['HHC'] = HHC['HHC'].apply(lambda x: x * feet_to_meter if np.isfinite(x) else np.nan)
    # Calculate quantiles excluding NaN values
    first_quartile = np.nanpercentile(HHC.HHC.dropna(), 25)
    median = np.nanpercentile(HHC.HHC.dropna(), 50)
    third_quartile = np.nanpercentile(HHC.HHC.dropna(), 75)
    # Print quantiles for verification

    print('first_quartile:', first_quartile)
    print('median:', median)
    print('third_quartile:', third_quartile)

    # Create a copy to avoid SettingWithCopyWarning
    Zone = HHC[['Cell_ID', 'HHC', 'geometry']].copy()
    # Initialize 'Zone' column and assign values considering NaN
    Zone['Zone'] = np.select(

        [   Zone['HHC'].isna(),
            Zone['HHC'] <= first_quartile,
            (Zone['HHC'] > first_quartile) & (Zone['HHC'] <= median),
            (Zone['HHC'] > median) & (Zone['HHC'] <= third_quartile),
            Zone['HHC'] > third_quartile],
        [
            np.nan, 1, 2, 3, 4
        ],
        default=np.nan
    )

    # Set 'Sy' values incrementally for each zone
    for zone_number in range(1, 5):
        increment = zone_number * 0.05
        Zone.loc[Zone['Zone'] == zone_number, 'Sy'] = increment

    # Calculate representative value (mean) for each zone
    for zone_number in range(1, 5):
        zone_HHC_values = Zone[Zone['Zone'] == zone_number]['HHC']
        k = np.nanmean(zone_HHC_values)
        Zone.loc[Zone['Zone'] == zone_number, 'k'] = k

    outpath = os.path.join(gis_folder, 'Zone_Shape.shp')
    Zone.to_file(outpath)
    print('Zones are created and saved:', 'Zone_Shape.shp')

    Zone['n'] = 0.3

    ZONE_K_TABLE = Zone[['Zone', 'k', 'Sy', 'n']]
    ZONE_K_TABLE = ZONE_K_TABLE.groupby('Zone').mean().reset_index(drop=False)

    ZONE_K_TABLE['Zone'] = ZONE_K_TABLE['Zone'].astype(int)
    ZONE_K_TABLE['k'] = ZONE_K_TABLE['k'].astype(float).round(2)
    ZONE_K_TABLE['Sy'] = ZONE_K_TABLE['Sy'].astype(float).round(2)
    ZONE_K_TABLE['n'] = ZONE_K_TABLE['n'].astype(float).round(2)


    return Zone, ZONE_K_TABLE, outpath

def write_table(table, file):
    try:
        table.to_csv(file, sep='\t', header=False, index=False, line_terminator='\n')
    except Exception as e:
        print(f"Error writing table: {e}")


def setting_env(BASE_PATH, LEVEL, NAME, start_year, end_year, MODEL_NAME, SWAT_MODEL_NAME):

    """
        Setting the environment
    """

    # Pathways for local and output files
    gwflow_target_path = os.path.join(BASE_PATH, f"SWAT_input/{LEVEL}/{NAME}/{MODEL_NAME}/Scenarios/Default/TxtInOut")
    gwflow_input_target_path = os.path.join(BASE_PATH, f"SWAT_input/{LEVEL}/{NAME}/{MODEL_NAME}")
    clear_directory(gwflow_target_path)
    gwflow_data = os.path.join(gwflow_target_path, "gwflow.data")
    gis_folder = os.path.join(gwflow_input_target_path, "gwflow_gis")
    rech_out_folder = os.path.join(gwflow_input_target_path, "recharg_output")

    os.makedirs(gis_folder, exist_ok=True)
    os.makedirs(rech_out_folder, exist_ok=True)

    ### copy necessary files to the target directory
    shutil.copytree (fr"/data/MyDataBase/SWATGenXAppData/SWAT_input/{LEVEL}/{NAME}/{SWAT_MODEL_NAME}/Scenarios/Default/TxtInOut", gwflow_target_path,dirs_exist_ok=True)
 #   shutil.copy2    (fr"/data/MyDataBase/SWATGenXAppData/bin/gwflow_input_files.exe", gwflow_target_path)
    shutil.copy2("/data/MyDataBase/SWATGenXAppData/bin/swatplus.exe", gwflow_target_path)

    ### make changes to the time
    with open (os.path.join(gwflow_target_path, 'time.sim'), 'w') as file:
        file.write('time.sim: written by Vahid for SWAT+gwflow\n')
        file.write('day_start  yrc_start   day_end   yrc_end      step  \n')
        file.write(f' 0   {start_year}     0   {end_year}      0  \n')

    #### info outout
    print('Target path:', gwflow_target_path)
    print('environment:',gis_folder)
    ### clear target directory for previous run
    clear_directory(gis_folder)


    return gwflow_data, gwflow_target_path, gis_folder, rech_out_folder

def defining_bound_and_active(subbasin_path, gis_folder, RESOLUTION):
    print('#################defining bound and active area using subbasin geometry#############')
    Subbasin = gpd.read_file(subbasin_path)

    active_domain = Subbasin.dissolve().reset_index(drop=True)
    active_domain = active_domain.set_geometry('geometry').copy()
    active_domain['Active'] = 1

    bound = active_domain.boundary.copy()
    bound = bound.explode(index_parts=False)
    bound = bound[bound.length == bound.length.max()]
    bound = bound.buffer(RESOLUTION)
    bound = gpd.GeoDataFrame(geometry=bound)
    bound.crs = active_domain.crs
    bound['Bound'] = 2

    bound_path = os.path.join(gis_folder, 'bound_shape.shp')
    active_domain_path = os.path.join(gis_folder, 'active_domain_shape.shp')
    bound[['Bound','geometry']].to_file(bound_path)
    active_domain[['Active','geometry']].to_file(active_domain_path)

    print('Generated shap saved to:',os.path.basename(bound_path))
    print('Generated shape saved to:',os.path.basename(active_domain_path))

    return bound_path, active_domain_path


def process_path_objects(path_objects, grids_point_path, gis_folder, grids_polygon):
    # sourcery skip: avoid-builtin-shadow

    """
      processing rasters and shapefiles.
      Rasters using Arcpy & centroid sampling
      Shapefiles using geopandas and overlay
    """

    outpath_object = {}

    for item in path_objects:
        path = item['path']
        name = item['name']
        Type = item['Type']

        if Type == 'raster':
            outpath = extract_raster_values_to_points(grids_point_path, path, f'{name}_shape.shp', gis_folder, name)
            outpath_object[name] = outpath

        elif Type == "shape":

            object = gpd.read_file(path)[[name, 'geometry']].set_geometry('geometry').to_crs('EPSG:26990')


            if name in ['HRUS', 'Channel']:
                object[name] = object[name].astype(int)
                print(f'length of {name} array:', len(object[name]))
                print(f'Maximum value of {name} ID:', object[name].max())
                print(f'Unique {name} IDs:', len(object[name].unique()))

            if name=='HRUS':
                object['HRUs_Area'] = object.geometry.area
                object['HRUs_Area'] = object['HRUs_Area'].astype(float).round(2)
                data = gpd.overlay(object[[name,'HRUs_Area' ,'geometry']], grids_polygon[["Cell_ID", 'geometry']], how='intersection')
                print(f"Extracted values saved to: {name}_shape.shp")
                outpath = os.path.join(gis_folder, f'{name}_shape.shp')
                data[['Cell_ID','HRUs_Area', name, 'geometry']].to_file(outpath)

            else:
                data = gpd.overlay(object[[name, 'geometry']], grids_polygon[["Cell_ID", 'geometry']], how='intersection')
                print(f"Extracted values saved to: {name}_shape.shp")
                outpath = os.path.join(gis_folder, f'{name}_shape.shp')
                data[['Cell_ID', name, 'geometry']].to_file(outpath)
            outpath_object[name] = outpath

    return outpath_object

def Setting_env_and_returning_object_path(BASE_PATH, NAME, LEVEL, RESOLUTION,start_year, end_year, MODEL_NAME,MODFLOW_MODEL_NAME, SWAT_MODEL_NAME):
    if "_ML_" in MODFLOW_MODEL_NAME:
        estimation_method = "predictions_ML"
    else:
        estimation_method = "_kriging_output"


    ## setting the environment
    gwflow_data, gwflow_target_path, gis_folder, rech_out_folder = setting_env(BASE_PATH, LEVEL, NAME, start_year, end_year, MODEL_NAME, SWAT_MODEL_NAME)
    arcpy.env.overwriteOutput = True
    #### Modflow and groundwater data
    Modflow_directory    = fr"/data/MyDataBase/SWATGenXAppData/SWAT_input/{LEVEL}/{NAME}/{MODFLOW_MODEL_NAME}/"                                ## directory
    grids_polygon_path   = os.path.join(Modflow_directory, "Grids_MODFLOW/Grids_MODFLOW.shp")                                    ## file
    GW_RASTERS_dire      = os.path.join(Modflow_directory,"rasters_input/")                                        ## directory
    thickrast_path       = os.path.join(GW_RASTERS_dire, f"{NAME}{estimation_method}_AQ_THK_1_{RESOLUTION}m.tif.tif")  # EPSG:26990
    perm_path            = os.path.join(GW_RASTERS_dire, f"{NAME}{estimation_method}_H_COND_1_{RESOLUTION}m.tif.tif")  # EPSG:26990
    aq_zone_path_original = os.path.join(BASE_PATH, fr"all_rasters/Aquifer_Characteristics_Of_Glacial_Drift_{RESOLUTION}m.tif") # EPSG:26990
    aq_zone_path         = os.path.join(GW_RASTERS_dire, fr"Aquifer_Characteristics_Of_Glacial_Drift_{RESOLUTION}m.tif") # EPSG:26990
    arcpy.Copy_management(aq_zone_path_original, aq_zone_path)
    SWL_path             = os.path.join(GW_RASTERS_dire, f"{NAME}{estimation_method}_SWL_{RESOLUTION}m.tif.tif")       # EPSG:26990
    domain_path          = os.path.join(GW_RASTERS_dire, f"{NAME}_raster.tif")                                     # EPSG:26990
    DEM_path             = os.path.join(GW_RASTERS_dire, f"{NAME}_DEM_{RESOLUTION}m.tif.tif")                               # EPSG:26990

    #### SWAT and its directories
    SWAT_model_directory = fr"/data/MyDataBase/SWATGenXAppData/SWAT_input/{LEVEL}/{NAME}/"
    SWAT_shape_dire      = os.path.join(SWAT_model_directory, f"{SWAT_MODEL_NAME}/Watershed/Shapes/")
    hrus_path            = os.path.join(SWAT_shape_dire, "hrus2.shp")            # EPSG:26990
    streams_path         = os.path.join(SWAT_shape_dire, "rivs1.shp")            # EPSG:26990
    subbasins_path       = os.path.join(SWAT_shape_dire, "subs1.shp")            # EPSG:26990
    lakes_path           = os.path.join(SWAT_shape_dire, "SWAT_plus_lakes.shp")  # EPSG:26990

    ### defining bound and active area using subbasin geometry
    bound_path, active_domain_path = defining_bound_and_active(subbasins_path, gis_folder, RESOLUTION)

    if os.path.exists(lakes_path):
        ### creating path&object dictionary
        input_path_object = [
            { 'Type':'shape' , 'path': hrus_path,              'name':'HRUS'    },
            { 'Type':'shape' , 'path': bound_path,             'name':'Bound'   },
            { 'Type':'shape' , 'path': active_domain_path,     'name':'Active'  },
            { 'Type':'shape' , 'path': streams_path,           'name':'Channel' },
            { 'Type':'shape' , 'path': subbasins_path,         'name':'Subbasin'},
            { 'Type':'shape' , 'path': lakes_path,             'name':'LakeId'  },
            { 'Type':'raster', 'path': thickrast_path,         'name':'Thick'   },
            { 'Type':'raster', 'path': aq_zone_path,           'name':'aq_zone' },
            { 'Type':'raster', 'path': perm_path,              'name':'HHC'     },
            { 'Type':'raster', 'path': SWL_path,               'name':'SWL'     },
            { 'Type':'raster', 'path': DEM_path,               'name':'Elev'    }    ]
    else:
        input_path_object = [
            { 'Type':'shape' , 'path': hrus_path,              'name':'HRUS'    },
            { 'Type':'shape' , 'path': bound_path,             'name':'Bound'   },
            { 'Type':'shape' , 'path': active_domain_path,     'name':'Active'  },
            { 'Type':'shape' , 'path': streams_path,           'name':'Channel' },
            { 'Type':'shape' , 'path': subbasins_path,         'name':'Subbasin'},
            { 'Type':'raster', 'path': thickrast_path,         'name':'Thick'   },
            { 'Type':'raster', 'path': aq_zone_path,           'name':'aq_zone' },
            { 'Type':'raster', 'path': perm_path,              'name':'HHC'     },
            { 'Type':'raster', 'path': SWL_path,               'name':'SWL'     },
            { 'Type':'raster', 'path': DEM_path,               'name':'Elev'    }    ]

    ### reading grids, creating centroid, saving, returning path and importing their polygon geometry
    grids_polygon , grids_point_path = read_grids (grids_polygon_path, gis_folder,  CRS=26990)
    #processing rasters and shapefiles. Rasters using Arcpy for centroid sampling & Shapefiles using geopandas for overlay on grids
    outpath_object = process_path_objects(input_path_object, grids_point_path, gis_folder, grids_polygon)

    return {
        'outpath_object': outpath_object,
        'gis_folder': gis_folder,
        'gwflow_target_path': gwflow_target_path,
        'grids_polygon': grids_polygon,
        'grids_point_path': grids_point_path,
        'grids_polygon_path': grids_polygon_path,
        'rech_out_folder': rech_out_folder,
    }



def read_grids(grids_polygon_path, gis_folder, CRS):

    temp_grids = gpd.read_file(grids_polygon_path)
    temp_grids ['Cell_ID'] = np.arange(1, len(temp_grids)+1)

    grids_polygon=  gpd.GeoDataFrame(temp_grids, crs=CRS, geometry='geometry').copy()

    temp_grids['centroid'] = temp_grids.geometry.centroid
    temp_grids.drop(columns='geometry', inplace=True)
    temp_grids.rename(columns={'centroid': 'geometry'}, inplace=True)
    grids_point = gpd.GeoDataFrame(temp_grids, geometry='geometry', crs=CRS)
    grids_point_path= os.path.join(gis_folder, 'grids_points.shp')
    grids_point.to_file(grids_point_path)

    return grids_polygon, grids_point_path


def execution(gwflow_target_path):

    os.chdir(gwflow_target_path)

    completion_status = 'not started'

#    try:
#        command = "gwflow_input_files.exe"    ####executing gwflow input creation program
#        subprocess.run(command, shell=False, cwd=gwflow_target_path)
#        completion_status = 'done'
#    except Exception as e:
#        completion_status= f"gwflow input creation failed {e}"
#        print(f"gwflow input creation failed due to {e}")

#    if completion_status == 'done':
    try:
        command = "swatplus.exe"
        subprocess.run(command, shell=False, cwd=gwflow_target_path)    ####executing SWAT+gwflow model
        completion_status = 'model execution finished'
    except Exception as e:
        completion_status= f"gwflow model creation failed {e}"
        print(f"gwflow input creation failed due to {e}")

    return completion_status

def getting_CELL_CHANNEL_HRU_TABLES(outpath_object, RESOLUTION,gwflow_target_path):

    HRUs         = gpd.read_file(outpath_object['HRUS'])
    DEM          = gpd.read_file(outpath_object['Elev'])
    Zone         = gpd.read_file(outpath_object['Zone'])
    Channels     = gpd.read_file(outpath_object['Channel'])
    chandeg      = pd.read_csv(os.path.join(gwflow_target_path,'chandeg.con'),skiprows=1, sep='\s+' )
    Channels     = Channels.merge(chandeg[['lcha','gis_id']], right_on='gis_id', left_on='Channel').drop(columns='Channel').rename(columns={'lcha':'Channel'})
    Subbasin     = gpd.read_file(outpath_object['Subbasin'])
    active_cells = Subbasin.Cell_ID.unique()
    HRUs         = HRUs.assign(poly_Area=HRUs.geometry.area, Cell_Area=RESOLUTION*RESOLUTION).drop(columns='geometry')
    HRUs = HRUs.merge(DEM[['Elev','Cell_ID']],  on='Cell_ID', how='left')



    columns_to_convert = {
        'HRUs_Area': (float, 2),
        'poly_Area': (float, 2),
        'Cell_Area': (int, None),
        'Cell_ID': (int, None),
        'HRUS': (int, None)
    }

    for column, (data_type, precision) in columns_to_convert.items():
        if precision is not None:
            HRUs[column] = HRUs[column].astype(data_type).round(precision)
        else:
            HRUs[column] = HRUs[column].astype(data_type)

    ### sorted first by CELL number then by HRU    ['Cell_ID','Area_m2','HRU','poly_Area']
    HRUs = HRUs.sort_values(by=['Cell_ID'], ascending=True).reset_index(drop=True)
    ### filtering small intersections and where Elev is 0
    HRUs = HRUs[HRUs.poly_Area>10].reset_index(drop=True)
    HRUs = HRUs[HRUs.HRUs_Area>10].reset_index(drop=True)
    HRUs = HRUs[HRUs.Elev>0].reset_index(drop=True)

    HRU_CELL_TABLE = HRUs[['Cell_ID', 'HRUs_Area', 'HRUS', 'poly_Area']].copy()

    #### sorted first by cell_id and then by HRU number  ['Cell_ID','Area_m2', 'HRU', 'poly_Area']
    HRUs = HRUs.sort_values(by=['HRUS'], ascending=True).reset_index(drop=True)
    CEll_HRU_TABLE = HRUs[['Cell_ID', 'HRUs_Area', 'HRUS', 'poly_Area']].copy()

    ## preparing channel table
    Channels = Channels.merge(DEM[['Elev','Cell_ID']],  on='Cell_ID', how='left')
    Channels = Channels.merge(Zone[['Zone','Cell_ID']], on='Cell_ID', how='left')
    Channels = Channels.assign(Length=Channels.geometry.length).drop(columns='geometry')
    Channels = Channels.drop_duplicates(subset=['Cell_ID','Channel'])
    Channels = Channels.sort_values(by=['Cell_ID'], ascending=True)
    Channels['Elev'   ] = Channels['Elev'].astype(float).round(2)
    Channels['Length' ] = Channels['Length'].astype(float).round(2)
    Channels['Cell_ID'] = Channels['Cell_ID'].astype(int)
    Channels['Zone'   ] = Channels['Zone'].astype(int)
    Channels['Channel'] = Channels['Channel'].astype(int)
    Channels = Channels[Channels.Elev>0].reset_index(drop=True)
    Channels = Channels[Channels.Length>10].reset_index(drop=True)
    CELL_CHANNEL_TABLE = Channels[['Cell_ID','Elev' ,'Channel','Length','Zone']].copy()

    print('Length of channels_cell array:' , len(Channels))
    print('Maximum number of Channels:'    , Channels['Channel'].max())
    print('Length of CELL_HRUs array:'     , len(HRUs))
    print('Maximum number of HRUs:'        , HRUs['HRUS'].max())


    return CEll_HRU_TABLE, HRU_CELL_TABLE, CELL_CHANNEL_TABLE


def process_and_plot_recharge(gwflow_target_path, num_rows, num_cols):
    recharge = pd.DataFrame(np.nan, index=range(num_rows), columns=range(num_cols))

    with open(os.path.join(gwflow_target_path, "gwflow_flux_recharge"), 'r') as file:
        row_index = 0
        for line in file:
            if "Annual" in line or "Recharge" in line:
                continue
            values = np.array(line.split(), dtype=float)
            if row_index < num_rows:
                recharge.iloc[row_index, :] = values[:num_cols]
                row_index += 1
            if row_index >= num_rows:
                break

    recharge.replace(0, np.nan, inplace=True)

    plt.imshow(np.array(recharge))
    plt.colorbar()
    plt.title("Recharge Data Visualization")
    plt.savefig(os.path.join(gwflow_target_path,'recharge.jpeg'), dpi=400)
    plt.show()
    return recharge




def writing_SWATplus_GW_input(BASE_PATH, LEVEL, NAME, RESOLUTION, CEll_HRU_TABLE, HRU_CELL_TABLE, CELL_CHANNEL_TABLE, GRIDS_TABLE, num_rows, num_cols, Zone, ZONE_K_TABLE, grids_point, gwflow_target_path, gis_folder, MODEL_NAME):

    shutil.copy2("/data/MyDataBase/SWATGenXAppData/bin/swatplus.exe", gis_folder)
    # Reading the file
    rout_unit = pd.read_csv(
        os.path.join(f"/data/MyDataBase/SWATGenXAppData/SWAT_input/{LEVEL}/{NAME}/{MODEL_NAME}/Scenarios/Default/TxtInOut/rout_unit.con"),
        skiprows=2,
        sep='\s+',
        names=['id', 'name', 'gis_id', 'area', 'lat', 'lon', 'elev', 'rtu', 'wst', 'cst', 'ovfl', 'rule', 'out_tot', 'obj_typ', 'obj_id', 'hyd_typ', 'frac','clo1', 'col2', 'col3', 'col4']
    )


    grids_point = grids_point.drop_duplicates(subset='Cell_ID').sort_values('Cell_ID').reset_index(drop=True)
    # Dropping unnecessary columns
    rout_unit = rout_unit.drop(columns=['clo1', 'col2', 'col3', 'col4'])
    rout_unit['out_tot'] = 1

    # Custom header with timestamp
    header = f"Routing unit configuration file, Edited by Vahid, Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    # Writing to the file with the header and formatted data

    with open(os.path.join(gwflow_target_path, r"rout_unit.con"), 'w') as file:

        file.write(header + '\n')  # Custom header with timestamp
        # Write column names
        col_line = '\t'.join(rout_unit.columns)
        file.write(col_line + '\n')
        # Write data with aligned columns
        max_widths = [
            max(len(str(x)) for x in rout_unit[col])
            for col in rout_unit.columns
        ]
        for _, row in rout_unit.iterrows():
            line = '\t'.join([str(row[col]).ljust(max_widths[i]) for i, col in enumerate(rout_unit.columns)])
            file.write(line + '\n')

    print(f"File saved to {gwflow_target_path}")

    object_cnt = pd.read_csv(
        os.path.join(gwflow_target_path, "object.cnt"), skiprows=1, sep='\s+'
    )

    object_cnt.aqu = 0                                         # Modifying the DataFrame
    object_cnt.mfl = len(CELL_CHANNEL_TABLE)
    object_cnt.obj = object_cnt.obj + len(CELL_CHANNEL_TABLE)

    header = f"Edited by Vahid, Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"               # Custom header
    max_widths = [max(len(str(object_cnt[col].iloc[0])), len(col)) for col in object_cnt.columns]            # Determine the maximum width for each column
    formatted_columns = ' '.join([col.ljust(width) for col, width in zip(object_cnt.columns, max_widths)])   # Prepare column names and data row with alignment
    formatted_data = ' '.join([str(val).ljust(width) for val, width in zip(object_cnt.iloc[0], max_widths)])

    with open(os.path.join(gwflow_target_path,r"object.cnt"), 'w') as file:  # Writing to the file
        file.write(header + '\n')
        file.write(formatted_columns + '\n')
        file.write(formatted_data)

    with open(fr'/data/MyDataBase/SWATGenXAppData/SWAT_input/{LEVEL}/{NAME}/{MODEL_NAME}/Scenarios/Default/TxtInOut/file.cio','r') as file:
        lines = file.readlines()
    lines[4] = 'connect\thru.con\tnull\trout_unit.con\tgwflow.con\tnull\tnull\tnull\treservoir.con\tnull\tnull\tnull\tnull\tchandeg.con\t\n'
    with open(os.path.join(gwflow_target_path,'file.cio'),'w') as file:
        file.writelines(lines)

    # Writing gwflow.hrucell file
    with open(os.path.join(gwflow_target_path,'gwflow.hrucell'), 'w') as file:
        file.write(f"CELL_HRU Connection Information\n\nHRUs connected to cells,  Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        hrus = CEll_HRU_TABLE.sort_values('HRUS', ascending=True).HRUS.unique()
        file.write(f'{len(hrus)}\n')
        for hru in hrus:
            file.write(f'\t{hru}\n')
        file.write('\n')
        file.write('HRU_ID    HRU_AREA   CELL_ID   overlap_area_m2\n')
        for cell_id, hru_area, hru, overlap_area in zip(CEll_HRU_TABLE.Cell_ID, CEll_HRU_TABLE.HRUs_Area, CEll_HRU_TABLE.HRUS, CEll_HRU_TABLE.poly_Area):
            file.write(f'\t{hru}\t{hru_area}\t{cell_id}\t{overlap_area}\n')

    # Writing gwflow.cellhru file
    with open(os.path.join(gwflow_target_path,'gwflow.cellhru'), 'w') as file:
        HRU_CELL_TABLE['cell_size'] = RESOLUTION * RESOLUTION
        file.write(f"HRU-CELL Connection Information\n Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write(f'{len(HRU_CELL_TABLE.Cell_ID.unique())}\n')
        file.write('CELL_ID    HRU_ID    CELL_AREA    overlap_area_m2\n')
        for cell_id, hru, cell_size, overlap_area in zip(HRU_CELL_TABLE.Cell_ID, HRU_CELL_TABLE.HRUS, HRU_CELL_TABLE.cell_size, HRU_CELL_TABLE.poly_Area):
            file.write(f'\t{cell_id}\t{hru}\t{cell_size}\t{overlap_area}\n')

    # Writing gwflow.rivcells file
    with open(os.path.join(gwflow_target_path,'gwflow.rivcells'), 'w') as file:
        file.write(f"Cell-Channel Connection Information\n Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write('CELL_ID  Elev_m  CHANNEL_ID  CHANNEL_length_m  ZONE\n')
        for Cell_ID, Elev, Channel, Length, Zone in zip(CELL_CHANNEL_TABLE.Cell_ID, CELL_CHANNEL_TABLE.Elev, CELL_CHANNEL_TABLE.Channel, CELL_CHANNEL_TABLE.Length, CELL_CHANNEL_TABLE.Zone):
            file.write(f'\t{Cell_ID}\t{Elev}\t{Channel}\t{Length}\t{Zone}\n')

    ## writing gwflow.input
    with open(os.path.join(gwflow_target_path, "gwflow.input"), 'w') as file:
        file.write('    INPUT FOR GWFLOW MODULE - written by Vahid\t')
        file.write(f" Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write('    Basic information\n')
        file.write(f"    {RESOLUTION}\n")                    ## cell size
        file.write(f"    {num_rows}\t{num_cols}\n")          ## row & column
        file.write('    3\n')                                ## water table initation plan
        file.write('    10\n')                               ## initial water table flag
        file.write('    1\n')                                ## boundary condition type
        file.write('    1\n')                                ## flag to simulate groundwater-soil interactions
        file.write('    1\n')                                ## flag to simulate saturation excess routing
        file.write('    1\n')                                ## flag to simulate groundwater et
        file.write('    1\n')                                ## flag to simulate groundwater-tile drain exchange
        file.write('    1\n')                                ## flag to simulate groundwater-lake exchange
        file.write('    0\n')                                ## flag to simulate specified groundwater pumping
        file.write('    1\n')                                ## flag to simulate recharge
        file.write('    3\n')                                ## GW delay
        file.write('    0\n')                                ## gw transport flag
        file.write('    1\n')                                ## user specified time step
        file.write('    1\t1\t1\n\n')                        ## groundwater day, year and annual average

        file.write(' Aquifer and Streambed Parameter Zones\n')
        file.write(' Aquifer Hydraulic Conductivity (m/day) Zones\n')

        file.write(f'           {len(ZONE_K_TABLE.Zone)}\n')   ## aq_ks zones and values

        for zone, k in zip(ZONE_K_TABLE.Zone, ZONE_K_TABLE.k):
            file.write(f"           {zone}\t{k}\n")

        file.write(' Aquifer Specific Yield Zones\n')
        file.write(f'           {len(ZONE_K_TABLE.Zone)}\n')    ## groundwater sy zones

        for zone, Sy in zip(ZONE_K_TABLE.Zone, ZONE_K_TABLE.Sy):
            file.write(f"\t{zone}\t{Sy}\n")

        file.write(' Aquifer Porosity Zones\n')                ## aquifer prosity zone and values
        file.write(f'           {len(ZONE_K_TABLE.Zone)}\n')

        for zone, n in zip(ZONE_K_TABLE.Zone, ZONE_K_TABLE.n):
            file.write(f"           {zone}\t{n}\n")

        file.write(' Streambed Hydraulic Conductivity (m/day) Zones\n')
        file.write('           4\n')
        file.write('           1  4.9999999E-03\n')
        file.write('           2  4.9999999E-03\n')
        file.write('           3  4.9999999E-03\n')
        file.write('           4  4.9999999E-03\n')
        file.write(' Streambed Thickness (m) Zones\n')
        file.write('           4\n')
        file.write('           1   1.000000\n')
        file.write('           2   1.000000\n')
        file.write('           3   1.000000\n')
        file.write('           4   1.000000\n\n')
        file.write(' Grid Cell Information\n')
        file.write('Cell_ID   Active Elev   Thick  K_zone   Sy_zone   n_zone  expP  ET_fix  Tile  IniNO3  InitP  Lake   Lake_bed   Lake_shape\n')

        GRIDS_TABLE = GRIDS_TABLE.drop_duplicates(subset='Cell_ID').sort_values('Cell_ID').reset_index(drop=True)
        GRIDS_TABLE['Active'] = np.where(GRIDS_TABLE.Elev==0,0, GRIDS_TABLE.Active )
        GRIDS_TABLE['Thick'] = np.where(GRIDS_TABLE.Thick<5,5, GRIDS_TABLE.Thick )


        for Cell_ID, Active, Elev, Thick, Zone, Zone, Zone, EXDP, ET_fix, Tile, InitNO3, InitP, Lake, Lake_bed, Lake_stage in zip(GRIDS_TABLE.Cell_ID, GRIDS_TABLE.Active,GRIDS_TABLE.Elev,GRIDS_TABLE.Thick,GRIDS_TABLE.Zone,GRIDS_TABLE.Zone,GRIDS_TABLE.Zone,GRIDS_TABLE.EXDP,GRIDS_TABLE.ET_fix,GRIDS_TABLE.Tile,GRIDS_TABLE.InitNO3,GRIDS_TABLE.InitP,GRIDS_TABLE.Lake,GRIDS_TABLE.Lake_bed,GRIDS_TABLE.Lake_stage):  #### THIS PROBLEMATO
            file.write(f'           {Cell_ID}\t{Active}\t{Elev}\t{Thick}\t{Zone}\t{Zone}\t{Zone}\t{EXDP}\t{ET_fix}\t{Tile}\t{InitNO3}\t{InitP}\t{Lake}\t{Lake_bed}\t{Lake_stage}\n')

        grids_point = grids_point.drop_duplicates(subset='Cell_ID').sort_values('Cell_ID').reset_index(drop=True)
        file.write(' Initial Groundwater Head\n')

        for row in grids_point.Row.unique():
            for col in grids_point.Col.unique():
                head = grids_point.loc[(grids_point.Row == row) & (grids_point.Col == col), 'Head'].values[0]
                file.write(f" {head}")
            file.write('\n')

        file.write(" Times for Groundwater Head Output\n")
        file.write(" 1\n")
        file.write("    2002      10\n")
        file.write(" Groundwater Observation Locations\n")
        file.write(" 0\n")
        file.write(" Cell for detailed daily sources/sink output\n")
        file.write(" Row     Column\n")
        file.write("           1           1\n")
        file.write(" River Cell Information\n")
        file.write("   5.000000\n")
        file.write(" Hydrograph separation\n")
        file.write(" 1\n")
        file.write(" 1\n")
        file.write(" Tile Drain Information\n")
        file.write("   1.220000\n")
        file.write("   50.00000\n")
        file.write("   5.000000\n")
        file.write("          0\n")
        file.write(" Lake Information\n")
        file.write("   2.000000\n")
        file.write("   4.9999999E-03\n")
        file.write("   2.000000\n")
        file.write("   9.9999998E-03\n")



def create_recharge_image_for_name(gwflow_target_path, LEVEL, NAME, RESOLUTION,gis_folder,rech_out_folder, start_year, end_year):

    """ this function create annual recharge figures
    processes:
        1- removing previous figures
        2- checking the gwflow_flux_recharge exists:
        3- reading row and columns from gwflow.inpu
        4- reading grid arrays for extracting active domain
        5- reading gwflow flux recharge
        6- removing out of active zone values
        7- plotting gwflow recharge figures
    """
    print('begin')
    number_of_years = end_year - start_year + 1

    files=glob.glob(os.path.join(gwflow_target_path , '*jpeg'))
    for file in files:
        os.remove(file)

    files=glob.glob(os.path.join(gwflow_target_path, '*shp'))
    for file in files:
        os.remove(file)

    path = os.path.join(gwflow_target_path,  "gwflow_flux_recharge")

    if not os.path.exists(path):
        message = 'gwflow_flux_recharge does not exist. check the print options'
        print(message)
        return message
    # Read nrows and ncols
    with open(os.path.join(gwflow_target_path ,"gwflow.input"), 'r') as file:
        lines = file.readlines()
        nrows, ncols = map(int, lines[3].split())

    # Initialize the recharge array
    domain = np.zeros([nrows, ncols])

    # Read the grid arrays file
    with open(os.path.join(gwflow_target_path,'gwflow_grid_arrays'), 'r') as file:
        m = 0  # row index
        for line in file:
            if "inactive" in line:
                continue
            for i, col_value in enumerate(line.split()):
                domain[m, i] = int(col_value)
            m += 1
            if m >= nrows:  # Stop if you've read enough rows
                break

    # Plot the domain
    plt.imshow(domain)
    plt.colorbar()
    plt.title("Active Domain")
    plt.show()
    recharge = np.zeros([number_of_years, nrows, ncols])

    # Read the recharge data
    year_index = 0  # Index for the year
    row_index = 0  # Index for the row within a year

    with open(os.path.join( gwflow_target_path,"gwflow_flux_recharge"), 'r') as file:
        for line in file:
            if "Recharge" in line or 'Annual' in line or line.strip() == '':
                if row_index != 0:  # If it's not the first line of a year
                    year_index += 1  # Move to the next year
                    row_index = 0  # Reset row index for the new year
                continue
            # Split the line into columns and assign to recharge array
            try:
                recharge[year_index, row_index, :] = np.array(line.split(), dtype=float)
                row_index += 1
            except Exception as e:
                print('failed reading year index in gwflow flux recharge')


    for i, year in enumerate(range(start_year, end_year)):  # Example years
        # Apply domain mask to recharge data
        if year==start_year:
            continue
        masked_recharge = np.where(domain == 1, recharge[i], np.nan)

        # Calculate 95th percentile of the non-NaN values
        percentile_95 = np.nanpercentile(masked_recharge, 95)

        # Plot
        plt.imshow(masked_recharge, vmax=percentile_95)
        plt.colorbar()
        plt.title(f"Recharge {year}")

        # Save plot
        plt.savefig(os.path.join(gwflow_target_path, f'recharge_{year}.jpeg'), dpi=400)
        plt.show()

        create_recharge_shapefile(LEVEL, NAME,RESOLUTION, masked_recharge, year,rech_out_folder)

        print('done')

def create_recharge_shapefile(LEVEL, NAME,RESOLUTION, recharge, year,rech_out_folder):
    input_grids_path = f'/data/MyDataBase/SWATGenXAppData/SWAT_input/{LEVEL}/{NAME}/{MODFLOW_MODEL_NAME}/Grids_MODFLOW/Grids_MODFLOW.shp'
    output_recharge_path = os.path.join(rech_out_folder, f'recharge_{year}.shp')
    SWAT_dem_path = f'/data/MyDataBase/SWATGenXAppData/SWAT_input/{LEVEL}/{NAME}/DEM_{RESOLUTION}m.tif'
    output_recharge_raster_path = os.path.join(rech_out_folder, f'recharge_{year}.tif')

    # Read the grid shapefile
    grids = gpd.read_file(input_grids_path)

    # Assign recharge values to each grid cell
    grids['Recharge'] = [recharge[row, col] for row, col in zip(grids['Row'], grids['Col'])]

    # Save the new shapefile
    grids.to_file(output_recharge_path)

    # Read DEM for spatial reference
    with rasterio.open(SWAT_dem_path) as dem:
        dem_meta = dem.meta.copy()

    shapes = ((geom, value) for geom, value in zip(grids.geometry, grids['Recharge']))

    # Create a raster image based on the shapefile
    with rasterio.open(output_recharge_raster_path, 'w', **dem_meta) as out_raster:
        out_raster.write_band(1, rasterize(shapes, out_shape=out_raster.shape, fill=np.nan, transform=out_raster.transform))
