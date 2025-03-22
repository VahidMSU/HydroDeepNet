import warnings
import flopy
import pyproj
import shutil
import numpy as np
from MODGenX.utils import *
from MODGenX.rivers import river_gen, river_correction
from MODGenX.lakes import lakes_to_drain
from MODGenX.visualization import plot_data, create_plots_and_return_metrics, plot_heads
from MODGenX.zonation import create_error_zones_and_save
from MODGenX.well_info import well_location, well_data_import
from MODGenX.rasterize_swat import rasterize_SWAT_features
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)

debug=True





def create_modflow_model(NAME, BASE_PATH, LEVEL,RESOLUTION, MODEL_NAME, ML, SWAT_MODEL_NAME):


    raster_folder = os.path.join(f'/data/SWATGenXApp/GenXAppData/{username}/', f"SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}/rasters_input")
    model_path = os.path.join(f'/data/SWATGenXApp/GenXApp/{username}', f'SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}')
    moflow_exe_path=os.path.join(model_path,"MODFLOW-NWT_64.exe")
    swat_lake_shapefile_path = os.path.join(f'/data/SWATGenXApp/GenXApp/{username}', f'SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/{SWAT_MODEL_NAME}/Watershed/Shapes/SWAT_plus_lakes.shp')
    ref_raster_path = os.path.join(f'/data/SWATGenXApp/GenXApp/{username}', f'SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/DEM_{RESOLUTION}m.tif')
    subbasin_path = os.path.join(f'/data/SWATGenXApp/GenXAppData/{username}/', f"SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/{SWAT_MODEL_NAME}/Watershed/Shapes/subs1.shp")
    SWAT_dem_path = os.path.join(f'/data/SWATGenXApp/GenXAppData/{username}/', f"SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/DEM_{RESOLUTION}m.tif")
    swat_river_raster_path=os.path.join(model_path, 'swat_river.tif')
    swat_lake_raster_path=os.path.join(model_path,'lake_raster.tif')


    # Check if the directory exists

    if os.path.exists(model_path):
        shutil.rmtree(model_path)
        print(f"Directory '{model_path}' has been removed.")

    os.makedirs(model_path)
    os.makedirs(raster_folder)
    print(f'model path: {model_path}')

    domain_raster_path, bound_raster_path = defining_bound_and_active(BASE_PATH, subbasin_path, raster_folder, RESOLUTION, SWAT_dem_path)
    load_raster_args = {
        'LEVEL': LEVEL,
        'RESOLUTION': RESOLUTION,
        'NAME': NAME,
        'ref_raster': ref_raster_path,
        'bound_raster': bound_raster_path,
        'active': domain_raster_path,
        'MODEL_NAME': MODEL_NAME,
        }
#if os.path.exists(f'/data/SWATGenXApp/Users/{username}/SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/best_solution.txt'):

    raster_paths = generate_raster_paths(RESOLUTION, ML)

    top = load_raster(raster_paths['DEM'], load_raster_args)
    print(f' ############## shape of top {top.shape} ############## ')
    basin = load_raster(domain_raster_path, load_raster_args)
    top = match_raster_dimensions(basin,top)
    print(f' ############## shape of top {top.shape} ############## ')
    #  plt.imshow(top)
    #  plt.close()
    if os.path.exists(swat_lake_shapefile_path):

        lake_flag = True

        rasterize_SWAT_features(BASE_PATH, "lakes", swat_lake_raster_path,  load_raster_args)
    else:
        print(' ############# NO LAKE ############## ')

        lake_flag = False
    print(swat_river_raster_path)
    rasterize_SWAT_features(BASE_PATH, "rivers", swat_river_raster_path, load_raster_args)

    nlay, nrow, ncol, n_sublay_1, n_sublay_2, k_bedrock, bedrock_thickness = discritization_configuration(top)

    print(f"lake status {lake_flag}")
    # debug arguments for the active domain
    active, lake_raster = active_domain(top, nlay, swat_lake_raster_path, swat_river_raster_path, load_raster_args, lake_flag, fitToMeter = 0.3048)


    z_botm, k_horiz, k_vert ,recharge_data, SWL, head = input_Data (
        active, top, load_raster_args,
        n_sublay_1,
        n_sublay_2,
        k_bedrock,
        bedrock_thickness, ML )

    ibound = remove_isolated_cells(active, load_raster_args)

    strt=GW_starting_head(
                        active,n_sublay_1,
                        n_sublay_2,z_botm,top,
                        head, nrow, ncol
                        )

    if lake_flag: drain_cells = lakes_to_drain(swat_lake_raster_path,
                                            top, k_horiz,
                                            load_raster_args)

    swat_river = river_correction(
        swat_river_raster_path, load_raster_args, basin, active
    )
    river_data = river_gen(nrow, ncol, swat_river, top, ibound)
    src,delr, delc = model_src(raster_paths['DEM'])

    os.makedirs(model_path, exist_ok=True)
    shutil.copy2(os.path.join(BASE_PATH,"bin/MODFLOW-NWT_64.exe"), model_path)

    mf = flopy.modflow.Modflow(
        MODEL_NAME,
        exe_name=moflow_exe_path,
        model_ws=model_path,
        version='mfnwt')

    dis = flopy.modflow.ModflowDis(
            mf, nlay, nrow, ncol,
            delr=delr, delc=delc,
            top=top, botm=z_botm,
            itmuni=4,  # time unit, 4 means days
            lenuni=2,  # length unit, 2 means meters
            nper=1,
            perlen=[365.25],
            nstp=[1],
            steady=[True],
            laycbd=[0] * (nlay - 1) + [1],
            crs = pyproj.CRS.from_user_input('EPSG:26990')
                                )

#        nper = 5000  # number of stress periods (one for each day)
#        perlen = [1] * nper  # each stress period is 1 day long
#        nstp = [1] * nper  # one time step per stress period
#        steady = [False] * nper  # all stress periods are transient

#        dis = flopy.modflow.ModflowDis(
#            mf, nlay, nrow, ncol,
#            delr=delr, delc=delc,
#            top=top, botm=z_botm,
#            itmuni=4, lenuni=2,  # 4: time unit (days), 2: length unit (meters)
#            nper=nper, perlen=perlen, nstp=nstp,
#            steady=steady,
#            laycbd=[0] * (nlay - 1) + [1],
#            crs=pyproj.CRS.from_user_input('EPSG:26990')
#        )

    mf.modelgrid.set_coord_info(xoff=src.bounds.left, yoff=src.bounds.bottom, angrot=0, crs = pyproj.CRS.from_user_input('EPSG:26990'))
    # Create solver object

    nwt = flopy.modflow.ModflowNwt(
        mf,
        headtol    = 0.01,  # Lower tolerance for head change
        fluxtol    = 0.001,  # Lower tolerance for flux imbalance
        maxiterout = 100,  # Increase the maximum number of outer iterations
        thickfact  = 1e-04,
        linmeth    = 1,
        iprnwt     = 1,
        ibotav     = 0,
        options    = 'MODERATE',
        Continue   = False,
        backflag   =1,
        maxbackiter =5
    )

    upw = flopy.modflow.ModflowUpw(
                                mf, hk=k_horiz, vka=k_vert,
                                laytyp=[1]*n_sublay_1 + [0]*n_sublay_2 + [0],
                                layavg= [2] + [2] * (nlay - 1)  ## a list the the first value is 2 and the rest is 1
                                )

    bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)
    rch = flopy.modflow.ModflowRch(mf, rech=recharge_data)

    oc = flopy.modflow.ModflowOc(
        mf,
        stress_period_data={(0, 0): [
            "SAVE HEAD",
            "SAVE DRAWDOWN",
            "SAVE BUDGET"
        ]}, compact=False
        )

    mf.write_input()

    out_shp = os.path.join(model_path, "Grids_MODFLOW")

    raster_path = os.path.join(raster_folder, f'{NAME}_DEM_{RESOLUTION}m.tif.tif')

    create_shapefile_from_modflow_grid_arcpy(BASE_PATH, model_path, MODEL_NAME, out_shp, raster_path)

    grids_path = f'{out_shp}.geojson'

    wel_data,obs_data, df_obs =  well_data_import(
                                        mf, top,
                                        load_raster_args,
                                        z_botm, active, grids_path,
                                        MODEL_NAME
                                        )

    if obs_data:
        wel = flopy.modflow.ModflowWel(mf, stress_period_data=wel_data)
        hob = flopy.modflow.ModflowHob(mf, iuhobsv = 41, hobdry = --999., obs_data=obs_data)

    rasterize_SWAT_features(BASE_PATH,"rivers", swat_river_raster_path, load_raster_args)

    swat_river = river_correction(
        swat_river_raster_path, load_raster_args, basin, active
    )
    print('nrows', nrow,'ncol', ncol)
    print(swat_river)

    river_data=river_gen(nrow, ncol, swat_river, top, ibound)

    riv = flopy.modflow.ModflowRiv(mf, stress_period_data=river_data)

    if lake_flag:
        rasterize_SWAT_features(BASE_PATH,"rivers", swat_river_raster_path, load_raster_args)
        drain_cells = lakes_to_drain(swat_lake_raster_path, top, k_horiz, load_raster_args)
        drn = flopy.modflow.ModflowDrn(mf, stress_period_data={0: drain_cells})

    mf.write_input()

    print('rivers are updated')

    mf.check()

    success, buff = mf.run_model()
    first_layer_simulated_head = plot_heads(LEVEL, NAME, RESOLUTION, MODEL_NAME)

    df_sim_obs = sim_obs (BASE_PATH, MODEL_NAME, mf, LEVEL, top,NAME, RESOLUTION, load_raster_args, df_obs)

    nse, mse, mae, pbias, kge = create_plots_and_return_metrics (df_sim_obs, LEVEL, NAME, MODEL_NAME)
    ## save the model performance metrics in a csv file
    metrics = [MODEL_NAME, NAME, RESOLUTION, nse, mse, mae, pbias, kge]
    metrics_path = os.path.join(f'/data/SWATGenXApp/GenXApp/{username}', f'SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}/metrics.csv')
    with open(metrics_path, 'w') as f:
        f.write('MODEL_NAME,NAME,RESOLUTION,NSE,MSE,MAE,PBIAS,KGE\n')
        f.write(','.join(str(metric) for metric in metrics))
    datasets = [
        well_location(df_sim_obs, active, str(NAME), LEVEL, RESOLUTION, load_raster_args),  smooth_invalid_thickness(top-strt[0]),
                strt[0] , ibound[0],    k_horiz[0] ,     k_horiz[1],       k_vert[0],          k_vert[1],
                recharge_data,swat_river,  top - z_botm[0], z_botm[0]-z_botm[1]
                ]

    titles = ['water wells location', "SWL initial",'Head',  'Active Cells','K Horizontal 1',
            'K Horizontal 2', 'K Vertical 1', 'K Vertical 2', 'Recharge','base flow','Thickness 1', 'thickness 2']

    model_input_figure_path = f"/data/SWATGenXApp/Users/{username}/SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}/input_figures.jpeg"

    plot_data(datasets, titles, model_input_figure_path)

    create_error_zones_and_save(model_path, load_raster_args, ML)
