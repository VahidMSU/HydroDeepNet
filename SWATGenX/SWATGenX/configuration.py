import pandas as pd 
import geopandas as gpd 
import rasterio 
import os 
#from SWATGenX.wrapped_build_geospatial_infrastructure import wrapped_build_geospatial_infrastructure

def check_configuration(VPUID,landuse_epoch) -> str:
    
    streams_path = os.path.join(f'/data/SWATGenXApp/GenXAppData/NHDPlusData/SWATPlus_NHDPlus/{VPUID}/streams.pkl')
    original_resolution = 30
    
    base_input_raster = f'/data/SWATGenXApp/GenXAppData/DEM/VPUID/{VPUID}/'
    
    streamflow_meta_data_directory = f"/data/SWATGenXApp/GenXAppData/streamflow_stations/VPUID/{VPUID}/meta_{VPUID}.csv"
    
    streamflow_stations_directory = f"/data/SWATGenXApp/GenXAppData/USGS/streamflow_stations/VPUID/{VPUID}/streamflow_stations_{VPUID}.shp"
    
    input_raster = os.path.join(base_input_raster, "USGS_DEM_30m.tif")
    landuse_raster = f"/data/SWATGenXApp/GenXAppData/LandUse/NLCD_CONUS/{VPUID}/NLCD_{VPUID}_{landuse_epoch}_250m.tif"
    soil_raster = f"/data/SWATGenXApp/GenXAppData/Soil/gSSURGO_CONUS/{VPUID}/soil_{VPUID}.tif"
    if not os.path.exists(input_raster) or not os.path.exists(landuse_raster):
        print(f"######### Building geospatial infrastructure: {VPUID} ###########")
        #if landuse_epoch in ["2001", "2004", "2006", "2008", "2011", "2013", "2016", "2019", "2021"]:
           # wrapped_build_geospatial_infrastructure(VPUID, landuse_epoch)    
        #else:
        #    raise Exception("Landuse epoch is not valid")
    if not os.path.exists(streams_path):
        print(streams_path)
        raise Exception("NHDPlus data are not unpacked. Please run NHDPlus_preprocessing.py")
    


        #raise Exception("DEM data are not downloaded. Please run DEM_extract_by_VPUID.py")
    
    streams = gpd.GeoDataFrame(pd.read_pickle(streams_path))
    print('Loaded streams CRS',streams.crs)

    with rasterio.open(input_raster) as src:
        print('Loaded raster CRS',src.crs)
        EPSG = src.crs.to_string()

    if streams.crs.to_string().split(' ')[1].split('=')[-1] != EPSG.split(':')[-1][-2:]:
        print('streams crs:',streams.crs.to_string().split(' ')[1].split('=')[-1])
        print('DEM crs:',EPSG.split(':')[-1])
        raise Exception("fatal error: CRS of the DEM raster and the streams are different")
    
    if not os.path.exists(streamflow_stations_directory):
        raise Exception("Streamflow data is not downloaded. Please run Building_monitoring_infrastructure.py")
    
    SWAT_MODEL_PRISM_path = f'/data/SWATGenXApp/GenXAppData/PRISM/VPUID/{VPUID}/PRISM_grid.shp'
    if not os.path.exists(SWAT_MODEL_PRISM_path):
        raise Exception("PRISM data is not clipped. Please run Bulding_climate_infrastructure.py")

    return EPSG
