import pandas as pd
import geopandas as gpd
import rasterio
import os


class MonitoringInfrastructureError(Exception):
    """Raised when monitoring infrastructure is not built."""
    pass


class GeospatialInfrastructureError(Exception):
    """Raised when geospatial infrastructure is not built."""
    pass


class NHDPlusDataError(Exception):
    """Raised when NHDPlus data is not unpacked."""
    pass


class CRSValidationError(Exception):
    """Raised when the CRS of the DEM raster and streams are different."""
    pass


class StreamflowDataError(Exception):
    """Raised when streamflow data is not downloaded."""
    pass


class PRISMDataError(Exception):
    """Raised when PRISM data is not clipped."""
    pass

try:
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
except ImportError:
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths

def check_configuration(VPUID, landuse_epoch) -> str:
    gSSUGO_path = SWATGenXPaths.gSSURGO_path
    NLCD_path = SWATGenXPaths.NLCD_path
    DEM_path = SWATGenXPaths.DEM_path
    extracted_nhd_swatplus_path = SWATGenXPaths.extracted_nhd_swatplus_path
    streamflow_path = SWATGenXPaths.streamflow_path
    PRISM_path = SWATGenXPaths.PRISM_path

    streams_path = os.path.join(f'{extracted_nhd_swatplus_path}/{VPUID}/streams.pkl')
    
    base_input_raster = f'{DEM_path}/VPUID/{VPUID}/'
    
    streamflow_meta_data_directory = f"{streamflow_path}/VPUID/{VPUID}/meta_{VPUID}.csv"
    
    streamflow_stations_directory = f"{streamflow_path}/VPUID/{VPUID}/streamflow_stations_{VPUID}.shp"
    
    if not os.path.exists(streamflow_meta_data_directory) or not os.path.exists(streamflow_stations_directory):
        print(f"######### Building monitoring infrastructure: {VPUID} ###########")
        raise MonitoringInfrastructureError("Monitoring infrastructure is not built. Please run Building_monitoring_infrastructure.py")

    input_raster = os.path.join(base_input_raster, "USGS_DEM_30m.tif")
    landuse_raster = f"{NLCD_path}/{VPUID}/NLCD_{VPUID}_{landuse_epoch}_250m.tif"
    soil_raster = f"{gSSUGO_path}/{VPUID}/soil_{VPUID}.tif"
    
    if not os.path.exists(input_raster) or not os.path.exists(landuse_raster) or not os.path.exists(soil_raster):
        print(f"######### Building geospatial infrastructure: {VPUID} ###########")
        raise GeospatialInfrastructureError("Geospatial infrastructure is not built. Please run Building_geospatial_infrastructure.py")

    if not os.path.exists(streams_path):
        print(streams_path)
        raise NHDPlusDataError("NHDPlus data are not unpacked. Please run NHDPlus_preprocessing.py")
    
    streams = gpd.GeoDataFrame(pd.read_pickle(streams_path))
    print('Loaded streams CRS', streams.crs)

    with rasterio.open(input_raster) as src:
        print('Loaded raster CRS', src.crs)
        EPSG = src.crs.to_string()

    print('DEBUG-----> DEM crs:', EPSG)
    print('DEBUG-----> streams crs:', streams.crs.to_string())
    if streams.crs.to_string().split(' ')[1].split('=')[-1] != EPSG.split(':')[-1][-2:]:
        print('streams crs:', streams.crs.to_string().split(' ')[1].split('=')[-1])
        print('DEM crs:', EPSG.split(':')[-1])
        raise CRSValidationError("Fatal error: CRS of the DEM raster and the streams are different.")
    
    if not os.path.exists(streamflow_stations_directory):
        raise StreamflowDataError("Streamflow data is not downloaded. Please run Building_monitoring_infrastructure.py")
    
    SWAT_MODEL_PRISM_path = f'{PRISM_path}/VPUID/{VPUID}/PRISM_grid.shp'
    if not os.path.exists(SWAT_MODEL_PRISM_path):
        raise PRISMDataError("PRISM data is not clipped. Please run Building_climate_infrastructure.py")

    return EPSG
