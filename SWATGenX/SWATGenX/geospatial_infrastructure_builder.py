from SWATGenX.gSSURGO_extraction import gSSURGO_extract_by_VPUID, gSSURGO_check_by_VPUID
from SWATGenX.NHDPlus_extract_by_VPUID import NHDPlus_extract_by_VPUID, NHDPlus_infra_check_by_VPUID
from SWATGenX.USGS_DEM_extraction import DEM_extract_by_VPUID, check_DEM_by_VPUID
from SWATGenX.NLCD_extraction import NLCD_extract_by_VPUID, check_NLCD_by_VPUID
from SWATGenX.NHDPlus_preprocessing import NHDPlus_preprocessing, check_NHDPlus_preprocessed_by_VPUID
from SWATGenX.extract_CONUS_gssurgo_raster import extract_CONUS_gssurgo_raster
from SWATGenX.download_USGS_DEM import download_USGS_DEM
from SWATGenX.SWATGenXConfigPars import SWATGenXPaths

def geospatial_infrastructure_builder(VPUID, landuse_epoch):
    
    """Build the geospatial infrastructure for a given VPUID.

    Note the you need to compile GDAL with FileGDB support to use the extract_CONUS_gssurgo_raster function.

    ## Checkers determine if a process has already been completed for a given VPUID. If the process has not been completed, 
    # the builder will run the process. If the process has been completed, the builder will skip the process.
    
    Args:
        VPUID (str): The VPUID to build the geospatial infrastructure for.
        landuse_epoch (str): The landuse epoch to build the geospatial infrastructure for.
        
        Returns:
            bool: True if the geospatial infrastructure was built successfully, False otherwise.                
    """

    try:
        extract_CONUS_gssurgo_raster()
    except Exception as e:
        return critical_error(VPUID, e)

    try:
        download_USGS_DEM()
    except Exception as e:
        return critical_error(VPUID, e)

    try:
        NHDPlus_infra_check_by_VPUID(VPUID)
        check_NHDPlus_preprocessed_by_VPUID(VPUID)
    except Exception:
        try:
            NHDPlus_extract_by_VPUID(VPUID) 
            NHDPlus_preprocessing(VPUID)
        except Exception as e:
            return critical_error(VPUID, e)
    try:
        check_DEM_by_VPUID(VPUID)
    except Exception:
        try:
            DEM_extract_by_VPUID(VPUID)
        except Exception as e:
            return critical_error(VPUID, e)
    try:
        check_NLCD_by_VPUID(VPUID, landuse_epoch)
    except Exception:
        try:
            NLCD_extract_by_VPUID(VPUID, landuse_epoch)
        except Exception as e:
            return critical_error(VPUID, e)
    try:
        gSSURGO_check_by_VPUID(VPUID)
    except Exception:
        try:
            gSSURGO_extract_by_VPUID(VPUID)
        except Exception as e:
            return critical_error(VPUID, e)

def critical_error(VPUID, e):
    error_file_path = SWATGenXPaths.critical_error_file_path
    with open(error_file_path, 'a') as file:
        file.write(f"Error in {VPUID}: {str(e)}\n")
    return False
