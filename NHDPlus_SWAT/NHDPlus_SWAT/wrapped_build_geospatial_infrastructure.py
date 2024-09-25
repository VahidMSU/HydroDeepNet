from NHDPlus_SWAT.gSSURGO_extraction import gSSURGO_extract_by_VPUID
from NHDPlus_SWAT.NHDPlus_extract_by_VPUID import NHDPlus_extract_by_VPUID
from NHDPlus_SWAT.USGS_DEM_extraction import DEM_extract_by_VPUID
from NHDPlus_SWAT.NLCD_extraction import NLCD_extract_by_VPUID
from NHDPlus_SWAT.NHDPlus_preprocessing import NHDPlus_preprocessing


def wrapped_build_geospatial_infrastructure(VPUID, landuse_epoch):
    print(f"Processing VPUID: {VPUID}")
    ## NOTE:  for all the following steps, we wont extract the data if it is already extracted
    NHDPlus_extract_by_VPUID(VPUID)
    NHDPlus_preprocessing(VPUID)
    DEM_extract_by_VPUID(VPUID)
    gSSURGO_extract_by_VPUID(VPUID)
    NLCD_extract_by_VPUID(VPUID, landuse_epoch)