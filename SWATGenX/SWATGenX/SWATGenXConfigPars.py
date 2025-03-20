from dataclasses import dataclass
import os
#/data/SWATGenXApp/codes/SWATGenX/SWATGenX/SWATGenXConfigPars.py
@dataclass
class SWATGenXPaths:
    overwrite: bool = True  
    base_path: str = "/data/SWATGenXApp/GenXAppData/"
    codes_path: str = "/data/SWATGenXApp/codes/SWATGenX/"
    SWATPlusEditor_path: str = '/usr/local/share/SWATPlusEditor/swatplus-editor/src/api'
    FPS_State_Territories: str = f'{base_path}USGS/FPS_States_and_Territories.csv'
    FPS_all_stations: str = f'{base_path}USGS/FPS_all_stations.csv'
    # Data Sources
    MODFLOW_MODEL_NAME: str = None
    SWAT_MODEL_NAME: str = None
    USGS_path: str = f"{base_path}USGS/"
    NLCD_path: str = f"{base_path}LandUse/NLCD_CONUS/"
    gSSURGO_path: str = f"{base_path}Soil/gSSURGO_CONUS/"
    PRISM_path: str = f"{base_path}PRISM/"
    DEM_path: str = f"{base_path}DEM/"
    Downloaded_CONUS_DEM_path: str = f"{base_path}DEM/CONUS/"
    NHDPlus_path: str = f"{base_path}NHDPlusData/"
    streamflow_path: str = f"{base_path}USGS/streamflow_stations/"
    database_dir: str = base_path
    camel_hydro_path: str = "/data/camel/camels_hydro.txt"
    # Output Paths
    swatgenx_outlet_path: str = f"{base_path}SWATplus_by_VPUID/"
    extracted_nhd_swatplus_path: str = f"{base_path}NHDPlusData/SWATPlus_NHDPlus/"
    report_path: str = f"{codes_path}/logs/"
    
    # Specific Data Files
    critical_error_file_path: str = f"{codes_path}critical_errors.txt"
    NLCD_release_path: str = f"{base_path}LandUse/NLCD_landcover_2021_release_all_files_20230630/"
    gSSURGO_raster: str = f"{base_path}Soil/gSSURGO_CONUS/MapunitRaster_30m.tif"
    swatplus_gssurgo_csv: str = f"{base_path}Soil/SWAT_gssurgo.csv"
    DEM_13_arc_second_list: str = f"{base_path}DEM/VPUID/DEM_13_arc_second.USGS"
    NHDPlus_VPU_National_path: str = f"{base_path}NHDPlusData/NHDPlus_VPU_National/"
    governmental_boundries_path: str = f"{base_path}USGS/GovernmentUnits_National_GDB/GovernmentUnits_National_GDB.gdb"
    USGS_CONUS_stations_path: str = f"{base_path}USGS/streamflow_stations/CONUS/streamflow_stations_CONUS.csv"
    USGS_CONUS_stations_shape_path: str = f"{base_path}USGS/streamflow_stations/CONUS/streamflow_stations_CONUS.shp"
    available_sites_path: str = f"{base_path}USGS/all_VPUIDs.csv"
    NSRDB_PRISM_path: str = f"{base_path}NSRDB/PRISM_NSRDB_CONUS.pkl"
    gSSURGO_CONUS_gdb_path: str = f"{base_path}Soil/gSSURGO_CONUS/gSSURGO_CONUS.gdb"
    
    # Software Paths
    wgn_db: str = "/usr/local/share/SWATPlus/Databases/swatplus_wgn.sqlite"
    QSWATPlus_env_path: str = '/usr/share/qgis/python/plugins/QSWATPlusLinux3_64'
    runQSWATPlus_path: str = "/data/SWATGenXApp/codes/scripts/runQSWATPlus.sh"
    swat_exe: str = "/data/SWATGenXApp/codes/bin/swatplus"
    bin_path: str = "/data/SWATGenXApp/codes/bin/"
    
    # PRISM Mesh Paths
    PRISM_mesh_path: str = f"{base_path}PRISM/prism_4km_mesh/prism_4km_mesh.shp"
    PRISM_mesh_pickle_path: str = f"{base_path}PRISM/prism_4km_mesh/prism_4km_mesh.pkl"
    PRISM_unzipped_path: str = f"{base_path}PRISM/unzipped_daily"
    PRISM_zipped_path: str = f"{base_path}PRISM/zipped_daily"
    PRISM_dem_path: str = f"{base_path}PRISM/PRISM_us_dem_4km_bil/PRISM_us_dem_4km_bil.bil"
    
    # Additional Paths
    LanduseTable: str = f"{base_path}LandUse/landuse_lookup.csv"

    niws_start_date: str = "2000-01-01"
    niws_end_date: str = "2021-01-01"

    prism_start_year: int = 1990
    prism_end_year: int = 2022

    start_year: int = 2015  # Changed to match `START_YEAR`
    end_year: int = 2022  # Changed to match `END_YEAR`
    single_model: bool = True
    sensitivity_flag: bool = False
    calibration_flag: bool = False
    verification_flag: bool = False
    nyskip: int = 3
    sen_total_evaluations: int = 1000
    sen_pool_size: int = 180
    num_levels: int = 10
    cal_pool_size: int = 50
    max_cal_iterations: int = 25
    termination_tolerance: int = 10
    epsilon: float = 0.0001
    Ver_START_YEAR: int = 2004
    Ver_END_YEAR: int = 2022
    Ver_nyskip: int = 3
    range_reduction_flag: bool = False
    pet: int = 2
    cn: int = 1
    no_value: float = 1000000.0
    verification_samples: int = 25
    VPUID: str = None   
    LEVEL: str = None
    NAME: str = None  
    MODEL_NAME: str = None
    BASE_PATH: str = None
    MAX_AREA: int = None
    MIN_AREA: int = None
    landuse_product: str = None
    landuse_epoch: str = None
    ls_resolution: str = None
    dem_resolution: str = None
    station_name: list = None
    GAP_percent: int = None
    username: str = None    
    START_YEAR: int = None
    END_YEAR: int = None
    RESOLUTION: int = None

    path: str = None

    def construct_path(self, *args) -> str:
        # If username is set, use the user-specific output path for SWAT_input paths
        if self.username is not None and args and args[0] == "SWAT_input":
            user_base = f"/data/SWATGenXApp/Users/{self.username}"
            # Replace SWAT_input with SWATplus_by_VPUID/VPUID
            if self.VPUID:
                return os.path.join(user_base, "SWATplus_by_VPUID", self.VPUID, *args[1:])
            else:
                return os.path.join(user_base, "SWATplus_by_VPUID", *args[1:])
        else:
            return os.path.join(self.base_path, *args)
    
    ### define post initi and if username is not None, then override the swatgex_outlet_path
    def __post_init__(self):
        if self.username is not None:
            self.swatgenx_outlet_path = f"/data/SWATGenXApp/Users/{self.username}/SWATplus_by_VPUID/"
            self.report_path: str = f"/data/SWATGenXApp/Users/{self.username}/SWATplus_by_VPUID/"