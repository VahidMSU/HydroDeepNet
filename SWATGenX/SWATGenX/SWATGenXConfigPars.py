from dataclasses import dataclass
import os
@dataclass
class SWATGenXPaths:
    base_path: str = "/data/SWATGenXApp/GenXAppData/"
    codes_path: str = "/data/SWATGenXApp/codes/SWATGenX/"
    SWATPlusEditor_path: str = '/data/SWATGenXApp/codes/SWATPlusEditor/swatplus.editor/src/api'
    # Data Sources
    USGS_path: str = f"{base_path}USGS/"
    NLCD_path: str = f"{base_path}LandUse/NLCD_CONUS/"
    gSSURGO_path: str = f"{base_path}Soil/gSSURGO_CONUS/"
    PRISM_path: str = f"{base_path}PRISM/"
    DEM_path: str = f"{base_path}DEM/"
    Downloaded_CONUS_DEM_path: str = "/data/SWATGenXApp/GenXAppData/DEM/CONUS/"
    NHDPlus_path: str = f"{base_path}NHDPlusData/"
    streamflow_path: str = f"{base_path}USGS/streamflow_stations/"
    database_dir: str = base_path
    
    # Output Paths
    swatgenx_outlet_path: str = f"{base_path}SWATplus_by_VPUID/"
    extracted_nhd_swatplus_path: str = f"{base_path}NHDPlusData/SWATPlus_NHDPlus/"
    report_path: str = f"{codes_path}SWATGenX/"
    
    # Specific Data Files
    critical_error_file_path: str = f"{codes_path}SWATGenX/critical_errors.txt"
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
    wgn_db: str = "/data/SWATGenXApp/swatplus_installation/swatplus_wgn.sqlite"
    QSWATPlus_env_path: str = '/home/rafieiva/.local/share/QGIS/QGIS3/profiles/default/python/plugins/QSWATPlusLinux3_64/'
    runQSWATPlus_path: str = "/data/SWATGenXApp/codes/SWATGenX/runQSWATPlus.sh"
    swat_exe: str = "/data/SWATGenXApp/codes/bin/swatplus"
    
    # PRISM Mesh Paths
    PRISM_mesh_path: str = f"{base_path}PRISM/prism_4km_mesh/prism_4km_mesh.shp"
    PRISM_mesh_pickle_path: str = f"{base_path}PRISM/prism_4km_mesh/prism_4km_mesh.pkl"
    PRISM_unzipped_path: str = f"{base_path}PRISM/unzipped_daily"
    PRISM_zipped_path: str = f"{base_path}PRISM/zipped_daily"
    PRISM_dem_path: str = f"{base_path}PRISM/PRISM_us_dem_4km_bil/PRISM_us_dem_4km_bil.bil"
    
    # Additional Paths
    LanduseTable: str = f"{base_path}LandUse/landuse_lookup.csv"

    start_date: str = "2000-01-01"
    end_date: str = "2021-01-01"

    start_year: int = 2000
    end_year: int = 2021

    exe_start_year: int = 2000
    exe_end_year: int = 2001


    NHDPlus_column_name_map: dict = {
        'ftype': 'FType',
        'fcode': 'FCode',
        'nhdplusid': 'NHDPlusID',
        'uphydroseq': 'UpHydroSeq',
        'hydroseq': 'HydroSeq',
        'dnhydroseq': 'DnHydroSeq',
        'startflag': 'StartFlag',
        'terminalfl': 'TerminalFl',
        'divergence': 'Divergence',
        "streamorde": "StreamOrde",
        "elevelev": "ElevElev", 
        'vpuid': 'VPUID',
        'permanent_identifier': 'Permanent_Identifier',
        "wbarea_permanent_identifier": "WBArea_Permanent_Identifier",
        'maxelevsmo': 'MaxElevSmo',
        'minelevsmo': 'MinElevSmo',
        'areasqkm': 'AreaSqKm',
        'lengthkm': 'LengthKM',
        'totdasqkm': 'TotDASqKm',
        'geometry': 'geometry'
    }


    def construct_path(self, *args) -> str:
        return os.path.join(self.base_path, *args)