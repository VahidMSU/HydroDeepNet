from SWATGenX.gssurgo_extraction import gSSURGO_extract_by_VPUID
from SWATGenX.NHDPlus_extract_by_VPUID import NHDPlus_extract_by_VPUID
from SWATGenX.USGS_DEM_extraction import DEM_extract_by_VPUID
from SWATGenX.NLCD_extraction import NLCD_extract_by_VPUID_helper
from SWATGenX.NHDPlus_preprocessing import NHDPlus_preprocessing
from SWATGenX.extract_gssurgo_conus_raster import extract_CONUS_gssurgo_raster
from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
from SWATGenX.SWATGenXLogging import LoggerSetup
import os
import glob

def validate_data_paths():
    """Validate that all required data paths exist.

    Returns:
        tuple: (bool, str) - (is_valid, error_message)
    """
    required_paths = {
        'DEM': SWATGenXPaths.DEM_path,
        'NHDPlus': SWATGenXPaths.NHDPlus_path,
        'NLCD': SWATGenXPaths.NLCD_path,
        'gSSURGO': SWATGenXPaths.gSSURGO_path
    }

    for name, path in required_paths.items():
        if not os.path.exists(path):
            return False, f"{name} base path does not exist: {path}"
    return True, ""

def create_vpuid_directories(VPUID, logger=None):
    """Create all required VPUID directories if they don't exist.

    Args:
        VPUID (str): The VPUID to create directories for
        logger: Logger object for logging

    Returns:
        bool: True if successful
    """
    required_dirs = {
        'DEM': os.path.join(SWATGenXPaths.DEM_path, "VPUID", VPUID),
        'NHDPlus': os.path.join(SWATGenXPaths.extracted_nhd_swatplus_path, VPUID),
        'NLCD': os.path.join(SWATGenXPaths.NLCD_path, "VPUID", VPUID),
        'gSSURGO': os.path.join(SWATGenXPaths.gSSURGO_path, "VPUID", VPUID)
    }

    for name, path in required_dirs.items():
        if not os.path.exists(path):
            if logger:
                logger.info(f"Creating directory for {name}: {path}")
            os.makedirs(path, exist_ok=True)
        elif logger:
            logger.info(f"Directory already exists for {name}: {path}")

    return True

def check_specific_files(VPUID, landuse_epoch, logger=None):
    """Check if all required files exist for each data type.

    Args:
        VPUID (str): The VPUID to check
        landuse_epoch (str): The landuse epoch to check
        logger: Logger object for logging

    Returns:
        dict: A dictionary with data type as key and a tuple of (bool, str) as value
              where bool indicates if all required files exist and str is an error message
    """
    results = {}

    if logger:
        logger.info(f"===== Checking required files for VPUID {VPUID} =====")

    # Check NHDPlus files
    nhd_dir = os.path.join(SWATGenXPaths.extracted_nhd_swatplus_path, VPUID)
    required_nhd_files = [
        'NHDFlowline.pkl',
        'NHDPlusCatchment.pkl',
        'NHDPlusFlowlineVAA.pkl',
        'WBDHU8.pkl',
        'WBDHU12.pkl',
        'NHDWaterbody.pkl',
        'streams.pkl'  # From preprocessing
    ]

    if logger:
        logger.info(f"Checking NHDPlus files in: {nhd_dir}")

    missing_nhd = []
    for f in required_nhd_files:
        file_path = os.path.join(nhd_dir, f)
        exists = os.path.exists(file_path)
        if logger:
            status = "EXISTS" if exists else "MISSING"
            logger.info(f"  - {f}: {status}")
        if not exists:
            missing_nhd.append(f)

    results['NHDPlus'] = (len(missing_nhd) == 0,
                          f"Missing NHDPlus files: {', '.join(missing_nhd)}" if missing_nhd else "")

    # Check DEM files
    dem_dir = os.path.join(SWATGenXPaths.DEM_path, "VPUID", VPUID)
    dem_pattern = f"*{VPUID}*.tif"

    if logger:
        logger.info(f"Checking DEM files in: {dem_dir} matching pattern: {dem_pattern}")

    # Check for any valid DEM files
    dem_files = glob.glob(os.path.join(dem_dir, dem_pattern))

    if logger:
        if dem_files:
            logger.info(f"  Found {len(dem_files)} DEM files:")
            for f in dem_files:
                logger.info(f"  - {os.path.basename(f)}: EXISTS")
        else:
            logger.info(f"  No DEM files found matching: {dem_pattern}")

    results['DEM'] = (len(dem_files) > 0,
                      f"No DEM files found for VPUID {VPUID}" if len(dem_files) == 0 else "")

    # Check NLCD files
    nlcd_dir = os.path.join(SWATGenXPaths.NLCD_path, "VPUID", VPUID)
    nlcd_pattern = f"NLCD_{VPUID}_{landuse_epoch}*.tif"

    if logger:
        logger.info(f"Checking NLCD files in: {nlcd_dir} matching pattern: {nlcd_pattern}")

    nlcd_files = glob.glob(os.path.join(nlcd_dir, nlcd_pattern))

    if logger:
        if nlcd_files:
            logger.info(f"  Found {len(nlcd_files)} NLCD files:")
            for f in nlcd_files:
                logger.info(f"  - {os.path.basename(f)}: EXISTS")
        else:
            logger.info(f"  No NLCD files found matching: {nlcd_pattern}")

    results['NLCD'] = (len(nlcd_files) > 0,
                       f"No NLCD files found for VPUID {VPUID} and epoch {landuse_epoch}" if len(nlcd_files) == 0 else "")

    # Check gSSURGO files - check for all resolutions as defined in gSSURGO_extraction.py
    gssu_dir = os.path.join(SWATGenXPaths.gSSURGO_path, "VPUID", VPUID)

    if logger:
        logger.info(f"Checking gSSURGO files in: {gssu_dir}")

    resolutions = [30, 100, 250, 500, 1000, 2000]
    expected_gssu_files = [f"gSSURGO_{VPUID}_{res}m.tif" for res in resolutions]

    missing_gssu = []
    for f in expected_gssu_files:
        file_path = os.path.join(gssu_dir, f)
        exists = os.path.exists(file_path)
        if logger:
            status = "EXISTS" if exists else "MISSING"
            logger.info(f"  - {f}: {status}")
        if not exists:
            missing_gssu.append(f)

    results['gSSURGO'] = (len(missing_gssu) == 0,
                          f"Missing gSSURGO files: {', '.join(missing_gssu)}" if missing_gssu else "")

    # Summary
    if logger:
        logger.info("===== File Check Summary =====")
        for data_type, (exists, msg) in results.items():
            status = "COMPLETE" if exists else "INCOMPLETE"
            logger.info(f"{data_type}: {status}")
            if not exists:
                logger.info(f"  Details: {msg}")

    return results

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
    logger = LoggerSetup(report_path='./logs', verbose=True, rewrite=True)
    logger = logger.setup_logger("geospatial_infrastructure_builder")
    logger.info(f"Building geospatial infrastructure for {VPUID}")

    # Validate base paths
    logger.info("Validating base data paths")
    is_valid, error_msg = validate_data_paths()
    if not is_valid:
        logger.error(error_msg)
        return critical_error(VPUID, error_msg)
    logger.info("All base data paths exist")

    # Create VPUID directories
    create_vpuid_directories(VPUID, logger)
    logger.info(f"Ensured all required directories exist for {VPUID}")

    # Check for existing files first
    file_check_results = check_specific_files(VPUID, landuse_epoch, logger)
    missing_data_types = [data_type for data_type, (exists, _) in file_check_results.items() if not exists]

    if not missing_data_types:
        logger.info(f"All required files already exist for VPUID {VPUID}")
        return True
    else:
        logger.info(f"Need to process: {', '.join(missing_data_types)} for VPUID {VPUID}")

    # Check and extract CONUS gSSURGO raster if needed
    if not os.path.exists(SWATGenXPaths.gSSURGO_raster):
        logger.info(f"CONUS gSSURGO raster is missing: {SWATGenXPaths.gSSURGO_raster}")
        logger.info("Extracting CONUS gSSURGO raster")
        if not extract_CONUS_gssurgo_raster():
            return critical_error(VPUID, "Failed to extract CONUS gSSURGO raster")
        logger.info("CONUS gSSURGO raster extracted")
    else:
        logger.info(f"CONUS gSSURGO raster already exists: {SWATGenXPaths.gSSURGO_raster}")

    # Process data types as needed

    # Process NHDPlus infrastructure if needed
    if 'NHDPlus' in missing_data_types:
        logger.info("========== Processing NHDPlus data ==========")

        # Extract NHDPlus infrastructure
        logger.info("Extracting NHDPlus infrastructure")
        if not NHDPlus_extract_by_VPUID(VPUID):
            return critical_error(VPUID, "Failed to extract NHDPlus infrastructure")
        logger.info("NHDPlus infrastructure extracted successfully")

        # Process NHDPlus preprocessing
        logger.info("Preprocessing NHDPlus infrastructure")
        if not NHDPlus_preprocessing(VPUID):
            return critical_error(VPUID, "Failed to preprocess NHDPlus infrastructure")
        logger.info("NHDPlus preprocessing completed successfully")

    # Process DEM if needed
    if 'DEM' in missing_data_types:
        logger.info("========== Processing DEM data ==========")
        logger.info("Extracting DEM")
        if not DEM_extract_by_VPUID(VPUID):
            return critical_error(VPUID, "Failed to extract DEM")
        logger.info("DEM extracted successfully")

    # Process NLCD if needed
    if 'NLCD' in missing_data_types:
        logger.info("========== Processing NLCD data ==========")
        logger.info(f"Extracting NLCD for epoch {landuse_epoch}")
        if not NLCD_extract_by_VPUID_helper(VPUID, landuse_epoch):
            return critical_error(VPUID, "Failed to extract NLCD")
        logger.info("NLCD extracted successfully")

    # Process gSSURGO if needed
    if 'gSSURGO' in missing_data_types:
        logger.info("========== Processing gSSURGO data ==========")
        logger.info("Extracting gSSURGO")
        if not gSSURGO_extract_by_VPUID(VPUID):
            return critical_error(VPUID, "Failed to extract gSSURGO")
        logger.info("gSSURGO extracted successfully")

    # Final validation - check all required files again
    logger.info("========== Performing final validation of all data ==========")
    final_check_results = check_specific_files(VPUID, landuse_epoch, logger)
    final_missing_data_types = [data_type for data_type, (exists, msg) in final_check_results.items() if not exists]

    if final_missing_data_types:
        logger.error("Final validation found missing files")
        error_messages = []
        for data_type, (exists, msg) in final_check_results.items():
            if not exists:
                error_messages.append(f"{data_type}: {msg}")

        error_summary = "\n".join(error_messages)
        logger.error(f"Final validation failed:\n{error_summary}")
        return critical_error(VPUID, f"Final validation failed for: {', '.join(final_missing_data_types)}\n{error_summary}")

    logger.info("========== All file checks passed ==========")
    logger.info(f"Successfully completed geospatial infrastructure building for {VPUID}")
    return True

def critical_error(VPUID, error_message):
    """Log a critical error and return False.

    Args:
        VPUID (str): The VPUID where the error occurred
        error_message (str): The error message to log

    Returns:
        bool: False to indicate failure
    """
    error_file_path = SWATGenXPaths.critical_error_file_path
    with open(error_file_path, 'a') as file:
        file.write(f"Error in {VPUID}: {error_message}\n")
    return False
