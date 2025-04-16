"""
Report Generator - Generate comprehensive reports from various data sources.

This script provides functionality to generate reports from different data sources
including PRISM climate data, MODIS satellite data, CDL crop data, groundwater data,
and governmental units.
"""
import os
import sys
import argparse
import logging
import time
import json
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable
# Add import for HTML conversion
try:
    from utils.html_report_converter import convert_markdown_to_html, create_report_index
    # Add import for plot utilities
    from utils.plot_utils import close_all_figures
except ImportError:
    from GeoReporter.utils.html_report_converter import convert_markdown_to_html, create_report_index
    from GeoReporter.utils.plot_utils import close_all_figures
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ReportGenerator")

# Add the parent directory to the path to help with imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

# Try importing with direct module path

try:
    from config import AgentConfig
    from HydroGeoDataset.loca2.climate_change_analysis import ClimateChangeAnalysis
    from HydroGeoDataset.usgs.governmental_units_report import analyze_governmental_units
    from HydroGeoDataset.prism.prism_report import batch_process_prism
    from HydroGeoDataset.modis.modis_report import run_comprehensive_report
    from HydroGeoDataset.cdl.cdl_report import analyze_cdl_data
    from HydroGeoDataset.nsrdb.nsrdb_report import batch_process_nsrdb
    from HydroGeoDataset.gssurgo.gssurgo_report import process_soil_data
    from HydroGeoDataset.wellogic.groundwater_report import GroundwaterAnalyzer
    from HydroGeoDataset.snodas.snowdas_report import batch_process_snodas
except ImportError as e:
    from GeoReporter.config import AgentConfig
    from GeoReporter.HydroGeoDataset.loca2.climate_change_analysis import ClimateChangeAnalysis
    from GeoReporter.HydroGeoDataset.usgs.governmental_units_report import analyze_governmental_units
    from GeoReporter.HydroGeoDataset.prism.prism_report import batch_process_prism
    from GeoReporter.HydroGeoDataset.modis.modis_report import run_comprehensive_report
    from GeoReporter.HydroGeoDataset.cdl.cdl_report import analyze_cdl_data
    from GeoReporter.HydroGeoDataset.nsrdb.nsrdb_report import batch_process_nsrdb
    from GeoReporter.HydroGeoDataset.gssurgo.gssurgo_report import process_soil_data
    from GeoReporter.HydroGeoDataset.wellogic.groundwater_report import GroundwaterAnalyzer
    from GeoReporter.HydroGeoDataset.snodas.snowdas_report import batch_process_snodas
    





def generate_prism_report(config: Dict[str, Any], output_dir: str) -> Optional[str]:
    """
    Generate a PRISM climate data report.
    
    Args:
        config: Configuration dictionary with processing parameters
        output_dir: Directory to save report files
        
    Returns:
        Path to the generated report or None if generation failed
    """
    try:
        logger.info("Generating PRISM climate report...")
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process PRISM data and generate report
        report_path = batch_process_prism(config, output_dir)
        
        if report_path:
            logger.info(f"PRISM report generated: {report_path}")
            return report_path
        else:
            logger.error("Failed to generate PRISM report")
            return None
            
    except Exception as e:
        logger.error(f"Error generating PRISM report: {e}", exc_info=True)
        return None

def generate_modis_report(config: Dict[str, Any], output_dir: str) -> Optional[str]:
    """
    Generate MODIS data reports.
    
    Args:
        config: Configuration dictionary
        output_dir: Directory to save report files
        
    Returns:
        Path to the generated report or None if generation failed
    """
    try:
        logger.info("Generating MODIS report...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate comprehensive report
        report_path = run_comprehensive_report(config, output_dir)
        
        if report_path:
            logger.info(f"MODIS report generated: {report_path}")
            return report_path
        else:
            logger.error("Failed to generate MODIS report")
            return None
            
    except Exception as e:
        logger.error(f"Error generating MODIS report: {e}", exc_info=True)
        return None

def generate_cdl_report(config: Dict[str, Any], output_dir: str) -> Optional[str]:
    """
    Generate Cropland Data Layer (CDL) report.
    
    Args:
        config: Configuration dictionary
        output_dir: Directory to save report files
        
    Returns:
        Path to the generated report or None if generation failed
    """
    try:
        logger.info("Generating CDL report...")    
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate CDL report
        report_path = analyze_cdl_data(config=config, output_dir=output_dir)
        
        if report_path:
            logger.info(f"CDL report generated: {report_path}")
            return report_path
        else:
            logger.error("Failed to generate CDL report")
            return None
            
    except Exception as e:
        logger.error(f"Error generating CDL report: {e}", exc_info=True)
        return None

def generate_groundwater_report(config: Dict[str, Any], output_dir: str) -> Optional[str]:
    """
    Generate groundwater data report.
    
    Args:
        config: Configuration dictionary
        output_dir: Directory to save report files
        
    Returns:
        Path to the generated report or None if generation failed
    """
    try:
        logger.info("Generating groundwater report...")
        
        # Import the necessary class

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize analyzer
        analyzer = GroundwaterAnalyzer(bounding_box=config.get('bounding_box'))
        
        # Extract data
        analyzer.extract_data()
        
        # Generate report
        report_path = analyzer.generate_report(output_dir)
        
        if report_path:
            logger.info(f"Groundwater report generated: {report_path}")
            return report_path
        else:
            logger.error("Failed to generate groundwater report")
            return None
            
    except Exception as e:
        logger.error(f"Error generating groundwater report: {e}", exc_info=True)
        return None

def generate_governmental_units_report(config: Dict[str, Any], output_dir: str) -> Optional[str]:
    """
    Generate governmental units report.
    
    Args:
        config: Configuration dictionary
        output_dir: Directory to save report files
        
    Returns:
        Path to the generated report or None if generation failed
    """
    try:
        logger.info("Generating governmental units report...")
        os.makedirs(output_dir, exist_ok=True)
        # Extract parameters from config
        gdb_path = config.get('gdb_path', AgentConfig.USGS_governmental_path)
        bbox = config.get('bounding_box')
        
        # Generate report
        report_path = analyze_governmental_units(
            gdb_path=gdb_path,
            bounding_box=bbox,
            output_dir=output_dir
        )
        
        if report_path:
            logger.info(f"Governmental units report generated: {report_path}")
            return report_path
        else:
            logger.error("Failed to generate governmental units report")
            return None
            
    except Exception as e:
        logger.error(f"Error generating governmental units report: {e}", exc_info=True)
        return None

def generate_climate_change_report(config: Dict[str, Any], output_dir: str) -> Optional[str]:
    """
    Generate climate change analysis report using LOCA2 data.
    
    Args:
        config: Configuration dictionary with processing parameters
        output_dir: Directory to save report files
        
    Returns:
        Path to the generated report or None if generation failed
    """
    try:
        logger.info("Generating climate change analysis report...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up configuration for climate change analysis
        cc_config = {
            "RESOLUTION": config.get('RESOLUTION', 250),
            "aggregation": config.get('aggregation', 'monthly'),
            "bounding_box": config.get('bounding_box'),
            "output_dir": output_dir,
            "use_synthetic_data_fallback": config.get('use_synthetic_data_fallback', True)
        }
        
        # Create analysis object
        analysis = ClimateChangeAnalysis(cc_config)
        
        # Get historical configuration
        hist_start_year = config.get('hist_start_year', 2000)
        hist_end_year = config.get('hist_end_year', 2014)
        model = config.get('cc_model', 'ACCESS-CM2')
        ensemble = config.get('cc_ensemble', 'r1i1p1f1')
        
        historical_config = {
            'start_year': hist_start_year,
            'end_year': hist_end_year,
            'model': model,
            'ensemble': ensemble
        }
        
        # Get future scenario configuration
        fut_start_year = config.get('fut_start_year', 2045)
        fut_end_year = config.get('fut_end_year', 2060)
        scenario = config.get('cc_scenario', 'ssp245')
        
        scenario_configs = [
            {
                'name': scenario,
                'start_year': fut_start_year,
                'end_year': fut_end_year,
                'model': model,
                'ensemble': ensemble
            }
        ]
        
        # Extract data and run analysis
        logger.info("Starting climate data extraction...")
        success = analysis.extract_data(historical_config, scenario_configs)
        
        if success:
            logger.info("Climate data extraction successful")
            
            # Calculate climate change metrics
            metrics = analysis.calculate_climate_change_metrics()
            logger.info("Climate change metrics calculated")
            
            # Generate visualizations
            logger.info("Generating visualizations...")
            
            # Time series plots
            analysis.plot_timeseries_comparison()
            
            # Spatial maps for key variables
            for var in ['pr', 'tmax', 'tmin', 'tmean']:
                analysis.plot_spatial_change_maps(variable=var)
            
            # Seasonal cycle plots if monthly data
            if analysis.aggregation == 'monthly':
                for var in ['pr', 'tmax', 'tmin']:
                    for scenario_name in metrics.keys():
                        analysis.plot_seasonal_cycle_comparison(
                            variable=var, 
                            scenario_name=scenario_name
                        )
            
            # Generate comprehensive report
            logger.info("Generating climate change report...")
            report_path = analysis.generate_climate_change_report()
            
            if report_path:
                logger.info(f"Climate change report generated: {report_path}")
                return report_path
            else:
                logger.error("Failed to generate climate change report")
                return None
        else:
            logger.error("Climate data extraction failed")
            return None
            
    except Exception as e:
        logger.error(f"Error generating climate change report: {e}", exc_info=True)
        return None

def generate_nsrdb_report(config: Dict[str, Any], output_dir: str) -> Optional[str]:
    """
    Generate a NSRDB solar radiation data report.
    
    Args:
        config: Configuration dictionary with processing parameters
        output_dir: Directory to save report files
        
    Returns:
        Path to the generated report or None if generation failed
    """
    try:
        logger.info("Generating NSRDB solar radiation report...")
        

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract parameters from config
        start_year = config.get('start_year', 2019)
        end_year = config.get('end_year', 2019)
        bbox = config.get('bounding_box', [-85.444332, 43.658148, -85.239256, 44.164683])
        
        # Create NSRDB-specific config
        nsrdb_config = {
            "bbox": bbox,
            "start_year": start_year,
            "end_year": end_year,
            "extract_for_swat": config.get('extract_for_swat', False)
        }
        
        # Process NSRDB data and generate report
        report_path = batch_process_nsrdb(nsrdb_config, output_dir)
        
        if report_path:
            logger.info(f"NSRDB report generated: {report_path}")
            return report_path
        else:
            logger.error("Failed to generate NSRDB report")
            return None
            
    except Exception as e:
        logger.error(f"Error generating NSRDB report: {e}", exc_info=True)
        return None

def generate_gssurgo_report(config: Dict[str, Any], output_dir: str) -> Optional[str]:
    """
    Generate a gSSURGO soil data report.
    
    Args:
        config: Configuration dictionary with processing parameters
        output_dir: Directory to save report files
        
    Returns:
        Path to the generated report or None if generation failed
    """
    try:
        logger.info("Generating gSSURGO soil report...")
        

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate gSSURGO report
        report_path = process_soil_data(config=config, output_dir=output_dir)
        
        if report_path:
            logger.info(f"gSSURGO soil report generated: {report_path}")
            return report_path
        else:
            logger.error("Failed to generate gSSURGO soil report")
            return None
            
    except Exception as e:
        logger.error(f"Error generating gSSURGO soil report: {e}", exc_info=True)
        return None

def generate_snodas_report(config: Dict[str, Any], output_dir: str) -> Optional[str]:
    """
    Generate a SNODAS snow data report.
    
    Args:
        config: Configuration dictionary with processing parameters
        output_dir: Directory to save report files
        
    Returns:
        Path to the generated report or None if generation failed
    """
    try:
        logger.info("Generating SNODAS snow data report...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process SNODAS data and generate report
        report_path = batch_process_snodas(config, output_dir)
        
        if report_path:
            logger.info(f"SNODAS report generated: {report_path}")
            return report_path
        else:
            logger.error("Failed to generate SNODAS report")
            return None
            
    except Exception as e:
        logger.error(f"Error generating SNODAS report: {e}", exc_info=True)
        return None

def generate_comprehensive_report(config: Dict[str, Any], output_dir: str, parallel: bool = True) -> List[str]:
    """
    Generate a comprehensive report including all data sources.
    
    Args:
        config: Configuration dictionary
        output_dir: Directory to save report files
        parallel: Whether to process reports in parallel
        
    Returns:
        List of paths to generated reports
    """
    reports = []
    html_reports = []
    
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define report configurations
    report_configs = [
        {
            'name': 'prism',
            'dir': os.path.join(output_dir, "prism"),
            'func': generate_prism_report,
        },
        {
            'name': 'nsrdb',
            'dir': os.path.join(output_dir, "nsrdb"),
            'func': generate_nsrdb_report,
        },
        {
            'name': 'modis',
            'dir': os.path.join(output_dir, "modis"),
            'func': generate_modis_report,
        },
        {
            'name': 'cdl',
            'dir': os.path.join(output_dir, "cdl"), 
            'func': generate_cdl_report,
        },
        {
            'name': 'groundwater',
            'dir': os.path.join(output_dir, "groundwater"),
            'func': generate_groundwater_report,
        },
        {
            'name': 'gov_units',
            'dir': os.path.join(output_dir, "gov_units"),
            'func': generate_governmental_units_report,
        },
        {
            'name': 'gssurgo',
            'dir': os.path.join(output_dir, "gssurgo"),
            'func': generate_gssurgo_report,
        },
        {
            'name': 'snodas',
            'dir': os.path.join(output_dir, "snodas"),
            'func': generate_snodas_report,
        }
    ]
    
    # Add climate change report if requested
    if config.get('include_climate_change', False):
        report_configs.append({
            'name': 'climate_change',
            'dir': os.path.join(output_dir, "climate_change"),
            'func': generate_climate_change_report,
        })
    
    try:
        if parallel:
            # Generate reports in parallel, but make sure we isolate each report type
            # to prevent figure interference
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                # Process one report type at a time to prevent figure conflicts
                for report_config in report_configs:
                    # Extract report-specific config if available
                    report_name = report_config['name']
                    report_specific_config = config.get(f"{report_name}_config", {})
                    
                    # Merge with global config, with report-specific values taking precedence
                    merged_config = {**config, **report_specific_config}
                    
                    future = executor.submit(report_config['func'], merged_config, report_config['dir'])
                    try:
                        report_path = future.result()
                        if report_path:
                            logger.info(f"{report_config['name']} report generation completed")
                            reports.append(report_path)
                            
                            # Make sure all figures are closed before converting to HTML
                            close_all_figures()
                            
                            # Convert to HTML if it's a Markdown file
                            if report_path.endswith('.md'):
                                html_path = convert_markdown_to_html(report_path)
                                if html_path:
                                    html_reports.append(html_path)
                                    logger.info(f"Converted to HTML: {html_path}")
                        else:
                            logger.warning(f"{report_config['name']} report generation failed")
                    except Exception as exc:
                        logger.error(f"{report_config['name']} report generation raised an exception: {exc}")
                    
                    # Ensure all figures are closed after each report type
                    close_all_figures()
        else:
            # Generate reports sequentially
            for report_config in report_configs:
                # Extract report-specific config if available
                report_name = report_config['name']
                report_specific_config = config.get(f"{report_name}_config", {})
                
                # Merge with global config, with report-specific values taking precedence
                merged_config = {**config, **report_specific_config}
                
                report_path = report_config['func'](merged_config, report_config['dir'])
                if report_path:
                    reports.append(report_path)
                    
                    # Convert to HTML if it's a Markdown file
                    if report_path.endswith('.md'):
                        html_path = convert_markdown_to_html(report_path)
                        if html_path:
                            html_reports.append(html_path)
                            logger.info(f"Converted to HTML: {html_path}")
    finally:
        # Ensure all matplotlib figures are closed
        close_all_figures()
    
    # Create an index of all HTML reports
    if html_reports:
        index_path = create_report_index(output_dir)
        if index_path:
            logger.info(f"Created HTML report index: {index_path}")
    
    logger.info(f"Generated {len(reports)} reports in {output_dir}")
    return reports

# Add input validation and logging to ensure coordinates are processed correctly

def run_report_generation(report_type: str, config: Dict[str, Any], output_dir: str, parallel: bool = True) -> List[str]:
    """
    Run the report generation for a specific type or all reports.
    
    Args:
        report_type: Type of report to generate ('prism', 'modis', etc. or 'all')
        config: Configuration dictionary
        output_dir: Directory to save report files
        parallel: Whether to process reports in parallel when type is 'all'
        
    Returns:
        List of paths to generated reports
    """
    reports = []
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Validate the bounding box coordinates
    try:
        bbox = config.get('bounding_box', [])
        if len(bbox) != 4:
            logger.error(f"Invalid bounding box format: {bbox}")
            with open(os.path.join(output_dir, "error_log.txt"), "a") as f:
                f.write(f"ERROR: Invalid bounding box format: {bbox}\n")
            return reports
            
        min_lon, min_lat, max_lon, max_lat = bbox
        
        # Convert to float if they are strings
        min_lon = float(min_lon)
        min_lat = float(min_lat)
        max_lon = float(max_lon)
        max_lat = float(max_lat)
        
        # Basic validation
        if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90 and 
                -180 <= min_lon <= 180 and -180 <= max_lon <= 180):
            logger.error(f"Bounding box coordinates out of valid range: {bbox}")
            with open(os.path.join(output_dir, "error_log.txt"), "a") as f:
                f.write(f"ERROR: Bounding box coordinates out of valid range: {bbox}\n")
            return reports
            
        # Ensure min is less than max
        if min_lat > max_lat or min_lon > max_lon:
            logger.warning(f"Swapping min/max coordinates in bounding box: {bbox}")
            min_lat, max_lat = min(min_lat, max_lat), max(min_lat, max_lat)
            min_lon, max_lon = min(min_lon, max_lon), max(min_lon, max_lon)
            
        # Update config with validated values
        config['bounding_box'] = [min_lon, min_lat, max_lon, max_lat]
        
        # Log the validated bounding box
        logger.info(f"Using bounding box: {min_lon}, {min_lat}, {max_lon}, {max_lat}")
        
    except Exception as e:
        logger.error(f"Error validating bounding box: {e}")
        with open(os.path.join(output_dir, "error_log.txt"), "a") as f:
            f.write(f"ERROR: Error validating bounding box: {e}\n")
        return reports
    
    # Save the config to a file for debugging
    try:
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            
            json.dump(config, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Could not save config to file: {e}")
    
    try:
        # Continue with report generation
        if report_type == 'all':
            # Generate all reports using parallel processing if enabled
            reports = generate_comprehensive_report(config, output_dir, parallel)
        else:
            # Generate a specific report
            report_dir = os.path.join(output_dir, report_type)
            
            # Map report type to its generator function
            report_funcs = {
                'prism': generate_prism_report,
                'nsrdb': generate_nsrdb_report,
                'modis': generate_modis_report,
                'cdl': generate_cdl_report,
                'groundwater': generate_groundwater_report,
                'gov_units': generate_governmental_units_report,
                'gssurgo': generate_gssurgo_report,
                'climate_change': generate_climate_change_report,
                'snodas': generate_snodas_report
            }
            
            if report_type in report_funcs:
                # Extract report-specific config if available
                report_specific_config = config.get(f"{report_type}_config", {})
                
                # Merge with global config, with report-specific values taking precedence
                merged_config = {**config, **report_specific_config}
                
                report_path = report_funcs[report_type](merged_config, report_dir)
                if report_path:
                    reports.append(report_path)
            else:
                logger.error(f"Unknown report type: {report_type}")
        
        # After generating reports, convert them to HTML
        logger.info("Converting Markdown reports to HTML format...")
        html_reports = []
        
        # Find all generated Markdown reports
        for root, _, files in os.walk(output_dir):
            for file in files:
                if file.endswith('.md'):
                    md_path = os.path.join(root, file)
                    html_path = convert_markdown_to_html(md_path)
                    if html_path:
                        html_reports.append(html_path)
        
        if html_reports:
            # Create HTML index
            index_path = create_report_index(output_dir)
            if index_path:
                logger.info(f"Created HTML report index: {index_path}")
                
            logger.info(f"Converted {len(html_reports)} reports to HTML format")
    except Exception as e:
        logger.error(f"Error in report generation: {e}")
    finally:
        # Always close all matplotlib figures to prevent memory leaks
        close_all_figures()
    
    return reports

# Define report-specific argument definitions
REPORT_SPECIFIC_ARGS = {
    'prism': [
        ('--prism-dataset', {'type': str, 'help': 'PRISM dataset type (daily, monthly, etc.)'}),
        ('--prism-variables', {'type': str, 'help': 'Comma-separated list of PRISM variables to include'}),
        ('--prism-fill-gaps', {'action': 'store_true', 'help': 'Fill gaps in PRISM data'})
    ],
    'nsrdb': [
        ('--nsrdb-extract-for-swat', {'action': 'store_true', 'help': 'Extract NSRDB data in SWAT format'}),
        ('--nsrdb-variables', {'type': str, 'help': 'Comma-separated list of NSRDB variables to include'})
    ],
    'modis': [
        ('--modis-product', {'type': str, 'help': 'MODIS product to use (MOD13Q1, etc.)'}),
        ('--modis-indices', {'type': str, 'help': 'Comma-separated list of MODIS indices to include'}),
        ('--modis-backend', {'type': str, 'choices': ['gdal', 'xarray'], 'help': 'Backend to use for processing'})
    ],
    'cdl': [
        ('--cdl-recode-crops', {'action': 'store_true', 'help': 'Recode crops into major categories'}),
        ('--cdl-top-n-crops', {'type': int, 'help': 'Number of top crops to analyze separately'})
    ],
    'groundwater': [
        ('--groundwater-max-depth', {'type': float, 'help': 'Maximum depth to include in analysis'}),
        ('--groundwater-min-samples', {'type': int, 'help': 'Minimum number of samples required for analysis'})
    ],
    'gov_units': [
        ('--gov-units-db-path', {'type': str, 'help': 'Path to governmental units database'}),
        ('--gov-units-layers', {'type': str, 'help': 'Comma-separated list of layers to include'})
    ],
    'gssurgo': [
        ('--gssurgo-properties', {'type': str, 'help': 'Comma-separated list of soil properties to analyze'}),
        ('--gssurgo-depth-range', {'type': str, 'help': 'Depth range to analyze (e.g., "0-30,30-100")'})
    ],
    'climate_change': [
        ('--hist-start-year', {'type': int, 'default': 2000, 'help': 'Start year for historical climate period'}),
        ('--hist-end-year', {'type': int, 'default': 2014, 'help': 'End year for historical climate period'}),
        ('--fut-start-year', {'type': int, 'default': 2045, 'help': 'Start year for future climate period'}),
        ('--fut-end-year', {'type': int, 'default': 2060, 'help': 'End year for future climate period'}),
        ('--cc-model', {'type': str, 'default': 'ACCESS-CM2', 'help': 'Climate model for climate change analysis'}),
        ('--cc-ensemble', {'type': str, 'default': 'r1i1p1f1', 'help': 'Ensemble member for climate change analysis'}),
        ('--cc-scenario', {'type': str, 'default': 'ssp245', 'help': 'Climate scenario for future projections'})
    ],
    'snodas': [
        ('--snodas-variables', {'type': str, 'help': 'Comma-separated list of SNODAS variables to include'}),
        ('--snodas-snow-season', {'type': str, 'help': 'Months to consider for snow season (e.g., "11,12,1,2,3,4")'})
    ]
}

def generate_reports():
    """Parse command line arguments and generate reports."""
    parser = argparse.ArgumentParser(description='Generate reports from various data sources')
    
    # Common arguments
    parser.add_argument('--type', choices=['prism', 'nsrdb', 'modis', 'cdl', 'groundwater', 
                                          'gov_units', 'gssurgo', 'climate_change', 'snodas', 'all'],
                        default='all', help='Type of report to generate')
    parser.add_argument('--output', type=str, default='reports',
                       help='Output directory for reports')
    parser.add_argument('--start-year', type=int, default=2010,
                       help='Start year for analysis')
    parser.add_argument('--end-year', type=int, default=2020,
                       help='End year for analysis')
    parser.add_argument('--min-lon', type=float, default=-85.444332,
                       help='Minimum longitude of bounding box')
    parser.add_argument('--min-lat', type=float, default=43.158148,
                       help='Minimum latitude of bounding box')
    parser.add_argument('--max-lon', type=float, default=-84.239256,
                       help='Maximum longitude of bounding box')
    parser.add_argument('--max-lat', type=float, default=44.164683,
                       help='Maximum latitude of bounding box')
    parser.add_argument('--resolution', type=int, default=250,
                       help='Resolution for data analysis')
    parser.add_argument('--aggregation', type=str, default='monthly',
                       choices=['daily', 'monthly', 'seasonal', 'annual'],
                       help='Temporal aggregation for climate data')
    
    # Add report-specific argument groups
    report_groups = {}
    for report_type, args_list in REPORT_SPECIFIC_ARGS.items():
        group = parser.add_argument_group(f'{report_type} report arguments')
        report_groups[report_type] = group
        
        for arg_name, arg_kwargs in args_list:
            group.add_argument(arg_name, **arg_kwargs)
    
    # Common arguments for climate data fallback
    parser.add_argument('--use-synthetic', action='store_true',
                       help='Use synthetic data if actual data not available')
    
    # Parallel processing control
    parser.add_argument('--sequential', action='store_true',
                       help='Run report generation sequentially (disable parallel processing)')
    
    args = parser.parse_args()
    
    # Create basic configuration with common arguments
    config = {
        'RESOLUTION': args.resolution,
        'resolution': args.resolution,
        'start_year': args.start_year,
        'end_year': args.end_year,
        'bounding_box': [args.min_lon, args.min_lat, args.max_lon, args.max_lat],
        'aggregation': args.aggregation,
        'use_synthetic_data_fallback': args.use_synthetic,
        'include_climate_change': args.type in ['climate_change', 'all']
    }
    
    # Process report-specific arguments and create dedicated config sections
    for report_type, args_list in REPORT_SPECIFIC_ARGS.items():
        report_config = {}
        
        for arg_name, _ in args_list:
            # Convert arg_name (e.g., --prism-dataset) to attribute name (prism_dataset)
            arg_attr = arg_name.lstrip('-').replace('-', '_')
            if hasattr(args, arg_attr) and getattr(args, arg_attr) is not None:
                value = getattr(args, arg_attr)
                # Handle comma-separated values
                if isinstance(value, str) and ',' in value:
                    if arg_attr.endswith('_variables') or arg_attr.endswith('_indices') or arg_attr.endswith('_layers'):
                        value = [v.strip() for v in value.split(',')]
                report_config[arg_attr.split('_', 1)[1]] = value
        
        # Only add the config section if it has values
        if report_config:
            config[f"{report_type}_config"] = report_config
            
    # Add climate change config as a special case since it has many parameters
    # that have both general and specific uses
    if 'climate_change' in config:
        cc_config = config.get('climate_change_config', {})
        cc_config.update({
            'hist_start_year': args.hist_start_year,
            'hist_end_year': args.hist_end_year,
            'fut_start_year': args.fut_start_year,
            'fut_end_year': args.fut_end_year,
            'cc_model': args.cc_model,
            'cc_ensemble': args.cc_ensemble,
            'cc_scenario': args.cc_scenario
        })
        config['climate_change_config'] = cc_config
        
        # Also add to main config for backward compatibility
        if args.type == 'climate_change':
            config.update(cc_config)
    
    # Determine if we should use parallel processing
    use_parallel = not args.sequential
    
    if use_parallel:
        logger.info("Using parallel processing for report generation")
    else:
        logger.info("Using sequential processing for report generation")
    
    try:
        # Run report generation
        reports = run_report_generation(args.type, config, args.output, parallel=use_parallel)
        
        logger.info(f"Report generation completed. Generated {len(reports)} reports in {args.output}")
    finally:
        # Always close all matplotlib figures
        close_all_figures()

if __name__ == "__main__":

    start_time = time.time()
    try:
        generate_reports()
    finally:
        # Ensure all matplotlib figures are closed
        close_all_figures()
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
