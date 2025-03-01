"""
Example script to demonstrate the use of MODIS report generation functionality.

This script shows how to use the MODIS data extraction, analysis, and reporting
capabilities for generating comprehensive environmental reports.
"""
import os
import sys
import numpy as np
import pandas as pd
import logging
import argparse

# Add parent directory to path to help with imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from AI_agent.AI_agent.config import AgentConfig
    from AI_agent.AI_agent.modis import MODIS_dataset
    from AI_agent.AI_agent.modis_analysis import (
        generate_modis_report, 
        generate_comprehensive_modis_report,
        analyze_modis_environmental_indicators,
        create_integrated_landcover_report,
        create_modis_climate_comparison
    )
    from AI_agent.AI_agent.cdl import CDL_dataset
except ImportError:
    # Handle both package and direct imports
    try:
        from AI_agent.config import AgentConfig
        from AI_agent.modis import MODIS_dataset
        from AI_agent.modis_analysis import (
            generate_modis_report, 
            generate_comprehensive_modis_report,
            analyze_modis_environmental_indicators,
            create_integrated_landcover_report,
            create_modis_climate_comparison
        )
        from AI_agent.cdl import CDL_dataset
    except ImportError:
        print("Error importing required modules. Make sure the path is correct.")
        sys.exit(1)

# Configure logger
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('MODIS_Example')

def extract_modis_products(config, products):
    """
    Extract MODIS data for multiple products.
    
    Args:
        config: Configuration dictionary
        products: List of MODIS product names to extract
        
    Returns:
        Dictionary mapping product names to data arrays
    """
    modis_data = {}
    
    for product_name in products:
        logger.info(f"Extracting MODIS data for {product_name}...")
        
        # Update config with the current product
        product_config = config.copy()
        product_config["data_product"] = product_name
        
        # Create MODIS dataset handler and extract data
        modis = MODIS_dataset(product_config)
        data = modis.MODIS_ET()
        
        if data is not None and data.size > 0:
            modis_data[product_name] = data
            logger.info(f"Successfully extracted data for {product_name} with shape {data.shape}")
        else:
            logger.warning(f"No data extracted for {product_name}")
    
    return modis_data

def load_sample_climate_data(region_name=None, start_year=2000, end_year=2020):
    """
    Create sample climate data for demonstration purposes.
    
    Args:
        region_name: Optional region name for data simulation
        start_year: Start year for data
        end_year: End year for data
        
    Returns:
        DataFrame with climate variables
    """
    # Generate dates at 8-day intervals to better match MODIS data frequency
    # This creates dates that are more likely to match with MODIS observations
    dates = []
    current_date = pd.Timestamp(f'{start_year}-01-01')
    end_date = pd.Timestamp(f'{end_year}-12-31')
    
    while current_date <= end_date:
        dates.append(current_date)
        current_date += pd.Timedelta(days=8)  # Use 8-day frequency to match MODIS
    
    n = len(dates)
    
    # Create random but reasonable climate data
    np.random.seed(42)  # For reproducibility
    
    # Temperature with seasonal pattern
    base_temp = 15  # Base temperature
    seasonal_amplitude = 10  # Seasonal variation
    year_fraction = np.array([(d.dayofyear / 366) for d in dates])  # Fraction of year
    season = -np.cos(2 * np.pi * year_fraction)  # Peaks in summer
    temperature = base_temp + seasonal_amplitude * season + np.random.normal(0, 2, n)
    
    # Precipitation with seasonal pattern (more in certain months)
    base_precip = 50  # Base precipitation per period
    precip_seasonal = 30 * (np.sin(2 * np.pi * (year_fraction - 0.25)) + 0.5)  # Peak in spring
    precipitation = np.maximum(0, base_precip + precip_seasonal + np.random.normal(0, 15, n))
    
    # Relative humidity related to temperature and precipitation
    rel_humidity = 60 + 0.3 * precipitation - 0.5 * temperature + np.random.normal(0, 5, n)
    rel_humidity = np.clip(rel_humidity, 20, 100)  # Keep within realistic bounds
    
    # Solar radiation with seasonal pattern
    solar_rad_base = 200  # Base radiation
    solar_rad_amplitude = 150  # Seasonal variation
    solar_radiation = solar_rad_base + solar_rad_amplitude * season + np.random.normal(0, 20, n)
    
    # Combine into DataFrame
    climate_data = pd.DataFrame({
        'date': dates,
        'temperature': temperature,
        'precipitation': precipitation,
        'rel_humidity': rel_humidity,
        'solar_radiation': solar_radiation
    })
    
    logger.info(f"Generated sample climate data with {len(climate_data)} records")
    return climate_data

def run_single_product_report(config, output_dir):
    """Generate a report for a single MODIS product."""
    # Extract MODIS data
    product_name = config["data_product"]
    modis = MODIS_dataset(config)
    data = modis.MODIS_ET()
    
    if data is None or data.size == 0:
        logger.error(f"No data extracted for {product_name}")
        return None
    
    # Generate report
    product_dir = os.path.join(output_dir, product_name)
    report_path = generate_modis_report(
        data=data,
        product_name=product_name,
        start_year=config['start_year'],
        end_year=config['end_year'],
        bounding_box=config['bounding_box'],
        output_dir=product_dir
    )
    
    return report_path

def run_comprehensive_report(config, output_dir):
    """Generate a comprehensive report with multiple MODIS products."""
    # List of products to include
    products = [
        'MOD13Q1_NDVI',
        'MOD13Q1_EVI',
        'MOD15A2H_Lai_500m',
        'MOD16A2_ET'
    ]
    
    # Extract data for all products
    modis_data = extract_modis_products(config, products)
    
    if not modis_data:
        logger.error("No MODIS data extracted for any product")
        return None
    
    # Create product configurations
    products_config = {
        product: {
            'start_year': config['start_year'],
            'end_year': config['end_year']
        }
        for product in modis_data.keys()
    }
    
    # Define baseline years (e.g., first 5 years of data)
    baseline_years = {
        product: list(range(config['start_year'], min(config['start_year'] + 5, config['end_year'] + 1)))
        for product in modis_data.keys()
    }
    
    # Generate comprehensive report
    comprehensive_dir = os.path.join(output_dir, "comprehensive")
    report_path = generate_comprehensive_modis_report(
        data=modis_data,
        products_config=products_config,
        region_name="Study Area",
        bounding_box=config['bounding_box'],
        output_dir=comprehensive_dir,
        include_animations=False,  # Set to True for GIF animations (resource intensive)
        baseline_years=baseline_years,
        cross_product_analysis=True
    )
    
    return report_path

def run_environmental_indicators(config, output_dir):
    """Generate environmental indicators from MODIS data."""
    # List of products to include
    products = [
        'MOD13Q1_NDVI',
        'MOD16A2_ET'
    ]
    
    # Extract data for vegetation and water products
    modis_data = extract_modis_products(config, products)
    
    if not modis_data:
        logger.error("No MODIS data extracted for environmental indicators")
        return None
    
    # Generate indicators
    indicators_dir = os.path.join(output_dir, "environmental_indicators")
    indicators = analyze_modis_environmental_indicators(
        modis_data=modis_data,
        config=config,
        output_dir=indicators_dir
    )
    
    return indicators.get('report_path') if indicators else None

def run_integrated_analysis(config, output_dir):
    """Generate an integrated analysis of MODIS and CDL data."""
    # Extract MODIS vegetation data
    modis_data = extract_modis_products(config, ['MOD13Q1_NDVI'])
    
    if not modis_data:
        logger.error("No MODIS vegetation data extracted for integrated analysis")
        return None
    
    # Extract CDL data
    # Make a copy of config for CDL to ensure it has all required fields
    cdl_config = config.copy()
    cdl_config["aggregation"] = "annual"
    
    cdl = CDL_dataset(cdl_config)
    cdl_data = cdl.cdl_trends()
    
    if not cdl_data:
        logger.error("No CDL data extracted for integrated analysis")
        return None
    
    # Generate integrated report
    integrated_dir = os.path.join(output_dir, "integrated_analysis")
    report_path = create_integrated_landcover_report(
        modis_data=modis_data,
        cdl_data=cdl_data,
        output_dir=integrated_dir
    )
    
    return report_path

def run_climate_comparison(config, output_dir):
    """Generate a comparison between MODIS data and climate variables."""
    # Extract MODIS vegetation data
    modis_data = extract_modis_products(config, ['MOD13Q1_NDVI', 'MOD13Q1_EVI'])
    
    if not modis_data:
        logger.error("No MODIS vegetation data extracted for climate comparison")
        return None
    
    # Load or generate climate data
    climate_data = load_sample_climate_data(
        start_year=config['start_year'], 
        end_year=config['end_year']
    )
    
    # Generate climate comparison report
    climate_dir = os.path.join(output_dir, "climate_comparison")
    report_path = create_modis_climate_comparison(
        modis_data=modis_data,
        climate_data=climate_data,
        output_dir=climate_dir
    )
    
    return report_path

def modis_report_gen():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate MODIS reports and analyses")
    parser.add_argument("--type", choices=["single", "comprehensive", "indicators", "integrated", "climate", "all"],
                       default="comprehensive", help="Type of report to generate")
    parser.add_argument("--output", type=str, default="modis_reports",
                       help="Output directory for reports")
    parser.add_argument("--start-year", type=int, default=2015,
                       help="Start year for analysis")
    parser.add_argument("--end-year", type=int, default=2020,
                       help="End year for analysis")
    parser.add_argument("--product", type=str, default="MOD13Q1_NDVI",
                       help="MODIS product to analyze (for single product report)")
    args = parser.parse_args()
    
    # Configure the analysis
    config = {
        "RESOLUTION": 250,
        "start_year": args.start_year,
        "end_year": args.end_year,
        'bounding_box': [-85.444332, 43.658148, -85.239256, 44.164683],  # Example: Michigan area
        "data_product": args.product,
    }
    
    # Create output directory
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the requested analysis
    report_path = None
    
    if args.type == "single" or args.type == "all":
        logger.info(f"Generating single product report for {args.product}")
        report_path = run_single_product_report(config, output_dir)
        if report_path:
            logger.info(f"Single product report generated: {report_path}")
    
    if args.type == "comprehensive" or args.type == "all":
        logger.info("Generating comprehensive MODIS report")
        report_path = run_comprehensive_report(config, output_dir)
        if report_path:
            logger.info(f"Comprehensive report generated: {report_path}")
    
    if args.type == "indicators" or args.type == "all":
        logger.info("Analyzing environmental indicators")
        report_path = run_environmental_indicators(config, output_dir)
        if report_path:
            logger.info(f"Environmental indicators report generated: {report_path}")
    
    if args.type == "integrated" or args.type == "all":
        logger.info("Creating integrated MODIS-CDL analysis")
        report_path = run_integrated_analysis(config, output_dir)
        if report_path:
            logger.info(f"Integrated analysis report generated: {report_path}")
    
    if args.type == "climate" or args.type == "all":
        logger.info("Creating climate-vegetation comparison")
        report_path = run_climate_comparison(config, output_dir)
        if report_path:
            logger.info(f"Climate comparison report generated: {report_path}")
    
    # Display final message
    if report_path:
        logger.info(f"All requested reports generated successfully in {output_dir}")
    else:
        logger.error("Failed to generate reports")


if __name__ == "__main__":
    modis_report_gen()
