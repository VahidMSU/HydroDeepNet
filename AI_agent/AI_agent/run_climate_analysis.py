"""
Script to run climate change analysis using LOCA2 data.

This script provides a command-line interface for running climate change analyses
with proper time period handling for the LOCA2 dataset.
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# Set up paths to import modules
sys.path.append('/data/SWATGenXApp/codes')

try:
    from AI_agent.config import AgentConfig
    from AI_agent.climate_change_analysis import ClimateChangeAnalysis
    from AI_agent.loca2_dataset import list_of_cc_models
except ImportError as e:
    from config import AgentConfig
    from climate_change_analysis import ClimateChangeAnalysis
    from loca2_dataset import list_of_cc_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("climate_analysis")

LOCA2_TIME_PERIODS = [(2015, 2044), (2045, 2074), (2075, 2100)]


def setup_parser():
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='Run climate change analysis with LOCA2 data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--hist-start', type=int, default=2000,
                        help='Historical period start year')
    parser.add_argument('--hist-end', type=int, default=2014,
                        help='Historical period end year')
    parser.add_argument('--fut-start', type=int, default=2020,
                        help='Future period start year')
    parser.add_argument('--fut-end', type=int, default=2030,
                        help='Future period end year')
    parser.add_argument('--model', type=str, default='ACCESS-CM2',
                        help='Climate model to use')
    parser.add_argument('--ensemble', type=str, default='r1i1p1f1',
                        help='Ensemble member to use')
    parser.add_argument('--scenario', type=str, default='ssp245',
                        help='Future scenario to analyze')
    parser.add_argument('--output-dir', type=str, default='climate_analysis_results',
                        help='Directory to store outputs')
    parser.add_argument('--agg', type=str, default='monthly',
                        choices=['daily', 'monthly', 'seasonal', 'annual'],
                        help='Temporal aggregation type')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data if real data not available')
    
    # Geographic extent argument (bounding box as min_lon,min_lat,max_lon,max_lat)
    parser.add_argument('--bbox', type=str, default='-85.444332,43.658148,-85.239256,44.164683',
                        help='Bounding box as min_lon,min_lat,max_lon,max_lat')
    
    return parser


def validate_future_years(start_year, end_year):
    """
    Validate that future years fall within the available LOCA2 time periods.
    Returns recommendations if they don't match a single period.
    """
    messages = []
    
    # Check if years fall within a single period
    in_single_period = False
    for period_start, period_end in LOCA2_TIME_PERIODS:
        if start_year >= period_start and end_year <= period_end:
            in_single_period = True
            messages.append(f"Years {start_year}-{end_year} fall within LOCA2 period {period_start}-{period_end}")
            break
    
    if not in_single_period:
        messages.append(f"WARNING: Years {start_year}-{end_year} cross multiple LOCA2 time periods")
        messages.append("The LOCA2 dataset has these fixed time periods: 2015-2044, 2045-2074, 2075-2100")
        
        # Find best matching period
        for period_start, period_end in LOCA2_TIME_PERIODS:
            if start_year >= period_start and start_year <= period_end:
                recommended_end = min(period_end, start_year + (end_year - start_year))
                messages.append(f"Recommendation: Use {start_year}-{recommended_end} which falls within {period_start}-{period_end}")
                break
    
    return messages


def run_analysis(args):
    """Run climate change analysis with provided arguments."""
    
    # Parse bounding box
    try:
        bbox = list(map(float, args.bbox.split(',')))
        if len(bbox) != 4:
            raise ValueError("Invalid bounding box format")
    except:
        logger.error("Bounding box must be specified as min_lon,min_lat,max_lon,max_lat")
        return False
    
    # Check if future years are valid
    messages = validate_future_years(args.fut_start, args.fut_end)
    for msg in messages:
        logger.info(msg)
    
    # Set up output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {output_dir}")
    
    # Configuration for climate change analysis
    cc_config = {
        "RESOLUTION": 250,
        "aggregation": args.agg,
        "bounding_box": bbox,
        "output_dir": os.path.join(output_dir, 'climate_change_analysis'),
        "use_synthetic_data_fallback": args.synthetic
    }
    
    # Create analysis object
    analysis = ClimateChangeAnalysis(cc_config)
    
    # Define historical data config
    historical_config = {
        'start_year': args.hist_start,
        'end_year': args.hist_end,
        'model': args.model,
        'ensemble': args.ensemble
    }
    
    # Define scenario data config
    scenario_configs = [
        {
            'name': args.scenario,
            'start_year': args.fut_start,
            'end_year': args.fut_end,
            'model': args.model,
            'ensemble': args.ensemble
        }
    ]
    
    # Print summary of analysis parameters
    logger.info("============== Analysis Parameters ==============")
    logger.info(f"Historical period: {args.hist_start}-{args.hist_end}")
    logger.info(f"Future period: {args.fut_start}-{args.fut_end}")
    logger.info(f"Climate model: {args.model}, Ensemble: {args.ensemble}")
    logger.info(f"Future scenario: {args.scenario}")
    logger.info(f"Aggregation: {args.agg}")
    logger.info(f"Geographic extent: {bbox}")
    logger.info("===============================================")
    
    try:
        # Extract data and run analysis
        logger.info("Starting climate data extraction...")
        success = analysis.extract_data(historical_config, scenario_configs)
        
        if success:
            logger.info("Climate data extraction successful")
            
            # Calculate climate change metrics
            metrics = analysis.calculate_climate_change_metrics()
            logger.info("Climate change metrics calculated")
            
            # Display key metrics
            for scenario, variables in metrics.items():
                logger.info(f"===== {scenario} Climate Change Projections =====")
                
                for var_name, var_metrics in variables.items():
                    var_label = {
                        'pr': 'Precipitation', 
                        'tmax': 'Max Temperature',
                        'tmin': 'Min Temperature',
                        'tmean': 'Mean Temperature'
                    }.get(var_name, var_name)
                    
                    abs_change = var_metrics['absolute_change']
                    pct_change = var_metrics['percent_change']
                    
                    units = "mm/day" if var_name == 'pr' else "°C"
                    direction = "increase" if abs_change > 0 else "decrease"
                    
                    logger.info(f"{var_label}: {abs_change:.2f} {units} {direction} ({pct_change:.1f}%)")
            
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
                    for scenario in metrics.keys():
                        analysis.plot_seasonal_cycle_comparison(
                            variable=var, 
                            scenario_name=scenario
                        )
            
            # Generate comprehensive report
            logger.info("Generating climate change report...")
            report_path = analysis.generate_climate_change_report()
            logger.info(f"Climate change report generated: {report_path}")
            
            logger.info(f"Complete climate change analysis saved to {cc_config['output_dir']}")
            return True
        else:
            logger.error("Climate data extraction failed")
            return False
    
    except Exception as e:
        logger.error(f"Error in climate change analysis: {e}", exc_info=True)
        return False


def main():
    """Main entry point for the script."""
    parser = setup_parser()
    args = parser.parse_args()
    
    start_time = datetime.now()
    logger.info(f"Starting climate change analysis at {start_time}")
    
    success = run_analysis(args)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    if success:
        logger.info(f"Analysis completed successfully in {duration}.")
        logger.info(f"Results saved to: {args.output_dir}")
    else:
        logger.error(f"Analysis failed after {duration}.")
    

if __name__ == "__main__":
    main()
