#!/usr/bin/env python3
"""
Detailed Analysis module for SWAT model results
This script performs advanced statistical analysis on the recharge data
"""

import os
import sys
import time
import argparse
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# Add the utilities module to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import required modules
try:
    from read_model_h5 import load_model_data
    from utils.statistics import (
        calculate_monthly_statistics, 
        calculate_seasonal_statistics, 
        calculate_annual_statistics,
        generate_statistics_tables,
        generate_statistics_plots
    )
    from utils.soil_analysis import analyze_soil_properties
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure all required modules are properly installed")
    sys.exit(1)

def run_detailed_analysis(landuse_data, var="perc", start_year=2000, end_year=2005, output_dir="./Michigan", 
                         include_soil_analysis=True, base_path=None):
    """
    Runs detailed statistical analysis on the recharge data
    
    Parameters:
    -----------
    landuse_data : dict
        Data structure containing model results
    var : str
        Variable being analyzed
    start_year, end_year : int
        Year range for analysis
    output_dir : str
        Output directory for results
    include_soil_analysis : bool
        Whether to include soil property analysis (True by default)
    base_path : str
        Base path to SWAT model directories (required for soil analysis)
    
    Returns:
    --------
    dict
        Statistics and file paths
    """
    print("Running detailed statistical analysis...")
    print("Note: Analysis includes both precipitation and snowfall in water input calculations")
    
    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figs"), exist_ok=True)
    
    # Step 1: Calculate monthly statistics
    print("Calculating monthly statistics...")
    monthly_stats = calculate_monthly_statistics(landuse_data, var, start_year, end_year)
    
    # Step 2: Calculate seasonal statistics
    print("Calculating seasonal statistics...")
    seasonal_stats = calculate_seasonal_statistics(landuse_data, var, start_year, end_year)
    
    # Step 3: Calculate annual statistics
    print("Calculating annual statistics...")
    annual_stats = calculate_annual_statistics(landuse_data, var, start_year, end_year)
    
    # Step 4: Generate statistics tables
    print("Generating statistics tables...")
    table_paths = generate_statistics_tables(
        monthly_stats, seasonal_stats, annual_stats, output_dir
    )
    
    # Step 5: Generate statistics plots
    print("Generating statistics plots...")
    plot_paths = generate_statistics_plots(
        monthly_stats, seasonal_stats, annual_stats, output_dir
    )
    
    # Prepare results dictionary
    results = {
        "statistics": {
            "monthly": monthly_stats,
            "seasonal": seasonal_stats,
            "annual": annual_stats
        },
        "tables": table_paths,
        "plots": plot_paths
    }
    
    # Step 6: Run soil analysis (now included by default)
    if include_soil_analysis and base_path:
        print("\nStep 6: Running soil property analysis...")
        try:
            import multiprocessing
            num_workers = max(1, int(multiprocessing.cpu_count() * 0.5))  # Use 50% of CPU cores
            
            soil_analysis_start = time.time()
            soil_results = analyze_soil_properties(
                base_path=base_path,
                output_dir=output_dir,
                model_limit=50,  # Limit to 50 models
                num_workers=num_workers
            )
            
            soil_analysis_time = time.time() - soil_analysis_start
            print(f"Soil property analysis completed in {soil_analysis_time:.2f} seconds")
            
            if soil_results:
                # Add soil analysis results to the main results
                results["soil_analysis"] = soil_results
                print(f"Added soil analysis with {len(soil_results['visualizations'])} visualizations")
            else:
                print("Soil analysis did not return any results")
        except Exception as e:
            print(f"Error during soil analysis: {e}")
            traceback.print_exc()
            print("Continuing without soil analysis results")
    
    print("Detailed analysis completed successfully")
    return results

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run detailed analysis on SWAT model results')
    parser.add_argument('--base-path', type=str, 
                        default="/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12",
                        help='Base path to SWAT model directories')
    parser.add_argument('--model-nums', type=int, default=50,
                        help='Number of models to analyze')
    parser.add_argument('--start-year', type=int, default=2000,
                        help='Start year for analysis')
    parser.add_argument('--end-year', type=int, default=2005,
                        help='End year for analysis (exclusive)')
    parser.add_argument('--var', type=str, default="perc",
                        help='Variable to analyze (e.g., "perc" for percolation)')
    parser.add_argument('--precip-threshold', type=float, default=10.0,
                        help='Precipitation threshold (mm) for ratio calculation')
    parser.add_argument('--output-dir', type=str, default="./Michigan",
                        help='Directory for output files')
    parser.add_argument('--cpus', type=float, default=0.8,
                        help='Fraction of CPU cores to use (0-1)')
    parser.add_argument('--skip-extraction', action='store_true',
                        help='Skip data extraction and use existing data')
    parser.add_argument('--skip-soil-analysis', action='store_true',
                        help='Skip soil property analysis in the report (soil analysis is included by default)')
    parser.add_argument('--report-name', type=str, default="",
                        help='Custom name for the report directory (default is timestamped name)')
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Calculate number of worker processes
    import multiprocessing
    max_workers = max(1, int(multiprocessing.cpu_count() * args.cpus))
    
    # Soil analysis is included by default unless explicitly skipped
    include_soil_analysis = not args.skip_soil_analysis
    
    # Create timestamp for report directories
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Set up reports directory structure
    reports_dir = os.path.join(args.output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Create a report name (custom or timestamped)
    if args.report_name:
        report_name = args.report_name
    else:
        report_name = f"detailed_report_{timestamp}"
    
    # Create full report directory path
    report_dir = os.path.join(reports_dir, report_name)
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(os.path.join(report_dir, "figs"), exist_ok=True)
    os.makedirs(os.path.join(report_dir, "soil_htmls"), exist_ok=True)
    
    # Configuration summary
    print("\n" + "="*80)
    print("Detailed Analysis Configuration")
    print("="*80)
    print(f"Base path:           {args.base_path}")
    print(f"Number of models:    {args.model_nums}")
    print(f"Analysis period:     {args.start_year}-{args.end_year-1}")
    print(f"Variable analyzed:   {args.var}")
    print(f"Precip. threshold:   {args.precip_threshold} mm")
    print(f"Output directory:    {args.output_dir}")
    print(f"Report name:         {report_name}")
    print(f"Workers:             {max_workers}")
    print(f"Include soil analysis: {include_soil_analysis}")
    print("="*80 + "\n")
    
    # Make sure output directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figs"), exist_ok=True)
    
    # Start timing
    start_time = time.time()
    
    try:
        # Step 1: Extract data (unless skipped)
        if not args.skip_extraction:
            print("Step 1: Extracting data from SWAT models...")
            landuse_data = load_model_data(
                base_path=args.base_path,
                model_nums=args.model_nums,
                var=args.var,
                precip_threshold=args.precip_threshold,
                max_workers=max_workers,
                start_year=args.start_year,
                end_year=args.end_year
            )
            # Save data for future use
            import pickle
            data_path = os.path.join(args.output_dir, "landuse_data.pkl")
            with open(data_path, 'wb') as f:
                pickle.dump(landuse_data, f)
            print(f"Data saved to {data_path}")
        else:
            print("Step 1: Loading existing data...")
            import pickle
            data_path = os.path.join(args.output_dir, "landuse_data.pkl")
            with open(data_path, 'rb') as f:
                landuse_data = pickle.load(f)
            print(f"Data loaded from {data_path}")
        
        # Step 2: Run detailed analysis
        print("\nStep 2: Running detailed analysis...")
        analysis_start_time = time.time()
        analysis_results = run_detailed_analysis(
            landuse_data,
            var=args.var,
            start_year=args.start_year,
            end_year=args.end_year,
            output_dir=args.output_dir,
            include_soil_analysis=include_soil_analysis,
            base_path=args.base_path
        )
        analysis_time = time.time() - analysis_start_time
        print(f"Analysis completed in {analysis_time:.2f} seconds")
        
        # Step 3: Generate the report with statistical tables
        print("\nStep 3: Generating the detailed report...")
        from generate_report import create_report
        
        # Create the report in the new report directory
        detailed_report_path = os.path.join(report_dir, 'detailed_report.html')
        report_path = create_report(
            output_path=detailed_report_path,
            start_year=args.start_year,
            end_year=args.end_year,
            var=args.var,
            precip_threshold=args.precip_threshold,
            include_statistics=True,
            statistics_results=analysis_results,
            include_soil_analysis=include_soil_analysis,
            soil_results=analysis_results.get("soil_analysis")
        )
        print(f"Detailed report generated: {report_path}")
        
        # Try to open the report in a browser
        try:
            import webbrowser
            webbrowser.open('file://' + os.path.abspath(report_path))
            print("Report opened in web browser")
        except Exception as e:
            print(f"Could not open report in browser: {e}")
            print(f"Please open the file manually: {os.path.abspath(report_path)}")
        
    except Exception as e:
        print(f"Error during detailed analysis: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Display total execution time
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("="*80)
    print("\nDetailed analysis completed successfully!")

if __name__ == "__main__":
    main()
