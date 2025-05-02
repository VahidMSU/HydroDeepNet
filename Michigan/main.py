#!/usr/bin/env python3
"""
Main script to streamline the analysis workflow from reading SWAT models to generating reports.
This script coordinates the entire process of data extraction, uncertainty analysis,
and report generation.
"""

import os
import sys
import argparse
import time
import traceback
import datetime

# Add the utilities module to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import necessary modules
try:
    from read_model_h5 import load_model_data
    from generate_report import create_report
    from utils.geolocate import landuse_lookup
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure all required modules are properly installed")
    sys.exit(1)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run SWAT model analysis and generate report')
    parser.add_argument('--base-path', type=str,
                        default="/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12",
                        help='Base path to SWAT model directories')
    parser.add_argument('--model-nums', type=int, default=50,
                        help='Number of models to analyze')
    parser.add_argument('--start-year', type=int, default=2000,
                        help='Start year for analysis')
    parser.add_argument('--end-year', type=int, default=2020,
                        help='End year for analysis (exclusive)')
    parser.add_argument('--var', type=str, default="perc",
                        help='Variable to analyze (e.g., "perc" for percolation)')
    parser.add_argument('--precip-threshold', type=float, default=10.0,
                        help='Precipitation threshold (mm) for ratio calculation')
    parser.add_argument('--output-dir', type=str, default="./Michigan",
                        help='Directory for output files')
    parser.add_argument('--cpus', type=float, default=0.8,
                        help='Fraction of CPU cores to use (0-1)')
    parser.add_argument('--skip-analysis', action='store_true',
                        help='Skip analysis and only generate report from existing data')
    parser.add_argument('--lookup-table', type=str,
                        default="/data/SWATGenXApp/GenXAppData/NLCD/landuse_lookup.csv",
                        help='Path to landuse lookup table')
    parser.add_argument('--detailed-analysis', action='store_true',
                        help='Perform detailed statistical analysis')
    parser.add_argument('--save-data', action='store_true',
                        help='Save model data for future analysis')
    parser.add_argument('--skip-soil-analysis', action='store_true',
                        help='Skip soil property analysis in the report (soil analysis is included by default)')
    parser.add_argument('--report-name', type=str, default="",
                        help='Custom name for the report directory (default is timestamped name)')

    return parser.parse_args()

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'geopandas',
        'shapely', 'h5py', 'tqdm', 'PIL'
    ]

    optional_packages = [
        'contextily',  # For Michigan map with basemap
        'seaborn'      # For enhanced visualizations
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print("WARNING: Missing required packages:", ", ".join(missing))
        print("Please install them using: pip install " + " ".join(missing))
        return False

    for package in optional_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"NOTE: Optional package '{package}' is not installed.")
            print(f"      Some features may be limited. To install: pip install {package}")

    return len(missing) == 0

def main():
    """Main function to coordinate the workflow"""
    # Import multiprocessing locally to ensure it's available
    import multiprocessing

    # Check dependencies
    if not check_dependencies():
        print("WARNING: Some dependencies are missing. The script may not work correctly.")
        response = input("Do you want to continue anyway? (y/n): ")
        if response.lower() not in ['y', 'yes']:
            sys.exit(1)

    # Parse arguments
    args = parse_arguments()

    # Create timestamp for report directories
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # Set up output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figs"), exist_ok=True)

    # Set up reports directory
    reports_dir = os.path.join(args.output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # Create a report name (custom or timestamped)
    if args.report_name:
        report_name = args.report_name
    else:
        report_name = f"report_{timestamp}"

    # Create full report directory path
    report_dir = os.path.join(reports_dir, report_name)
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(os.path.join(report_dir, "figs"), exist_ok=True)
    os.makedirs(os.path.join(report_dir, "soil_htmls"), exist_ok=True)

    # Calculate number of worker processes
    max_workers = max(1, int(multiprocessing.cpu_count() * args.cpus))
    print(f"Using {max_workers} workers out of {multiprocessing.cpu_count()} available cores")

    # Soil analysis is included by default unless explicitly skipped
    include_soil_analysis = not args.skip_soil_analysis

    # Configuration summary
    print("\n" + "="*80)
    print("SWAT Model Analysis Configuration")
    print("="*80)
    print(f"Base path:           {args.base_path}")
    print(f"Number of models:    {args.model_nums}")
    print(f"Analysis period:     {args.start_year}-{args.end_year-1}")
    print(f"Variable analyzed:   {args.var}")
    print(f"Water input threshold: {args.precip_threshold} mm (precipitation + snowfall)")
    print(f"Output directory:    {args.output_dir}")
    print(f"Report name:         {report_name}")
    print(f"Workers:             {max_workers}")
    print(f"Include soil analysis: {include_soil_analysis}")
    print("="*80 + "\n")

    # Start timing
    start_time = time.time()

    # Store the landuse_data object for later use if needed
    landuse_data = None

    # Step 1: Run analysis (unless skipped)
    if not args.skip_analysis:
        print("Step 1: Running SWAT model analysis...")
        try:
            # Load landuse lookup table
            lookup = landuse_lookup(args.lookup_table)
            print(f"Loaded landuse lookup table with {len(lookup)} entries")

            # Run data extraction and analysis
            landuse_data = load_model_data(
                base_path=args.base_path,
                model_nums=args.model_nums,
                var=args.var,
                precip_threshold=args.precip_threshold,
                max_workers=max_workers,
                start_year=args.start_year,
                end_year=args.end_year
            )

            analysis_time = time.time() - start_time
            print(f"Analysis completed in {analysis_time:.2f} seconds")

            # Save the data if requested
            if args.save_data:
                import pickle
                data_path = os.path.join(args.output_dir, "landuse_data.pkl")
                with open(data_path, 'wb') as f:
                    pickle.dump(landuse_data, f)
                print(f"Data saved to {data_path}")
        except Exception as e:
            print(f"Error during analysis: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Try to load data if detailed analysis is requested
        if args.detailed_analysis:
            import pickle
            data_path = os.path.join(args.output_dir, "landuse_data.pkl")
            try:
                with open(data_path, 'rb') as f:
                    landuse_data = pickle.load(f)
                print(f"Data loaded from {data_path}")
            except Exception as e:
                print(f"Error loading data for detailed analysis: {e}")
                print("Please run the analysis with --save-data first or skip --skip-analysis")
                sys.exit(1)
        print("Skipping analysis as requested")

    # Step 2: Run detailed analysis if requested
    if args.detailed_analysis:
        if landuse_data is None:
            print("Error: Data is not available for detailed analysis")
            print("Please run without --skip-analysis or with --save-data first")
            sys.exit(1)

        print("\nStep 2: Running detailed statistical analysis...")
        try:
            from detailed_analysis import run_detailed_analysis

            detailed_start_time = time.time()

            analysis_results = run_detailed_analysis(
                landuse_data,
                var=args.var,
                start_year=args.start_year,
                end_year=args.end_year,
                output_dir=args.output_dir,
                include_soil_analysis=include_soil_analysis,
                base_path=args.base_path
            )

            detailed_time = time.time() - detailed_start_time
            print(f"Detailed analysis completed in {detailed_time:.2f} seconds")

            # Generate report with statistics
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

            print(f"Detailed report saved to: {report_path}")

            # Open the detailed report in a browser
            try:
                import webbrowser
                webbrowser.open('file://' + os.path.abspath(report_path))
                print("Detailed report opened in web browser")
            except Exception as e:
                print(f"Could not open detailed report in browser: {e}")

        except Exception as e:
            print(f"Error during detailed analysis: {e}")
            traceback.print_exc()

        # Skip the regular report generation
        print("Skipping regular report generation as detailed report was created")
    else:
        # Step 3: Generate regular report
        print("\nStep 3: Generating HTML report...")
        try:
            report_start_time = time.time()

            # Run soil analysis (always run unless explicitly skipped)
            soil_results = None
            if include_soil_analysis:
                print("Running soil property analysis for report...")
                try:
                    from utils.soil_analysis import analyze_soil_properties
                    import multiprocessing
                    num_workers = max(1, int(multiprocessing.cpu_count() * 0.5))

                    soil_results = analyze_soil_properties(
                        base_path=args.base_path,
                        output_dir=args.output_dir,
                        model_limit=args.model_nums,
                        num_workers=num_workers
                    )

                    if soil_results:
                        print(f"Soil analysis completed with {len(soil_results['visualizations'])} visualizations")
                except Exception as e:
                    print(f"Error during soil analysis: {e}")
                    traceback.print_exc()
                    print("Continuing without soil analysis")

            # Create the report
            regular_report_path = os.path.join(report_dir, 'report.html')
            report_path = create_report(
                output_path=regular_report_path,
                start_year=args.start_year,
                end_year=args.end_year,
                var=args.var,
                precip_threshold=args.precip_threshold,
                include_soil_analysis=(include_soil_analysis and soil_results is not None),
                soil_results=soil_results
            )

            report_time = time.time() - report_start_time
            print(f"Report generation completed in {report_time:.2f} seconds")
            print(f"Report saved to: {report_path}")

            # Try to open the report in a browser
            try:
                import webbrowser
                webbrowser.open('file://' + os.path.abspath(report_path))
                print("Report opened in web browser")
            except Exception as e:
                print(f"Could not open report in browser: {e}")
                print(f"Please open the file manually: {os.path.abspath(report_path)}")
        except Exception as e:
            print(f"Error generating report: {e}")
            traceback.print_exc()
            sys.exit(1)

    # Display total execution time
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("="*80)
    print("\nAnalysis workflow completed successfully!")

if __name__ == "__main__":
    main()
