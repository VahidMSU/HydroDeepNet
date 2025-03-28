#!/usr/bin/env python3
"""
Streamflow Analysis for SWAT models
This script analyzes streamflow data from SWAT models and generates reports
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import local modules
from utils.streamflow_analyzer import StreamflowAnalyzer, analyze_streamflow
from utils.streamflow_cluster_analysis import analyze_streamflow_clusters

def main():
    """Main function for streamflow analysis"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze streamflow data from SWAT models')
    parser.add_argument('--base-path', type=str, 
                        default='/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12',
                        help='Path to the watershed models')
    parser.add_argument('--start-year', type=int, default=2000,
                        help='Start year for analysis')
    parser.add_argument('--end-year', type=int, default=2005,
                        help='End year for analysis')
    parser.add_argument('--output-dir', type=str, default='./Michigan',
                        help='Directory to save output files')
    parser.add_argument('--cluster-analysis', action='store_true',
                        help='Perform cluster analysis on streamflow results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Starting streamflow analysis using data from {args.base_path}")
    print(f"Time period: {args.start_year}-{args.end_year}")
    
    # Run the streamflow analysis
    try:
        results = analyze_streamflow(args.base_path, args.start_year, args.end_year)
        
        if not results:
            print("Streamflow analysis failed or no data was found.")
            return 1
        
        print(f"Successfully analyzed {results['num_analyzed']} streamflow stations")
        print(f"Summary table saved to {os.path.join(args.output_dir, results['table_path'])}")
        
        # Perform cluster analysis if requested
        if args.cluster_analysis:
            print("Performing cluster analysis on streamflow results...")
            cluster_results = analyze_streamflow_clusters(results, args.output_dir)
            
            if cluster_results:
                print("Streamflow cluster analysis completed successfully")
                print(f"Report saved to {os.path.join(args.output_dir, cluster_results['report'])}")
                print(f"Generated {len(cluster_results['visualizations'])} visualizations")
                print(f"Generated {len(cluster_results['tables'])} tables")
            else:
                print("Streamflow cluster analysis failed or no clusters were found")
        
        print("Streamflow analysis completed successfully")
        return 0
    
    except Exception as e:
        print(f"Error during streamflow analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())