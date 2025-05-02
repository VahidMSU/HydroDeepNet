#!/usr/bin/env python3
"""
Baseflow Processor Module

This module interfaces with the Fortran baseflow separation program
and provides functions to process streamflow data through it.
"""

import os
import subprocess
import pandas as pd
import numpy as np
import tempfile
import shutil
from datetime import datetime
import time

def run_baseflow_separation(streamflow_df, station_name="streamflow", min_days=10, max_days=30, daily_output=True):
    """
    Run the Fortran baseflow separation program on a pandas DataFrame with streamflow data.
    
    Parameters:
    -----------
    streamflow_df : pandas.DataFrame
        DataFrame containing 'date' and 'streamflow' columns
    station_name : str
        Name of the station (used for file naming)
    min_days : int
        Minimum number of days for alpha calculation
    max_days : int
        Maximum number of days for alpha calculation
    daily_output : bool
        Whether to include daily baseflow values in the output
        
    Returns:
    --------
    dict
        Dictionary containing baseflow separation results including:
        - baseflow_fractions: DataFrame with fractions for 3 passes
        - alpha_factor: recession constant (if available)
        - baseflow_days: baseflow days (if available)
        - daily_values: DataFrame with daily baseflow values (if daily_output=True)
    """
    # Get base directory for the baseflow program
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'baseflow')
    baseflow_exe = os.path.join(base_dir, 'baseflow')
    
    # Check if the baseflow executable exists
    if not os.path.exists(baseflow_exe):
        print(f"Error: Baseflow executable not found at {baseflow_exe}")
        print("Using digital filter method as fallback")
        return None
    
    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Remove invalid characters from station name for file naming
        safe_station_name = ''.join(c for c in station_name if c.isalnum() or c in ['-', '_']).strip()
        if not safe_station_name:
            safe_station_name = "station"
        
        # Create the input file with the station name
        prn_file = f"{safe_station_name}.prn"
        out_file = f"{safe_station_name}.out"
        
        input_file_path = os.path.join(temp_dir, prn_file)
        output_file_path = os.path.join(temp_dir, out_file)
        
        # Format the streamflow data into the required format for the Fortran program
        formatted_df = streamflow_df.copy()
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(formatted_df['date']):
            formatted_df['date'] = pd.to_datetime(formatted_df['date'])
            
        # Create YYYYMMDD format (required by the Fortran program)
        formatted_df['date_str'] = formatted_df['date'].dt.strftime('%Y%m%d')
        
        # Handle no-value data - Fortran program expects a specific value (999.0, 9999.0 or -1.0)
        # Using 9999.0 which is recognized as a missing value by the program
        formatted_df.loc[formatted_df['streamflow'] < 0, 'streamflow'] = 9999.0
        formatted_df.loc[np.isnan(formatted_df['streamflow']), 'streamflow'] = 9999.0
        
        # Sort by date (crucial for the Fortran program)
        formatted_df = formatted_df.sort_values('date')
        
        # Check if there's enough data for analysis
        valid_data = formatted_df[formatted_df['streamflow'] < 9000].copy()
        if len(valid_data) < min_days:
            print(f"Warning: Not enough valid data points ({len(valid_data)}) for station {safe_station_name}. Minimum required: {min_days}")
            return None
            
        # Calculate the percentage of valid data
        valid_percentage = (len(valid_data) / len(formatted_df)) * 100
        if valid_percentage < 50:
            print(f"Warning: Only {valid_percentage:.1f}% of data is valid for station {safe_station_name}. Using digital filter method.")
            return None
        
        # Write the formatted data to the input file in the exact format required by the Fortran program
        # Format: YYYYMMDD  Flow  (with fixed width spacing)
        with open(input_file_path, 'w') as f:
            f.write('Date    Flow\n')
            for _, row in formatted_df.iterrows():
                # Format the flow with 1 decimal place, right-justified in a field of 10 characters
                f.write(f"{row['date_str']}    {row['streamflow']:10.1f}\n")
        
        # Create the file.lst configuration file
        file_lst_path = os.path.join(temp_dir, 'file.lst')
        with open(file_lst_path, 'w') as f:
            f.write('!!Input for baseflow program:\n')
            f.write(f"    {min_days} !NDMIN: minimum number of days for alpha calculation\n")
            f.write(f"    {max_days} !NDMAX: maximum number of days for alpha calculation\n")
            f.write(f"     {'1' if daily_output else '0'} !IPRINT: daily print option (0-no; 1-yes)\n")
            f.write("\n!!Daily stream data files\n")
            f.write(f"{prn_file}     {out_file}\n")
        
        # Copy the executable to the temp directory
        local_exe = os.path.join(temp_dir, 'baseflow')
        try:
            shutil.copy2(baseflow_exe, local_exe)
            os.chmod(local_exe, 0o755)  # Make executable
        except Exception as e:
            print(f"Warning: Could not copy baseflow executable: {e}")
            local_exe = baseflow_exe  # Use the original path
        
        # Change to the temporary directory to run the program
        # This is critical since the baseflow program expects to find files in the current directory
        cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Run the baseflow program
            print(f"Running baseflow program for station {safe_station_name}...")
            
            # First check the input file exists and has data
            if not os.path.exists(prn_file) or os.path.getsize(prn_file) < 20:
                print(f"Error: Input file {prn_file} does not exist or is too small")
                return None
                
            # Run with a timeout to prevent hanging
            subprocess.run(['./baseflow'], check=True, stderr=subprocess.PIPE, timeout=60)
            
            # Give the program a moment to complete writing files
            time.sleep(0.5)
            
            # Check if the output file and baseflow.dat files were created
            if not os.path.exists(out_file):
                print(f"Warning: Output file not found: {out_file}")
                print(f"The baseflow program did not generate the expected output for station {safe_station_name}")
                return None
                
            baseflow_dat = os.path.join(temp_dir, 'baseflow.dat')
            if not os.path.exists(baseflow_dat):
                print(f"Warning: Baseflow.dat file not found for station {safe_station_name}")
                return None
            
            # Parse the baseflow.dat file which contains recession constants and baseflow fractions
            with open(baseflow_dat, 'r') as f:
                lines = f.readlines()
            
            # Extract baseflow fractions and other parameters
            result = {}
            
            # Find the line with the results for our station
            data_line = None
            for i, line in enumerate(lines):
                if prn_file in line:
                    data_line = line
                    break
            
            if data_line:
                # Parse the data line
                parts = data_line.strip().split()
                
                # Create a DataFrame for the baseflow fractions
                if len(parts) >= 4:  # Must have at least the input filename and 3 fractions
                    baseflow_fractions = pd.DataFrame({
                        'pass': [1, 2, 3],
                        'fraction': [float(parts[1]), float(parts[2]), float(parts[3])]
                    })
                    
                    result['baseflow_fractions'] = baseflow_fractions
                    
                    # Extract alpha factor and baseflow days if available
                    if len(parts) > 4:
                        result['npr'] = int(parts[4])
                        result['alpha_factor'] = float(parts[5])
                        result['baseflow_days'] = float(parts[6])
                else:
                    print(f"Warning: Unexpected format in baseflow.dat for station {safe_station_name}")
            else:
                print(f"Warning: Could not find results for {prn_file} in baseflow.dat")
                return None
            
            # Load daily values if requested
            if daily_output and os.path.exists(output_file_path):
                # Skip header lines and read the data
                daily_data = []
                with open(output_file_path, 'r') as f:
                    lines = f.readlines()
                    
                # Parse the data lines (skip the first two header lines)
                for line in lines[2:]:
                    parts = line.strip().split()
                    if len(parts) >= 7:
                        try:
                            year = int(parts[0])
                            month = int(parts[1])
                            day = int(parts[2])
                            total_flow = float(parts[3])
                            baseflow_pass1 = float(parts[4])
                            baseflow_pass2 = float(parts[5])
                            baseflow_pass3 = float(parts[6])
                            
                            # Create datetime
                            try:
                                date = datetime(year, month, day)
                                
                                daily_data.append({
                                    'date': date,
                                    'total_flow': total_flow,
                                    'baseflow_pass1': baseflow_pass1,
                                    'baseflow_pass2': baseflow_pass2,
                                    'baseflow_pass3': baseflow_pass3,
                                    'quickflow_pass3': total_flow - baseflow_pass3
                                })
                            except ValueError as e:
                                print(f"Warning: Invalid date ({year}-{month}-{day}) in output file: {e}")
                        except (ValueError, IndexError) as e:
                            print(f"Warning: Could not parse line in output file: {line.strip()} - {e}")
                
                # Create DataFrame
                if daily_data:
                    result['daily_values'] = pd.DataFrame(daily_data)
                    # Sort by date to ensure correct ordering
                    result['daily_values'] = result['daily_values'].sort_values('date')
                else:
                    print(f"Warning: No daily values could be parsed from output file for {safe_station_name}")
                    return None
            else:
                print(f"Warning: No daily output file found for {safe_station_name}")
                return None
            
            return result
            
        except subprocess.CalledProcessError as e:
            print(f"Error running baseflow program for station {safe_station_name}: {e}")
            if e.stderr:
                print(f"Stderr: {e.stderr.decode('utf-8')}")
            return None
        except subprocess.TimeoutExpired:
            print(f"Error: Baseflow program execution timed out after 60 seconds for station {safe_station_name}")
            return None
        except Exception as e:
            print(f"Error in baseflow processing for station {safe_station_name}: {e}")
            return None
        finally:
            # Change back to the original directory
            os.chdir(cwd)

def process_streamflow_data(file_path, station_name=None, min_days=10, max_days=30, daily_output=True):
    """
    Process a streamflow CSV file and run baseflow separation.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing streamflow data
    station_name : str, optional
        Name of the station (if None, derived from file name)
    min_days : int
        Minimum number of days for alpha calculation
    max_days : int
        Maximum number of days for alpha calculation
    daily_output : bool
        Whether to include daily baseflow values in the output
        
    Returns:
    --------
    dict
        Dictionary containing baseflow separation results
    """
    try:
        # Read the streamflow data
        df = pd.read_csv(file_path)
        
        # Check for required columns
        if 'date' not in df.columns or 'streamflow' not in df.columns:
            print(f"Error: CSV file {file_path} must contain 'date' and 'streamflow' columns")
            return None
        
        # Ensure date is in datetime format
        df['date'] = pd.to_datetime(df['date'])
        
        # Derive station name from file name if not provided
        if station_name is None:
            station_name = os.path.basename(file_path).split('.')[0]
        
        # Run baseflow separation
        return run_baseflow_separation(df, station_name, min_days, max_days, daily_output)
    except Exception as e:
        print(f"Error processing streamflow data from {file_path}: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        station_name = None
        
        if len(sys.argv) > 2:
            station_name = sys.argv[2]
        else:
            station_name = os.path.basename(file_path).split('.')[0]
            
        result = process_streamflow_data(file_path, station_name)
        
        if result:
            print("\nBaseflow Separation Results:")
            print("---------------------------")
            print("Baseflow Fractions:")
            print(result['baseflow_fractions'])
            
            if 'alpha_factor' in result:
                print(f"\nAlpha Factor: {result['alpha_factor']:.4f}")
                print(f"Baseflow Days: {result['baseflow_days']:.2f}")
                print(f"Number of Recessions (NPR): {result['npr']}")
            
            if 'daily_values' in result:
                print(f"\nDaily Values (first 5 rows):")
                print(result['daily_values'].head())
        else:
            print("Error processing streamflow data")
    else:
        print("Usage: python baseflow_processor.py <streamflow_csv_file> [station_name]")