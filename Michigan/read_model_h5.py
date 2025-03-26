import h5py 
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import concurrent.futures
from functools import partial
import tqdm
import sys

# Add the utilities module to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.geolocate import landuse_lookup, get_model_cluster_mapping
    from utils.plots import create_plot
except ImportError as e:
    print(f"Error importing utility modules: {e}")
    print("Creating local directory structure...")
    os.makedirs('./Michigan/utils', exist_ok=True)
    print("Please ensure utility modules are properly installed")
    sys.exit(1)

def process_model_ensemble(args, base_path, NAMES, name_to_cluster, var="perc", start_year=2000, end_year=2010, precip_threshold=10):
    model_num, ensamble = args
    
    # Handle index out of range
    if model_num >= len(NAMES):
        print(f"Warning: model_num {model_num} is out of range for NAMES list of length {len(NAMES)}")
        return (model_num, ensamble, -1, [np.nan] * ((end_year - start_year) * 12))
    
    result = []
    model_name = NAMES[model_num]
    cluster = name_to_cluster.get(model_name, -1)  # -1 for models not in cluster mapping
    
    path = f"{base_path}/{model_name}/SWAT_gwflow_MODEL/Scenarios/verification_stage_{ensamble}/SWATplus_output.h5"
    try:
        with h5py.File(path, 'r') as f:
            # Check if watershed has agricultural land
            landuse = f['Landuse/landuse_30m'][:]
            mask = np.where(landuse == 82, 1, 0)  ## agriculture
            
            # If there's no agricultural land, return a list of NaN values
            if np.sum(mask) == 0:
                print(f"No agricultural land in model {model_num} ({model_name}), ensemble {ensamble}")
                # Return NaN values for each month
                return (model_num, ensamble, cluster, [np.nan] * ((end_year - start_year) * 12))
            
            for year in range(start_year, end_year):
                for month in range(1, 13):
                    try:
                        # Check if the data exists for this year/month
                        if f'hru_wb_30m/{year}/{month}' not in f:
                            print(f"Warning: Data for {year}/{month} not found in model {model_num} ({model_name}), ensemble {ensamble}")
                            result.append(np.nan)
                            continue
                            
                        data_perc = f[f'hru_wb_30m/{year}/{month}/{var}'][:]
                        data_pcp = f[f'hru_wb_30m/{year}/{month}/precip'][:]
                        
                        # Also get snowfall data if available
                        try:
                            data_snow = f[f'hru_wb_30m/{year}/{month}/snofall'][:]
                        except KeyError:
                            # If snowfall data is not available, use zeros
                            print(f"Warning: Snowfall data for {year}/{month} not found in model {model_num}, using zeros")
                            data_snow = np.zeros_like(data_pcp)
                        
                        # Calculate total water input (precipitation + snowfall)
                        total_water_input = data_pcp + data_snow
                        
                        # Create a mask for areas with total water input > threshold
                        water_mask = np.where(total_water_input > precip_threshold, 1, 0)
                        
                        # Combine agricultural land mask with water input mask
                        combined_mask = mask * water_mask
                        
                        # Skip if no cells meet criteria
                        if np.sum(combined_mask) == 0:
                            result.append(0)  # No contribution when no water input or no agriculture
                            continue
                        
                        # Handle division by zero safely
                        with np.errstate(divide='ignore', invalid='ignore'):
                            data_perc_ratio = 100*(data_perc/total_water_input)
                        
                        # Apply the combined mask - only include agricultural areas with water input > threshold
                        average = data_perc_ratio * combined_mask
                        average = np.where(average>100, 100, average)  # Cap at 100%
                        
                        # For areas with water input <= threshold, set ratio to 0 instead of NaN
                        # This is outside the combined mask (where water input <= threshold but land is agriculture)
                        low_water_ag_mask = mask * (1 - water_mask)
                        average = np.where(low_water_ag_mask == 1, 0, average)
                        
                        # Replace 0 with nan for non-agricultural areas for calculation purposes
                        # But keep 0s for agricultural areas with low water input
                        non_ag_mask = 1 - mask
                        average = np.where(non_ag_mask == 1, np.nan, average)
                        
                        # Check if we have any valid data after masking
                        if np.all(np.isnan(average)):
                            result.append(np.nan)
                        else:
                            # Calculate median only for agricultural areas
                            result.append(np.nanmedian(average))
                    except Exception as e:
                        print(f"Error processing {year}/{month} for model {model_num} ({model_name}), ensemble {ensamble}: {e}")
                        result.append(np.nan)
    except Exception as e:
        print(f"Error opening file for model {model_num} ({model_name}), ensemble {ensamble}: {e}")
        return (model_num, ensamble, cluster, [np.nan] * ((end_year - start_year) * 12))
    
    return (model_num, ensamble, cluster, result)

def analyze_cluster_uncertainty(landuse_data, cluster, total_months, start_year, end_year, var):
    # Get models in this cluster
    models_in_cluster = landuse_data["cluster"][cluster]["models"]
    
    if not models_in_cluster:
        print(f"No models in cluster {cluster}, skipping analysis")
        return
    
    print(f"Cluster {cluster} has {len(models_in_cluster)} models")
    
    # Calculate statistics for each month
    monthly_data = []
    for month_idx in range(total_months):
        month_values = []
        for model_num in models_in_cluster:
            for ensamble in range(5):
                if (model_num in landuse_data["model"] and 
                    "ensamble" in landuse_data["model"][model_num] and
                    ensamble in landuse_data["model"][model_num]["ensamble"]):
                    data = landuse_data["model"][model_num]["ensamble"][ensamble]
                    if month_idx < len(data):
                        value = data[month_idx]
                        # Only include non-NaN values
                        if not np.isnan(value):
                            month_values.append(value)
        monthly_data.append(month_values)
    
    # Calculate statistics for each month, handling empty lists
    median = []
    upperband = []
    lowerband = []
    
    for month in monthly_data:
        if len(month) > 0:
            median.append(np.nanmedian(month))
            upperband.append(np.nanpercentile(month, 97.5))
            lowerband.append(np.nanpercentile(month, 2.5))
        else:
            # No valid data for this month across all models and ensembles
            print(f"Warning: No valid data for a month in cluster {cluster}, using 0 for plotting")
            median.append(0)
            upperband.append(0)
            lowerband.append(0)
    
    # Create plot
    create_plot(median, upperband, lowerband, total_months, start_year, end_year, var, cluster, len(models_in_cluster))

def analyze_overall_uncertainty(landuse_data, total_months, start_year, end_year, var):
    # Calculate statistics for all models
    monthly_data = []
    valid_models = 0
    
    # Count models with valid data
    for model_num in landuse_data["model"]:
        if "ensamble" in landuse_data["model"][model_num] and landuse_data["model"][model_num]["ensamble"]:
            has_valid_data = False
            for ensamble in range(5):
                if (ensamble in landuse_data["model"][model_num]["ensamble"] and 
                    any(not np.isnan(val) for val in landuse_data["model"][model_num]["ensamble"][ensamble])):
                    has_valid_data = True
                    break
            if has_valid_data:
                valid_models += 1
    
    for month_idx in range(total_months):
        month_values = []
        for model_num in landuse_data["model"]:
            if "ensamble" not in landuse_data["model"][model_num]:
                continue
                
            for ensamble in range(5):
                if ensamble in landuse_data["model"][model_num]["ensamble"]:
                    data = landuse_data["model"][model_num]["ensamble"][ensamble]
                    if month_idx < len(data):
                        value = data[month_idx]
                        # Only include non-NaN values
                        if not np.isnan(value):
                            month_values.append(value)
        monthly_data.append(month_values)
    
    # Calculate statistics for each month, handling empty lists
    median = []
    upperband = []
    lowerband = []
    
    for month in monthly_data:
        if len(month) > 0:
            median.append(np.nanmedian(month))
            upperband.append(np.nanpercentile(month, 97.5))
            lowerband.append(np.nanpercentile(month, 2.5))
        else:
            # No valid data for this month across all models and ensembles
            median.append(0)
            upperband.append(0)
            lowerband.append(0)
    
    # Create plot
    create_plot(median, upperband, lowerband, total_months, start_year, end_year, var, "all", valid_models)

def load_model_data(base_path, model_nums, var="perc", precip_threshold=10, max_workers=None, start_year=2000, end_year=2010):
    # Get model names and cluster mapping
    NAMES = os.listdir(base_path) 
    NAMES = [name for name in NAMES if os.path.isdir(os.path.join(base_path, name))]  
    
    print(f"Found {len(NAMES)} model directories in {base_path}")
    
    print("Generating cluster mapping...")
    name_to_cluster = get_model_cluster_mapping(base_path)
    print(f"Generated clusters for {len(name_to_cluster)} models")
    
    # Initialize landuse_data with proper nested structure
    landuse_data = {
        "model": {},
        "cluster": {}
    }
    
    # Initialize clusters
    for cluster in range(5):
        landuse_data["cluster"][cluster] = {"models": []}
    
    # Calculate total number of months
    num_years = end_year - start_year
    total_months = num_years * 12
    print(f"Analyzing {total_months} months from {start_year} to {end_year-1}")
    
    # Initialize model dictionaries
    for model_num in range(0, min(model_nums, len(NAMES))):
        model_name = NAMES[model_num]
        cluster = name_to_cluster.get(model_name, -1)
        landuse_data["model"][model_num] = {"ensamble": {}, "cluster": cluster}
        if cluster >= 0:
            landuse_data["cluster"][cluster]["models"].append(model_num)
        
        for ensamble in range(0, 5):
            landuse_data["model"][model_num]["ensamble"][ensamble] = []
    
    # Create a list of all (model_num, ensemble) pairs to process
    tasks = [(model_num, ensamble) for model_num in range(0, min(model_nums, len(NAMES))) for ensamble in range(0, 5)]
    
    # Process tasks in parallel
    print(f"Starting parallel processing with {max_workers or 'default'} workers")
    print(f"Extracting data for years {start_year} to {end_year-1}")
    print(f"Precipitation threshold: {precip_threshold}mm")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        process_func = partial(process_model_ensemble, base_path=base_path, NAMES=NAMES, 
                              name_to_cluster=name_to_cluster, var=var, 
                              start_year=start_year, end_year=end_year, 
                              precip_threshold=precip_threshold)
        results = list(tqdm.tqdm(executor.map(process_func, tasks), total=len(tasks)))
    
    # Populate the data structure with results
    for model_num, ensamble, cluster, data in results:
        if model_num in landuse_data["model"] and "ensamble" in landuse_data["model"][model_num]:
            landuse_data["model"][model_num]["ensamble"][ensamble] = data
    
    # For each cluster, analyze uncertainty
    for cluster in range(5):
        print(f"Analyzing cluster {cluster}...")
        analyze_cluster_uncertainty(landuse_data, cluster, total_months, start_year, end_year, var)
    
    # Also do the overall analysis
    print("Analyzing overall uncertainty...")
    analyze_overall_uncertainty(landuse_data, total_months, start_year, end_year, var)
    
    return landuse_data

if __name__ == "__main__":
    base_path = "/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12"
    model_nums = 50
    import multiprocessing
    # Use 80% of available CPU cores for parallel processing
    max_workers = max(1, int(multiprocessing.cpu_count() * 0.8))
    print(f"Using {max_workers} workers out of {multiprocessing.cpu_count()} available cores")
    
    # Define year range to extract
    start_year = 2000
    end_year = 2005  # This is exclusive (will extract 2000-2004)
    
    # Set precipitation threshold (mm)
    precip_threshold = 10
    
    # Set variable to analyze (percolation)
    var = "perc"
    
    # Make sure the Michigan directory exists
    os.makedirs('./Michigan', exist_ok=True)
    
    try:
        load_model_data(base_path, model_nums, var=var, precip_threshold=precip_threshold,
                       max_workers=max_workers, start_year=start_year, end_year=end_year)
        print("Processing completed successfully")
    except Exception as e:
        import traceback
        print(f"Error during processing: {e}")
        traceback.print_exc()






