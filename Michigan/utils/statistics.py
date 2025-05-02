import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy import stats

def calculate_monthly_statistics(landuse_data, var="perc", start_year=2000, end_year=2005):
    """
    Calculate monthly statistics across all models and clusters.
    
    Parameters:
    -----------
    landuse_data : dict
        Data structure containing all model results
    var : str
        Variable analyzed
    start_year, end_year : int
        Year range for analysis
    
    Returns:
    --------
    dict
        Dictionary containing monthly statistics
    """
    # Calculate number of months
    num_years = end_year - start_year
    total_months = num_years * 12
    
    # Create month-year labels
    month_labels = []
    month_abbr = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for year in range(start_year, end_year):
        for month in month_abbr:
            month_labels.append(f"{month} {year}")
    
    # Initialize results dictionary
    stats = {
        "overall": {},
        "clusters": {}
    }
    
    # Get all data by month across all models
    monthly_data = [[] for _ in range(total_months)]
    
    # Populate monthly data
    for model_num in landuse_data["model"]:
        if "ensamble" not in landuse_data["model"][model_num]:
            continue
            
        for ensamble in range(5):
            if ensamble not in landuse_data["model"][model_num]["ensamble"]:
                continue
                
            data = landuse_data["model"][model_num]["ensamble"][ensamble]
            
            for month_idx in range(min(len(data), total_months)):
                if not np.isnan(data[month_idx]):
                    monthly_data[month_idx].append(data[month_idx])
    
    # Calculate overall statistics
    overall_mean = [np.mean(month) if month else np.nan for month in monthly_data]
    overall_median = [np.median(month) if month else np.nan for month in monthly_data]
    overall_std = [np.std(month) if month else np.nan for month in monthly_data]
    overall_cv = [np.std(month)/np.mean(month) if month and np.mean(month) != 0 else np.nan for month in monthly_data]
    overall_q25 = [np.percentile(month, 25) if month else np.nan for month in monthly_data]
    overall_q75 = [np.percentile(month, 75) if month else np.nan for month in monthly_data]
    overall_min = [np.min(month) if month else np.nan for month in monthly_data]
    overall_max = [np.max(month) if month else np.nan for month in monthly_data]
    
    # Store overall statistics
    stats["overall"] = {
        "month_labels": month_labels,
        "mean": overall_mean,
        "median": overall_median,
        "std": overall_std,
        "cv": overall_cv,  # Coefficient of variation
        "q25": overall_q25,
        "q75": overall_q75,
        "min": overall_min,
        "max": overall_max,
        "sample_sizes": [len(month) for month in monthly_data]
    }
    
    # Calculate statistics by cluster
    for cluster in range(5):
        # Get list of models in this cluster
        if cluster not in landuse_data["cluster"] or "models" not in landuse_data["cluster"][cluster]:
            continue
            
        models_in_cluster = landuse_data["cluster"][cluster]["models"]
        
        # Skip empty clusters
        if not models_in_cluster:
            continue
        
        # Initialize monthly data arrays for this cluster
        cluster_monthly_data = [[] for _ in range(total_months)]
        
        # Populate cluster monthly data
        for model_num in models_in_cluster:
            if model_num not in landuse_data["model"] or "ensamble" not in landuse_data["model"][model_num]:
                continue
                
            for ensamble in range(5):
                if ensamble not in landuse_data["model"][model_num]["ensamble"]:
                    continue
                    
                data = landuse_data["model"][model_num]["ensamble"][ensamble]
                
                for month_idx in range(min(len(data), total_months)):
                    if not np.isnan(data[month_idx]):
                        cluster_monthly_data[month_idx].append(data[month_idx])
        
        # Calculate cluster statistics
        cluster_mean = [np.mean(month) if month else np.nan for month in cluster_monthly_data]
        cluster_median = [np.median(month) if month else np.nan for month in cluster_monthly_data]
        cluster_std = [np.std(month) if month else np.nan for month in cluster_monthly_data]
        cluster_cv = [np.std(month)/np.mean(month) if month and np.mean(month) != 0 else np.nan for month in cluster_monthly_data]
        cluster_q25 = [np.percentile(month, 25) if month else np.nan for month in cluster_monthly_data]
        cluster_q75 = [np.percentile(month, 75) if month else np.nan for month in cluster_monthly_data]
        cluster_min = [np.min(month) if month else np.nan for month in cluster_monthly_data]
        cluster_max = [np.max(month) if month else np.nan for month in cluster_monthly_data]
        
        # Store cluster statistics
        stats["clusters"][cluster] = {
            "month_labels": month_labels,
            "mean": cluster_mean,
            "median": cluster_median,
            "std": cluster_std,
            "cv": cluster_cv,  # Coefficient of variation
            "q25": cluster_q25,
            "q75": cluster_q75,
            "min": cluster_min,
            "max": cluster_max,
            "sample_sizes": [len(month) for month in cluster_monthly_data]
        }
    
    return stats

def calculate_seasonal_statistics(landuse_data, var="perc", start_year=2000, end_year=2005):
    """
    Calculate seasonal statistics across all models and clusters.
    
    Parameters:
    -----------
    landuse_data : dict
        Data structure containing all model results
    var : str
        Variable analyzed
    start_year, end_year : int
        Year range for analysis
    
    Returns:
    --------
    dict
        Dictionary containing seasonal statistics
    """
    # Calculate number of months
    num_years = end_year - start_year
    total_months = num_years * 12
    
    # Define seasons (meteorological seasons in Northern Hemisphere)
    # Winter: Dec, Jan, Feb
    # Spring: Mar, Apr, May
    # Summer: Jun, Jul, Aug
    # Fall: Sep, Oct, Nov
    seasons = {
        "winter": [12, 1, 2],
        "spring": [3, 4, 5],
        "summer": [6, 7, 8],
        "fall": [9, 10, 11]
    }
    
    # Initialize results dictionary
    stats = {
        "overall": {},
        "clusters": {}
    }
    
    # Initialize seasonal data containers
    seasonal_data = {season: [] for season in seasons}
    
    # Populate seasonal data from all models
    for model_num in landuse_data["model"]:
        if "ensamble" not in landuse_data["model"][model_num]:
            continue
            
        for ensamble in range(5):
            if ensamble not in landuse_data["model"][model_num]["ensamble"]:
                continue
                
            data = landuse_data["model"][model_num]["ensamble"][ensamble]
            
            for month_idx in range(min(len(data), total_months)):
                if np.isnan(data[month_idx]):
                    continue
                    
                # Calculate year and month
                year_offset = month_idx // 12
                month = (month_idx % 12) + 1  # Convert to 1-12 range
                current_year = start_year + year_offset
                
                # Determine season
                for season, months in seasons.items():
                    if month in months:
                        seasonal_data[season].append(data[month_idx])
                        break
    
    # Calculate overall seasonal statistics
    stats["overall"] = {
        season: {
            "mean": np.mean(data) if data else np.nan,
            "median": np.median(data) if data else np.nan,
            "std": np.std(data) if data else np.nan,
            "cv": np.std(data)/np.mean(data) if data and np.mean(data) != 0 else np.nan,
            "q25": np.percentile(data, 25) if data else np.nan,
            "q75": np.percentile(data, 75) if data else np.nan,
            "min": np.min(data) if data else np.nan,
            "max": np.max(data) if data else np.nan,
            "sample_size": len(data)
        }
        for season, data in seasonal_data.items()
    }
    
    # Initialize cluster statistics
    stats["clusters"] = {cluster: {} for cluster in range(5)}
    
    # Calculate statistics by cluster
    for cluster in range(5):
        # Get list of models in this cluster
        if cluster not in landuse_data["cluster"] or "models" not in landuse_data["cluster"][cluster]:
            continue
            
        models_in_cluster = landuse_data["cluster"][cluster]["models"]
        
        # Skip empty clusters
        if not models_in_cluster:
            continue
        
        # Initialize seasonal data containers for this cluster
        cluster_seasonal_data = {season: [] for season in seasons}
        
        # Populate cluster seasonal data
        for model_num in models_in_cluster:
            if model_num not in landuse_data["model"] or "ensamble" not in landuse_data["model"][model_num]:
                continue
                
            for ensamble in range(5):
                if ensamble not in landuse_data["model"][model_num]["ensamble"]:
                    continue
                    
                data = landuse_data["model"][model_num]["ensamble"][ensamble]
                
                for month_idx in range(min(len(data), total_months)):
                    if np.isnan(data[month_idx]):
                        continue
                        
                    # Calculate year and month
                    year_offset = month_idx // 12
                    month = (month_idx % 12) + 1  # Convert to 1-12 range
                    
                    # Determine season
                    for season, months in seasons.items():
                        if month in months:
                            cluster_seasonal_data[season].append(data[month_idx])
                            break
        
        # Calculate cluster seasonal statistics
        for season, data in cluster_seasonal_data.items():
            stats["clusters"][cluster][season] = {
                "mean": np.mean(data) if data else np.nan,
                "median": np.median(data) if data else np.nan,
                "std": np.std(data) if data else np.nan,
                "cv": np.std(data)/np.mean(data) if data and np.mean(data) != 0 else np.nan,
                "q25": np.percentile(data, 25) if data else np.nan,
                "q75": np.percentile(data, 75) if data else np.nan,
                "min": np.min(data) if data else np.nan,
                "max": np.max(data) if data else np.nan,
                "sample_size": len(data)
            }
    
    return stats

def calculate_annual_statistics(landuse_data, var="perc", start_year=2000, end_year=2005):
    """
    Calculate annual statistics across all models and clusters.
    
    Parameters:
    -----------
    landuse_data : dict
        Data structure containing all model results
    var : str
        Variable analyzed
    start_year, end_year : int
        Year range for analysis
    
    Returns:
    --------
    dict
        Dictionary containing annual statistics
    """
    # Calculate number of years
    num_years = end_year - start_year
    
    # Initialize results dictionary
    stats = {
        "overall": {},
        "clusters": {}
    }
    
    # Initialize annual data containers
    annual_data = {year: [] for year in range(start_year, end_year)}
    
    # Populate annual data from all models
    for model_num in landuse_data["model"]:
        if "ensamble" not in landuse_data["model"][model_num]:
            continue
            
        for ensamble in range(5):
            if ensamble not in landuse_data["model"][model_num]["ensamble"]:
                continue
                
            data = landuse_data["model"][model_num]["ensamble"][ensamble]
            
            for year_idx in range(num_years):
                year = start_year + year_idx
                
                # Get data for this year (12 months)
                year_start_idx = year_idx * 12
                year_end_idx = year_start_idx + 12
                
                if year_end_idx > len(data):
                    continue
                    
                year_data = data[year_start_idx:year_end_idx]
                
                # Skip if all values are NaN
                if all(np.isnan(val) for val in year_data):
                    continue
                
                # Calculate annual average (skipping NaN values)
                valid_values = [val for val in year_data if not np.isnan(val)]
                if valid_values:
                    annual_avg = np.mean(valid_values)
                    annual_data[year].append(annual_avg)
    
    # Calculate overall annual statistics
    stats["overall"] = {
        year: {
            "mean": np.mean(data) if data else np.nan,
            "median": np.median(data) if data else np.nan,
            "std": np.std(data) if data else np.nan,
            "cv": np.std(data)/np.mean(data) if data and np.mean(data) != 0 else np.nan,
            "q25": np.percentile(data, 25) if data else np.nan,
            "q75": np.percentile(data, 75) if data else np.nan,
            "min": np.min(data) if data else np.nan,
            "max": np.max(data) if data else np.nan,
            "sample_size": len(data)
        }
        for year, data in annual_data.items()
    }
    
    # Initialize cluster statistics
    stats["clusters"] = {cluster: {} for cluster in range(5)}
    
    # Calculate statistics by cluster
    for cluster in range(5):
        # Get list of models in this cluster
        if cluster not in landuse_data["cluster"] or "models" not in landuse_data["cluster"][cluster]:
            continue
            
        models_in_cluster = landuse_data["cluster"][cluster]["models"]
        
        # Skip empty clusters
        if not models_in_cluster:
            continue
        
        # Initialize annual data containers for this cluster
        cluster_annual_data = {year: [] for year in range(start_year, end_year)}
        
        # Populate cluster annual data
        for model_num in models_in_cluster:
            if model_num not in landuse_data["model"] or "ensamble" not in landuse_data["model"][model_num]:
                continue
                
            for ensamble in range(5):
                if ensamble not in landuse_data["model"][model_num]["ensamble"]:
                    continue
                    
                data = landuse_data["model"][model_num]["ensamble"][ensamble]
                
                for year_idx in range(num_years):
                    year = start_year + year_idx
                    
                    # Get data for this year (12 months)
                    year_start_idx = year_idx * 12
                    year_end_idx = year_start_idx + 12
                    
                    if year_end_idx > len(data):
                        continue
                        
                    year_data = data[year_start_idx:year_end_idx]
                    
                    # Skip if all values are NaN
                    if all(np.isnan(val) for val in year_data):
                        continue
                    
                    # Calculate annual average (skipping NaN values)
                    valid_values = [val for val in year_data if not np.isnan(val)]
                    if valid_values:
                        annual_avg = np.mean(valid_values)
                        cluster_annual_data[year].append(annual_avg)
        
        # Calculate cluster annual statistics
        for year, data in cluster_annual_data.items():
            stats["clusters"][cluster][year] = {
                "mean": np.mean(data) if data else np.nan,
                "median": np.median(data) if data else np.nan,
                "std": np.std(data) if data else np.nan,
                "cv": np.std(data)/np.mean(data) if data and np.mean(data) != 0 else np.nan,
                "q25": np.percentile(data, 25) if data else np.nan,
                "q75": np.percentile(data, 75) if data else np.nan,
                "min": np.min(data) if data else np.nan,
                "max": np.max(data) if data else np.nan,
                "sample_size": len(data)
            }
    
    return stats

def generate_statistics_tables(monthly_stats, seasonal_stats, annual_stats, output_dir="./Michigan"):
    """
    Generate HTML tables from statistics.
    
    Parameters:
    -----------
    monthly_stats : dict
        Monthly statistics
    seasonal_stats : dict
        Seasonal statistics
    annual_stats : dict
        Annual statistics
    output_dir : str
        Directory to save HTML tables
    
    Returns:
    --------
    dict
        Dictionary containing paths to HTML tables
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "monthly": {},
        "seasonal": {},
        "annual": {}
    }
    
    # Generate monthly statistics tables
    monthly_overall_df = pd.DataFrame({
        "Month": monthly_stats["overall"]["month_labels"],
        "Mean (%)": monthly_stats["overall"]["mean"],
        "Median (%)": monthly_stats["overall"]["median"],
        "Std Dev (%)": monthly_stats["overall"]["std"],
        "CV": monthly_stats["overall"]["cv"],
        "25th Perc (%)": monthly_stats["overall"]["q25"],
        "75th Perc (%)": monthly_stats["overall"]["q75"],
        "Min (%)": monthly_stats["overall"]["min"],
        "Max (%)": monthly_stats["overall"]["max"],
        "Sample Size": monthly_stats["overall"]["sample_sizes"]
    })
    
    # Save monthly overall table
    monthly_overall_path = os.path.join(output_dir, "monthly_stats_overall.html")
    monthly_overall_df.to_html(monthly_overall_path, index=False, float_format="%.2f")
    results["monthly"]["overall"] = os.path.basename(monthly_overall_path)
    
    # Generate monthly cluster tables
    for cluster, data in monthly_stats["clusters"].items():
        monthly_cluster_df = pd.DataFrame({
            "Month": data["month_labels"],
            "Mean (%)": data["mean"],
            "Median (%)": data["median"],
            "Std Dev (%)": data["std"],
            "CV": data["cv"],
            "25th Perc (%)": data["q25"],
            "75th Perc (%)": data["q75"],
            "Min (%)": data["min"],
            "Max (%)": data["max"],
            "Sample Size": data["sample_sizes"]
        })
        
        # Save monthly cluster table
        monthly_cluster_path = os.path.join(output_dir, f"monthly_stats_cluster{cluster}.html")
        monthly_cluster_df.to_html(monthly_cluster_path, index=False, float_format="%.2f")
        results["monthly"][f"cluster{cluster}"] = os.path.basename(monthly_cluster_path)
    
    # Generate seasonal statistics table
    seasons = list(seasonal_stats["overall"].keys())
    seasonal_overall_df = pd.DataFrame({
        "Season": seasons,
        "Mean (%)": [seasonal_stats["overall"][season]["mean"] for season in seasons],
        "Median (%)": [seasonal_stats["overall"][season]["median"] for season in seasons],
        "Std Dev (%)": [seasonal_stats["overall"][season]["std"] for season in seasons],
        "CV": [seasonal_stats["overall"][season]["cv"] for season in seasons],
        "25th Perc (%)": [seasonal_stats["overall"][season]["q25"] for season in seasons],
        "75th Perc (%)": [seasonal_stats["overall"][season]["q75"] for season in seasons],
        "Min (%)": [seasonal_stats["overall"][season]["min"] for season in seasons],
        "Max (%)": [seasonal_stats["overall"][season]["max"] for season in seasons],
        "Sample Size": [seasonal_stats["overall"][season]["sample_size"] for season in seasons]
    })
    
    # Save seasonal overall table
    seasonal_overall_path = os.path.join(output_dir, "seasonal_stats_overall.html")
    seasonal_overall_df.to_html(seasonal_overall_path, index=False, float_format="%.2f")
    results["seasonal"]["overall"] = os.path.basename(seasonal_overall_path)
    
    # Generate seasonal cluster tables
    for cluster, cluster_data in seasonal_stats["clusters"].items():
        if not cluster_data:  # Skip empty clusters
            continue
            
        seasons = list(cluster_data.keys())
        seasonal_cluster_df = pd.DataFrame({
            "Season": seasons,
            "Mean (%)": [cluster_data[season]["mean"] for season in seasons],
            "Median (%)": [cluster_data[season]["median"] for season in seasons],
            "Std Dev (%)": [cluster_data[season]["std"] for season in seasons],
            "CV": [cluster_data[season]["cv"] for season in seasons],
            "25th Perc (%)": [cluster_data[season]["q25"] for season in seasons],
            "75th Perc (%)": [cluster_data[season]["q75"] for season in seasons],
            "Min (%)": [cluster_data[season]["min"] for season in seasons],
            "Max (%)": [cluster_data[season]["max"] for season in seasons],
            "Sample Size": [cluster_data[season]["sample_size"] for season in seasons]
        })
        
        # Save seasonal cluster table
        seasonal_cluster_path = os.path.join(output_dir, f"seasonal_stats_cluster{cluster}.html")
        seasonal_cluster_df.to_html(seasonal_cluster_path, index=False, float_format="%.2f")
        results["seasonal"][f"cluster{cluster}"] = os.path.basename(seasonal_cluster_path)
    
    # Generate annual statistics table
    years = list(annual_stats["overall"].keys())
    annual_overall_df = pd.DataFrame({
        "Year": years,
        "Mean (%)": [annual_stats["overall"][year]["mean"] for year in years],
        "Median (%)": [annual_stats["overall"][year]["median"] for year in years],
        "Std Dev (%)": [annual_stats["overall"][year]["std"] for year in years],
        "CV": [annual_stats["overall"][year]["cv"] for year in years],
        "25th Perc (%)": [annual_stats["overall"][year]["q25"] for year in years],
        "75th Perc (%)": [annual_stats["overall"][year]["q75"] for year in years],
        "Min (%)": [annual_stats["overall"][year]["min"] for year in years],
        "Max (%)": [annual_stats["overall"][year]["max"] for year in years],
        "Sample Size": [annual_stats["overall"][year]["sample_size"] for year in years]
    })
    
    # Save annual overall table
    annual_overall_path = os.path.join(output_dir, "annual_stats_overall.html")
    annual_overall_df.to_html(annual_overall_path, index=False, float_format="%.2f")
    results["annual"]["overall"] = os.path.basename(annual_overall_path)
    
    # Generate annual cluster tables
    for cluster, cluster_data in annual_stats["clusters"].items():
        if not cluster_data:  # Skip empty clusters
            continue
            
        years = list(cluster_data.keys())
        annual_cluster_df = pd.DataFrame({
            "Year": years,
            "Mean (%)": [cluster_data[year]["mean"] for year in years],
            "Median (%)": [cluster_data[year]["median"] for year in years],
            "Std Dev (%)": [cluster_data[year]["std"] for year in years],
            "CV": [cluster_data[year]["cv"] for year in years],
            "25th Perc (%)": [cluster_data[year]["q25"] for year in years],
            "75th Perc (%)": [cluster_data[year]["q75"] for year in years],
            "Min (%)": [cluster_data[year]["min"] for year in years],
            "Max (%)": [cluster_data[year]["max"] for year in years],
            "Sample Size": [cluster_data[year]["sample_size"] for year in years]
        })
        
        # Save annual cluster table
        annual_cluster_path = os.path.join(output_dir, f"annual_stats_cluster{cluster}.html")
        annual_cluster_df.to_html(annual_cluster_path, index=False, float_format="%.2f")
        results["annual"][f"cluster{cluster}"] = os.path.basename(annual_cluster_path)
    
    return results

def generate_statistics_plots(monthly_stats, seasonal_stats, annual_stats, output_dir="./Michigan"):
    """
    Generate plots from statistics.
    
    Parameters:
    -----------
    monthly_stats : dict
        Monthly statistics
    seasonal_stats : dict
        Seasonal statistics
    annual_stats : dict
        Annual statistics
    output_dir : str
        Directory to save plots
    
    Returns:
    --------
    dict
        Dictionary containing paths to plot files
    """
    # Create directory structure
    os.makedirs(output_dir, exist_ok=True)
    figs_dir = os.path.join(output_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    
    results = {
        "boxplot": {},
        "seasonal": {},
        "annual": {},
        "cluster_comparison": {}
    }
    
    # Set plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Generate monthly boxplot for overall data
    month_names = [label.split()[0] for label in monthly_stats["overall"]["month_labels"]]
    years = [int(label.split()[1]) for label in monthly_stats["overall"]["month_labels"]]
    unique_years = sorted(set(years))
    
    # Create monthly boxplot
    fig, ax = plt.subplots(figsize=(15, 8))
    boxplot = ax.boxplot([monthly_stats["overall"]["mean"][i:i+12] for i in range(0, len(month_names), 12)], 
               labels=unique_years, whis=(2.5, 97.5), patch_artist=True)
    
    # Customize boxplot appearance
    for box in boxplot['boxes']:
        box.set(facecolor='lightblue', alpha=0.8)
    for whisker in boxplot['whiskers']:
        whisker.set(color='navy', linewidth=1.5, linestyle='--')
    for cap in boxplot['caps']:
        cap.set(color='navy', linewidth=1.5)
    for median in boxplot['medians']:
        median.set(color='darkred', linewidth=2)
    for flier in boxplot['fliers']:
        flier.set(marker='o', markersize=5, markerfacecolor='red', alpha=0.5)
    
    ax.set_title("Annual Variability in Percolation/Precipitation Ratio", fontsize=14, fontweight='bold')
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Monthly Ratio (%)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add statistical annotations
    yearly_medians = [np.median(monthly_stats["overall"]["mean"][i:i+12]) for i in range(0, len(month_names), 12)]
    yearly_means = [np.mean(monthly_stats["overall"]["mean"][i:i+12]) for i in range(0, len(month_names), 12)]
    yearly_iqrs = [np.percentile(monthly_stats["overall"]["mean"][i:i+12], 75) - 
                  np.percentile(monthly_stats["overall"]["mean"][i:i+12], 25) 
                  for i in range(0, len(month_names), 12)]
    
    # Add a text box with summary statistics
    stats_text = "Annual Statistics:\n"
    for i, year in enumerate(unique_years):
        stats_text += f"{year}: Median={yearly_medians[i]:.2f}%, Mean={yearly_means[i]:.2f}%, IQR={yearly_iqrs[i]:.2f}%\n"
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=props)
    
    # Add horizontal percentile lines for boxplot
    overall_mean_values = monthly_stats["overall"]["mean"]
    valid_values = [x for x in overall_mean_values if not np.isnan(x)]
    
    percentile_25 = np.percentile(valid_values, 25)
    percentile_50 = np.percentile(valid_values, 50)
    percentile_95 = np.percentile(valid_values, 95)
    
    ax.axhline(y=percentile_25, color='#ff7f0e', linestyle=':', linewidth=1.5, 
               label=f'25th Percentile: {percentile_25:.1f}%')
    ax.axhline(y=percentile_50, color='#1f77b4', linestyle=':', linewidth=1.5, 
               label=f'50th Percentile: {percentile_50:.1f}%')
    ax.axhline(y=percentile_95, color='#2ca02c', linestyle=':', linewidth=1.5, 
               label=f'95th Percentile: {percentile_95:.1f}%')
    
    # Add legend for percentile lines
    ax.legend(loc='upper left', fontsize=9)
    
    # Save boxplot
    boxplot_path = os.path.join(figs_dir, "annual_variability_boxplot.png")
    plt.tight_layout()
    plt.savefig(boxplot_path, dpi=300)
    plt.close()
    results["boxplot"]["annual"] = os.path.basename(boxplot_path)
    
    # Generate seasonal comparison plot
    seasons = list(seasonal_stats["overall"].keys())
    season_means = [seasonal_stats["overall"][season]["mean"] for season in seasons]
    season_medians = [seasonal_stats["overall"][season]["median"] for season in seasons]
    season_stds = [seasonal_stats["overall"][season]["std"] for season in seasons]
    season_q25 = [seasonal_stats["overall"][season]["q25"] for season in seasons]
    season_q75 = [seasonal_stats["overall"][season]["q75"] for season in seasons]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # First, create the bar chart with means
    bars = ax.bar(seasons, season_means, capsize=10, color='skyblue', edgecolor='navy', alpha=0.7, label='Mean')
    
    # Then add error bars showing standard deviation
    ax.errorbar(seasons, season_means, yerr=season_stds, fmt='none', color='black', capsize=5, label='±1 Std Dev')
    
    # Add a line for the median values
    ax.plot(seasons, season_medians, 'ro-', linewidth=2, markersize=8, label='Median')
    
    # Add IQR range as a line
    for i, season in enumerate(seasons):
        ax.vlines(x=i, ymin=season_q25[i], ymax=season_q75[i], colors='darkred', linestyles='--', linewidth=2)
    
    ax.set_title("Seasonal Variation in Percolation/Precipitation Ratio", fontsize=14, fontweight='bold')
    ax.set_xlabel("Season", fontsize=12)
    ax.set_ylabel("Ratio (%)", fontsize=12)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='upper left')
    
    # Add value labels with more detailed statistics
    for i, v in enumerate(season_means):
        # Get sample size
        sample_size = seasonal_stats["overall"][seasons[i]]["sample_size"]
        # Calculate coefficient of variation
        cv = seasonal_stats["overall"][seasons[i]]["cv"]
        stats_str = f"Mean: {v:.1f}%\nMedian: {season_medians[i]:.1f}%\nCV: {cv:.2f}\nn={sample_size}"
        
        # Position text to the right of the bar
        ax.text(i, v + 0.5, stats_str, ha='center', va='bottom', fontsize=8, 
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # Add horizontal percentile lines for seasonal plot
    all_season_means = np.array(season_means)
    valid_season_means = all_season_means[~np.isnan(all_season_means)]
    
    season_p25 = np.percentile(valid_season_means, 25)
    season_p50 = np.percentile(valid_season_means, 50)
    season_p95 = np.percentile(valid_season_means, 95)
    
    ax.axhline(y=season_p25, color='#ff7f0e', linestyle=':', linewidth=1.5, 
              label=f'25th: {season_p25:.1f}%')
    ax.axhline(y=season_p50, color='#1f77b4', linestyle=':', linewidth=1.5, 
              label=f'50th: {season_p50:.1f}%')
    ax.axhline(y=season_p95, color='#2ca02c', linestyle=':', linewidth=1.5, 
              label=f'95th: {season_p95:.1f}%')
    
    # Update legend to include percentile lines
    ax.legend(loc='upper left')
    
    # Save seasonal plot
    seasonal_path = os.path.join(figs_dir, "seasonal_comparison.png")
    plt.tight_layout()
    plt.savefig(seasonal_path, dpi=300)
    plt.close()
    results["seasonal"]["overall"] = os.path.basename(seasonal_path)
    
    # Generate annual trends plot
    years = list(annual_stats["overall"].keys())
    year_means = [annual_stats["overall"][year]["mean"] for year in years]
    year_medians = [annual_stats["overall"][year]["median"] for year in years]
    year_q25 = [annual_stats["overall"][year]["q25"] for year in years]
    year_q75 = [annual_stats["overall"][year]["q75"] for year in years]
    year_mins = [annual_stats["overall"][year]["min"] for year in years]
    year_maxs = [annual_stats["overall"][year]["max"] for year in years]
    year_sample_sizes = [annual_stats["overall"][year]["sample_size"] for year in years]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the mean line
    ax.plot(years, year_means, marker='o', linewidth=2, color='navy', label="Mean Ratio")
    
    # Plot the median line
    ax.plot(years, year_medians, marker='s', linewidth=2, linestyle='--', color='darkred', label="Median Ratio")
    
    # Shade the IQR area
    ax.fill_between(years, year_q25, year_q75, alpha=0.3, color='royalblue', label="Interquartile Range (25-75th)")
    
    # Add min-max range with lighter shading
    ax.fill_between(years, year_mins, year_maxs, alpha=0.1, color='gray', label="Min-Max Range")
    
    ax.set_title("Annual Trends in Percolation/Precipitation Ratio", fontsize=14, fontweight='bold')
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Annual Ratio (%)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper left')
    
    # Add annotations with more detailed statistics
    for i, year in enumerate(years):
        stats_str = (f"Mean: {year_means[i]:.1f}%\n"
                    f"Median: {year_medians[i]:.1f}%\n"
                    f"IQR: [{year_q25[i]:.1f}%-{year_q75[i]:.1f}%]\n"
                    f"Range: [{year_mins[i]:.1f}%-{year_maxs[i]:.1f}%]\n"
                    f"n={year_sample_sizes[i]}")
        
        # Position text above the point
        y_pos = max(year_maxs[i] + 1, year_means[i] + 10)  # Ensure text is visible
        ax.annotate(stats_str, xy=(year, year_means[i]), xytext=(year, y_pos),
                   fontsize=8, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7),
                   arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    
    # Add horizontal percentile lines for annual plot
    all_year_means = np.array(year_means)
    valid_year_means = all_year_means[~np.isnan(all_year_means)]
    
    year_p25 = np.percentile(valid_year_means, 25)
    year_p50 = np.percentile(valid_year_means, 50)
    year_p95 = np.percentile(valid_year_means, 95)
    
    ax.axhline(y=year_p25, color='#ff7f0e', linestyle=':', linewidth=1.5, 
              label=f'25th: {year_p25:.1f}%')
    ax.axhline(y=year_p50, color='#1f77b4', linestyle=':', linewidth=1.5, 
              label=f'50th: {year_p50:.1f}%')
    ax.axhline(y=year_p95, color='#2ca02c', linestyle=':', linewidth=1.5, 
              label=f'95th: {year_p95:.1f}%')
    
    # Add trend line in center of plot
    try:
        x_numeric = np.array([int(year) for year in years])
        z = np.polyfit(x_numeric, year_means, 1)
        p = np.poly1d(z)
        trend_line = p(x_numeric)
        
        # Calculate trend percentage change
        total_change = z[0] * len(years)
        percent_change = (total_change / np.nanmean(year_means)) * 100 if np.nanmean(year_means) != 0 else 0
        
        ax.plot(years, trend_line, "r-", linewidth=2, alpha=0.7, 
                label=f"Trend: {z[0]:.2f}%/year")
        
        # Add trend information to the center of the plot
        if z[0] > 0:
            trend_text = f"↗ Increasing trend: +{z[0]:.2f}%/year\nTotal change: +{total_change:.1f}% ({percent_change:.1f}%)"
            trend_color = 'darkgreen'
        else:
            trend_text = f"↘ Decreasing trend: {z[0]:.2f}%/year\nTotal change: {total_change:.1f}% ({percent_change:.1f}%)"
            trend_color = 'darkred'
            
        # Position text in the center of the plot
        ax.text(0.5, 0.5, trend_text, transform=ax.transAxes, fontsize=12,
                color='white', fontweight='bold', ha='center', va='center',
                bbox=dict(facecolor=trend_color, alpha=0.7, boxstyle='round,pad=0.5'),
                zorder=10)  # Make sure it's on top
    except Exception as e:
        print(f"Could not calculate trend line: {e}")
    
    # Update legend to include trend line
    ax.legend(loc='upper left')
    
    # Save annual trends plot
    annual_path = os.path.join(figs_dir, "annual_trends.png")
    plt.tight_layout()
    plt.savefig(annual_path, dpi=300)
    plt.close()
    results["annual"]["overall"] = os.path.basename(annual_path)
    
    # Generate cluster comparison plot (using seasonal data)
    valid_clusters = [cluster for cluster in seasonal_stats["clusters"] if seasonal_stats["clusters"][cluster]]
    
    if valid_clusters:
        # Extract seasonal means and medians for each cluster
        cluster_seasonal_means = {
            cluster: [seasonal_stats["clusters"][cluster][season]["mean"] for season in seasons]
            for cluster in valid_clusters
        }
        
        cluster_seasonal_medians = {
            cluster: [seasonal_stats["clusters"][cluster][season]["median"] for season in seasons]
            for cluster in valid_clusters
        }
        
        cluster_seasonal_stds = {
            cluster: [seasonal_stats["clusters"][cluster][season]["std"] for season in seasons]
            for cluster in valid_clusters
        }
        
        # Create grouped bar chart with error bars
        fig, ax = plt.subplots(figsize=(14, 10))
        bar_width = 0.15
        index = np.arange(len(seasons))
        
        # Use a different color for each cluster
        colors = plt.cm.tab10(np.linspace(0, 1, len(valid_clusters)))
        
        # Plot bars and collect handles for legend
        handles = []
        for i, cluster in enumerate(valid_clusters):
            offset = (i - len(valid_clusters)/2) * bar_width
            bars = ax.bar(index + offset, cluster_seasonal_means[cluster], bar_width, 
                    color=colors[i], label=f'Cluster {cluster} Mean', alpha=0.7)
            handles.append(bars)
            
            # Add error bars for standard deviation
            ax.errorbar(index + offset, cluster_seasonal_means[cluster], 
                       yerr=cluster_seasonal_stds[cluster], fmt='none', 
                       color='black', capsize=3)
            
            # Add line connecting median values
            median_line = ax.plot(index + offset, cluster_seasonal_medians[cluster], 
                                 marker='o', linestyle='-', linewidth=1.5, 
                                 color=colors[i], alpha=0.9, label=f'Cluster {cluster} Median')
            handles.append(median_line[0])
        
        # Add cluster-specific statistical annotations in a legend-like format
        legend_text = "Cluster Statistics (Mean±StdDev):\n"
        for cluster in valid_clusters:
            cluster_text = f"Cluster {cluster}: "
            for i, season in enumerate(seasons):
                mean = seasonal_stats["clusters"][cluster][season]["mean"]
                std = seasonal_stats["clusters"][cluster][season]["std"]
                cluster_text += f"{season}={mean:.1f}±{std:.1f}%, "
            legend_text += cluster_text + "\n"
        
        ax.text(0.5, -0.15, legend_text, transform=ax.transAxes, fontsize=10, 
                ha='center', va='center', bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.9))
        
        ax.set_title("Seasonal Patterns by Cluster", fontsize=16, fontweight='bold')
        ax.set_xlabel("Season", fontsize=14)
        ax.set_ylabel("Average Ratio (%)", fontsize=14)
        ax.set_xticks(index)
        ax.set_xticklabels(seasons, fontsize=12)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Create custom legend
        ax.legend(loc='upper left', fontsize=10)
        
        # Save cluster comparison plot
        cluster_comp_path = os.path.join(figs_dir, "cluster_seasonal_comparison.png")
        plt.tight_layout()
        plt.savefig(cluster_comp_path, dpi=300, bbox_inches='tight')
        plt.close()
        results["cluster_comparison"]["seasonal"] = os.path.basename(cluster_comp_path)
        
        # Calculate overall percentiles across all clusters and seasons
        all_means = []
        for cluster in valid_clusters:
            all_means.extend(cluster_seasonal_means[cluster])
        
        cluster_p25 = np.percentile(all_means, 25)
        cluster_p50 = np.percentile(all_means, 50)
        cluster_p95 = np.percentile(all_means, 95)
        
        # Add horizontal percentile lines
        ax.axhline(y=cluster_p25, color='#ff7f0e', linestyle=':', linewidth=1.5, 
                  label=f'Overall 25th: {cluster_p25:.1f}%')
        ax.axhline(y=cluster_p50, color='#1f77b4', linestyle=':', linewidth=1.5, 
                  label=f'Overall 50th: {cluster_p50:.1f}%')
        ax.axhline(y=cluster_p95, color='#2ca02c', linestyle=':', linewidth=1.5, 
                  label=f'Overall 95th: {cluster_p95:.1f}%')
        
        # Update legend
        ax.legend(loc='upper left', fontsize=10)
        
        # Add a new heatmap visualization for cluster comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a data matrix for heatmap (clusters x seasons)
        data_matrix = np.array([cluster_seasonal_means[cluster] for cluster in valid_clusters])
        
        # Create the heatmap
        im = ax.imshow(data_matrix, cmap='viridis')
        
        # Add colorbar with percentile markers
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Percolation/Precipitation Ratio (%)', fontsize=12)
        
        # Add percentile markers to colorbar
        cbar.ax.axhline(y=(cluster_p25 - im.norm.vmin) / (im.norm.vmax - im.norm.vmin), 
                      color='#ff7f0e', linestyle=':', linewidth=1.5)
        cbar.ax.axhline(y=(cluster_p50 - im.norm.vmin) / (im.norm.vmax - im.norm.vmin), 
                      color='#1f77b4', linestyle=':', linewidth=1.5)
        cbar.ax.axhline(y=(cluster_p95 - im.norm.vmin) / (im.norm.vmax - im.norm.vmin), 
                      color='#2ca02c', linestyle=':', linewidth=1.5)
        
        cbar.ax.text(1.5, (cluster_p25 - im.norm.vmin) / (im.norm.vmax - im.norm.vmin), 
                   f'25th: {cluster_p25:.1f}%', ha='left', va='center', fontsize=9)
        cbar.ax.text(1.5, (cluster_p50 - im.norm.vmin) / (im.norm.vmax - im.norm.vmin), 
                   f'50th: {cluster_p50:.1f}%', ha='left', va='center', fontsize=9)
        cbar.ax.text(1.5, (cluster_p95 - im.norm.vmin) / (im.norm.vmax - im.norm.vmin), 
                   f'95th: {cluster_p95:.1f}%', ha='left', va='center', fontsize=9)
        
        # Set axis labels and ticks
        ax.set_xticks(np.arange(len(seasons)))
        ax.set_yticks(np.arange(len(valid_clusters)))
        ax.set_xticklabels(seasons, fontsize=12)
        ax.set_yticklabels([f'Cluster {c}' for c in valid_clusters], fontsize=12)
        
        # Add text annotations in each cell
        for i in range(len(valid_clusters)):
            for j in range(len(seasons)):
                mean = data_matrix[i, j]
                std = cluster_seasonal_stds[valid_clusters[i]][j]
                sample_size = seasonal_stats["clusters"][valid_clusters[i]][seasons[j]]["sample_size"]
                text = ax.text(j, i, f"{mean:.1f}±{std:.1f}\nn={sample_size}", 
                              ha="center", va="center", color="white" if mean > 30 else "black",
                              fontsize=10, fontweight='bold')
        
        ax.set_title(f"Cluster-Season Heatmap of Percolation/Precipitation Ratio\n" +
                    f"25th: {cluster_p25:.1f}%, 50th: {cluster_p50:.1f}%, 95th: {cluster_p95:.1f}%", 
                    fontsize=16, fontweight='bold')
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Save the heatmap
        heatmap_path = os.path.join(figs_dir, "cluster_season_heatmap.png")
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
        results["cluster_comparison"]["heatmap"] = os.path.basename(heatmap_path)
    
    return results
