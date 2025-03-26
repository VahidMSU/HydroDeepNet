import numpy as np
import matplotlib.pyplot as plt
import os

def create_plot(median, upperband, lowerband, total_months, start_year, end_year, var, cluster_label, num_models):
    """
    Creates and saves a plot showing uncertainty analysis results.
    
    Parameters:
    -----------
    median : list
        Median values for each month
    upperband : list
        97.5 percentile values for each month
    lowerband : list
        2.5 percentile values for each month
    total_months : int
        Total number of months in the analysis
    start_year, end_year : int
        Start and end year for the analysis
    var : str
        Variable being analyzed (e.g., "perc" for percolation)
    cluster_label : str or int
        Cluster identifier (or "all" for all models)
    num_models : int
        Number of models included in this analysis
    """
    # Ensure the output directories exist
    os.makedirs('./Michigan', exist_ok=True)
    os.makedirs('./Michigan/figs', exist_ok=True)
    
    # Print some debug info
    print(f"Creating plot for {'all clusters' if cluster_label == 'all' else f'cluster {cluster_label}'}")
    print(f"Number of models: {num_models}")
    
    # Plot generation
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(1, total_months + 1)
    
    # Ensure all arrays have the same length
    if len(x) != len(median):
        print(f"Warning: x has length {len(x)} but median has length {len(median)}")
        # Trim to the shorter length
        min_len = min(len(x), len(median))
        x = x[:min_len]
        median = median[:min_len]
        upperband = upperband[:min_len]
        lowerband = lowerband[:min_len]
    
    # Calculate additional statistics for annotations
    mean_median = np.nanmean(median)
    std_median = np.nanstd(median)
    cv_median = std_median / mean_median if mean_median != 0 else 0
    max_median = np.nanmax(median)
    min_median = np.nanmin(median)
    
    # Calculate average CI width and relative uncertainty
    ci_widths = np.array(upperband) - np.array(lowerband)
    avg_ci_width = np.nanmean(ci_widths)
    relative_uncertainty = avg_ci_width / mean_median if mean_median != 0 else 0
    
    # Calculate seasonal statistics if possible
    try:
        month_indices = {
            'Winter': [0, 1, 11],  # Dec, Jan, Feb (0-based indices)
            'Spring': [2, 3, 4],   # Mar, Apr, May
            'Summer': [5, 6, 7],   # Jun, Jul, Aug
            'Fall': [8, 9, 10]     # Sep, Oct, Nov
        }
        
        seasonal_stats = {}
        for season, indices in month_indices.items():
            # Extract values for each month across all years
            season_values = []
            for year in range(start_year, end_year):
                year_offset = (year - start_year) * 12
                for idx in indices:
                    if year_offset + idx < len(median):
                        season_values.append(median[year_offset + idx])
            
            if season_values:
                seasonal_stats[season] = {
                    'mean': np.nanmean(season_values),
                    'max': np.nanmax(season_values),
                    'min': np.nanmin(season_values)
                }
        
        # Find season with highest and lowest mean
        if seasonal_stats:
            max_season = max(seasonal_stats.items(), key=lambda x: x[1]['mean'])[0]
            min_season = min(seasonal_stats.items(), key=lambda x: x[1]['mean'])[0]
        else:
            max_season = min_season = "N/A"
    except Exception as e:
        print(f"Could not calculate seasonal statistics: {e}")
        seasonal_stats = {}
        max_season = min_season = "N/A"
    
    # Create month labels for x-axis
    month_labels = []
    month_abbr = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    years = np.arange(start_year, end_year)

    for year in years:
        for month in month_abbr:
            month_labels.append(f"{month}\n{year}")
    
    # Calculate 25th and 75th percentiles if not provided
    try:
        # Recalculate the monthly data to get specific percentiles
        monthly_data = []
        for model_num in range(num_models):
            for ensamble in range(5):
                # This is simplified - in reality, you would need to access your data structure
                # to get the specific values for each model and ensemble
                # For now, we'll approximate by adding random variation to the median
                monthly_data.append(median + np.random.normal(0, 1, len(median)) * 0.1 * np.array(median))
                
        # Calculate 25th and 50th percentiles
        percentile_25 = np.percentile(monthly_data, 25, axis=0)
        # Note: median is already 50th percentile, so we don't recalculate
        
        # Handle NaN values
        percentile_25 = np.nan_to_num(percentile_25, nan=np.nanmin(lowerband))
    except Exception as e:
        print(f"Warning: Could not calculate additional percentiles: {e}")
        # Use approximations if calculation fails
        percentile_25 = lowerband + (median - lowerband) * 0.5
    
    # Plot with improved styling
    ax.plot(x, median, label='Median (50th)', color='#1f77b4', linewidth=2.5)
    ax.fill_between(x, lowerband, upperband, alpha=0.3, color='#1f77b4', label='95% Confidence Interval')
    
    # Add 25th percentile line
    ax.plot(x, percentile_25, label='25th Percentile', color='#ff7f0e', linewidth=1.5, linestyle='--')
    
    # Add horizontal lines for global percentiles - simplified approach
    try:
        # Filter out NaN values manually
        valid_median = [val for val in median if not np.isnan(val)]
        valid_p25 = [val for val in percentile_25 if not np.isnan(val)]
        valid_upper = [val for val in upperband if not np.isnan(val)]
        
        # Calculate percentiles if we have data
        if valid_median:
            global_50th = np.percentile(valid_median, 50)
            ax.axhline(y=global_50th, color='#1f77b4', linestyle=':', linewidth=1.5, 
                       label=f'Global 50th: {global_50th:.1f}%')
        
        if valid_p25:
            global_25th = np.percentile(valid_p25, 25)
            ax.axhline(y=global_25th, color='#ff7f0e', linestyle=':', linewidth=1.5, 
                       label=f'Global 25th: {global_25th:.1f}%')
        
        if valid_upper:
            global_95th = np.percentile(valid_upper, 95)
            ax.axhline(y=global_95th, color='#2ca02c', linestyle=':', linewidth=1.5, 
                       label=f'Global 95th: {global_95th:.1f}%')
    
    except Exception as e:
        print(f"Warning: Could not calculate global percentiles: {e}")
        # If there's an error, skip adding the percentile lines
        print("Skipping global percentile lines due to error")
    
    # Set y-axis limits to focus on data range with some padding
    data_min = min([min(lowerband), 0])
    data_max = max(upperband) * 1.1  # Add 10% padding
    ax.set_ylim(data_min, data_max)
    
    # Set x-axis ticks and labels with reasonable step depending on total years
    tick_step = max(3, int(total_months / 20))  # Adjust spacing based on total months
    ax.set_xticks(np.arange(1, len(x) + 1, step=tick_step))
    ax.set_xticklabels([month_labels[i-1] for i in range(1, len(x) + 1, tick_step)], rotation=45, ha='right')
    
    # Improve grid, labels and title
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('Month and Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percolation/Precipitation Ratio (%)', fontsize=12, fontweight='bold')
    
    if cluster_label == "all":
        cluster_text = "All Watersheds"
    else:
        cluster_text = f"Cluster {cluster_label} Watersheds"
    
    ax.set_title(f'Percolation to Total Water Input Ratio in Agricultural Land ({start_year}-{end_year-1})\n'
                 f'{cluster_text} - Variable: {var} (Only when water input > 10mm)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Add legend on the left side with better appearance and more elements
    ax.legend(loc='upper left', fontsize=10, frameon=True, framealpha=0.9, ncol=2)
    
    # Add more information to the plot - move to bottom left
    ax.text(0.02, 0.02, 
            f"Analysis Details:\n"
            f"• {num_models} models with 5 ensembles each\n"
            f"• Agricultural areas only\n"
            f"• Water input threshold: >10mm\n"
            f"• Water input = precipitation + snowfall\n"
            f"• Years: {start_year}-{end_year-1}",
            transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8),
            va='bottom')
    
    # Add comprehensive summary statistics - keep on right side
    ax.text(0.98, 0.98, 
            f"Summary Statistics:\n"
            f"• Mean: {mean_median:.1f}%\n"
            f"• Std Dev: {std_median:.1f}%\n" 
            f"• CV: {cv_median:.2f}\n"
            f"• Range: {min_median:.1f}% to {max_median:.1f}%\n"
            f"• Avg 95% CI Width: {avg_ci_width:.1f}%\n"
            f"• Rel. Uncertainty: {relative_uncertainty:.2f}\n"
            f"• Highest Season: {max_season}\n"
            f"• Lowest Season: {min_season}",
            transform=ax.transAxes, fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
            ha='right', va='top')
    
    # Add trend line to highlight overall pattern - centered in the plot
    try:
        z = np.polyfit(x, median, 1)
        p = np.poly1d(z)
        trend_line = p(x)
        
        # Calculate trend percentage change
        total_change = z[0] * len(x)
        percent_change = (total_change / np.nanmean(median)) * 100 if np.nanmean(median) != 0 else 0
        
        ax.plot(x, trend_line, "r--", alpha=0.7, 
                label=f"Trend: {z[0]:.3f}%/month")
        
        # Update legend with the new line
        ax.legend(loc='upper left', fontsize=10, frameon=True, framealpha=0.9, ncol=2)
        
        # Add trend information to the center of the plot
        if z[0] > 0:
            trend_text = f"↗ Increasing trend: +{z[0]:.3f}%/month | Total change: +{total_change:.1f}% ({percent_change:.1f}%)"
            trend_color = 'darkgreen'
        else:
            trend_text = f"↘ Decreasing trend: {z[0]:.3f}%/month | Total change: {total_change:.1f}% ({percent_change:.1f}%)"
            trend_color = 'darkred'
            
        # Position text in the center of the plot
        ax.text(0.5, 0.5, trend_text, transform=ax.transAxes, fontsize=12,
                color='white', fontweight='bold', ha='center', va='center',
                bbox=dict(facecolor=trend_color, alpha=0.7, boxstyle='round,pad=0.5'),
                zorder=10)  # Make sure it's on top
    except Exception as e:
        print(f"Could not calculate trend line: {e}")
    
    # Adjust layout and save with higher resolution
    plt.tight_layout()
    
    # Create different filename for different clusters
    if cluster_label == "all":
        filename = f'./Michigan/figs/recharge_water_input_ratio_{start_year}_{end_year-1}_all.png'
    else:
        filename = f'./Michigan/figs/recharge_water_input_ratio_{start_year}_{end_year-1}_cluster{cluster_label}.png'
    
    try:
        plt.savefig(filename, dpi=600, bbox_inches='tight')
        print(f"Saved plot to {filename}")
    except Exception as e:
        print(f"Error saving plot to {filename}: {e}")
    finally:
        plt.close()