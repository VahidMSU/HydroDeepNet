import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from matplotlib.colors import LinearSegmentedColormap
try:
    from loca2_dataset import DataImporter, list_of_cc_models
except ImportError:
    from AI_agent.loca2_dataset import DataImporter, list_of_cc_models

def full_climate_change_data(bbox):
    df = list_of_cc_models()

    print(f"cc_models: {df['cc_model'].unique()}")
    print(f"scenarios: {df['scenario'].unique()}")
    print(f"ensembles: {df['ensemble'].unique()}")
    
    config = {
        "RESOLUTION": 250,
        "huc8": None,
        "video": False,
        "aggregation": "monthly",
        'bounding_box': bbox,
    }

    importer = DataImporter(config)
    
    # Define scenarios configuration
    scenarios_config = {
        "historical": {
            "periods": [{"start_year": 1950, "end_year": 2015}],
            "cc_model": "ACCESS-CM2",
            "ensemble": "r2i1p1f1"
        },
        "ssp245": {  # low emission
            "periods": [
                {"start_year": 2016, "end_year": 2045},  # near future
                {"start_year": 2045, "end_year": 2075},  # mid future
                {"start_year": 2075, "end_year": 2100}   # far future
            ],
            "cc_model": "ACCESS-CM2",
            "ensemble": "r2i1p1f1"
        },
        "ssp370": {  # medium emission
            "periods": [
                {"start_year": 2016, "end_year": 2045},
                {"start_year": 2045, "end_year": 2075},
                {"start_year": 2075, "end_year": 2100}
            ],
            "cc_model": "ACCESS-CM2",
            "ensemble": "r2i1p1f1"
        },
        "ssp585": {  # high emission
            "periods": [
                {"start_year": 2016, "end_year": 2045},
                {"start_year": 2045, "end_year": 2075},
                {"start_year": 2075, "end_year": 2100}
            ],
            "cc_model": "ACCESS-CM2",
            "ensemble": "r2i1p1f1"
        }
    }
    
    # Dictionary to store results
    results = {}
    
    # Loop through scenarios and periods
    for scenario_name, scenario_data in scenarios_config.items():
        results[scenario_name] = []
        
        for period in scenario_data["periods"]:
            start_year = period["start_year"]
            end_year = period["end_year"]
            cc_model = scenario_data["cc_model"]
            ensemble = scenario_data["ensemble"]
            
            print(f"Processing: {scenario_name}, {start_year}-{end_year}")
            
            # Extract data
            ppt_loca2, tmax_loca2, tmin_loca2 = importer.LOCA2(
                start_year=start_year, 
                end_year=end_year, 
                cc_model=cc_model, 
                scenario=scenario_name, 
                ensemble=ensemble, 
                cc_time_step='daily'
            )
            
            # Store results
            period_result = {
                "period": f"{start_year}-{end_year}",
                "precipitation": ppt_loca2,
                "tmax": tmax_loca2,
                "tmin": tmin_loca2
            }
            
            results[scenario_name].append(period_result)
    
    # Now you can access data like: results["ssp245"][0]["precipitation"] for the first period's data
    # Example of accessing data
    print(f"Number of scenarios processed: {len(results)}")
    for scenario, periods in results.items():
        print(f"Scenario: {scenario}, Periods: {len(periods)}")


    return results

def calculate_climate_changes(results):
    """
    Calculate changes in temperature and precipitation compared to historical data
    Returns a dictionary with differences organized by scenario and period
    """
    climate_changes = {}
    
    # Get historical data (assuming it's the first period of the historical scenario)
    if 'historical' in results and results['historical']:
        historical_data = results['historical'][0]
        historical_ppt = historical_data['precipitation']
        historical_tmax = historical_data['tmax']
        historical_tmin = historical_data['tmin']
        
        # Calculate annual averages for historical period
        hist_annual_ppt = np.mean(historical_ppt, axis=0)  # Average across time dimension
        hist_annual_tmax = np.mean(historical_tmax, axis=0)
        hist_annual_tmin = np.mean(historical_tmin, axis=0)
        
        # Calculate changes for each scenario and period
        for scenario, periods in results.items():
            if scenario == 'historical':
                continue
                
            climate_changes[scenario] = []
            
            for period_data in periods:
                period = period_data['period']
                
                # Calculate annual averages
                period_ppt = np.mean(period_data['precipitation'], axis=0)
                period_tmax = np.mean(period_data['tmax'], axis=0)
                period_tmin = np.mean(period_data['tmin'], axis=0)
                
                # Calculate differences
                ppt_change_pct = ((period_ppt - hist_annual_ppt) / hist_annual_ppt) * 100
                tmax_change = period_tmax - hist_annual_tmax
                tmin_change = period_tmin - hist_annual_tmin
                
                # Store changes
                climate_changes[scenario].append({
                    'period': period,
                    'ppt_change_pct': ppt_change_pct,  # Percentage change
                    'tmax_change': tmax_change,        # Absolute change in °C
                    'tmin_change': tmin_change         # Absolute change in °C
                })
    
    return climate_changes

def create_summary_table(climate_changes):
    """
    Create a summary table of climate changes across scenarios and periods
    """
    table_data = []
    
    for scenario, periods in climate_changes.items():
        for period_data in periods:
            period = period_data['period']
            
            # Calculate spatial means
            mean_ppt_change = np.nanmean(period_data['ppt_change_pct'])
            mean_tmax_change = np.nanmean(period_data['tmax_change'])
            mean_tmin_change = np.nanmean(period_data['tmin_change'])
            
            table_data.append([
                scenario,
                period,
                f"{mean_ppt_change:.2f}%",
                f"{mean_tmax_change:.2f}°C",
                f"{mean_tmin_change:.2f}°C"
            ])
    
    # Create a DataFrame for better formatting
    df = pd.DataFrame(
        table_data, 
        columns=['Scenario', 'Period', 'PPT Change (%)', 'Tmax Change (°C)', 'Tmin Change (°C)']
    )
    
    return df

def plot_climate_changes(climate_changes, output_dir=None, prefix=""):
    """
    Create visualizations of climate changes
    
    Args:
        climate_changes: Dictionary containing climate change data
        output_dir: Directory to save the figure
        prefix: Prefix to add to output filename to avoid conflicts
        
    Returns:
        Matplotlib Figure object
    """
    # Set up the figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Prepare data for plotting
    scenarios = []
    periods = []
    ppt_changes = []
    tmax_changes = []
    tmin_changes = []
    
    for scenario, periods_data in climate_changes.items():
        for period_data in periods_data:
            period = period_data['period']
            
            # Calculate spatial means
            mean_ppt_change = np.nanmean(period_data['ppt_change_pct'])
            mean_tmax_change = np.nanmean(period_data['tmax_change'])
            mean_tmin_change = np.nanmean(period_data['tmin_change'])
            
            scenarios.append(scenario)
            periods.append(period)
            ppt_changes.append(mean_ppt_change)
            tmax_changes.append(mean_tmax_change)
            tmin_changes.append(mean_tmin_change)
    
    # Create DataFrame for easier plotting
    plot_df = pd.DataFrame({
        'Scenario': scenarios,
        'Period': periods,
        'Precipitation Change (%)': ppt_changes,
        'Tmax Change (°C)': tmax_changes,
        'Tmin Change (°C)': tmin_changes
    })
    
    # Plotting
    # 1. Precipitation change plot
    sns.barplot(x='Period', y='Precipitation Change (%)', hue='Scenario', data=plot_df, ax=axes[0])
    axes[0].set_title('Precipitation Change Compared to Historical Period')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # 2. Tmax change plot
    sns.barplot(x='Period', y='Tmax Change (°C)', hue='Scenario', data=plot_df, ax=axes[1])
    axes[1].set_title('Maximum Temperature Change Compared to Historical Period')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # 3. Tmin change plot
    sns.barplot(x='Period', y='Tmin Change (°C)', hue='Scenario', data=plot_df, ax=axes[2])
    axes[2].set_title('Minimum Temperature Change Compared to Historical Period')
    axes[2].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(f"{output_dir}/{prefix}climate_change_summary.png", dpi=300, bbox_inches='tight')
    
    return fig

def plot_spatial_changes(climate_changes, scenario='ssp585', period_idx=2):
    """
    Create spatial visualizations of climate changes for a specific scenario and period
    """
    if scenario not in climate_changes or len(climate_changes[scenario]) <= period_idx:
        print(f"Scenario {scenario} or period index {period_idx} not available.")
        return None
    
    period_data = climate_changes[scenario][period_idx]
    period = period_data['period']
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Custom colormap for precipitation (blue to brown)
    ppt_cmap = LinearSegmentedColormap.from_list('ppt_cmap', ['brown', 'white', 'blue'])
    
    # 1. Precipitation change map
    im1 = axes[0].imshow(period_data['ppt_change_pct'], cmap=ppt_cmap)
    axes[0].set_title(f'Precipitation Change (%) - {scenario} {period}')
    plt.colorbar(im1, ax=axes[0], label='%')
    
    # 2. Tmax change map
    im2 = axes[1].imshow(period_data['tmax_change'], cmap='coolwarm')
    axes[1].set_title(f'Max Temperature Change (°C) - {scenario} {period}')
    plt.colorbar(im2, ax=axes[1], label='°C')
    
    # 3. Tmin change map
    im3 = axes[2].imshow(period_data['tmin_change'], cmap='coolwarm')
    axes[2].set_title(f'Min Temperature Change (°C) - {scenario} {period}')
    plt.colorbar(im3, ax=axes[2], label='°C')
    
    plt.tight_layout()
    return fig

def analyze_climate_changes(results, output_dir=None, prefix=""):
    """
    Analyze climate changes and create visualizations
    
    Args:
        results: Dictionary of climate data
        output_dir: Directory to save outputs
        prefix: Prefix to add to output filenames to avoid conflicts
        
    Returns:
        Dictionary containing analysis results
    """
    # Calculate changes compared to historical data
    climate_changes = calculate_climate_changes(results)
    
    # Create a summary table
    summary_table = create_summary_table(climate_changes)
    print("\nClimate Change Summary Table:")
    print(tabulate(summary_table, headers='keys', tablefmt='grid'))
    
    if output_dir:
        summary_table.to_csv(f"{output_dir}/{prefix}climate_change_summary.csv", index=False)
    
    # Create summary plots
    plot_fig = plot_climate_changes(climate_changes, output_dir, prefix)
    
    # Create spatial plots for the worst-case scenario (high emission, far future)
    spatial_fig = plot_spatial_changes(climate_changes)
    
    if output_dir and spatial_fig:
        spatial_fig.savefig(f"{output_dir}/{prefix}spatial_changes_worst_case.png", dpi=300, bbox_inches='tight')
    
    return {
        'summary_table': summary_table,
        'climate_changes': climate_changes,
        'plots': {
            'summary_plot': plot_fig,
            'spatial_plot': spatial_fig
        }
    }

if __name__ == "__main__":
    import os
    # Create output directory for visualizations - use the main directory
    output_dir = "/data/SWATGenXApp/codes/climate_change_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define bounding box
    bbox = [-85.444332, 43.658148, -85.239256, 44.164683]
    
    # Get climate data
    results = full_climate_change_data(bbox)
    
    # Analyze climate changes and create visualizations
    # Use "multi_scenario_" prefix for output files
    analysis_results = analyze_climate_changes(results, output_dir, prefix="multi_scenario_")
    
    print("\nAnalysis complete. Visualizations saved to:", output_dir)
    
    # Display some statistics about the data
    print("\nScenarios analyzed:", list(results.keys()))
    print("\nExample statistics for high-emission scenario (SSP585):")
    if 'ssp585' in results and results['ssp585']:
        last_period = results['ssp585'][-1]['period']
        print(f"Far future period ({last_period}):")
        
        # Get the corresponding climate change data
        if 'ssp585' in analysis_results['climate_changes']:
            far_future_changes = analysis_results['climate_changes']['ssp585'][-1]
            print(f"  Average precipitation change: {np.nanmean(far_future_changes['ppt_change_pct']):.2f}%")
            print(f"  Average max temperature increase: {np.nanmean(far_future_changes['tmax_change']):.2f}°C")
            print(f"  Average min temperature increase: {np.nanmean(far_future_changes['tmin_change']):.2f}°C")
            
            # Calculate extreme values
            print(f"  Max precipitation change: {np.nanmax(far_future_changes['ppt_change_pct']):.2f}%")
            print(f"  Max temperature increase: {np.nanmax(far_future_changes['tmax_change']):.2f}°C")
    
    plt.show()  # Display the plots if running in an interactive environment
