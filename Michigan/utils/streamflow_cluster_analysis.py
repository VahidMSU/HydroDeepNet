import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Tuple, Optional
import glob
from collections import Counter
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import clustering function
try:
    from utils.geolocate import get_model_cluster_mapping
except ImportError:
    print("Warning: Could not import get_model_cluster_mapping")

def analyze_streamflow_clusters(results_dict: Dict, 
                               output_dir: str = './Michigan',
                               significance_threshold: float = 0.05):
    """
    Analyze streamflow stations by cluster to identify regional patterns.
    
    Parameters:
    -----------
    results_dict : Dict
        Results dictionary from StreamflowAnalyzer
    output_dir : str
        Directory to save output files
    significance_threshold : float
        P-value threshold for statistical significance
        
    Returns:
    --------
    Dict
        Dictionary with cluster analysis results and paths to generated visualizations
    """
    print("Starting streamflow cluster analysis...")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    figs_dir = os.path.join(output_dir, 'figs/streamflow_clusters')
    os.makedirs(figs_dir, exist_ok=True)
    
    # Extract station results
    stations_results = results_dict.get('results', {})
    
    if not stations_results:
        print("No station results found for analysis")
        return None
    
    # Get cluster mapping for watersheds
    try:
        cluster_mapping = get_model_cluster_mapping('/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12')
        print(f"Retrieved cluster mapping for {len(cluster_mapping)} watersheds")
    except Exception as e:
        print(f"Error retrieving cluster mapping: {e}")
        # Create empty mapping as fallback
        cluster_mapping = {}
    
    # Map stations to clusters based on their watershed
    station_clusters = {}
    for station_name, result in stations_results.items():
        watershed = result.get('watershed', '')
        # Assign cluster based on watershed
        cluster = cluster_mapping.get(watershed, -1)  # Use -1 for unknown clusters
        station_clusters[station_name] = cluster
    
    print(f"Mapped {len(station_clusters)} stations to clusters")
    
    # Process trend data
    trend_data = []
    
    for station_name, result in stations_results.items():
        cluster = station_clusters.get(station_name, -1)
        watershed = result.get('watershed', '')
        
        # Get trend analysis results
        streamflow_trend = result.get('streamflow_trend', {})
        baseflow_trend = result.get('baseflow_trend', {})
        quickflow_trend = result.get('quickflow_trend', {})
        
        # Calculate if trends are significant (p-value < significance_threshold)
        streamflow_significant = streamflow_trend.get('p_value', 1.0) < significance_threshold
        baseflow_significant = baseflow_trend.get('p_value', 1.0) < significance_threshold
        quickflow_significant = quickflow_trend.get('p_value', 1.0) < significance_threshold
        
        # Determine trend direction (increase, decrease, or no significant change)
        streamflow_direction = 'no change'
        if streamflow_significant:
            streamflow_direction = 'increase' if streamflow_trend.get('trend_percent', 0) > 0 else 'decrease'
            
        baseflow_direction = 'no change'
        if baseflow_significant:
            baseflow_direction = 'increase' if baseflow_trend.get('trend_percent', 0) > 0 else 'decrease'
            
        quickflow_direction = 'no change'
        if quickflow_significant:
            quickflow_direction = 'increase' if quickflow_trend.get('trend_percent', 0) > 0 else 'decrease'
        
        # Create record
        trend_data.append({
            'station_name': station_name,
            'watershed': watershed,
            'cluster': cluster,
            'streamflow_mean': streamflow_trend.get('mean', np.nan),  # in cfs
            'baseflow_mean': baseflow_trend.get('mean', np.nan),  # in cfs
            'quickflow_mean': quickflow_trend.get('mean', np.nan),  # in cfs
            'baseflow_index': result.get('baseflow_index', np.nan),
            'streamflow_trend': streamflow_trend.get('trend_percent', np.nan),
            'baseflow_trend': baseflow_trend.get('trend_percent', np.nan),
            'quickflow_trend': quickflow_trend.get('trend_percent', np.nan),
            'streamflow_p_value': streamflow_trend.get('p_value', np.nan),
            'baseflow_p_value': baseflow_trend.get('p_value', np.nan),
            'quickflow_p_value': quickflow_trend.get('p_value', np.nan),
            'streamflow_significant': streamflow_significant,
            'baseflow_significant': baseflow_significant,
            'quickflow_significant': quickflow_significant,
            'streamflow_direction': streamflow_direction,
            'baseflow_direction': baseflow_direction,
            'quickflow_direction': quickflow_direction
        })
    
    # Create DataFrame
    df = pd.DataFrame(trend_data)
    
    # Save the full dataset
    csv_path = os.path.join(output_dir, 'streamflow_cluster_analysis.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved complete analysis to {csv_path}")
    
    # Analysis results
    results = {
        'visualizations': {},
        'tables': {},
        'cluster_summaries': {}
    }
    
    # Analyze trend patterns by cluster
    cluster_trends = analyze_cluster_trends(df, figs_dir)
    results['visualizations'].update(cluster_trends['visualizations'])
    results['tables'].update(cluster_trends['tables'])
    results['cluster_summaries']['trends'] = cluster_trends['summaries']
    
    # Analyze seasonal patterns by cluster
    seasonal_patterns = analyze_seasonal_patterns(stations_results, station_clusters, figs_dir)
    results['visualizations'].update(seasonal_patterns['visualizations'])
    results['tables'].update(seasonal_patterns['tables'])
    results['cluster_summaries']['seasonal'] = seasonal_patterns['summaries']
    
    # Analyze baseflow index by cluster
    bfi_analysis = analyze_baseflow_index(df, figs_dir)
    results['visualizations'].update(bfi_analysis['visualizations'])
    results['tables'].update(bfi_analysis['tables'])
    results['cluster_summaries']['bfi'] = bfi_analysis['summaries']
    
    # Generate cluster comparison HTML report
    report_path = generate_cluster_report(df, results, output_dir)
    results['report'] = report_path
    
    print("Streamflow cluster analysis completed")
    return results

def analyze_cluster_trends(df: pd.DataFrame, output_dir: str) -> Dict:
    """
    Analyze trend patterns by cluster.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with streamflow analysis results
    output_dir : str
        Directory to save output files
        
    Returns:
    --------
    Dict
        Dictionary with analysis results
    """
    print("Analyzing trend patterns by cluster...")
    
    results = {
        'visualizations': {},
        'tables': {},
        'summaries': {}
    }
    
    # Only consider valid clusters (not -1)
    valid_clusters = sorted([c for c in df['cluster'].unique() if c >= 0])
    
    if not valid_clusters:
        print("No valid clusters found for trend analysis")
        return results
    
    # Create summary of trend directions by cluster
    cluster_trend_summary = {}
    
    for cluster in valid_clusters:
        cluster_df = df[df['cluster'] == cluster]
        
        # Count trend directions for each flow component
        streamflow_counts = Counter(cluster_df['streamflow_direction'])
        baseflow_counts = Counter(cluster_df['baseflow_direction'])
        quickflow_counts = Counter(cluster_df['quickflow_direction'])
        
        # Calculate percentages
        total = len(cluster_df)
        
        streamflow_pct = {k: (v/total)*100 for k, v in streamflow_counts.items()} if total > 0 else {}
        baseflow_pct = {k: (v/total)*100 for k, v in baseflow_counts.items()} if total > 0 else {}
        quickflow_pct = {k: (v/total)*100 for k, v in quickflow_counts.items()} if total > 0 else {}
        
        # Store summary
        cluster_trend_summary[cluster] = {
            'n_stations': total,
            'streamflow': {
                'counts': dict(streamflow_counts),
                'percentages': streamflow_pct,
                'mean_trend': cluster_df['streamflow_trend'].mean(),
                'median_trend': cluster_df['streamflow_trend'].median()
            },
            'baseflow': {
                'counts': dict(baseflow_counts),
                'percentages': baseflow_pct,
                'mean_trend': cluster_df['baseflow_trend'].mean(),
                'median_trend': cluster_df['baseflow_trend'].median()
            },
            'quickflow': {
                'counts': dict(quickflow_counts),
                'percentages': quickflow_pct,
                'mean_trend': cluster_df['quickflow_trend'].mean(),
                'median_trend': cluster_df['quickflow_trend'].median()
            }
        }
    
    # Store summary in results
    results['summaries'] = cluster_trend_summary
    
    # Create summary table
    summary_rows = []
    
    for cluster, summary in cluster_trend_summary.items():
        # Streamflow trend percentages
        sf_increase = summary['streamflow']['percentages'].get('increase', 0)
        sf_decrease = summary['streamflow']['percentages'].get('decrease', 0)
        sf_no_change = summary['streamflow']['percentages'].get('no change', 0)
        
        # Baseflow trend percentages
        bf_increase = summary['baseflow']['percentages'].get('increase', 0)
        bf_decrease = summary['baseflow']['percentages'].get('decrease', 0)
        bf_no_change = summary['baseflow']['percentages'].get('no change', 0)
        
        # Quickflow trend percentages
        qf_increase = summary['quickflow']['percentages'].get('increase', 0)
        qf_decrease = summary['quickflow']['percentages'].get('decrease', 0)
        qf_no_change = summary['quickflow']['percentages'].get('no change', 0)
        
        # Create row
        summary_rows.append({
            'Cluster': f'Cluster {cluster}',
            'Stations': summary['n_stations'],
            'Streamflow Increase (%)': sf_increase,
            'Streamflow Decrease (%)': sf_decrease,
            'Streamflow No Change (%)': sf_no_change,
            'Baseflow Increase (%)': bf_increase,
            'Baseflow Decrease (%)': bf_decrease,
            'Baseflow No Change (%)': bf_no_change,
            'Quickflow Increase (%)': qf_increase,
            'Quickflow Decrease (%)': qf_decrease,
            'Quickflow No Change (%)': qf_no_change,
            'Mean Streamflow Trend (%)': summary['streamflow']['mean_trend'],
            'Mean Baseflow Trend (%)': summary['baseflow']['mean_trend'],
            'Mean Quickflow Trend (%)': summary['quickflow']['mean_trend']
        })
    
    # Create and save summary table
    summary_df = pd.DataFrame(summary_rows)
    
    # Save summary table
    summary_table_path = os.path.join(output_dir, 'cluster_trend_summary.html')
    summary_df.to_html(summary_table_path, index=False, float_format='%.2f')
    results['tables']['trend_summary'] = os.path.basename(summary_table_path)
    
    # Create visualizations
    
    # 1. Bar chart of trend directions by cluster
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Colors for trend directions
    colors = {'Increase': 'green', 'Decrease': 'red', 'No change': 'gray'}
    
    # Streamflow trends
    streamflow_data = []
    for cluster, summary in cluster_trend_summary.items():
        for direction, pct in summary['streamflow']['percentages'].items():
            streamflow_data.append({
                'Cluster': f'Cluster {cluster}',
                'Direction': direction.capitalize(),
                'Percentage': pct
            })
    
    if streamflow_data:
        streamflow_df = pd.DataFrame(streamflow_data)
        
        # Sort by cluster
        streamflow_df['Cluster_num'] = streamflow_df['Cluster'].str.extract(r'(\d+)').astype(int)
        streamflow_df = streamflow_df.sort_values('Cluster_num')
        streamflow_df = streamflow_df.drop('Cluster_num', axis=1)
        
        # Plot
        streamflow_plot = sns.barplot(x='Cluster', y='Percentage', hue='Direction', 
                                     data=streamflow_df, ax=axes[0], palette=colors)
        axes[0].set_title('Streamflow Trend Directions by Cluster', fontsize=14)
        axes[0].set_ylabel('Percentage of Stations (%)', fontsize=12)
        axes[0].set_ylim(0, 100)
        
        # Add data labels
        for p in streamflow_plot.patches:
            if p.get_height() > 0:
                streamflow_plot.annotate(f'{p.get_height():.1f}%', 
                                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                                        ha='center', va='bottom', fontsize=8)
    
    # Baseflow trends
    baseflow_data = []
    for cluster, summary in cluster_trend_summary.items():
        for direction, pct in summary['baseflow']['percentages'].items():
            baseflow_data.append({
                'Cluster': f'Cluster {cluster}',
                'Direction': direction.capitalize(),
                'Percentage': pct
            })
    
    if baseflow_data:
        baseflow_df = pd.DataFrame(baseflow_data)
        
        # Sort by cluster
        baseflow_df['Cluster_num'] = baseflow_df['Cluster'].str.extract(r'(\d+)').astype(int)
        baseflow_df = baseflow_df.sort_values('Cluster_num')
        baseflow_df = baseflow_df.drop('Cluster_num', axis=1)
        
        # Plot
        baseflow_plot = sns.barplot(x='Cluster', y='Percentage', hue='Direction', 
                                   data=baseflow_df, ax=axes[1], palette=colors)
        axes[1].set_title('Baseflow Trend Directions by Cluster', fontsize=14)
        axes[1].set_ylabel('Percentage of Stations (%)', fontsize=12)
        axes[1].set_ylim(0, 100)
        
        # Add data labels
        for p in baseflow_plot.patches:
            if p.get_height() > 0:
                baseflow_plot.annotate(f'{p.get_height():.1f}%', 
                                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                                      ha='center', va='bottom', fontsize=8)
    
    # Quickflow trends
    quickflow_data = []
    for cluster, summary in cluster_trend_summary.items():
        for direction, pct in summary['quickflow']['percentages'].items():
            quickflow_data.append({
                'Cluster': f'Cluster {cluster}',
                'Direction': direction.capitalize(),
                'Percentage': pct
            })
    
    if quickflow_data:
        quickflow_df = pd.DataFrame(quickflow_data)
        
        # Sort by cluster
        quickflow_df['Cluster_num'] = quickflow_df['Cluster'].str.extract(r'(\d+)').astype(int)
        quickflow_df = quickflow_df.sort_values('Cluster_num')
        quickflow_df = quickflow_df.drop('Cluster_num', axis=1)
        
        # Plot
        quickflow_plot = sns.barplot(x='Cluster', y='Percentage', hue='Direction', 
                                    data=quickflow_df, ax=axes[2], palette=colors)
        axes[2].set_title('Quickflow Trend Directions by Cluster', fontsize=14)
        axes[2].set_xlabel('Cluster', fontsize=12)
        axes[2].set_ylabel('Percentage of Stations (%)', fontsize=12)
        axes[2].set_ylim(0, 100)
        
        # Add data labels
        for p in quickflow_plot.patches:
            if p.get_height() > 0:
                quickflow_plot.annotate(f'{p.get_height():.1f}%', 
                                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    trend_bar_path = os.path.join(output_dir, 'cluster_trend_directions.png')
    plt.savefig(trend_bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    results['visualizations']['trend_directions'] = os.path.basename(trend_bar_path)
    
    # 2. Boxplot of trend percentages by cluster
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Streamflow trends boxplot
    streamflow_trends = []
    for cluster in valid_clusters:
        cluster_df = df[df['cluster'] == cluster]
        for trend in cluster_df['streamflow_trend']:
            streamflow_trends.append({
                'Cluster': f'Cluster {cluster}',
                'Trend': trend
            })
    
    if streamflow_trends:
        streamflow_trend_df = pd.DataFrame(streamflow_trends)
        
        # Sort by cluster
        streamflow_trend_df['Cluster_num'] = streamflow_trend_df['Cluster'].str.extract(r'(\d+)').astype(int)
        streamflow_trend_df = streamflow_trend_df.sort_values('Cluster_num')
        streamflow_trend_df = streamflow_trend_df.drop('Cluster_num', axis=1)
        
        # Plot
        sns.boxplot(x='Cluster', y='Trend', data=streamflow_trend_df, ax=axes[0])
        axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.7)
        axes[0].set_title('Streamflow Trend Percentages by Cluster', fontsize=14)
        axes[0].set_ylabel('Trend (%)', fontsize=12)
    
    # Baseflow trends boxplot
    baseflow_trends = []
    for cluster in valid_clusters:
        cluster_df = df[df['cluster'] == cluster]
        for trend in cluster_df['baseflow_trend']:
            baseflow_trends.append({
                'Cluster': f'Cluster {cluster}',
                'Trend': trend
            })
    
    if baseflow_trends:
        baseflow_trend_df = pd.DataFrame(baseflow_trends)
        
        # Sort by cluster
        baseflow_trend_df['Cluster_num'] = baseflow_trend_df['Cluster'].str.extract(r'(\d+)').astype(int)
        baseflow_trend_df = baseflow_trend_df.sort_values('Cluster_num')
        baseflow_trend_df = baseflow_trend_df.drop('Cluster_num', axis=1)
        
        # Plot
        sns.boxplot(x='Cluster', y='Trend', data=baseflow_trend_df, ax=axes[1])
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.7)
        axes[1].set_title('Baseflow Trend Percentages by Cluster', fontsize=14)
        axes[1].set_ylabel('Trend (%)', fontsize=12)
    
    # Quickflow trends boxplot
    quickflow_trends = []
    for cluster in valid_clusters:
        cluster_df = df[df['cluster'] == cluster]
        for trend in cluster_df['quickflow_trend']:
            quickflow_trends.append({
                'Cluster': f'Cluster {cluster}',
                'Trend': trend
            })
    
    if quickflow_trends:
        quickflow_trend_df = pd.DataFrame(quickflow_trends)
        
        # Sort by cluster
        quickflow_trend_df['Cluster_num'] = quickflow_trend_df['Cluster'].str.extract(r'(\d+)').astype(int)
        quickflow_trend_df = quickflow_trend_df.sort_values('Cluster_num')
        quickflow_trend_df = quickflow_trend_df.drop('Cluster_num', axis=1)
        
        # Plot
        sns.boxplot(x='Cluster', y='Trend', data=quickflow_trend_df, ax=axes[2])
        axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.7)
        axes[2].set_title('Quickflow Trend Percentages by Cluster', fontsize=14)
        axes[2].set_xlabel('Cluster', fontsize=12)
        axes[2].set_ylabel('Trend (%)', fontsize=12)
    
    plt.tight_layout()
    
    # Save figure
    trend_box_path = os.path.join(output_dir, 'cluster_trend_percentages.png')
    plt.savefig(trend_box_path, dpi=300, bbox_inches='tight')
    plt.close()
    results['visualizations']['trend_percentages'] = os.path.basename(trend_box_path)
    
    # 3. Heatmap of mean trend percentages by cluster
    trend_heatmap_data = []
    
    for cluster in valid_clusters:
        trend_heatmap_data.append({
            'Cluster': f'Cluster {cluster}',
            'Streamflow': cluster_trend_summary[cluster]['streamflow']['mean_trend'],
            'Baseflow': cluster_trend_summary[cluster]['baseflow']['mean_trend'],
            'Quickflow': cluster_trend_summary[cluster]['quickflow']['mean_trend']
        })
    
    if trend_heatmap_data:
        trend_heatmap_df = pd.DataFrame(trend_heatmap_data)
        
        # Sort by cluster
        trend_heatmap_df['Cluster_num'] = trend_heatmap_df['Cluster'].str.extract(r'(\d+)').astype(int)
        trend_heatmap_df = trend_heatmap_df.sort_values('Cluster_num')
        trend_heatmap_df = trend_heatmap_df.drop('Cluster_num', axis=1)
        
        # Set Cluster as index for the heatmap
        trend_heatmap_df = trend_heatmap_df.set_index('Cluster')
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Create custom colormap centered at zero
        colors = ["#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"]
        cmap = LinearSegmentedColormap.from_list("trend_cmap", colors)
        
        # Determine bounds for the colormap
        vmax = max(abs(trend_heatmap_df.values.min()), abs(trend_heatmap_df.values.max()))
        
        # Create heatmap
        sns.heatmap(trend_heatmap_df, annot=True, cmap=cmap, center=0, vmin=-vmax, vmax=vmax,
                   fmt='.2f', cbar_kws={'label': 'Mean Trend (%)'})
        
        plt.title('Mean Flow Trend Percentages by Cluster', fontsize=14)
        plt.tight_layout()
        
        # Save figure
        trend_heatmap_path = os.path.join(output_dir, 'cluster_trend_heatmap.png')
        plt.savefig(trend_heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        results['visualizations']['trend_heatmap'] = os.path.basename(trend_heatmap_path)
    
    return results

def analyze_seasonal_patterns(stations_results: Dict, 
                             station_clusters: Dict, 
                             output_dir: str) -> Dict:
    """
    Analyze seasonal patterns by cluster.
    
    Parameters:
    -----------
    stations_results : Dict
        Dictionary with results for each station
    station_clusters : Dict
        Dictionary mapping station names to cluster IDs
    output_dir : str
        Directory to save output files
        
    Returns:
    --------
    Dict
        Dictionary with seasonal analysis results
    """
    print("Analyzing seasonal patterns by cluster...")
    
    results = {
        'visualizations': {},
        'tables': {},
        'summaries': {}
    }
    
    # Group stations by cluster
    cluster_stations = {}
    for station_name, cluster in station_clusters.items():
        if cluster not in cluster_stations:
            cluster_stations[cluster] = []
        cluster_stations[cluster].append(station_name)
    
    # Only consider valid clusters (not -1)
    valid_clusters = sorted([c for c in cluster_stations.keys() if c >= 0])
    
    if not valid_clusters:
        print("No valid clusters found for seasonal analysis")
        return results
    
    # Analyze seasonal patterns for each cluster
    cluster_seasonal_data = {}
    
    for cluster in valid_clusters:
        # Get stations in this cluster
        stations = cluster_stations.get(cluster, [])
        
        if not stations:
            continue
        
        # Seasonal data for this cluster
        cluster_data = {
            'winter': {'streamflow': [], 'baseflow': [], 'quickflow': []},
            'spring': {'streamflow': [], 'baseflow': [], 'quickflow': []},
            'summer': {'streamflow': [], 'baseflow': [], 'quickflow': []},
            'fall': {'streamflow': [], 'baseflow': [], 'quickflow': []}
        }
        
        # Process each station
        for station_name in stations:
            # Get station results
            station_result = stations_results.get(station_name, {})
            
            # Get seasonal averages
            seasonal_avg = station_result.get('seasonal_averages')
            
            if not isinstance(seasonal_avg, pd.DataFrame):
                continue
                
            # Process each season
            for _, row in seasonal_avg.iterrows():
                season = row.get('season')
                
                if season not in cluster_data:
                    continue
                
                # Add flow values to the appropriate lists
                streamflow = row.get('streamflow')
                baseflow = row.get('baseflow')
                quickflow = row.get('quickflow')
                
                if not np.isnan(streamflow):
                    cluster_data[season]['streamflow'].append(streamflow)
                if not np.isnan(baseflow):
                    cluster_data[season]['baseflow'].append(baseflow)
                if not np.isnan(quickflow):
                    cluster_data[season]['quickflow'].append(quickflow)
        
        # Calculate summary statistics for each season and flow component
        cluster_seasonal_stats = {}
        
        for season, components in cluster_data.items():
            season_stats = {}
            
            for component, values in components.items():
                if not values:
                    # No data for this component
                    season_stats[component] = {
                        'mean': np.nan,
                        'median': np.nan,
                        'std': np.nan,
                        'min': np.nan,
                        'max': np.nan,
                        'count': 0
                    }
                else:
                    # Calculate statistics
                    season_stats[component] = {
                        'mean': np.mean(values),
                        'median': np.median(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }
            
            cluster_seasonal_stats[season] = season_stats
        
        # Store in cluster data
        cluster_seasonal_data[cluster] = cluster_seasonal_stats
    
    # Store summaries
    results['summaries'] = cluster_seasonal_data
    
    # Create visualizations
    
    # 1. Seasonal patterns by cluster (bar chart)
    # Create figure with one row per flow component
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Define seasons in order
    seasons = ['winter', 'spring', 'summer', 'fall']
    season_colors = ['#d1e5f0', '#92c5de', '#2166ac', '#67a9cf']  # Winter to fall colors
    
    # Set consistent y-axis limits across clusters
    max_streamflow = 0
    max_baseflow = 0
    max_quickflow = 0
    
    for cluster, seasonal_stats in cluster_seasonal_data.items():
        for season, stats in seasonal_stats.items():
            if not np.isnan(stats['streamflow']['mean']):
                max_streamflow = max(max_streamflow, stats['streamflow']['mean'] + stats['streamflow']['std'])
            if not np.isnan(stats['baseflow']['mean']):
                max_baseflow = max(max_baseflow, stats['baseflow']['mean'] + stats['baseflow']['std'])
            if not np.isnan(stats['quickflow']['mean']):
                max_quickflow = max(max_quickflow, stats['quickflow']['mean'] + stats['quickflow']['std'])
    
    # Add some margin to the max values
    max_streamflow *= 1.1
    max_baseflow *= 1.1
    max_quickflow *= 1.1
    
    # Create plot data
    streamflow_data = []
    baseflow_data = []
    quickflow_data = []
    
    for cluster in valid_clusters:
        seasonal_stats = cluster_seasonal_data.get(cluster, {})
        
        for season in seasons:
            stats = seasonal_stats.get(season, {})
            
            # Streamflow
            streamflow_stats = stats.get('streamflow', {})
            streamflow_data.append({
                'Cluster': f'Cluster {cluster}',
                'Season': season.capitalize(),
                'Mean': streamflow_stats.get('mean', np.nan),
                'Std': streamflow_stats.get('std', np.nan),
                'Count': streamflow_stats.get('count', 0)
            })
            
            # Baseflow
            baseflow_stats = stats.get('baseflow', {})
            baseflow_data.append({
                'Cluster': f'Cluster {cluster}',
                'Season': season.capitalize(),
                'Mean': baseflow_stats.get('mean', np.nan),
                'Std': baseflow_stats.get('std', np.nan),
                'Count': baseflow_stats.get('count', 0)
            })
            
            # Quickflow
            quickflow_stats = stats.get('quickflow', {})
            quickflow_data.append({
                'Cluster': f'Cluster {cluster}',
                'Season': season.capitalize(),
                'Mean': quickflow_stats.get('mean', np.nan),
                'Std': quickflow_stats.get('std', np.nan),
                'Count': quickflow_stats.get('count', 0)
            })
    
    # Convert to DataFrames
    streamflow_df = pd.DataFrame(streamflow_data)
    baseflow_df = pd.DataFrame(baseflow_data)
    quickflow_df = pd.DataFrame(quickflow_data)
    
    # Sort by cluster
    for df_name in ['streamflow_df', 'baseflow_df', 'quickflow_df']:
        df = locals()[df_name]
        if not df.empty:
            df['Cluster_num'] = df['Cluster'].str.extract(r'(\d+)').astype(int)
            locals()[df_name] = df.sort_values(['Cluster_num', 'Season'])
    
    # Plot streamflow seasonal patterns
    if not streamflow_df.empty:
        bars = sns.barplot(x='Cluster', y='Mean', hue='Season', data=streamflow_df, 
                          ax=axes[0], palette=season_colors)
        
        # Add error bars - FIXING THE INDEXING ISSUE
        for i, bar in enumerate(bars.patches):
            if i < len(streamflow_df):  # Make sure index is valid
                row = streamflow_df.iloc[i]
                if not np.isnan(row['Mean']) and not np.isnan(row['Std']):
                    bars.errorbar(bar.get_x() + bar.get_width()/2, row['Mean'], 
                                 yerr=row['Std'], color='black', capsize=3)
        
        axes[0].set_title('Seasonal Streamflow Patterns by Cluster', fontsize=14)
        axes[0].set_ylabel('Average Streamflow (cfs)', fontsize=12)
        axes[0].set_ylim(0, max_streamflow)
    
    # Plot baseflow seasonal patterns
    if not baseflow_df.empty:
        bars = sns.barplot(x='Cluster', y='Mean', hue='Season', data=baseflow_df, 
                          ax=axes[1], palette=season_colors)
        
        # Add error bars - FIXING THE INDEXING ISSUE
        for i, bar in enumerate(bars.patches):
            if i < len(baseflow_df):  # Make sure index is valid
                row = baseflow_df.iloc[i]
                if not np.isnan(row['Mean']) and not np.isnan(row['Std']):
                    bars.errorbar(bar.get_x() + bar.get_width()/2, row['Mean'], 
                                 yerr=row['Std'], color='black', capsize=3)
        
        axes[1].set_title('Seasonal Baseflow Patterns by Cluster', fontsize=14)
        axes[1].set_ylabel('Average Baseflow (cfs)', fontsize=12)
        axes[1].set_ylim(0, max_baseflow)
    
    # Plot quickflow seasonal patterns
    if not quickflow_df.empty:
        bars = sns.barplot(x='Cluster', y='Mean', hue='Season', data=quickflow_df, 
                          ax=axes[2], palette=season_colors)
        
        # Add error bars - FIXING THE INDEXING ISSUE
        for i, bar in enumerate(bars.patches):
            if i < len(quickflow_df):  # Make sure index is valid
                row = quickflow_df.iloc[i]
                if not np.isnan(row['Mean']) and not np.isnan(row['Std']):
                    bars.errorbar(bar.get_x() + bar.get_width()/2, row['Mean'], 
                                 yerr=row['Std'], color='black', capsize=3)
        
        axes[2].set_title('Seasonal Quickflow Patterns by Cluster', fontsize=14)
        axes[2].set_xlabel('Cluster', fontsize=12)
        axes[2].set_ylabel('Average Quickflow (cfs)', fontsize=12)
        axes[2].set_ylim(0, max_quickflow)
    
    plt.tight_layout()
    
    # Save figure
    seasonal_bar_path = os.path.join(output_dir, 'cluster_seasonal_patterns.png')
    plt.savefig(seasonal_bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    results['visualizations']['seasonal_patterns'] = os.path.basename(seasonal_bar_path)
    
    # 2. Seasonal flow proportions by cluster (stacked bar chart)
    seasonal_prop_data = []
    
    for cluster in valid_clusters:
        seasonal_stats = cluster_seasonal_data.get(cluster, {})
        
        for season in seasons:
            stats = seasonal_stats.get(season, {})
            
            # Get flow components
            streamflow_mean = stats.get('streamflow', {}).get('mean', np.nan)
            baseflow_mean = stats.get('baseflow', {}).get('mean', np.nan)
            quickflow_mean = stats.get('quickflow', {}).get('mean', np.nan)
            
            # Calculate proportions
            if not np.isnan(streamflow_mean) and streamflow_mean > 0:
                baseflow_prop = (baseflow_mean / streamflow_mean) * 100 if not np.isnan(baseflow_mean) else np.nan
                quickflow_prop = (quickflow_mean / streamflow_mean) * 100 if not np.isnan(quickflow_mean) else np.nan
            else:
                baseflow_prop = np.nan
                quickflow_prop = np.nan
            
            # Add to data
            seasonal_prop_data.append({
                'Cluster': f'Cluster {cluster}',
                'Season': season.capitalize(),
                'Baseflow (%)': baseflow_prop,
                'Quickflow (%)': quickflow_prop
            })
    
    # Convert to DataFrame
    seasonal_prop_df = pd.DataFrame(seasonal_prop_data)
    
    # Sort by cluster and season
    if not seasonal_prop_df.empty:
        seasonal_prop_df['Cluster_num'] = seasonal_prop_df['Cluster'].str.extract(r'(\d+)').astype(int)
        seasonal_prop_df['Season_order'] = pd.Categorical(seasonal_prop_df['Season'], 
                                                          categories=['Winter', 'Spring', 'Summer', 'Fall'], 
                                                          ordered=True)
        seasonal_prop_df = seasonal_prop_df.sort_values(['Cluster_num', 'Season_order'])
        
        # Melt for stacked bar chart
        melted_df = pd.melt(seasonal_prop_df, 
                           id_vars=['Cluster', 'Season'], 
                           value_vars=['Baseflow (%)', 'Quickflow (%)'],
                           var_name='Component', value_name='Percentage')
        
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # Create stacked bar chart
        sns.barplot(x='Cluster', y='Percentage', hue='Component', data=melted_df, 
                   palette=['green', 'skyblue'])
        
        plt.title('Flow Component Proportions by Season and Cluster', fontsize=14)
        plt.xlabel('Cluster', fontsize=12)
        plt.ylabel('Percentage of Total Streamflow (%)', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Flow Component', loc='upper right')
        
        # Add season labels above each group of bars
        bar_width = 0.4  # Approximate width of a bar
        unique_clusters = seasonal_prop_df['Cluster'].unique()
        
        for i, cluster in enumerate(unique_clusters):
            cluster_df = seasonal_prop_df[seasonal_prop_df['Cluster'] == cluster]
            seasons_in_cluster = cluster_df['Season'].unique()
            
            for j, season in enumerate(seasons_in_cluster):
                x_pos = i + (j / len(seasons_in_cluster)) - 0.4
                plt.text(x_pos, 105, season, fontsize=8, ha='center', rotation=90)
        
        plt.tight_layout()
        
        # Save figure
        seasonal_prop_path = os.path.join(output_dir, 'cluster_seasonal_proportions.png')
        plt.savefig(seasonal_prop_path, dpi=300, bbox_inches='tight')
        plt.close()
        results['visualizations']['seasonal_proportions'] = os.path.basename(seasonal_prop_path)
    
    # 3. Create seasonal summary table
    seasonal_table_rows = []
    
    for cluster in valid_clusters:
        seasonal_stats = cluster_seasonal_data.get(cluster, {})
        
        for season in seasons:
            stats = seasonal_stats.get(season, {})
            
            # Get flow components stats
            streamflow_stats = stats.get('streamflow', {})
            baseflow_stats = stats.get('baseflow', {})
            quickflow_stats = stats.get('quickflow', {})
            
            # Add to table
            seasonal_table_rows.append({
                'Cluster': f'Cluster {cluster}',
                'Season': season.capitalize(),
                'Streamflow Mean (cfs)': streamflow_stats.get('mean', np.nan),
                'Streamflow Median (cfs)': streamflow_stats.get('median', np.nan),
                'Streamflow Std (cfs)': streamflow_stats.get('std', np.nan),
                'Baseflow Mean (cfs)': baseflow_stats.get('mean', np.nan),
                'Baseflow Median (cfs)': baseflow_stats.get('median', np.nan),
                'Baseflow Std (cfs)': baseflow_stats.get('std', np.nan),
                'Quickflow Mean (cfs)': quickflow_stats.get('mean', np.nan),
                'Quickflow Median (cfs)': quickflow_stats.get('median', np.nan),
                'Quickflow Std (cfs)': quickflow_stats.get('std', np.nan),
                'Sample Size': streamflow_stats.get('count', 0)
            })
    
    # Create and save table
    seasonal_table_df = pd.DataFrame(seasonal_table_rows)
    
    # Sort by cluster and season
    if not seasonal_table_df.empty:
        seasonal_table_df['Cluster_num'] = seasonal_table_df['Cluster'].str.extract(r'(\d+)').astype(int)
        seasonal_table_df['Season_order'] = pd.Categorical(seasonal_table_df['Season'], 
                                                          categories=['Winter', 'Spring', 'Summer', 'Fall'], 
                                                          ordered=True)
        seasonal_table_df = seasonal_table_df.sort_values(['Cluster_num', 'Season_order'])
        seasonal_table_df = seasonal_table_df.drop(['Cluster_num', 'Season_order'], axis=1)
        
        # Save table
        seasonal_table_path = os.path.join(output_dir, 'cluster_seasonal_summary.html')
        seasonal_table_df.to_html(seasonal_table_path, index=False, float_format='%.2f')
        results['tables']['seasonal_summary'] = os.path.basename(seasonal_table_path)
    
    return results

def analyze_baseflow_index(df: pd.DataFrame, output_dir: str) -> Dict:
    """
    Analyze baseflow index patterns by cluster.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with streamflow analysis results
    output_dir : str
        Directory to save output files
        
    Returns:
    --------
    Dict
        Dictionary with baseflow index analysis results
    """
    print("Analyzing baseflow index patterns by cluster...")
    
    results = {
        'visualizations': {},
        'tables': {},
        'summaries': {}
    }
    
    # Only consider valid clusters (not -1)
    valid_clusters = sorted([c for c in df['cluster'].unique() if c >= 0])
    
    if not valid_clusters:
        print("No valid clusters found for BFI analysis")
        return results
    
    # Calculate summary statistics for baseflow index by cluster
    bfi_summary = {}
    
    for cluster in valid_clusters:
        cluster_df = df[df['cluster'] == cluster]
        
        # Calculate BFI statistics
        bfi_values = cluster_df['baseflow_index'].dropna().values
        
        if len(bfi_values) > 0:
            bfi_summary[cluster] = {
                'mean': np.mean(bfi_values),
                'median': np.median(bfi_values),
                'std': np.std(bfi_values),
                'min': np.min(bfi_values),
                'max': np.max(bfi_values),
                'q25': np.percentile(bfi_values, 25),
                'q75': np.percentile(bfi_values, 75),
                'count': len(bfi_values)
            }
        else:
            bfi_summary[cluster] = {
                'mean': np.nan,
                'median': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'q25': np.nan,
                'q75': np.nan,
                'count': 0
            }
    
    # Store summary
    results['summaries'] = bfi_summary
    
    # Create BFI summary table
    bfi_table_rows = []
    
    for cluster, stats in bfi_summary.items():
        bfi_table_rows.append({
            'Cluster': f'Cluster {cluster}',
            'Mean BFI': stats['mean'],
            'Median BFI': stats['median'],
            'Std Dev': stats['std'],
            'Min BFI': stats['min'],
            'Max BFI': stats['max'],
            '25th Percentile': stats['q25'],
            '75th Percentile': stats['q75'],
            'Sample Size': stats['count']
        })
    
    # Create and save table
    bfi_table_df = pd.DataFrame(bfi_table_rows)
    
    # Sort by cluster
    if not bfi_table_df.empty:
        bfi_table_df['Cluster_num'] = bfi_table_df['Cluster'].str.extract(r'(\d+)').astype(int)
        bfi_table_df = bfi_table_df.sort_values('Cluster_num')
        bfi_table_df = bfi_table_df.drop('Cluster_num', axis=1)
        
        # Save table
        bfi_table_path = os.path.join(output_dir, 'cluster_bfi_summary.html')
        bfi_table_df.to_html(bfi_table_path, index=False, float_format='%.3f')
        results['tables']['bfi_summary'] = os.path.basename(bfi_table_path)
    
    # Create BFI visualizations
    
    # 1. BFI boxplot by cluster
    bfi_data = []
    
    for cluster in valid_clusters:
        cluster_df = df[df['cluster'] == cluster]
        
        for bfi in cluster_df['baseflow_index'].dropna():
            bfi_data.append({
                'Cluster': f'Cluster {cluster}',
                'Baseflow Index': bfi
            })
    
    if bfi_data:
        bfi_df = pd.DataFrame(bfi_data)
        
        # Sort by cluster
        bfi_df['Cluster_num'] = bfi_df['Cluster'].str.extract(r'(\d+)').astype(int)
        bfi_df = bfi_df.sort_values('Cluster_num')
        bfi_df = bfi_df.drop('Cluster_num', axis=1)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create boxplot
        ax = sns.boxplot(x='Cluster', y='Baseflow Index', data=bfi_df)
        
        # Add individual points
        sns.stripplot(x='Cluster', y='Baseflow Index', data=bfi_df, 
                     size=4, color='black', alpha=0.4)
        
        # Add mean markers
        for i, cluster in enumerate(bfi_df['Cluster'].unique()):
            if cluster in [f'Cluster {c}' for c in bfi_summary]:
                c = int(cluster.replace('Cluster ', ''))
                mean_bfi = bfi_summary[c]['mean']
                ax.plot(i, mean_bfi, marker='*', markersize=10, color='red')
        
        plt.title('Baseflow Index Distribution by Cluster', fontsize=14)
        plt.xlabel('Cluster', fontsize=12)
        plt.ylabel('Baseflow Index', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add reference lines for BFI interpretation
        plt.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, 
                   label='BFI=0.3 (low baseflow contribution)')
        plt.axhline(y=0.6, color='green', linestyle='--', alpha=0.5, 
                   label='BFI=0.6 (high baseflow contribution)')
        
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        bfi_box_path = os.path.join(output_dir, 'cluster_bfi_boxplot.png')
        plt.savefig(bfi_box_path, dpi=300, bbox_inches='tight')
        plt.close()
        results['visualizations']['bfi_boxplot'] = os.path.basename(bfi_box_path)
    
    # 2. BFI vs. Streamflow Mean scatter plot by cluster
    scatter_data = []
    
    for _, row in df.iterrows():
        if row['cluster'] >= 0 and not np.isnan(row['baseflow_index']) and not np.isnan(row['streamflow_mean']):
            scatter_data.append({
                'Cluster': f'Cluster {int(row["cluster"])}',
                'Baseflow Index': row['baseflow_index'],
                'Mean Streamflow (cfs)': row['streamflow_mean']
            })
    
    if scatter_data:
        scatter_df = pd.DataFrame(scatter_data)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot
        sns.scatterplot(x='Mean Streamflow (cfs)', y='Baseflow Index', 
                       hue='Cluster', data=scatter_df, s=80)
        
        plt.title('Baseflow Index vs. Mean Streamflow by Cluster', fontsize=14)
        plt.xlabel('Mean Streamflow (cubic feet per second)', fontsize=12)
        plt.ylabel('Baseflow Index', fontsize=12)
        plt.grid(linestyle='--', alpha=0.7)
        
        # Add reference lines for BFI interpretation
        plt.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, 
                   label='BFI=0.3 (low baseflow contribution)')
        plt.axhline(y=0.6, color='green', linestyle='--', alpha=0.5, 
                   label='BFI=0.6 (high baseflow contribution)')
        
        # Log scale for streamflow
        plt.xscale('log')
        
        # Add annotations for highest and lowest values in each cluster
        for cluster in scatter_df['Cluster'].unique():
            cluster_data = scatter_df[scatter_df['Cluster'] == cluster]
            
            # Highest BFI
            max_bfi_row = cluster_data.loc[cluster_data['Baseflow Index'].idxmax()]
            plt.annotate(f"{max_bfi_row['Baseflow Index']:.2f}", 
                        (max_bfi_row['Mean Streamflow (cfs)'], max_bfi_row['Baseflow Index']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8)
            
            # Lowest BFI
            min_bfi_row = cluster_data.loc[cluster_data['Baseflow Index'].idxmin()]
            plt.annotate(f"{min_bfi_row['Baseflow Index']:.2f}", 
                        (min_bfi_row['Mean Streamflow (cfs)'], min_bfi_row['Baseflow Index']),
                        xytext=(5, -10), textcoords='offset points',
                        fontsize=8)
        
        plt.tight_layout()
        
        # Save figure
        bfi_scatter_path = os.path.join(output_dir, 'cluster_bfi_vs_streamflow.png')
        plt.savefig(bfi_scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        results['visualizations']['bfi_scatter'] = os.path.basename(bfi_scatter_path)
    
    return results

def generate_cluster_report(df: pd.DataFrame, results: Dict, output_dir: str) -> str:
    """
    Generate an HTML report comparing streamflow characteristics across clusters.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with streamflow analysis results
    results : Dict
        Dictionary with analysis results
    output_dir : str
        Directory to save the report
        
    Returns:
    --------
    str
        Path to the generated HTML report
    """
    print("Generating cluster comparison report...")
    
    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Streamflow Cluster Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .section {{ margin-top: 30px; margin-bottom: 30px; }}
            .figure {{ text-align: center; margin: 20px 0; }}
            .figure img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ padding: 8px; text-align: left; border: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .note {{ font-style: italic; color: #666; }}
            .highlight {{ background-color: #ffffcc; padding: 10px; border-left: 4px solid #ffeb3b; }}
        </style>
    </head>
    <body>
        <h1>Streamflow Cluster Analysis Report</h1>
        <p>This report analyzes streamflow characteristics across watershed clusters, highlighting regional patterns and trends.</p>
        
        <div class="highlight">
            <p><strong>Note:</strong> All streamflow measurements are in cubic feet per second (cfs).</p>
        </div>
        
        <div class="section">
            <h2>1. Trend Analysis by Cluster</h2>
            <p>This section examines the trends in streamflow, baseflow, and quickflow across different watershed clusters.</p>
            
            <div class="figure">
                <h3>Trend Directions by Cluster</h3>
                <img src="figs/streamflow_clusters/{results['visualizations'].get('trend_directions', '')}" alt="Trend Directions by Cluster">
                <p>The chart shows the percentage of stations in each cluster exhibiting increasing, decreasing, or no significant change in streamflow components.</p>
            </div>
            
            <div class="figure">
                <h3>Trend Magnitudes by Cluster</h3>
                <img src="figs/streamflow_clusters/{results['visualizations'].get('trend_percentages', '')}" alt="Trend Percentages by Cluster">
                <p>Box plots showing the distribution of trend percentages for each flow component across clusters.</p>
            </div>
            
            <div class="figure">
                <h3>Mean Trend Comparison</h3>
                <img src="figs/streamflow_clusters/{results['visualizations'].get('trend_heatmap', '')}" alt="Trend Heatmap by Cluster">
                <p>Heatmap comparing mean trend percentages for each flow component across clusters. Negative values (red) indicate decreasing trends, while positive values (green) indicate increasing trends.</p>
            </div>
        </div>
        
        <div class="section">
            <h2>2. Seasonal Flow Patterns by Cluster</h2>
            <p>This section examines seasonal variations in streamflow components across different watershed clusters.</p>
            
            <div class="figure">
                <h3>Seasonal Flow Patterns</h3>
                <img src="figs/streamflow_clusters/{results['visualizations'].get('seasonal_patterns', '')}" alt="Seasonal Patterns by Cluster">
                <p>Bar charts showing average streamflow, baseflow, and quickflow for each season across clusters.</p>
            </div>
            
            <div class="figure">
                <h3>Flow Component Proportions</h3>
                <img src="figs/streamflow_clusters/{results['visualizations'].get('seasonal_proportions', '')}" alt="Seasonal Proportions by Cluster">
                <p>Stacked bar chart showing the relative proportion of baseflow and quickflow components by season for each cluster.</p>
            </div>
        </div>
        
        <div class="section">
            <h2>3. Baseflow Index Analysis by Cluster</h2>
            <p>This section examines the baseflow index (BFI) patterns across different watershed clusters.</p>
            
            <div class="figure">
                <h3>Baseflow Index Distribution</h3>
                <img src="figs/streamflow_clusters/{results['visualizations'].get('bfi_boxplot', '')}" alt="BFI Boxplot by Cluster">
                <p>Box plot showing the distribution of baseflow index values across clusters. The baseflow index represents the proportion of streamflow that comes from groundwater sources. Higher values indicate greater groundwater contribution.</p>
                <p class="note">Reference lines: BFI = 0.3 (low baseflow contribution), BFI = 0.6 (high baseflow contribution)</p>
            </div>
            
            <div class="figure">
                <h3>Baseflow Index vs. Mean Streamflow</h3>
                <img src="figs/streamflow_clusters/{results['visualizations'].get('bfi_scatter', '')}" alt="BFI vs Streamflow by Cluster">
                <p>Scatter plot showing the relationship between baseflow index and mean streamflow for each station, colored by cluster.</p>
            </div>
        </div>
        
        <div class="section">
            <h2>4. Summary Tables</h2>
            
            <h3>Trend Summary by Cluster</h3>
            <iframe src="{results['tables'].get('trend_summary', '')}" width="100%" height="300px"></iframe>
            
            <h3>Seasonal Flow Summary by Cluster</h3>
            <iframe src="{results['tables'].get('seasonal_summary', '')}" width="100%" height="300px"></iframe>
            
            <h3>Baseflow Index Summary by Cluster</h3>
            <iframe src="{results['tables'].get('bfi_summary', '')}" width="100%" height="300px"></iframe>
        </div>
        
        <div class="section">
            <h2>5. Conclusions</h2>
            <p>Based on the analysis of streamflow patterns across watershed clusters, the following conclusions can be drawn:</p>
            <ul>
                <li><strong>Regional Trends:</strong> Clusters show distinct patterns in streamflow trends, suggesting regional differences in hydrological response.</li>
                <li><strong>Seasonal Patterns:</strong> The seasonal distribution of streamflow components varies by cluster, reflecting differences in watershed characteristics and climate response.</li>
                <li><strong>Baseflow Contribution:</strong> Variations in baseflow index across clusters indicate differences in groundwater contribution to streamflow, which may be related to geological and soil characteristics.</li>
            </ul>
        </div>
        
        <div class="note">
            <p>Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d')}</p>
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    report_path = os.path.join(output_dir, 'streamflow_cluster_analysis_report.html')
    
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"Report saved to {report_path}")
    return os.path.basename(report_path)

# If run as a script
if __name__ == "__main__":
    import sys
    
    # Add parent directory to path for imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    try:
        from Michigan.utils.streamflow_analyzer import StreamflowAnalyzer
        
        # Example usage
        print("Running streamflow cluster analysis example...")
        
        # Path to streamflow data
        base_path = "/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12"
        
        # Create analyzer and run analysis
        analyzer = StreamflowAnalyzer(base_path, start_year=2000, end_year=2005)
        results = analyzer.run_full_analysis()
        
        if results:
            # Run cluster analysis
            cluster_results = analyze_streamflow_clusters(results)
            
            if cluster_results:
                print("Cluster analysis completed successfully")
                print(f"Report: {cluster_results['report']}")
                print(f"Visualizations: {len(cluster_results['visualizations'])}")
                print(f"Tables: {len(cluster_results['tables'])}")
            else:
                print("Cluster analysis failed")
        else:
            print("Streamflow analysis failed")
    except Exception as e:
        print(f"Error running example: {e}")