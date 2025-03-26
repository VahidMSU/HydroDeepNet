import os
import glob
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def create_report(output_path='./Michigan/report.html', start_year=2000, end_year=2005, 
                 var="perc", precip_threshold=10, include_statistics=False, statistics_results=None):
    """
    Generate an HTML report with analysis results and visualizations.
    
    Parameters:
    -----------
    output_path : str
        Path for the output HTML report
    start_year, end_year : int
        Start and end year used in the analysis
    var : str
        Variable analyzed (e.g., "perc" for percolation)
    precip_threshold : int
        Threshold used for precipitation in mm
    include_statistics : bool
        Whether to include detailed statistics in the report
    statistics_results : dict
        Dictionary with statistics results
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Find all generated images - update to new filename pattern
    cluster_images = glob.glob(f'./Michigan/figs/recharge_water_input_ratio_{start_year}_{end_year-1}_cluster*.png')
    all_image = glob.glob(f'./Michigan/figs/recharge_water_input_ratio_{start_year}_{end_year-1}_all.png')
    
    # If new pattern doesn't find files, try the old pattern as fallback
    if not cluster_images:
        cluster_images = glob.glob(f'./Michigan/figs/recharge_percolation_ratio_{start_year}_{end_year-1}_cluster*.png')
    if not all_image:
        all_image = glob.glob(f'./Michigan/figs/recharge_percolation_ratio_{start_year}_{end_year-1}_all.png')
    
    cluster_map = glob.glob('./Michigan/figs/clusters.png')
    michigan_map = glob.glob('./Michigan/figs/michigan_clusters_map.png')
    
    # Find statistics plots if available
    stats_plots = {}
    if include_statistics:
        stats_plots['boxplot'] = glob.glob('./Michigan/figs/annual_variability_boxplot.png')
        stats_plots['seasonal'] = glob.glob('./Michigan/figs/seasonal_comparison.png')
        stats_plots['annual'] = glob.glob('./Michigan/figs/annual_trends.png')
        stats_plots['cluster_comparison'] = glob.glob('./Michigan/figs/cluster_seasonal_comparison.png')
        stats_plots['heatmap'] = glob.glob('./Michigan/figs/cluster_season_heatmap.png')
    
    # Convert image paths to relative paths (keep "figs/" in path)
    cluster_images = [os.path.join("figs", os.path.basename(img)) for img in cluster_images]
    all_image = [os.path.join("figs", os.path.basename(img)) for img in all_image]
    cluster_map = [os.path.join("figs", os.path.basename(img)) for img in cluster_map]
    michigan_map = [os.path.join("figs", os.path.basename(img)) for img in michigan_map]
    
    # Convert statistics plots paths as well
    if include_statistics:
        for key in stats_plots:
            stats_plots[key] = [os.path.join("figs", os.path.basename(img)) for img in stats_plots[key]]
    
    # Extract basic stats from images if possible
    cluster_stats = {}
    try:
        # For each cluster, try to extract some metrics from the images
        for i in range(5):
            img_path = f'./Michigan/figs/recharge_percolation_ratio_{start_year}_{end_year-1}_cluster{i}.png'
            if os.path.exists(img_path):
                img = Image.open(img_path)
                width, height = img.size
                cluster_stats[i] = {
                    'width': width,
                    'height': height,
                    'size_kb': os.path.getsize(img_path) / 1024
                }
    except Exception as e:
        print(f"Error extracting image stats: {e}")
    
    # Generate HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Watershed Cluster Analysis Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .header {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            border-left: 5px solid #3498db;
        }}
        .section {{
            margin-bottom: 30px;
            padding: 20px;
            background-color: #fff;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .cluster-section {{
            margin-bottom: 40px;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 8px;
            border-left: 4px solid #2980b9;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.2);
            margin: 10px 0;
        }}
        .img-container {{
            text-align: center;
            margin: 20px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .timestamp {{
            font-style: italic;
            color: #7f8c8d;
            text-align: right;
            margin-top: 20px;
        }}
        .footnote {{
            font-size: 0.9em;
            color: #7f8c8d;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #eee;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Watershed Cluster Analysis Report</h1>
        <p>Analysis of Percolation to Total Water Input Ratio in Agricultural Lands</p>
    </div>
    
    <div class="section">
        <h2>Analysis Overview</h2>
        <p>This report presents the results of a comprehensive analysis of the percolation to total water input ratio in agricultural lands across different watershed clusters in Michigan. The analysis was conducted using the SWAT model for years {start_year}-{end_year-1}.</p>
        
        <p><strong>Key Parameters:</strong></p>
        <ul>
            <li>Variable analyzed: {var} (percolation)</li>
            <li>Water input threshold: {precip_threshold}mm</li>
            <li>Total water input: precipitation + snowfall</li>
            <li>Year range: {start_year}-{end_year-1}</li>
            <li>Land use focus: Agricultural land (code 82)</li>
        </ul>
        
        <p>The watersheds were clustered based on their geographical location to identify regional patterns in hydrological behavior.</p>
    </div>
    
    <div class="section">
        <h2>Watershed Clustering</h2>
        <p>The watersheds were grouped into 5 clusters based on their geographical coordinates using the K-means clustering algorithm. The spatial distribution of these clusters is shown below:</p>
        
        <div class="img-container">
            {f'<img src="{michigan_map[0]}" alt="Michigan Watershed Clusters Map" style="max-width: 80%;" />' if michigan_map else ''}
            {f'<img src="{cluster_map[0]}" alt="Watershed Clusters Map" style="max-width: 80%;" />' if cluster_map else '<p>Cluster map not available</p>'}
        </div>
        
        <p>Each cluster represents watersheds with similar geographical characteristics, allowing us to analyze regional variations in hydrological behavior across Michigan's Lower Peninsula.</p>
        <p>The map shows the location of each watershed centroid, colored by its assigned cluster. This spatial organization helps identify regional patterns in the percolation to precipitation ratio.</p>
    </div>
    
    <div class="section">
        <h2>Overall Results</h2>
        <p>The following chart shows the percolation to total water input ratio across all watersheds, providing a comprehensive view of the overall patterns:</p>
        
        <div class="img-container">
            {f'<img src="{all_image[0]}" alt="Overall Percolation Analysis" />' if all_image else '<p>Overall analysis chart not available</p>'}
        </div>
        
        <p>The shaded area represents the 95% confidence interval, indicating the uncertainty in the estimates across all watershed models. The calculation includes both precipitation and snowfall as total water input.</p>
    </div>
"""

    # Add sections for each cluster
    for i in range(5):
        cluster_image = [img for img in cluster_images if f"cluster{i}" in img]
        
        html_content += f"""
    <div class="cluster-section">
        <h2>Cluster {i} Analysis</h2>
        <p>Results for watersheds in geographic cluster {i}:</p>
        
        <div class="img-container">
            {f'<img src="{cluster_image[0]}" alt="Cluster {i} Analysis" />' if cluster_image else f'<p>Chart for Cluster {i} not available</p>'}
        </div>
        
        <p>The chart above shows the temporal variation in percolation to precipitation ratio for agricultural lands in watersheds belonging to Cluster {i}.</p>
"""
        
        # Add stats if available
        if i in cluster_stats:
            html_content += f"""
        <h3>Image Details</h3>
        <table>
            <tr>
                <th>Property</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Image Resolution</td>
                <td>{cluster_stats[i]['width']} x {cluster_stats[i]['height']} pixels</td>
            </tr>
            <tr>
                <td>File Size</td>
                <td>{cluster_stats[i]['size_kb']:.2f} KB</td>
            </tr>
        </table>
"""
        
        # Close the section
        html_content += """
    </div>
"""
    
    # Add detailed statistics section if requested
    if include_statistics and statistics_results:
        html_content += """
    <div class="section">
        <h2>Detailed Statistical Analysis</h2>
        <p>This section presents a more detailed statistical analysis of the percolation to precipitation ratio data.</p>
    """
        
        # Add annual trends plot
        if stats_plots.get('annual'):
            html_content += f"""
        <h3>Annual Trends</h3>
        <p>The following chart shows the annual trends in percolation to precipitation ratio:</p>
        
        <div class="img-container">
            <img src="{stats_plots['annual'][0]}" alt="Annual Trends" style="max-width: 80%;" />
        </div>
        
        <p>This chart shows how the annual average ratio changes over the study period, with the shaded area representing the 25th to 75th percentile range.</p>
        """
        
        # Add seasonal comparison plot
        if stats_plots.get('seasonal'):
            html_content += f"""
        <h3>Seasonal Patterns</h3>
        <p>The following chart shows the seasonal patterns in percolation to precipitation ratio:</p>
        
        <div class="img-container">
            <img src="{stats_plots['seasonal'][0]}" alt="Seasonal Comparison" style="max-width: 80%;" />
        </div>
        
        <p>This chart illustrates the seasonal variations in the ratio, with error bars indicating the standard deviation.</p>
        """
        
        # Add annual variability boxplot
        if stats_plots.get('boxplot'):
            html_content += f"""
        <h3>Annual Variability</h3>
        <p>The following boxplot shows the variability in monthly ratios for each year:</p>
        
        <div class="img-container">
            <img src="{stats_plots['boxplot'][0]}" alt="Annual Variability Boxplot" style="max-width: 80%;" />
        </div>
        
        <p>The boxplots display the distribution of monthly values within each year, with whiskers extending to the 2.5th and 97.5th percentiles.</p>
        """
        
        # Add cluster comparison plot
        if stats_plots.get('cluster_comparison'):
            html_content += f"""
        <h3>Cluster Comparison</h3>
        <p>The following chart compares seasonal patterns across different watershed clusters:</p>
        
        <div class="img-container">
            <img src="{stats_plots['cluster_comparison'][0]}" alt="Cluster Seasonal Comparison" style="max-width: 90%;" />
        </div>
        
        <p>This chart illustrates how the seasonal patterns differ between the geographic clusters, highlighting regional variations.</p>
        """
        
        # Add new heatmap visualization
        if stats_plots.get('heatmap'):
            html_content += f"""
        <h3>Cluster-Season Heatmap</h3>
        <p>The following heatmap provides a visual representation of how percolation/precipitation ratios vary across clusters and seasons:</p>
        
        <div class="img-container">
            <img src="{stats_plots['heatmap'][0]}" alt="Cluster-Season Heatmap" style="max-width: 90%;" />
        </div>
        
        <p>This visualization highlights spatial and temporal patterns, with values shown as mean ± standard deviation. The color intensity represents the magnitude of the ratio.</p>
        """
        
        # Add a statistical summary section
        html_content += """
        <h3>Statistical Summary</h3>
        <p>The analysis reveals several key statistical insights:</p>
        
        <ul>
        """
        
        # Add insights based on available statistics
        if 'statistics' in statistics_results:
            # Seasonal insights
            if 'seasonal' in statistics_results['statistics']:
                seasonal_data = statistics_results['statistics']['seasonal']['overall']
                if seasonal_data:
                    # Find season with highest and lowest mean
                    seasons = list(seasonal_data.keys())
                    max_season = max(seasons, key=lambda s: seasonal_data[s]['mean'])
                    min_season = min(seasons, key=lambda s: seasonal_data[s]['mean'])
                    
                    html_content += f"""
            <li><strong>Seasonal Patterns:</strong> The {max_season.capitalize()} season shows the highest percolation/precipitation ratio 
                (mean: {seasonal_data[max_season]['mean']:.1f}%), while {min_season.capitalize()} shows the lowest 
                (mean: {seasonal_data[min_season]['mean']:.1f}%).</li>
            """
            
            # Annual insights
            if 'annual' in statistics_results['statistics']:
                annual_data = statistics_results['statistics']['annual']['overall']
                if annual_data:
                    years = sorted(annual_data.keys())
                    if len(years) > 1:
                        first_year = years[0]
                        last_year = years[-1]
                        first_mean = annual_data[first_year]['mean']
                        last_mean = annual_data[last_year]['mean']
                        change = last_mean - first_mean
                        
                        html_content += f"""
            <li><strong>Annual Trends:</strong> From {first_year} to {last_year}, the average ratio 
                {'increased' if change > 0 else 'decreased'} by {abs(change):.1f}% 
                (from {first_mean:.1f}% to {last_mean:.1f}%).</li>
            """
            
            # Variability insights
            try:
                # Calculate coefficient of variation across all data
                all_means = []
                for year in statistics_results['statistics']['annual']['overall']:
                    all_means.append(statistics_results['statistics']['annual']['overall'][year]['mean'])
                
                mean_of_means = np.mean(all_means)
                std_of_means = np.std(all_means)
                cv = std_of_means / mean_of_means if mean_of_means != 0 else 0
                
                html_content += f"""
            <li><strong>Variability:</strong> The coefficient of variation (CV) across years is {cv:.2f}, 
                indicating {'high' if cv > 0.3 else 'moderate' if cv > 0.1 else 'low'} variability in the percolation/precipitation ratio.</li>
            """
            except Exception as e:
                print(f"Error calculating variability metrics: {e}")
        
        # Add cluster insights
        if 'statistics' in statistics_results and 'seasonal' in statistics_results['statistics']:
            clusters_data = statistics_results['statistics']['seasonal'].get('clusters', {})
            if clusters_data:
                # Find cluster with highest and lowest overall means
                cluster_means = {}
                for cluster, data in clusters_data.items():
                    if not data:
                        continue
                    seasons = list(data.keys())
                    season_means = [data[season]['mean'] for season in seasons]
                    cluster_means[cluster] = np.mean(season_means)
                
                if cluster_means:
                    max_cluster = max(cluster_means.items(), key=lambda x: x[1])[0]
                    min_cluster = min(cluster_means.items(), key=lambda x: x[1])[0]
                    
                    html_content += f"""
            <li><strong>Spatial Patterns:</strong> Cluster {max_cluster} exhibits the highest average ratio ({cluster_means[max_cluster]:.1f}%), 
                while Cluster {min_cluster} shows the lowest ({cluster_means[min_cluster]:.1f}%).</li>
            """
        
        html_content += """
        </ul>
        
        <p>These findings suggest that percolation to precipitation ratios vary significantly both spatially and temporally, 
        with important implications for water resource management across Michigan's watersheds.</p>
        """
        
        # Add statistics tables
        html_content += """
        <h3>Statistical Tables</h3>
        <p>The following tables provide detailed statistics for the percolation to precipitation ratio:</p>
        """
        
        # Add overall tables
        if statistics_results and 'tables' in statistics_results:
            # Annual statistics
            if 'annual' in statistics_results['tables'] and 'overall' in statistics_results['tables']['annual']:
                annual_table_path = statistics_results['tables']['annual']['overall']
                try:
                    with open(os.path.join('./Michigan', annual_table_path), 'r') as f:
                        annual_table_html = f.read()
                    
                    html_content += f"""
        <h4>Annual Statistics</h4>
        {annual_table_html}
        <p>The table above shows annual statistics for the percolation to precipitation ratio across all watersheds.</p>
        """
                except Exception as e:
                    print(f"Error including annual table: {e}")
            
            # Seasonal statistics
            if 'seasonal' in statistics_results['tables'] and 'overall' in statistics_results['tables']['seasonal']:
                seasonal_table_path = statistics_results['tables']['seasonal']['overall']
                try:
                    with open(os.path.join('./Michigan', seasonal_table_path), 'r') as f:
                        seasonal_table_html = f.read()
                    
                    html_content += f"""
        <h4>Seasonal Statistics</h4>
        {seasonal_table_html}
        <p>The table above shows seasonal statistics for the percolation to precipitation ratio across all watersheds.</p>
        """
                except Exception as e:
                    print(f"Error including seasonal table: {e}")
        
        # Close detailed statistics section
        html_content += """
    </div>
    """
    
    # Add conclusion and footer
    html_content += f"""
    <div class="section">
        <h2>Conclusions</h2>
        <p>The analysis reveals several key insights about the percolation to precipitation ratio in agricultural lands across different watershed clusters:</p>
        
        <ul>
            <li>There is significant spatial variability across the watershed clusters, indicating that geographic location plays an important role in determining the hydrological behavior.</li>
            <li>Temporal patterns show seasonal variations in the percolation to precipitation ratio, with higher values typically observed during certain months.</li>
            <li>The width of the confidence intervals indicates the level of uncertainty in the estimates, which varies across different clusters.</li>
        </ul>
        
        <p>These findings can help inform water resource management strategies and agricultural practices in different regions of Michigan.</p>
    </div>
    
    <div class="footnote">
        <p>Analysis conducted using SWAT model outputs. Percolation to total water input ratio was calculated only for agricultural areas with combined precipitation and snowfall exceeding {precip_threshold}mm.</p>
        <p>For more information about the methodology and data sources, please refer to the technical documentation.</p>
        <p class="timestamp">Report generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
</body>
</html>
"""
    
    # Write the HTML content to a file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Report generated successfully: {output_path}")
    return output_path

if __name__ == "__main__":
    # Define parameters
    start_year = 2000
    end_year = 2005
    var = "perc"
    precip_threshold = 10
    
    # Generate report
    report_path = create_report(
        output_path='./Michigan/report.html',
        start_year=start_year,
        end_year=end_year,
        var=var,
        precip_threshold=precip_threshold
    )
    
    # Try to open the report in a browser
    try:
        import webbrowser
        webbrowser.open('file://' + os.path.abspath(report_path))
        print("Report opened in web browser")
    except Exception as e:
        print(f"Could not open report in browser: {e}")
        print(f"Please open the following file manually: {os.path.abspath(report_path)}")
