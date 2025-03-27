import os
import glob
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import shutil

def create_report(output_path='./Michigan/reports/default/report.html', start_year=2000, end_year=2005, 
                 var="perc", precip_threshold=10, include_statistics=False, statistics_results=None,
                 include_soil_analysis=False, soil_results=None):
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
    include_soil_analysis : bool
        Whether to include soil property analysis
    soil_results : dict
        Dictionary with soil analysis results
    """
    # Determine report directory structure
    report_dir = os.path.dirname(output_path)
    figs_dir = os.path.join(report_dir, 'figs')
    soil_htmls_dir = os.path.join(report_dir, 'soil_htmls')
    
    # Ensure all directories exist
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(soil_htmls_dir, exist_ok=True)
    
    print(f"Creating report in {report_dir}")
    
    # Find and copy all generated images from the Michigan/figs directory to report figs directory
    src_figs_dir = './Michigan/figs'
    if os.path.exists(src_figs_dir):
        # Create patterns for finding files
        cluster_pattern = f'recharge_water_input_ratio_{start_year}_{end_year-1}_cluster*.png'
        all_pattern = f'recharge_water_input_ratio_{start_year}_{end_year-1}_all.png'
        
        # Find files matching patterns
        cluster_images_src = glob.glob(os.path.join(src_figs_dir, cluster_pattern))
        all_image_src = glob.glob(os.path.join(src_figs_dir, all_pattern))
        cluster_map_src = glob.glob(os.path.join(src_figs_dir, 'clusters.png'))
        michigan_map_src = glob.glob(os.path.join(src_figs_dir, 'michigan_clusters_map.png'))
        
        # If new pattern doesn't find files, try the old pattern as fallback
        if not cluster_images_src:
            cluster_pattern = f'recharge_percolation_ratio_{start_year}_{end_year-1}_cluster*.png'
            cluster_images_src = glob.glob(os.path.join(src_figs_dir, cluster_pattern))
        if not all_image_src:
            all_pattern = f'recharge_percolation_ratio_{start_year}_{end_year-1}_all.png'
            all_image_src = glob.glob(os.path.join(src_figs_dir, all_pattern))
            
        # Copy files to report directory
        for src_file in cluster_images_src + all_image_src + cluster_map_src + michigan_map_src:
            if os.path.exists(src_file):
                dest_file = os.path.join(figs_dir, os.path.basename(src_file))
                shutil.copy2(src_file, dest_file)
                print(f"Copied {src_file} to {dest_file}")
    
    # Create relative paths for images (for HTML)
    cluster_images = [os.path.join("figs", os.path.basename(img)) for img in glob.glob(os.path.join(figs_dir, f"*cluster*.png"))]
    all_image = [os.path.join("figs", os.path.basename(img)) for img in glob.glob(os.path.join(figs_dir, f"*all.png"))]
    cluster_map = [os.path.join("figs", os.path.basename(img)) for img in glob.glob(os.path.join(figs_dir, "clusters.png"))]
    michigan_map = [os.path.join("figs", os.path.basename(img)) for img in glob.glob(os.path.join(figs_dir, "michigan_clusters_map.png"))]
    
    # Find statistics plots and copy if available
    stats_plots = {}
    if include_statistics:
        stats_file_patterns = {
            'boxplot': 'annual_variability_boxplot.png',
            'seasonal': 'seasonal_comparison.png',
            'annual': 'annual_trends.png',
            'cluster_comparison': 'cluster_seasonal_comparison.png',
            'heatmap': 'cluster_season_heatmap.png'
        }
        
        for key, pattern in stats_file_patterns.items():
            src_files = glob.glob(os.path.join(src_figs_dir, pattern))
            if src_files:
                for src_file in src_files:
                    dest_file = os.path.join(figs_dir, os.path.basename(src_file))
                    shutil.copy2(src_file, dest_file)
                    stats_plots[key] = [os.path.join("figs", os.path.basename(src_file))]
    
    # Handle soil analysis files
    soil_plots = {}
    if include_soil_analysis and soil_results and 'visualizations' in soil_results:
        # Copy soil visualization files
        soil_visuals = soil_results['visualizations']
        for visual_type, src_path in soil_visuals.items():
            if os.path.exists(os.path.join('./Michigan', src_path)):
                # Determine source and destination paths
                src_file = os.path.join('./Michigan', src_path)
                # All visualization files go to figs directory
                dest_file = os.path.join(figs_dir, os.path.basename(src_file))
                shutil.copy2(src_file, dest_file)
                # Update path for HTML reference
                soil_plots[visual_type] = os.path.join("figs", os.path.basename(src_file))
        
        # Copy soil HTML tables
        if 'tables' in soil_results:
            for table_type, src_path in soil_results['tables'].items():
                if os.path.exists(os.path.join('./Michigan', src_path)):
                    # Determine source and destination paths
                    src_file = os.path.join('./Michigan', src_path)
                    # HTML tables go to soil_htmls directory
                    dest_file = os.path.join(soil_htmls_dir, os.path.basename(src_file))
                    shutil.copy2(src_file, dest_file)
                    # Update path reference
                    soil_results['tables'][table_type] = os.path.join("soil_htmls", os.path.basename(src_file))
    
    # Extract basic stats from images if possible
    cluster_stats = {}
    try:
        # For each cluster, try to extract some metrics from the images
        for i in range(5):
            # Look for images in the new report's figs directory
            img_path = os.path.join(figs_dir, f'recharge_water_input_ratio_{start_year}_{end_year-1}_cluster{i}.png')
            # Fallback to old naming pattern
            if not os.path.exists(img_path):
                img_path = os.path.join(figs_dir, f'recharge_percolation_ratio_{start_year}_{end_year-1}_cluster{i}.png')
            
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
        .soil-section {{
            margin-bottom: 40px;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 8px;
            border-left: 4px solid #27ae60;
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
        .nav-tabs {{
            display: flex;
            list-style: none;
            padding: 0;
            margin: 0 0 20px 0;
            border-bottom: 1px solid #ddd;
        }}
        .nav-tabs li {{
            margin-right: 10px;
        }}
        .nav-tabs a {{
            display: block;
            padding: 10px 15px;
            text-decoration: none;
            color: #555;
            background-color: #f5f5f5;
            border-radius: 5px 5px 0 0;
            transition: background-color 0.3s;
        }}
        .nav-tabs a:hover {{
            background-color: #e0e0e0;
        }}
        .nav-tabs a.active {{
            background-color: #3498db;
            color: white;
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
    
    # Add soil analysis section if requested
    if include_soil_analysis and soil_results and soil_results.get('visualizations'):
        html_content += """
    <div class="soil-section">
        <h2>Soil Properties Analysis</h2>
        <p>This section analyzes the soil properties of agricultural lands across different watershed clusters. 
        Understanding soil characteristics is crucial for interpreting percolation patterns, as soil composition 
        directly affects water movement through the subsurface.</p>
        
        <ul class="nav-tabs">
            <li><a href="#soil-overview" class="active">Overview</a></li>
            <li><a href="#soil-clusters">Cluster Comparison</a></li>
            <li><a href="#soil-correlations">Property Correlations</a></li>
            <li><a href="#soil-boxplots">Distribution Analysis</a></li>
        </ul>
        
        <div id="soil-overview">
            <h3>Soil Properties Overview</h3>
            <p>The following analysis examines key soil properties in agricultural areas across all watershed clusters:</p>
"""
        
        # Add soil tables if available
        if 'tables' in soil_results:
            overall_table_path = soil_results['tables'].get('overall')
            if overall_table_path:
                try:
                    with open(os.path.join(report_dir, overall_table_path), 'r') as f:
                        soil_table_html = f.read()
                    
                    html_content += f"""
            <div class="table-container">
                {soil_table_html}
            </div>
            <p>The table above summarizes the key soil properties across all agricultural areas in the analyzed watersheds.</p>
"""
                except Exception as e:
                    print(f"Error including soil table: {e}")
                    html_content += "<p>Soil properties table not available</p>"
        
        # Add soil heatmap visualization
        if 'heatmap' in soil_plots:
            heatmap_path = soil_plots['heatmap']
            html_content += f"""
            <h3>Soil Properties by Cluster</h3>
            <div class="img-container">
                <img src="{heatmap_path}" alt="Soil Properties Heatmap" style="max-width: 90%;" />
            </div>
            <p>This heatmap visualizes how different soil properties vary across the watershed clusters. Color intensity represents the mean value of each property within a cluster.</p>
"""
        
        # Close soil overview and start cluster comparison
        html_content += """
        </div>
        
        <div id="soil-clusters" style="display: none;">
            <h3>Cluster Comparison of Soil Properties</h3>
            <p>The following visualization compares key soil properties across different watershed clusters:</p>
"""
        
        # Add bar chart for cluster comparison
        if 'bar_chart' in soil_plots:
            bar_chart_path = soil_plots['bar_chart']
            html_content += f"""
            <div class="img-container">
                <img src="{bar_chart_path}" alt="Soil Properties Cluster Comparison" style="max-width: 90%;" />
            </div>
            <p>The chart above compares key soil properties across the five watershed clusters. Error bars represent standard deviation within each cluster.</p>
"""
        
        # Add comparison table if available
        if 'tables' in soil_results and 'comparison' in soil_results['tables']:
            comparison_table_path = soil_results['tables']['comparison']
            try:
                with open(os.path.join(report_dir, comparison_table_path), 'r') as f:
                    comparison_table_html = f.read()
                
                html_content += f"""
            <div class="table-container">
                {comparison_table_html}
            </div>
            <p>This table provides a direct comparison of mean values for key soil properties across the five clusters.</p>
"""
            except Exception as e:
                print(f"Error including comparison table: {e}")
        
        # Close cluster comparison and start correlations
        html_content += """
        </div>
        
        <div id="soil-correlations" style="display: none;">
            <h3>Correlations Between Soil Properties</h3>
            <p>Understanding the relationships between different soil properties can provide insights into soil behavior and water movement:</p>
"""
        
        # Add correlation heatmap
        if 'correlation' in soil_plots:
            correlation_path = soil_plots['correlation']
            html_content += f"""
            <div class="img-container">
                <img src="{correlation_path}" alt="Soil Properties Correlation" style="max-width: 85%;" />
            </div>
            <p>This correlation matrix shows the Pearson correlation coefficients between different soil properties. 
            Strong positive correlations appear in dark blue, while strong negative correlations appear in dark red.</p>
            
            <h4>Key Insights from Correlation Analysis:</h4>
            <ul>
                <li>Clay and silt content often show negative correlation with hydraulic conductivity, affecting water movement</li>
                <li>Organic carbon content typically correlates positively with available water capacity</li>
                <li>Bulk density often shows negative correlation with organic carbon and porosity</li>
            </ul>
"""
        else:
            html_content += "<p>Correlation analysis not available</p>"
        
        # Close correlations and start boxplots
        html_content += """
        </div>
        
        <div id="soil-boxplots" style="display: none;">
            <h3>Distribution of Soil Properties by Cluster</h3>
            <p>The following boxplots show the distribution of key soil properties across different watershed clusters:</p>
"""
        
        # Add boxplots for key properties
        boxplot_keys = [k for k in soil_plots.keys() if k.startswith('boxplot_')]
        if boxplot_keys:
            for key in boxplot_keys[:3]:  # Limit to 3 boxplots to avoid overwhelming the report
                boxplot_path = soil_plots[key]
                html_content += f"""
            <div class="img-container">
                <img src="{boxplot_path}" alt="Soil Property Distribution" style="max-width: 80%;" />
            </div>
"""
            
            html_content += """
            <p>These boxplots illustrate the median (center line), interquartile range (box), and range (whiskers) of values 
            for soil properties within each cluster. Outliers are shown as individual points.</p>
            
            <h4>Interpretations for Water Movement:</h4>
            <ul>
                <li>Higher clay content generally leads to slower infiltration but better water retention</li>
                <li>Higher sand content typically results in faster infiltration but lower water holding capacity</li>
                <li>Organic matter improves soil structure and water holding capacity</li>
                <li>Available water capacity directly affects plant water availability and indirectly influences percolation</li>
            </ul>
"""
        else:
            html_content += "<p>Distribution analysis not available</p>"
        
        # Close boxplots and add JavaScript for tab navigation
        html_content += """
        </div>
        
        <script>
            // Simple tab navigation
            document.addEventListener('DOMContentLoaded', function() {
                const tabs = document.querySelectorAll('.nav-tabs a');
                
                tabs.forEach(tab => {
                    tab.addEventListener('click', function(e) {
                        e.preventDefault();
                        
                        // Hide all content
                        document.querySelectorAll('[id^="soil-"]').forEach(content => {
                            content.style.display = 'none';
                        });
                        
                        // Remove active class
                        tabs.forEach(t => {
                            t.classList.remove('active');
                        });
                        
                        // Show selected content and mark tab as active
                        const targetId = this.getAttribute('href').substring(1);
                        document.getElementById(targetId).style.display = 'block';
                        this.classList.add('active');
                    });
                });
            });
        </script>
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
                annual_table_path = os.path.join(soil_htmls_dir, os.path.basename(statistics_results['tables']['annual']['overall']))
                # Copy the file if it exists
                src_annual_table = os.path.join('./Michigan', statistics_results['tables']['annual']['overall'])
                if os.path.exists(src_annual_table):
                    shutil.copy2(src_annual_table, annual_table_path)
                    
                try:
                    with open(annual_table_path, 'r') as f:
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
                seasonal_table_path = os.path.join(soil_htmls_dir, os.path.basename(statistics_results['tables']['seasonal']['overall']))
                # Copy the file if it exists
                src_seasonal_table = os.path.join('./Michigan', statistics_results['tables']['seasonal']['overall'])
                if os.path.exists(src_seasonal_table):
                    shutil.copy2(src_seasonal_table, seasonal_table_path)
                    
                try:
                    with open(seasonal_table_path, 'r') as f:
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
            {"<li>Soil properties, particularly texture and organic matter content, significantly influence percolation patterns across the watersheds.</li>" if include_soil_analysis else ""}
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
    
    # Create a timestamped report name
    report_name = f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = f'./Michigan/reports/{report_name}'
    
    # Generate report
    report_path = create_report(
        output_path=os.path.join(output_dir, 'report.html'),
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
