import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import sys
import warnings
import glob
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from scipy import stats

# Suppress warnings during processing
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import the clustering function
try:
    from utils.geolocate import get_model_cluster_mapping
except ImportError:
    print("Warning: Could not import get_model_cluster_mapping")

# Define soil properties and their user-friendly names and units
SOIL_PROPERTIES = {
    'alb_30m': {'name': 'Albedo', 'unit': '', 'desc': 'Surface reflectivity'},
    'awc_30m': {'name': 'Available Water Capacity', 'unit': 'mm/mm', 'desc': 'Amount of water available to plants'},
    'bd_30m': {'name': 'Bulk Density', 'unit': 'g/cm³', 'desc': 'Soil mass per volume'},
    'caco3_30m': {'name': 'Calcium Carbonate', 'unit': '%', 'desc': 'Lime content in soil'},
    'carbon_30m': {'name': 'Organic Carbon', 'unit': '%', 'desc': 'Organic matter content'},
    'clay_30m': {'name': 'Clay Content', 'unit': '%', 'desc': 'Percentage of clay particles'},
    'dp_30m': {'name': 'Soil Depth', 'unit': 'mm', 'desc': 'Depth of soil profile'},
    'ec_30m': {'name': 'Electrical Conductivity', 'unit': 'dS/m', 'desc': 'Measure of soil salinity'},
    'ph_30m': {'name': 'Soil pH', 'unit': '', 'desc': 'Acidity/alkalinity of soil'},
    'rock_30m': {'name': 'Rock Content', 'unit': '%', 'desc': 'Percentage of rock fragments'},
    'silt_30m': {'name': 'Silt Content', 'unit': '%', 'desc': 'Percentage of silt particles'},
    'soil_30m': {'name': 'Soil Type', 'unit': '', 'desc': 'Soil classification code'},
    'soil_k_30m': {'name': 'Hydraulic Conductivity', 'unit': 'mm/hr', 'desc': 'Water transmission rate'}
}

def process_model_soil_data(model_path, model_name, cluster=-1):
    """
    Process soil data for a single SWAT model.
    
    Parameters:
    -----------
    model_path : str
        Path to the model directory
    model_name : str
        Name of the model
    cluster : int
        Cluster number the model belongs to
        
    Returns:
    --------
    dict
        Dictionary with soil property statistics for agricultural land
    """
    # Path to the h5 file
    h5_file = os.path.join(model_path, model_name, "SWAT_gwflow_MODEL", "Scenarios", "verification_stage_0", "SWATplus_output.h5")
    
    # Check if file exists
    if not os.path.exists(h5_file):
        print(f"Warning: H5 file not found for model {model_name}")
        return None
    
    try:
        # Open the h5 file
        with h5py.File(h5_file, 'r') as f:
            # Check if required datasets exist
            if 'Landuse/landuse_30m' not in f or 'Soil' not in f:
                print(f"Warning: Required datasets not found in {model_name}")
                return None
            
            # Read landuse data to identify agricultural land (code 82)
            landuse = f['Landuse/landuse_30m'][:]
            ag_mask = np.where(landuse == 82, 1, 0)
            
            # If no agricultural land, return empty dict
            if np.sum(ag_mask) == 0:
                print(f"No agricultural land in model {model_name}")
                return None
            
            # Process soil properties
            soil_stats = {}
            
            # Get list of available soil properties
            available_properties = list(f['Soil'].keys())
            
            for prop in available_properties:
                if prop in SOIL_PROPERTIES:
                    try:
                        # Read soil property data
                        soil_data = f['Soil'][prop][:]
                        
                        # Check and fix size mismatch between soil data and landuse data
                        if soil_data.shape != landuse.shape:
                            print(f"Size mismatch for {prop} in {model_name}. Soil: {soil_data.shape}, Landuse: {landuse.shape}")
                            
                            # Ensure soil_data has the same shape as landuse data
                            if len(soil_data.shape) == 2 and len(landuse.shape) == 2:
                                # If soil data is smaller, pad it with zeros
                                if soil_data.shape[0] < landuse.shape[0] or soil_data.shape[1] < landuse.shape[1]:
                                    padded_data = np.zeros(landuse.shape, dtype=soil_data.dtype)
                                    # Copy the available data
                                    padded_data[:min(soil_data.shape[0], landuse.shape[0]), 
                                               :min(soil_data.shape[1], landuse.shape[1])] = soil_data[:min(soil_data.shape[0], landuse.shape[0]), 
                                                                                                      :min(soil_data.shape[1], landuse.shape[1])]
                                    soil_data = padded_data
                                    print(f"Padded {prop} data to match landuse shape")
                                # If soil data is larger, crop it
                                elif soil_data.shape[0] > landuse.shape[0] or soil_data.shape[1] > landuse.shape[1]:
                                    soil_data = soil_data[:landuse.shape[0], :landuse.shape[1]]
                                    print(f"Cropped {prop} data to match landuse shape")
                        
                        # Apply agricultural mask
                        masked_data = soil_data * ag_mask
                        
                        # Replace 0 with NaN for areas that aren't agricultural
                        masked_data = np.where(ag_mask == 0, np.nan, masked_data)
                        
                        # Calculate statistics
                        valid_data = masked_data[~np.isnan(masked_data)]
                        
                        if len(valid_data) > 0:
                            soil_stats[prop] = {
                                'mean': float(np.mean(valid_data)),
                                'median': float(np.median(valid_data)),
                                'std': float(np.std(valid_data)),
                                'min': float(np.min(valid_data)),
                                'max': float(np.max(valid_data)),
                                'q25': float(np.percentile(valid_data, 25)),
                                'q75': float(np.percentile(valid_data, 75)),
                                'count': int(len(valid_data)),
                                'ag_percent': float(100 * len(valid_data) / np.sum(ag_mask))
                            }
                    except Exception as e:
                        print(f"Error processing {prop} for {model_name}: {e}")
            
            # Add metadata
            soil_stats['metadata'] = {
                'model_name': model_name,
                'cluster': cluster,
                'ag_pixels': int(np.sum(ag_mask)),
                'total_pixels': int(landuse.size)
            }
            
            return soil_stats
    
    except Exception as e:
        print(f"Error processing model {model_name}: {e}")
        return None

def process_args_wrapper(args, base_path):
    """Helper function to unpack arguments for process_model_soil_data"""
    name, cluster = args
    return process_model_soil_data(base_path, name, cluster)

def analyze_soil_properties(base_path, output_dir="./Michigan", model_limit=50, num_workers=4):
    """
    Analyze soil properties for all models across clusters.
    
    Parameters:
    -----------
    base_path : str
        Base path to SWAT model directories
    output_dir : str
        Directory to save output files
    model_limit : int
        Maximum number of models to process
    num_workers : int
        Number of parallel workers
        
    Returns:
    --------
    dict
        Dictionary with soil analysis results and paths to generated figures
    """
    print("Starting soil property analysis...")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figs'), exist_ok=True)
    
    # Get model names
    model_names = []
    try:
        model_names = [name for name in os.listdir(base_path) 
                      if os.path.isdir(os.path.join(base_path, name))]
        print(f"Found {len(model_names)} models in {base_path}")
    except Exception as e:
        print(f"Error listing model directories: {e}")
        return None
    
    # Limit number of models if specified
    if model_limit and len(model_names) > model_limit:
        model_names = model_names[:model_limit]
        print(f"Limiting analysis to {model_limit} models")
    
    # Get cluster mapping
    try:
        name_to_cluster = get_model_cluster_mapping(base_path)
        print(f"Retrieved cluster mapping for {len(name_to_cluster)} models")
    except Exception as e:
        print(f"Error retrieving cluster mapping: {e}")
        name_to_cluster = {}  # Empty dict if clustering fails
    
    # Process models in parallel
    print(f"Processing soil data from {len(model_names)} models with {num_workers} workers...")
    
    all_soil_data = {}
    
    # Prepare arguments for parallel processing
    args_list = [(name, name_to_cluster.get(name, -1)) for name in model_names]
    
    # Create a function wrapper that includes the base_path
    func_with_path = partial(process_args_wrapper, base_path=base_path)
    
    # Process models in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Use the wrapped function
        results = list(executor.map(func_with_path, args_list))
    
    # Filter out None results and add to data structure
    results = [r for r in results if r is not None]
    for soil_stats in results:
        model_name = soil_stats['metadata']['model_name']
        all_soil_data[model_name] = soil_stats
    
    print(f"Successfully processed soil data for {len(all_soil_data)} models")
    
    # Organize data by clusters
    cluster_data = {}
    for model_name, soil_stats in all_soil_data.items():
        cluster = soil_stats['metadata']['cluster']
        if cluster not in cluster_data:
            cluster_data[cluster] = []
        cluster_data[cluster].append(soil_stats)
    
    # Calculate aggregate statistics by cluster
    aggregate_stats = {}
    property_data = {}
    
    # Initialize property_data with arrays for each soil property and cluster
    for prop in SOIL_PROPERTIES:
        property_data[prop] = {
            'all': [],  # All models
            'clusters': {i: [] for i in range(5)}  # Each cluster
        }
    
    # Populate property_data with values from each model
    for model_name, soil_stats in all_soil_data.items():
        cluster = soil_stats['metadata']['cluster']
        
        for prop in SOIL_PROPERTIES:
            if prop in soil_stats:
                # Add to overall list
                property_data[prop]['all'].append(soil_stats[prop]['mean'])
                
                # Add to cluster list if valid cluster
                if 0 <= cluster < 5:
                    property_data[prop]['clusters'][cluster].append(soil_stats[prop]['mean'])
    
    # Calculate statistics for each property
    for prop in SOIL_PROPERTIES:
        # Overall statistics
        if property_data[prop]['all']:
            aggregate_stats[prop] = {
                'all': {
                    'mean': np.mean(property_data[prop]['all']),
                    'median': np.median(property_data[prop]['all']),
                    'std': np.std(property_data[prop]['all']),
                    'min': np.min(property_data[prop]['all']),
                    'max': np.max(property_data[prop]['all']),
                    'count': len(property_data[prop]['all'])
                },
                'clusters': {}
            }
            
            # Cluster statistics
            for cluster in range(5):
                if property_data[prop]['clusters'][cluster]:
                    aggregate_stats[prop]['clusters'][cluster] = {
                        'mean': np.mean(property_data[prop]['clusters'][cluster]),
                        'median': np.median(property_data[prop]['clusters'][cluster]),
                        'std': np.std(property_data[prop]['clusters'][cluster]),
                        'min': np.min(property_data[prop]['clusters'][cluster]),
                        'max': np.max(property_data[prop]['clusters'][cluster]),
                        'count': len(property_data[prop]['clusters'][cluster])
                    }
    
    # Generate visualizations
    print("Generating soil property visualizations...")
    visualization_paths = generate_soil_visualizations(aggregate_stats, property_data, output_dir)
    
    # Generate HTML tables
    print("Generating soil property tables...")
    table_paths = generate_soil_tables(aggregate_stats, output_dir)
    
    # Return results
    return {
        'aggregate_stats': aggregate_stats,
        'visualizations': visualization_paths,
        'tables': table_paths
    }

def generate_soil_visualizations(aggregate_stats, property_data, output_dir):
    """Generate visualizations for soil properties"""
    visualization_paths = {}
    
    # Create directory for figures
    os.makedirs(os.path.join(output_dir, 'figs'), exist_ok=True)
    
    # 1. Bar chart comparing clusters for key soil properties
    key_properties = ['clay_30m', 'silt_30m', 'carbon_30m', 'awc_30m', 'soil_k_30m', 'ph_30m']
    filtered_properties = [p for p in key_properties if p in aggregate_stats]
    
    if filtered_properties:
        # Create bar chart
        plt.figure(figsize=(14, 10))
        
        # Number of properties and clusters
        n_props = len(filtered_properties)
        n_clusters = 5
        
        # Set up the plot
        fig, axes = plt.subplots(figsize=(12, 8), nrows=1, ncols=1)
        
        # Width of a bar 
        width = 0.15
        
        # Position of bars on x-axis
        r = np.arange(n_props)
        
        # Set color palette
        colors = plt.cm.viridis(np.linspace(0, 0.9, n_clusters))
        
        # Plot bars for each cluster
        for i in range(n_clusters):
            values = []
            errors = []
            
            for prop in filtered_properties:
                if prop in aggregate_stats and 'clusters' in aggregate_stats[prop] and i in aggregate_stats[prop]['clusters']:
                    values.append(aggregate_stats[prop]['clusters'][i]['mean'])
                    errors.append(aggregate_stats[prop]['clusters'][i]['std'])
                else:
                    values.append(0)
                    errors.append(0)
            
            # Plot bars with error bars
            axes.bar(r + i*width, values, width, yerr=errors, capsize=3, 
                    color=colors[i], label=f'Cluster {i}', alpha=0.7)
        
        # Add labels and title
        axes.set_xlabel('Soil Properties', fontweight='bold', fontsize=12)
        axes.set_ylabel('Mean Value', fontweight='bold', fontsize=12)
        axes.set_title('Comparison of Key Soil Properties Across Clusters', fontsize=16, fontweight='bold')
        
        # Set x-ticks in the middle of the bars of each property
        prop_labels = [SOIL_PROPERTIES[p]['name'] + f" ({SOIL_PROPERTIES[p]['unit']})" if SOIL_PROPERTIES[p]['unit'] else SOIL_PROPERTIES[p]['name'] for p in filtered_properties]
        axes.set_xticks(r + width * (n_clusters-1) / 2)
        axes.set_xticklabels(prop_labels, rotation=45, ha='right')
        
        # Add a legend
        axes.legend(title="Watershed Clusters")
        
        # Add grid
        axes.grid(True, linestyle='--', alpha=0.7)
        
        # Tight layout to prevent label cutoff
        plt.tight_layout()
        
        # Save figure
        bar_chart_path = os.path.join(output_dir, 'figs', 'soil_properties_clusters.png')
        plt.savefig(bar_chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        visualization_paths['bar_chart'] = os.path.join('figs', 'soil_properties_clusters.png')
    
    # 2. Heatmap for key soil properties by cluster
    if filtered_properties:
        # Create data for heatmap
        heatmap_data = []
        
        # For each property
        for prop in filtered_properties:
            row = []
            for cluster in range(5):
                if prop in aggregate_stats and 'clusters' in aggregate_stats[prop] and cluster in aggregate_stats[prop]['clusters']:
                    row.append(aggregate_stats[prop]['clusters'][cluster]['mean'])
                else:
                    row.append(np.nan)
            heatmap_data.append(row)
        
        # Convert to numpy array
        heatmap_array = np.array(heatmap_data)
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        
        # Create custom colormap
        cmap = LinearSegmentedColormap.from_list('soil_cmap', ['#f7fbff', '#08306b'])
        
        # Create the heatmap
        sns.heatmap(heatmap_array, cmap=cmap, annot=True, fmt='.2f', linewidths=.5, 
                  xticklabels=[f'Cluster {i}' for i in range(5)],
                  yticklabels=[SOIL_PROPERTIES[p]['name'] for p in filtered_properties])
        
        # Add labels and title
        plt.title('Soil Properties by Watershed Cluster', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        heatmap_path = os.path.join(output_dir, 'figs', 'soil_properties_heatmap.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        visualization_paths['heatmap'] = os.path.join('figs', 'soil_properties_heatmap.png')
    
    # 3. Box plots for selected soil properties
    for prop in filtered_properties:
        if property_data[prop]['all']:
            # Create box plot data structure
            box_data = []
            labels = []
            
            # Add data for each cluster
            for cluster in range(5):
                if property_data[prop]['clusters'][cluster]:
                    box_data.append(property_data[prop]['clusters'][cluster])
                    labels.append(f'Cluster {cluster}')
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Create box plot
            plt.boxplot(box_data, patch_artist=True, labels=labels)
            
            # Add title and labels
            plt.title(f'Distribution of {SOIL_PROPERTIES[prop]["name"]} by Cluster', fontsize=14, fontweight='bold')
            plt.ylabel(f'{SOIL_PROPERTIES[prop]["name"]} ({SOIL_PROPERTIES[prop]["unit"]})' if SOIL_PROPERTIES[prop]["unit"] else SOIL_PROPERTIES[prop]["name"], fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Save figure
            boxplot_path = os.path.join(output_dir, 'figs', f'soil_{prop}_boxplot.png')
            plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            visualization_paths[f'boxplot_{prop}'] = os.path.join('figs', f'soil_{prop}_boxplot.png')
    
    # 4. Correlation heatmap
    if filtered_properties and len(filtered_properties) > 1:
        # Prepare correlation data
        corr_data = np.zeros((len(filtered_properties), len(filtered_properties)))
        
        # Fill correlation matrix
        for i, prop1 in enumerate(filtered_properties):
            for j, prop2 in enumerate(filtered_properties):
                # Get all values for both properties
                values1 = property_data[prop1]['all']
                values2 = property_data[prop2]['all']
                
                # Calculate correlation
                if values1 and values2 and len(values1) == len(values2):
                    # Remove NaN values
                    valid_indices = ~np.isnan(values1) & ~np.isnan(values2)
                    valid_values1 = np.array(values1)[valid_indices]
                    valid_values2 = np.array(values2)[valid_indices]
                    
                    if len(valid_values1) > 1:
                        corr, _ = stats.pearsonr(valid_values1, valid_values2)
                        corr_data[i, j] = corr
                    else:
                        corr_data[i, j] = np.nan
                else:
                    corr_data[i, j] = np.nan
        
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_data, dtype=bool))
        
        # Create custom colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        # Plot heatmap
        sns.heatmap(corr_data, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                  square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True,
                  xticklabels=[SOIL_PROPERTIES[p]['name'] for p in filtered_properties],
                  yticklabels=[SOIL_PROPERTIES[p]['name'] for p in filtered_properties])
        
        plt.title('Correlation Between Soil Properties', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        corr_path = os.path.join(output_dir, 'figs', 'soil_properties_correlation.png')
        plt.savefig(corr_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        visualization_paths['correlation'] = os.path.join('figs', 'soil_properties_correlation.png')
    
    return visualization_paths

def generate_soil_tables(aggregate_stats, output_dir):
    """Generate HTML tables for soil properties"""
    table_paths = {}
    
    # Create directory for tables
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Overall statistics table for all soil properties
    overall_stats = []
    
    # Collect overall statistics for each property
    for prop, stats in aggregate_stats.items():
        if 'all' in stats:
            overall_stats.append({
                'Property': SOIL_PROPERTIES[prop]['name'],
                'Unit': SOIL_PROPERTIES[prop]['unit'],
                'Description': SOIL_PROPERTIES[prop]['desc'],
                'Mean': stats['all']['mean'],
                'Median': stats['all']['median'],
                'Std Dev': stats['all']['std'],
                'Min': stats['all']['min'],
                'Max': stats['all']['max'],
                'Count': stats['all']['count']
            })
    
    # Create DataFrame
    if overall_stats:
        df_overall = pd.DataFrame(overall_stats)
        
        # Sort by property name
        df_overall = df_overall.sort_values('Property')
        
        # Save to HTML
        overall_table_path = os.path.join(output_dir, 'soil_properties_overall.html')
        df_overall.to_html(overall_table_path, index=False, float_format='%.3f')
        table_paths['overall'] = 'soil_properties_overall.html'
    
    # 2. Cluster comparison tables for each soil property
    for prop, stats in aggregate_stats.items():
        if 'clusters' in stats:
            cluster_stats = []
            
            # Collect statistics for each cluster
            for cluster in range(5):
                if cluster in stats['clusters']:
                    cluster_stats.append({
                        'Cluster': cluster,
                        'Mean': stats['clusters'][cluster]['mean'],
                        'Median': stats['clusters'][cluster]['median'],
                        'Std Dev': stats['clusters'][cluster]['std'],
                        'Min': stats['clusters'][cluster]['min'],
                        'Max': stats['clusters'][cluster]['max'],
                        'Count': stats['clusters'][cluster]['count']
                    })
            
            # Create DataFrame
            if cluster_stats:
                df_cluster = pd.DataFrame(cluster_stats)
                
                # Sort by cluster
                df_cluster = df_cluster.sort_values('Cluster')
                
                # Save to HTML
                cluster_table_path = os.path.join(output_dir, f'soil_{prop}_clusters.html')
                df_cluster.to_html(cluster_table_path, index=False, float_format='%.3f')
                table_paths[f'cluster_{prop}'] = f'soil_{prop}_clusters.html'
    
    # 3. Summary comparison table across all clusters
    comparison_data = []
    
    # Select key properties for comparison
    key_properties = ['clay_30m', 'silt_30m', 'carbon_30m', 'awc_30m', 'soil_k_30m', 'ph_30m']
    filtered_properties = [p for p in key_properties if p in aggregate_stats]
    
    # Collect data for each cluster
    for cluster in range(5):
        row = {'Cluster': cluster}
        
        for prop in filtered_properties:
            if 'clusters' in aggregate_stats[prop] and cluster in aggregate_stats[prop]['clusters']:
                row[SOIL_PROPERTIES[prop]['name']] = aggregate_stats[prop]['clusters'][cluster]['mean']
            else:
                row[SOIL_PROPERTIES[prop]['name']] = np.nan
        
        comparison_data.append(row)
    
    # Create DataFrame
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        
        # Sort by cluster
        df_comparison = df_comparison.sort_values('Cluster')
        
        # Save to HTML
        comparison_table_path = os.path.join(output_dir, 'soil_properties_comparison.html')
        df_comparison.to_html(comparison_table_path, index=False, float_format='%.3f')
        table_paths['comparison'] = 'soil_properties_comparison.html'
    
    return table_paths

# Example usage
if __name__ == "__main__":
    base_path = "/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12"
    import multiprocessing
    num_workers = max(1, multiprocessing.cpu_count() // 2)  # Use half of available CPU cores
    
    # Run the analysis
    results = analyze_soil_properties(base_path, "./Michigan", model_limit=50, num_workers=num_workers)
    
    if results:
        print("Soil analysis completed successfully.")
        print(f"Generated {len(results['visualizations'])} visualizations and {len(results['tables'])} tables.")
        print("Visualization paths:", results['visualizations'])
        print("Table paths:", results['tables'])
    else:
        print("Soil analysis failed.")