import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np

def plot_distribution(pfas_gw, logger, name='gw'):
    logger.info("stage:plot_distribution ==##== Plotting sum of PFAS distribution")
    plt.figure()
    pfas_gw['sum_PFAS'].plot.hist(bins=100)
    plt.xlabel('Sum of PFAS (ng/L)')
    plt.ylabel('Frequency')
    plt.yscale("symlog")
    plt.title('Distribution of Sum of PFAS')
    plt.grid(axis='both', linestyle='--', alpha=0.6)
    os.makedirs('figs', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'figs/sum_pfas_{name}_distribution.png', dpi=300)
    plt.close()

def plot_site_samples(train_gw, val_gw, test_gw, pfas_sites, logger, name='gw'):
    logger.info("stage:plot_site_samples ==##== Plotting sampled sites")
    ax, fig = plt.subplots(figsize=(8, 8))
    huron_bounds = pd.read_pickle("/data/MyDataBase/HuronRiverPFAS/Huron_River_basin_bound.pkl").to_crs("EPSG:4326")
    huron_bounds.boundary.plot(ax=plt.gca(), facecolor='none', edgecolor='black', linewidth=1)
    pfas_sites[['Industry','geometry']].to_crs("EPSG:4326").plot(ax=plt.gca(), color='lightgray', edgecolor='black', linewidth=0.5, marker='^')

    train_gw[['geometry']].to_crs("EPSG:4326").plot(ax=plt.gca(), color='blue', edgecolor='black', linewidth=0.5)
    val_gw[['geometry']].to_crs("EPSG:4326").plot(ax=plt.gca(), color='green', edgecolor='black', linewidth=0.5)
    test_gw[['geometry']].to_crs("EPSG:4326").plot(ax=plt.gca(), color='red', edgecolor='black', linewidth=0.5)
    number_of_sites = len(pfas_sites)
    number_of_train = len(train_gw)
    number_of_val = len(val_gw)
    number_of_test = len(test_gw)
    ## annotate
    plt.annotate(f"Number of sites: {number_of_sites}\nNumber of train: {number_of_train}\nNumber of val: {number_of_val}\nNumber of test: {number_of_test}", xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, ha='left', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    #plt.xlabel('Longitude')
    #plt.ylabel('Latitude')
    ## remove x and y axis
    plt.xticks([])
    plt.yticks([])
    ## turn off axes
    plt.axis('off')
    #plt.title('Splitted PFAS samples in Huron River Basin')
    plt.grid(axis='both', linestyle='--', alpha=0.6)
    plt.legend(["Huron River Basin", 'PFAS sites', 'Train', 'Validation', 'Test'], loc = 'lower left')
    plt.tight_layout()
    plt.savefig(f'figs/{name}_sampled_sites.png', dpi=600)

    plt.close()

def plot_sum_pfas(pfas_gw, node_name):
    assert "geometry" in pfas_gw.columns, "geometry column is missing in pfas_gw"
    assert len(pfas_gw) > 0, "pfas_gw is empty"
    assert "sum_PFAS" in pfas_gw.columns, "sum_PFAS column is missing in pfas_gw"
    ## assert no negative values in sum_PFAS
    assert pfas_gw['sum_PFAS'].min() >= 0,"There are negative values in sum_PFAS"
    bounds = pd.read_pickle("/data/MyDataBase/HuronRiverPFAS/Huron_River_basin_bound.pkl").to_crs("EPSG:4326")

    ### verify all classes are present
   # assert all(
   #     i in pfas_gw['sum_PFAS_class'].unique() for i in [0, 1, 2]
    #), "There are missing classes in sum_PFAS_class"
    colors = ['green', 'blue', 'red' , 'grey']
    pfas_gw = gpd.GeoDataFrame(pfas_gw, geometry='geometry', crs='EPSG:26990').to_crs("EPSG:4326")

    plt.figure(figsize=(8, 8))
    bounds.boundary.plot(ax=plt.gca(), facecolor='none', edgecolor='black', linewidth=1)
    if node_name == 'sw_stations':
        legend_labels = ['0-200', '200-500', f'500-{pfas_gw["sum_PFAS"].max():.1f}', 'unknown']
    else:
        legend_labels = ['0-1','1-10',f'10-{pfas_gw["sum_PFAS"].max():.1f}', 'unknown']
    ### sort before plotting to show higher values on top
    #pfas_gw = pfas_gw.sort_values('sum_PFAS', ascending=False)
    ## sort in a way that we first see red, then blue, then green, then gray
    pfas_gw = pfas_gw.sort_values('sum_PFAS_class')
    for i, (name, group) in enumerate(pfas_gw.groupby('sum_PFAS_class', observed=True)):
        if colors[i] == 'grey':
            group.plot(ax=plt.gca(), facecolor='none', edgecolor=colors[i], linewidth=0.5, alpha=0.8, markersize=25)
        else:
            group.plot(ax=plt.gca(), color=colors[i],alpha=0.8, markersize=35)

    #plt.xlabel('Longitude')
    #plt.ylabel('Latitude')
    ## remove x and y axis
    plt.xticks([])
    plt.yticks([])
    plt.legend(["Huron River Basin"] + legend_labels, loc='lower left')
    #if node_name == 'gw_wells':
    #    plt.title(f'#{pfas_gw["WSSN"].nunique()} water wells with unique WSSN')
    #elif node_name == 'sw_stations':
    #    plt.title(f'#{pfas_gw["SiteCode"].nunique()} sites with unique SiteCode')
    #else:
    #    plt.title('#ERROR IN TITLE')
    ## turn off axes
    plt.axis('off')
    
    plt.grid(axis='both', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'figs/sum_{node_name}_PFAS.png', dpi=600)
    plt.close()



def plot_pred_sum_pfas_with_log_colorbar(pfas_gw, node_name):
    assert "geometry" in pfas_gw.columns, "geometry column is missing in pfas_gw"
    assert len(pfas_gw) > 0, "pfas_gw is empty"
    assert "pred_sum_PFAS" in pfas_gw.columns, "pred_sum_PFAS column is missing in pfas_gw"
    
    # Load the boundaries
    bounds = pd.read_pickle("/data/MyDataBase/HuronRiverPFAS/Huron_River_basin_bound.pkl").to_crs("EPSG:4326")
    
    # Replace negative values with 0
    pfas_gw['pred_sum_PFAS'] = pfas_gw['pred_sum_PFAS'].clip(lower=0.0)  # Avoid log(0) by clipping to a small positive value
    
    # Convert to GeoDataFrame and ensure correct CRS
    pfas_gw = gpd.GeoDataFrame(pfas_gw, geometry='geometry', crs='EPSG:26990').to_crs("EPSG:4326")
    
    # Plot setup
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    
    # Plot the boundaries
    bounds.boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)
    
    # Sort values for better visualization
    pfas_gw = pfas_gw.sort_values('pred_sum_PFAS')
    
    # Plot the data with a logarithmic color map
    pfas_gw.plot(ax=ax, column='pred_sum_PFAS', cmap='viridis', linewidth=0.5, edgecolor='none', alpha=1, norm=LogNorm(vmin=pfas_gw['pred_sum_PFAS'].min(), vmax=pfas_gw['pred_sum_PFAS'].max()), markersize=5)
    
    # Set axis labels
    #plt.xlabel('Longitude')
    #plt.ylabel('Latitude')
    ## remove x and y axis
    plt.xticks([])
    plt.yticks([])
    
    # Set axis ticks and format
    #plt.xticks(np.linspace(pfas_gw.total_bounds[0], pfas_gw.total_bounds[2], 4))
    #plt.yticks(np.linspace(pfas_gw.total_bounds[1], pfas_gw.total_bounds[3], 4))
    #plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.2f}'.format(x)))
    #plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))
    
    # Add logarithmic color bar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=LogNorm(vmin=pfas_gw['pred_sum_PFAS'].min(), vmax=pfas_gw['pred_sum_PFAS'].max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Predicted Sum of PFAS (Log Scale)')
    
    # Add title
    plt.title(f'Predicted Sum of PFAS for {len(pfas_gw)} samples with range: {pfas_gw["pred_sum_PFAS"].min():.2f} - {pfas_gw["pred_sum_PFAS"].max():.2f}')
    nse, rmse, mse, mae = calculate_error_metrics(pfas_gw, name="plot_pred_sum_pfas_with_log_colorbar")
    plt.annotate(f"NSE: {nse:.2f}\nRMSE: {rmse:.2f}\nMSE: {mse:.2f}\nMAE: {mae:.2f}", xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, ha='left', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    # Add grid
    plt.grid(axis='both', linestyle='--', alpha=0.6)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f'figs/pred_{node_name}_sum_PFAS_log_colorbar.png', dpi=300)
    plt.close()

def plot_pred_sum_pfas_with_colorbar(pfas_gw, node_name):
    assert "geometry" in pfas_gw.columns, "geometry column is missing in pfas_gw"
    assert len(pfas_gw) > 0, "pfas_gw is empty"
    assert "pred_sum_PFAS" in pfas_gw.columns, "pred_sum_PFAS column is missing in pfas_gw"
    
    # Load the boundaries
    bounds = pd.read_pickle("/data/MyDataBase/HuronRiverPFAS/Huron_River_basin_bound.pkl").to_crs("EPSG:4326")
    
    # Replace negative values with 0
    pfas_gw['pred_sum_PFAS'] = pfas_gw['pred_sum_PFAS'].clip(lower=0)
    
    # Convert to GeoDataFrame and ensure correct CRS
    pfas_gw = gpd.GeoDataFrame(pfas_gw, geometry='geometry', crs='EPSG:26990').to_crs("EPSG:4326")
    
    # Plot setup
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    
    # Plot the boundaries
    bounds.boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)
    
    # Sort values for better visualization
    pfas_gw = pfas_gw.sort_values('pred_sum_PFAS', ascending=False)
    
    # Plot the data with a color map
    pfas_gw.plot(ax=ax, column='pred_sum_PFAS', cmap='viridis', linewidth=0.5, edgecolor='black', alpha=0.5)
    
    # Set axis labels
    #plt.xlabel('Longitude')
    #plt.ylabel('Latitude')
    ## remove x and y axis
    plt.xticks([])
    plt.yticks([])
    
    # Set axis ticks and format
    #plt.xticks(np.linspace(pfas_gw.total_bounds[0], pfas_gw.total_bounds[2], 4))
    #plt.yticks(np.linspace(pfas_gw.total_bounds[1], pfas_gw.total_bounds[3], 4))
    #plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.2f}'.format(x)))
    #plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))
    
    # Add color bar
    sm = plt.cm.ScalarMappable(cmap='cividis', norm=plt.Normalize(vmin=pfas_gw['pred_sum_PFAS'].min(), vmax=pfas_gw['pred_sum_PFAS'].max()))
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Predicted Sum of PFAS')
    
    # Add title
    plt.title(f'Predicted Sum of PFAS for {len(pfas_gw)} samples with range: {pfas_gw["pred_sum_PFAS"].min():.2f} - {pfas_gw["pred_sum_PFAS"].max():.2f}')
    nse, rmse, mse, mae = calculate_error_metrics(pfas_gw, name="plot_pred_sum_pfas_with_colorbar")
    plt.annotate(f"NSE: {nse:.2f}\nRMSE: {rmse:.2f}\nMSE: {mse:.2f}\nMAE: {mae:.2f}", xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, ha='left', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    # Add grid
    plt.grid(axis='both', linestyle='--', alpha=0.6)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f'figs/pred_{node_name}_sum_PFAS_colorbar.png', dpi=300)
    plt.close()


def plot_pred_sum_pfas(pfas_gw, node_name):
    assert "geometry" in pfas_gw.columns, "geometry column is missing in pfas_gw"
    assert len(pfas_gw) > 0, "pfas_gw is empty"
    assert "pred_sum_PFAS" in pfas_gw.columns, "pred_sum_PFAS column is missing in pfas_gw"
    
    # Load the boundaries
    bounds = pd.read_pickle("/data/MyDataBase/HuronRiverPFAS/Huron_River_basin_bound.pkl").to_crs("EPSG:4326")
    
    # Replace negative values with 0
    pfas_gw['pred_sum_PFAS'] = pfas_gw['pred_sum_PFAS'].clip(lower=0)
    
    # Define color and label mappings based on node_name
    if node_name == 'gw_wells':
        class_bins = [-1, 1, 10, 50, 200]
        class_colors = {0: 'green', 1: 'blue', 2: 'red'}
        class_labels = {0: '0-1', 1: '1-10', 2: '10-200'}
    else:
        class_bins = [-1, 0, 200, 500, 2000]
        class_colors = {0: 'green', 1: 'yellow', 2: 'blue', 3: 'red'}
        class_labels = {0: '0', 1: '0-200', 2: '200-500', 3: '500-2000'}
    
    # Classify the data
    pfas_gw['pred_sum_PFAS_class'] = pd.cut(pfas_gw['pred_sum_PFAS'], bins=class_bins, labels=range(len(class_bins)-1))
    
    # Convert to GeoDataFrame and ensure correct CRS
    pfas_gw = gpd.GeoDataFrame(pfas_gw, geometry='geometry', crs='EPSG:26990').to_crs("EPSG:4326")
    
    # Plot setup
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    
    # Plot the boundaries
    bounds.boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)
    
    # Sort values for better visualization
    pfas_gw = pfas_gw.sort_values('pred_sum_PFAS')
    
    # Plot each class with its assigned color
    for class_value, color in class_colors.items():
        class_group = pfas_gw[pfas_gw['pred_sum_PFAS_class'] == class_value]
        if not class_group.empty:
            class_group.plot(ax=ax, color=color, edgecolor='black', linewidth=0.5, alpha=0.5, label=class_labels[class_value], markersize=60)
    
    # Set axis labels
    #plt.xlabel('Longitude')
    #plt.ylabel('Latitude')
    ## remove x and y axis
    plt.xticks([])
    plt.yticks([])
    
    # Set axis ticks and format
    #plt.xticks(np.linspace(pfas_gw.total_bounds[0], pfas_gw.total_bounds[2], 4))
    #plt.yticks(np.linspace(pfas_gw.total_bounds[1], pfas_gw.total_bounds[3], 4))
    #plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.2f}'.format(x)))
    #plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))
    
    # Add legend and title
    plt.legend(loc='lower left', title='PFAS Range (ng/l)')
   # plt.title(f'Predicted Sum of PFAS for {len(pfas_gw)} samples with range: {pfas_gw["pred_sum_PFAS"].min():.2f} - {pfas_gw["pred_sum_PFAS"].max():.2f}')
    ## turn of axes
    plt.axis('off')
    plt.grid(axis='both', linestyle='--', alpha=0.6)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f'figs/pred_{node_name}_sum_PFAS.png', dpi=600)
    plt.close()




def plot_pred_sum_pfas_kmeans(pfas_gw, node_name, n_clusters=5):
    
    assert "geometry" in pfas_gw.columns, "geometry column is missing in pfas_gw"
    assert len(pfas_gw) > 0, "pfas_gw is empty"
    assert "pred_sum_PFAS" in pfas_gw.columns, "pred_sum_PFAS column is missing in pfas_gw"
    
    # Load the boundaries
    bounds = pd.read_pickle("/data/MyDataBase/HuronRiverPFAS/Huron_River_basin_bound.pkl").to_crs("EPSG:4326")
    
    # Replace negative values with 0
    pfas_gw['pred_sum_PFAS'] = pfas_gw['pred_sum_PFAS'].clip(lower=0)

    # Perform K-means clustering on the PFAS predictions
    kmeans = KMeans(n_clusters=4)
    pfas_gw['pred_sum_PFAS_class'] = kmeans.fit_predict(pfas_gw[['pred_sum_PFAS']])
    
    # Sort the clusters by their mean values to ensure correct color assignment
    cluster_means = pfas_gw.groupby('pred_sum_PFAS_class')['pred_sum_PFAS'].mean().sort_values()
    sorted_clusters = cluster_means.index
    
    # Define color mappings based on sorted clusters
    class_colors = {sorted_clusters[0]: 'green', sorted_clusters[1]: 'blue', sorted_clusters[2]: 'orange', sorted_clusters[3]: 'red'}
    
    # Create range-based labels for each cluster
    class_labels = {}
    for class_value in sorted_clusters:
        class_group = pfas_gw[pfas_gw['pred_sum_PFAS_class'] == class_value]
        class_min = class_group['pred_sum_PFAS'].min()
        class_max = class_group['pred_sum_PFAS'].max()
        class_labels[class_value] = f'{class_min:.2f} - {class_max:.2f}'
    
    # Convert to GeoDataFrame and ensure correct CRS
    pfas_gw = gpd.GeoDataFrame(pfas_gw, geometry='geometry', crs='EPSG:26990').to_crs("EPSG:4326")
    
    # Plot setup
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    
    # Plot the boundaries
    bounds.boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)
    
    # Sort values for better visualization
    pfas_gw = pfas_gw.sort_values('pred_sum_PFAS', ascending=False)
    pfas_gw[['pred_sum_PFAS','geometry']].to_file('pfas_gw.geojson', driver='GeoJSON')
    # Plot each class with its assigned color and label in the correct order
    for class_value in sorted_clusters:
        color = class_colors[class_value]
        class_group = pfas_gw[pfas_gw['pred_sum_PFAS_class'] == class_value]
        if not class_group.empty:
            class_group.plot(ax=ax, color=color, edgecolor='black', linewidth=0.5, alpha=0.5, label=class_labels[class_value], markersize=60)
    
    # Set axis labels
    #plt.xlabel('Longitude')
    #plt.ylabel('Latitude')
    ## remove x and y axis
    plt.xticks([])
    plt.yticks([])
    ## turn off axes
    plt.axis('off')

    # Set axis ticks and format
    #plt.xticks(np.linspace(pfas_gw.total_bounds[0], pfas_gw.total_bounds[2], 4))
    #plt.yticks(np.linspace(pfas_gw.total_bounds[1], pfas_gw.total_bounds[3], 4))
    #plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
    #plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}'.format(y)))
    
    # Add legend and title
    plt.legend(loc='lower left', title='PFAS Range (ng/l)')
    #plt.title(f'Predicted Sum of PFAS for {len(pfas_gw)} samples with range: {pfas_gw["pred_sum_PFAS"].min():.2f} - {pfas_gw["pred_sum_PFAS"].max():.2f}')
    
    # Add grid
    plt.grid(axis='both', linestyle='--', alpha=0.6)
    #### add error metrics based on pred_sum_PFAS and sum_PFAS
    # Save the plot
    plt.tight_layout()  
    plt.savefig(f'figs/pred_{node_name}_sum_PFAS_kmeans_custom_ordered_colors.png', dpi=600)
    plt.close()


def calculate_error_metrics(pfas_gw, name='gw'):
    pred = pfas_gw[pfas_gw.sum_PFAS.isnull() == False]['pred_sum_PFAS']
    true = pfas_gw[pfas_gw.sum_PFAS.isnull() == False]['sum_PFAS']
    def NSE(pred, true):
        return 1 - np.sum((pred - true)**2) / np.sum((true - np.mean(true))**2)
    def RMSE(pred, true):
        return np.sqrt(np.mean((pred - true)**2))
    def MSE(pred, true):    
        return np.mean((pred - true)**2)
    def MAE(pred, true):
        return np.mean(np.abs(pred - true))
    nse = NSE(pfas_gw['pred_sum_PFAS'], pfas_gw['sum_PFAS'])
    rmse = RMSE(pfas_gw['pred_sum_PFAS'], pfas_gw['sum_PFAS'])
    mse = MSE(pfas_gw['pred_sum_PFAS'], pfas_gw['sum_PFAS'])
    mae = MAE(pfas_gw['pred_sum_PFAS'], pfas_gw['sum_PFAS'])

    ## scatter plot 
    plt.figure()
    plt.scatter(true, pred, alpha=0.5)
    plt.xlabel('True Sum of PFAS (ng/L)')
    plt.ylabel('Predicted Sum of PFAS (ng/L)')
    plt.title('Predicted vs True Sum of PFAS')
    plt.grid(axis='both', linestyle='--', alpha=0.6)
    plt.tight_layout()
    import time 
    plt.savefig(f'temp/{name}_scatter.png', dpi=300)
    return nse, rmse, mse, mae
def plot_predictions(train_target, train_pred, val_target, val_pred, test_target, test_pred, logger, node_name):
    import numpy as np
    import matplotlib.pyplot as plt

    assert len(train_target) == len(train_pred), "train_target and train_pred have different lengths"
    assert len(val_target) == len(val_pred), "val_target and val_pred have different lengths"
    assert len(test_target) == len(test_pred), "test_target and test_pred have different lengths"    
    
    logger.info(f"length of train_target: {len(train_target)}")
    logger.info(f"length of val_target: {len(val_target)}")
    logger.info(f"length of test_target: {len(test_target)}")
    logger.info(f"length of train_pred: {len(train_pred)}")
    logger.info(f"length of val_pred: {len(val_pred)}")
    logger.info(f"length of test_pred: {len(test_pred)}")

    # Convert tensors to numpy arrays for metric calculations
    train_target = train_target.cpu().numpy()
    train_pred = train_pred.cpu().numpy()
    val_target = val_target.cpu().numpy()
    val_pred = val_pred.cpu().numpy()
    test_target = test_target.cpu().numpy()
    test_pred = test_pred.cpu().numpy()


    # replace small values with a 1e-1
    train_target[train_target < 1e-1] = 1e-1
    train_pred[train_pred < 1e-1] = 1e-1
    val_target[val_target < 1e-1] = 1e-1
    val_pred[val_pred < 1e-1] = 1e-1
    test_target[test_target < 1e-1] = 1e-1
    test_pred[test_pred < 1e-1] = 1e-1






    def nse(pred, true):
        return 1 - np.sum((pred - true)**2) / np.sum((true - np.mean(true))**2)
    def rmse(pred, true):
        return np.sqrt(np.mean((pred - true)**2))
    def mse(pred, true):
        return np.mean((pred - true)**2)
    def mae(pred, true):
        return np.mean(np.abs(pred - true))

    train_nse = nse(train_pred, train_target)
    train_rmse = rmse(train_pred, train_target)
    train_mse = mse(train_pred, train_target)
    train_mae = mae(train_pred, train_target)

    val_nse = nse(val_pred, val_target)
    val_rmse = rmse(val_pred, val_target)
    val_mse = mse(val_pred, val_target)
    val_mae = mae(val_pred, val_target)

    test_nse = nse(test_pred, test_target)
    test_rmse = rmse(test_pred, test_target)
    test_mse = mse(test_pred, test_target)
    test_mae = mae(test_pred, test_target)

    logger.info("###################################################")
    logger.info(f"Number of train samples: {len(train_target)} with range: {train_target.min():.2f} - {train_target.max():.2f}")
    logger.info(f"Number of val samples: {len(val_target)} with range: {val_target.min():.2f} - {val_target.max():.2f}")
    logger.info(f"Number of test samples: {len(test_target)} with range: {test_target.min():.2f} - {test_target.max():.2f}")
    logger.info("###################################################")

    plt.figure()
    plt.scatter(train_target, train_pred, label='Training', alpha=0.5)
    plt.scatter(val_target, val_pred, label='Validation', alpha=0.5)
    plt.scatter(test_target, test_pred, label='Test', alpha=0.5)
    
    plt.title(f'Predicted vs True Sum of PFAS for\n{len(test_target)+len(val_target)+len(train_target)} water wells with unique {node_name}')
    plt.xlabel('True Sum of PFAS (ng/L)')
    plt.ylabel('Predicted Sum of PFAS (ng/L)')
    plt.legend(loc='lower right')
    plt.grid(axis='both', linestyle='--', alpha=0.6)
    if node_name == 'gw_wells':
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(1e-1, 1000)
        plt.ylim(1e-1, 1000)
    else:
        plt.xlim(0, 2000)
        plt.ylim(0, 2000)

    plt.annotate(f"Train:\nNSE: {train_nse:.2f}\nRMSE: {train_rmse:.2f}\nMSE: {train_mse:.2f}\nMAE: {train_mae:.2f}\n\nValidation:\nNSE: {val_nse:.2f}\nRMSE: {val_rmse:.2f}\nMSE: {val_mse:.2f}\nMAE: {val_mae:.2f}\n\nTest:\nNSE: {test_nse:.2f}\nRMSE: {test_rmse:.2f}\nMSE: {test_mse:.2f}\nMAE: {test_mae:.2f}", xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, ha='left', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    plt.tight_layout()
    plt.savefig(f'figs/{node_name}_predictions.png', dpi=300)
    plt.close()


def plot_rivers(rivs, logger):
    rivs.plot()
    #plt.xlabel('Longitude')
    #plt.ylabel('Latitude')
    ## remove x and y axis
    plt.xticks([])
    plt.yticks([])
    plt.title('Huron River Basin Rivers')
    plt.tight_layout()
    plt.savefig('figs/sw_rivers.png',dpi=300)
    plt.close()
def plot_histogram(all_dataframes):
    plt.figure()
    all_dataframes['train_loss'].plot.hist(alpha=0.5, bins=50, label='train_loss')
    all_dataframes['val_loss'].plot.hist(alpha=0.5, bins=50, label='val_loss')
    all_dataframes['test_loss'].plot.hist(alpha=0.5, bins=50, label='test_loss')
    plt.legend()
    plt.xlabel('Loss')
    plt.ylabel('Frequency')
    plt.xscale('symlog')  ## other options: log, symlog, linear
    plt.grid(axis='both', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('figs/loss_histogram.png', dpi=600)
    plt.close()

def plot_loss_curve(train_losses, val_losses, logger):
    logger.info(f"stage:plot_loss_curve ==##== Number of epochs: {len(train_losses)}")
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(axis='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('figs/loss_curve.png', dpi=600)
    plt.close()

