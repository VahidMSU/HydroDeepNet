
import os
import glob
import logging
import pandas as pd
import numpy as np
import pandas as pd
import torch
import pandas as pd
from libs.plot_funs import plot_pred_sum_pfas, plot_sum_pfas, plot_pred_sum_pfas_with_colorbar, plot_pred_sum_pfas_with_log_colorbar, plot_pred_sum_pfas_kmeans

def cleanup_temp_files():
    ## all csv files
    files = glob.glob('temp/*.csv')
    for f in files:
        os.remove(f)

def setup_logging(path='GNN_gw_pfas.txt', verbose=True):
    # Clear the file at the start
    with open(path, 'w') as f:
        f.write('')

    # Get the root logger
    logger = logging.getLogger()

    # If the logger already has handlers, skip adding them again
    if not logger.hasHandlers():
        setup_logging_handlers(verbose, logger, path)
    return logger

def setup_logging_handlers(verbose, logger, path):
    if verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    # Create file handler which logs even debug messages
    fh = logging.FileHandler(path)
    fh.setLevel(logging.INFO if verbose else logging.ERROR)

    # Create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO if verbose else logging.ERROR)

    # Create formatter without timestamp and root
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    # Define error logger
    error_logger = logging.getLogger('error')
    error_logger.setLevel(logging.ERROR)
    error_logger.addHandler(fh)
    error_logger.addHandler(ch)


def get_features_string(gw_features):
    for feature in gw_features:
        geological_feature = "kriging" in feature
    if "DEM_250m" in gw_features and "kriging_output_SWL_250m" in gw_features and not geological_feature:
        return "SWL_DEM"
    elif "DEM_250m" in gw_features and "kriging_output_SWL_250m" not in gw_features and not geological_feature:
        return "DEM"
    elif "DEM_250m" not in gw_features and "kriging_output_SWL_250m" in gw_features and not geological_feature:
        return "SWL"
    elif "DEM_250m" in gw_features and "kriging_output_SWL_250m" in gw_features:
        return "SWL_DEM_lithological"
    elif "DEM_250m" in gw_features:
        return "DEM_lithological"
    elif "kriging_output_SWL_250m" in gw_features:
        return "SWL_lithological"
    

def cleanup_models():
    for f in glob.glob('models/*.pth'):
        os.remove(f)
    

def remove_torch_geometry_garbage():
    ##get current directory
    current_dir = os.getcwd()
    ## remove all files with torch_geometric.nn
    files = glob.glob(f'{current_dir}/torch_geometric.nn*')
    for f in files:
        os.remove(f)
    files = glob.glob(f'{current_dir}/__pycache__/torch_geometric.nn*')
    for f in files:
        os.remove(f)
   
def remove_predictions():
    os.makedirs('predictions_results', exist_ok=True)
    files = glob.glob('predictions_results/*.csv')
    for f in files:
        os.remove(f)



def logging_fitting_results(train_loss, val_loss, test_loss, train_target, train_pred, val_target, val_pred, test_target, test_pred, serial_number, logger):
    logger.info(f"######Serial Number: {serial_number}######")
    logger.info(f"######Training Loss: {train_loss:.2f}######")
    logger.info(f"######Validation Loss: {val_loss:.2f}######")
    logger.info(f"######Test Loss: {test_loss:.2f}######")

    def report_zero_and_small_values(tensor):
        numpy_array = tensor.cpu().numpy()
        zero_values = np.sum(numpy_array == 0)
        small_values = np.sum(numpy_array < 0.01)
        return zero_values, small_values

    data = []

    for name, target, pred in [('train', train_target, train_pred), ('val', val_target, val_pred), ('test', test_target, test_pred)]:
        target_zeros, target_smalls = report_zero_and_small_values(target)
        pred_zeros, pred_smalls = report_zero_and_small_values(pred)

        data.append({
            'Set': name,
            'Target Shape': target.cpu().numpy().shape,
            'Target Range Min': target.cpu().numpy().min(),
            'Target Range Max': target.cpu().numpy().max(),
            'Target Zero Values': target_zeros,
            'Target Small Values <0.01': target_smalls,
            'Pred Shape': pred.cpu().numpy().shape,
            'Pred Range Min': pred.cpu().numpy().min(),
            'Pred Range Max': pred.cpu().numpy().max(),
            'Pred Zero Values': pred_zeros,
            'Pred Small Values <0.01': pred_smalls
        })

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)

    # Print the DataFrame
    print(df)

    # Optionally, log the DataFrame (you can also save it to a file if needed)
    logger.info(f"\n{df}")

    return df




def save_predictions(pfas_df, train_pred, val_pred, test_pred, data, unsampled_pred, device, args, serial_number, node_name):


    # Concatenate predictions and WSSN values
    if node_name == 'gw_wells':
        
        # Get node indices for train, validation, and test samples
        train_gw_node_index = data[node_name].x[data[node_name].train_mask, 1].to(device)
        val_gw_node_index = data[node_name].x[data[node_name].val_mask, 1].to(device)
        test_gw_node_index = data[node_name].x[data[node_name].test_mask, 1].to(device)
        unsampled_node_index = data[node_name].x[data[node_name].unsampled_mask, 1].to(device)
        all_pred = torch.cat([train_pred, val_pred, test_pred, unsampled_pred], dim=0)
        all_wssn = torch.cat([train_gw_node_index, val_gw_node_index, test_gw_node_index, unsampled_node_index], dim=0)
        
        # Convert to DataFrame
        all_pred_wssn = pd.DataFrame({
            'gw_node_index': all_wssn.cpu().numpy().astype(int),  # Ensure WSSN is of type string
            'pred_sum_PFAS': all_pred.cpu().numpy().flatten()
        })

        #all_pred_wssn.to_csv('all_pred_gw_node_index.csv')
        pfas_df = pfas_df.merge(all_pred_wssn, on='gw_node_index', how='left')
        ## save pfas_df
        pfas_df[['WSSN','sum_PFAS', 'pred_sum_PFAS']].to_csv(f'predictions_results/pfas_df_pred_{serial_number}.csv', index=False, float_format='%.4f')
        
        if args.get("plot", False):
            try:
                plot_pred_sum_pfas_kmeans(pfas_df, node_name=node_name)
            except Exception as e:
                print(e)

            #plot_pred_sum_pfas_with_log_colorbar(pfas_df, node_name=node_name)
            #plot_pred_sum_pfas_with_colorbar(pfas_df, node_name=node_name)
            plot_pred_sum_pfas(pfas_df, node_name=node_name)
            plot_sum_pfas(pfas_df, node_name=node_name)
            
    elif node_name == 'sw_stations':
        # Get node indices for train, validation, and test samples
        train_sw_node_index = data[node_name].x[data[node_name].train_mask, 1].to(device)
        val_sw_node_index = data[node_name].x[data[node_name].val_mask, 1].to(device)
        test_sw_node_index = data[node_name].x[data[node_name].test_mask, 1].to(device)



        all_pred = torch.cat([train_pred, val_pred, test_pred], dim=0)
        all_wssn = torch.cat([train_sw_node_index, val_sw_node_index, test_sw_node_index], dim=0)

        # Convert to DataFrame
        all_pred_wssn = pd.DataFrame({
            'sw_node_index': all_wssn.cpu().numpy().astype(int),  # Ensure WSSN is of type string
            'pred_sum_PFAS': all_pred.cpu().numpy().flatten()
        })

        #all_pred_wssn.to_csv('all_pred_gw_node_index.csv')
        pfas_df = pfas_df.merge(all_pred_wssn, on='sw_node_index', how='left')

        ## save pfas_df

        pfas_df[['SiteCode','sum_PFAS', 'pred_sum_PFAS']].to_csv(f'predictions_results/pfas_sw_pred_{serial_number}.csv', index=False, float_format='%.4f')
        if args.get("plot", False):
            plot_pred_sum_pfas(pfas_df, node_name=node_name)
            plot_sum_pfas(pfas_df, node_name=node_name)