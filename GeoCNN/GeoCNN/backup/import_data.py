import h5py
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import logging
import torch
import torch
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from GeoCNN.global_var import get_var_name
    from GeoCNN.viz import plot_feature
except:
    from global_var import get_var_name
    from viz import plot_feature

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_years(year_range, huc8, config, geodata, all_inputs, all_targets):
    for year in year_range:
        result = process_huc8_data(huc8, config, geodata, year)
        if result is not None:
            target_array, inputs = result
            all_inputs.append(inputs)
            all_targets.append(target_array)
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
import torch
def data_loader(config):
    # Split HUC8 regions into train, validation, and test sets
    train_huc8, temp_huc8 = train_test_split(config['list_of_huc8'], test_size=0.5, random_state=42)
    val_huc8, test_huc8 = train_test_split(temp_huc8, test_size=0.5, random_state=42)

    # Initialize data lists
    all_train_inputs, all_train_targets = [], []
    all_val_inputs, all_val_targets = [], []
    all_test_inputs, all_test_targets = [], []

    geodata = GeoDataProcessor(config)
    def process_and_store_data(huc8_list, year_range, all_inputs, all_targets):
        for huc8 in huc8_list:
            inputs_by_year = []
            targets_by_year = []
            for year in year_range:
                result = process_huc8_data(huc8, config, geodata, year)
                if result is not None:
                    target_array, inputs = result
                    inputs_by_year.append(inputs)  # Append inputs for this year
                    targets_by_year.append(target_array)  # Append target for this year

            # Check if there are any valid inputs/targets for this HUC8 region
            if inputs_by_year and targets_by_year:
                # Stack data across the time_steps dimension (years)
                inputs_by_year = torch.stack(inputs_by_year, dim=0)  # Shape: [time_steps, channels, height, width]
                targets_by_year = torch.stack(targets_by_year, dim=0)  # Shape: [time_steps, 1, height, width]

                all_inputs.append(inputs_by_year)
                all_targets.append(targets_by_year)
            else:
                logging.info(f"Skipping HUC8 {huc8} as it contains no valid data for the specified years.")
        return all_inputs, all_targets        

    # Process training, validation, and test sets sequentially
#    process_and_store_data(train_huc8, range(2016, 2020), all_train_inputs, all_train_targets)
#    process_and_store_data(val_huc8, range(2016, 2020), all_val_inputs, all_val_targets)
#    process_and_store_data(test_huc8, range(2016, 2020), all_test_inputs, all_test_targets)
    all_train_inputs, all_train_targets = process_and_store_data(train_huc8, range(2016, 2020), all_train_inputs, all_train_targets)    
    all_val_inputs, all_val_targets = process_and_store_data(val_huc8, range(2016, 2020), all_val_inputs, all_val_targets)  
    all_test_inputs, all_test_targets = process_and_store_data(test_huc8, range(2016, 2020), all_test_inputs, all_test_targets)
    from multiprocessing import Pool    



    # Convert lists to tensors and reshape
    all_train_inputs = torch.stack(all_train_inputs, dim=0)  # Shape: [batch_size, time_steps, channels, height, width]
    all_train_targets = torch.stack(all_train_targets, dim=0)  # Shape: [batch_size, time_steps, 1, height, width]
    
    all_val_inputs = torch.stack(all_val_inputs, dim=0)
    all_val_targets = torch.stack(all_val_targets, dim=0)
    
    all_test_inputs = torch.stack(all_test_inputs, dim=0)
    all_test_targets = torch.stack(all_test_targets, dim=0)

    # Reshape the target to be independent of time steps (for regression task):
    all_train_targets = all_train_targets[:, -1, ...]  # Only take the final time step target for training
    all_val_targets = all_val_targets[:, -1, ...]
    all_test_targets = all_test_targets[:, -1, ...]

    return all_train_inputs, all_train_targets, all_val_inputs, all_val_targets, all_test_inputs, all_test_targets


def process_huc8_data(huc8, config, geodata, year):
    config['huc8'] = huc8
    target, numerical_data, categorical_data, groups = geodata.import_data(year)
    target_array, inputs = geodata.prepare_data(target, numerical_data, categorical_data)
    
    if target_array is None:
        return None

    inputs = torch.from_numpy(inputs).float().unsqueeze(0)  # Add batch dimension (channel)
    target_array = torch.from_numpy(target_array).float().unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    return target_array, inputs


def process_huc8_data(huc8, config, geodata, year):
    config['huc8'] = huc8
    target, numerical_data, categorical_data, groups = geodata.import_data(year)
    target_array, inputs = geodata.prepare_data(target, numerical_data, categorical_data)
    
    if target_array is None:
        return None

    inputs = torch.from_numpy(inputs).float().unsqueeze(0)  # Add batch dimension
    target_array = torch.from_numpy(target_array).float().unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    return target_array, inputs

class GeoDataProcessor:
    def __init__(self, config):
        self.report_stat = "report/feature_stat.txt",
        self.resolution = config['RESOLUTION']
        self.divisor = config['divisor']
        self.config = config
        self.no_value = config['no_value']
        os.makedirs("input_figs", exist_ok=True)


    def target_apply_numerical_scale(self, array2d, array_name):
        logging.info(f"Processing numerical feature: {array_name}, shape: {array2d.shape}")

        array2d = np.where(array2d <= 0, self.no_value, array2d)

        return array2d


    def apply_numerical_scale(self, array2d, array_name):
        logging.info(f"Processing numerical feature: {array_name}, shape: {array2d.shape}")

        # Mark entries where array2d is <= 0 as no_value
        if "ppt" in array_name:
            array2d = np.where(array2d <= 0, self.no_value, array2d/365)
        else:
            array2d = np.where(array2d <= 0, self.no_value, array2d)


        return array2d

    def get_mask(self, database_path, huc8_select):
        with h5py.File(database_path, 'r') as f:
            mask = f[f'BaseRaster_{self.resolution}m'][:]
            mask = np.where(mask == self.no_value, 0, 1)
            return mask != 0

    def apply_categorical_encoding(self, array2d, array_name):
        logging.info(f"Processing categorical feature: {array_name}, shape: {array2d.shape}")
        logging.info(f"Unique values: {np.unique(array2d)}")
        ## starting from 1
        le = LabelEncoder()
        array2d = le.fit_transform(array2d.ravel()).reshape(array2d.shape)

        return array2d
    def get_huc8_ranges(self, database_path, huc8_select=False):
        with h5py.File(database_path, 'r') as f:
            huc8 = np.array(f[f'COUNTY_{self.resolution}m'][:])
        
        print(f"All unique huc8 values: {np.unique(huc8)}")
        
        # Get the indices of cells that match the selected huc8
        rows, cols = np.where(huc8 == int(huc8_select))
        
        # Find the center point of the selected HUC8 region
        center_row = (rows.max() + rows.min()) // 2
        center_col = (cols.max() + cols.min()) // 2
        
        logging.info(f"Center of HUC8 region - Row: {center_row}, Col: {center_col}")
        
        # Define the desired fixed size
        desired_rows = self.config['desired_rows'] + 1
        desired_cols = self.config['desired_cols'] + 1
        
        # Calculate half the size to expand on each side
        half_rows = desired_rows // 2
        half_cols = desired_cols // 2
        
        # Calculate the min and max rows and columns by expanding symmetrically
        row_min = max(0, center_row - half_rows)
        row_max = min(huc8.shape[0] - 1, center_row + half_rows)
        
        col_min = max(0, center_col - half_cols)
        col_max = min(huc8.shape[1] - 1, center_col + half_cols)
        
        # Adjust row size if it doesn't meet the desired number of rows
        current_row_size = row_max - row_min + 1
        if current_row_size < desired_rows:
            if row_min == 0:
                row_max = min(huc8.shape[0] - 1, row_max + (desired_rows - current_row_size))
            elif row_max == huc8.shape[0] - 1:
                row_min = max(0, row_min - (desired_rows - current_row_size))

        # Adjust column size if it doesn't meet the desired number of columns
        current_col_size = col_max - col_min + 1
        if current_col_size < desired_cols:
            if col_min == 0:
                col_max = min(huc8.shape[1] - 1, col_max + (desired_cols - current_col_size))
            elif col_max == huc8.shape[1] - 1:
                col_min = max(0, col_min - (desired_cols - current_col_size))

        # Check if the resulting sizes are greater than desired and adjust accordingly
        if row_max - row_min + 1 > desired_rows:
            row_max = row_min + desired_rows - 1
        
        if col_max - col_min + 1 > desired_cols:
            col_max = col_min + desired_cols - 1

        # Ensure that the resulting size is correct
        assert row_max - row_min + 1 == desired_rows, f"Row size is not {desired_rows}: {row_max - row_min + 1}"
        assert col_max - col_min + 1 == desired_cols, f"Col size is not {desired_cols}: {col_max - col_min + 1}"

        logging.info(f"Final HUC8 range - Row: {row_min} to {row_max}, Col: {col_min} to {col_max}")
        
        return row_min, row_max, col_min, col_max


    def import_simulated_data(self, database_path, target_array, numerical_arrays, categorical_arrays, huc8=None, pfas_database_path=None, year=None):
        logging.info(f"Importing data from: {database_path} with resolution {self.resolution}m")

        if huc8:
            row_min, row_max, col_min, col_max = self.get_huc8_ranges(database_path, huc8)
        else:
            row_min, row_max, col_min, col_max = 0, -1, 0, -1


        with h5py.File(database_path, 'r') as f:
            if pfas_database_path:
                with h5py.File(pfas_database_path, 'r') as f_pfas:
                    pfas = np.array(f_pfas["/Max/PFOS"][:][row_min:row_max, col_min:col_max])

            if target_array:
                if "recharge" in target_array:
                    target_array = f"recharge_{year}_{self.resolution}m"
                target = self.target_apply_numerical_scale(np.array(f[target_array][:][row_min:row_max, col_min:col_max]), "TARGET VAR")
            else:
                target = None

            ### if ppt in numerical arrays, rename it to ppt_{year}_{resolution}m
            for i, array_name in enumerate(numerical_arrays):
                if "ppt" in array_name:
                    print(f"ppt in array_name: {array_name}")   
                    numerical_arrays[i] = f"ppt_{year}_{self.resolution}m"

            numerical_data = [

                    self.apply_numerical_scale(np.array(f[array_name][:][row_min:row_max, col_min:col_max]), array_name)

                for array_name in numerical_arrays
            ]

            categorical_data = [

                    self.apply_categorical_encoding(np.array(f[array_name][:][row_min:row_max, col_min:col_max]), array_name)
                for array_name in categorical_arrays
            ]

            groups = self.apply_categorical_encoding(np.array(f[f"COUNTY_{self.resolution}m"][:][row_min:row_max, col_min:col_max]),
                                                f"COUNTY_{self.resolution}m")


        ### apply mask
        mask = self.get_mask(database_path, huc8)[row_min:row_max, col_min:col_max]
        if self.config['plotting']:
            plot_feature(mask, "Mask", no_value=self.no_value)


        target = np.where(mask, target, self.no_value)
        if self.config['plotting']:
            plot_feature(target, "TARGET VAR", no_value=self.no_value)
        logging.info(f"Number of values in TARGET VAR (except no_value): {np.sum(target != self.no_value)}")
        numerical_data = [np.where(mask, array, self.no_value) for array in numerical_data]
        categorical_data = [np.where(mask, array, self.no_value) for array in categorical_data]
        groups = np.where(mask, groups, self.no_value)

        if self.config['plotting']:
            ## now plot all the features
            for array_name, array2d in zip(numerical_arrays + categorical_arrays, numerical_data + categorical_data):
                plot_feature(array2d, array_name, no_value=self.no_value)

        return target, numerical_data, categorical_data, groups

    
    def prepare_data(self, target, numerical_data, categorical_data):
        # Check if the entire target array is no_value
        if np.all(target == self.no_value) or np.all(target == 0) or np.all(target == np.nan):
            logging.info("Target array contains all no_value entries. Skipping this batch.")
            return None, None

        valid_numerical_data = []
        valid_categorical_data = []

        # Filter out any numerical arrays that are fully no_value
        for array in numerical_data:
            if not np.all(array == self.no_value):
                valid_numerical_data.append(array)
            else:
                logging.info(f"Excluding numerical array with all no_value entries: shape {array.shape}")

        # Filter out any categorical arrays that are fully no_value
        for array in categorical_data:
            if not np.all(array == self.no_value):
                valid_categorical_data.append(array)
            else:
                logging.info(f"Excluding categorical array with all no_value entries: shape {array.shape}")

        # Convert valid numerical and categorical data into numpy arrays
        numerical_tensors = [np.array(nd, dtype=np.float32) for nd in valid_numerical_data]
        categorical_tensors = [np.array(cd, dtype=np.float32) for cd in valid_categorical_data]

        # Combine numerical and categorical data into a single 3D input: [channels, height, width]
        combined_input = np.stack(numerical_tensors + categorical_tensors, axis=0)
        
        logging.info(f"Combined input shape (after filtering): {combined_input.shape}")

        # Convert target to numpy array and add a new axis for batch dimension
        target_array = np.array(target, dtype=np.float32)[np.newaxis, ...]

        return target_array, combined_input

    def import_data(self, year):
        
        numerical_arrays = get_var_name("numerical", self.config['target_array'], self.config['RESOLUTION'])
        categorical_arrays = get_var_name("categorical", self.config['target_array'], self.config['RESOLUTION'])
        # Import data along with group information
        target, numerical_data, categorical_data, groups = self.import_simulated_data(
            database_path=self.config['database_path'],
            target_array=self.config['target_array'],
            numerical_arrays=numerical_arrays,
            categorical_arrays=categorical_arrays,
            huc8=self.config['huc8'],
            year=year
        )

        target[target == self.no_value] = 0  # Replace any missing data if necessary
        return target, numerical_data, categorical_data, groups

if __name__ == "__main__":
    
    RESOLUTION = 250
    from global_var import huc8_list_loader, county_list_loader


    config = {
        
        'desired_rows': 64,
        'desired_cols': 64,
            'plotting': False,  # Set to True to plot the data
            'target_array': "recharge",
            'database_path': f"/data/MyDataBase/HydroGeoDataset_ML_{RESOLUTION}.h5",
            'RESOLUTION': RESOLUTION,
            'list_of_huc8': county_list_loader(),
            'divisor': 16,
            'no_value': -999,
            'fig_path': 'figs',
        }

    ## use data loader
    all_train_input, all_train_target, all_val_input, all_val_target, all_test_input, all_test_target = data_loader(config)
    print(all_train_input.shape, all_train_target.shape, all_val_input.shape, all_val_target.shape, all_test_input.shape, all_test_target.shape)