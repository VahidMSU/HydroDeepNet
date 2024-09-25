import os
import h5py
import numpy as np
import torch
try:
    from GeoCNN.global_var import get_var_name
except:
    from global_var import get_var_name 
class GeoTemporalDataLoader:
    def __init__(self, config):
        self.config = config
        self.numerical_var = get_var_name("numerical", config['RESOLUTION'])
        self.categorical_var = get_var_name("categorical", config['RESOLUTION'])
        self.all_feature_names = self.numerical_var + self.categorical_var
        self.target_var = config['target_array']
        self.years_of_target = [config['start_year'], config['end_year']]
        self.dataset_path = config['database_path']
        self.size = config['batch_window']
        self.overwrite = config['overwrite']
        assert os.path.exists(self.dataset_path), "H5 file not found"

        # Load the dataset and process the data
        self.all_train_features, self.all_train_targets, self.all_val_features, self.all_val_targets, self.all_test_features, self.all_test_targets = self.process()

    def data_loader(self, f, batch_region, mask, unique_values):
        num_years = self.years_of_target[1] - self.years_of_target[0] + 1
        num_features = len(self.numerical_var) + len(self.categorical_var) + 1  # +1 for ppt

        # Use lists to store valid features and targets
        valid_features = []
        valid_targets = []

        for batch_idx, value in enumerate(unique_values):
            # Get the region for each unique batch
            region = np.where(batch_region == value)
            min_x, max_x = region[0].min(), region[0].max() + 1
            min_y, max_y = region[1].min(), region[1].max() + 1
            local_mask = mask[min_x:max_x, min_y:max_y]

            # Skip regions that are not 64x64
            if (max_x - min_x != self.size) or (max_y - min_y != self.size):
                print(f"Skipping batch {batch_idx + 1}/{len(unique_values)} due to size mismatch: ({max_x - min_x}, {max_y - min_y})")
                continue

            batch_features = torch.zeros((num_years, num_features, self.size, self.size))
            batch_targets = torch.zeros((num_years, 1, self.size, self.size))

            valid_batch = True  # To track if the batch is valid

            # Check the recharge data for the first year
            first_year = self.years_of_target[0]
            if "recharge" in self.target_var:
                var_recharge_first_year = f"recharge_{first_year}_250m"
                recharge_data_first_year = f[var_recharge_first_year][min_x:max_x, min_y:max_y]
                recharge_data_first_year[local_mask != 1] = 0  # Set no-data regions to 0

                # Check if more than 80% of the target data is zero for the first year
                if np.sum(recharge_data_first_year == 0) > 0.8 * recharge_data_first_year.size:
                    print(f"Skipping entire batch {batch_idx + 1}/{len(unique_values)} due to more than 80% zeros in the first year ({first_year})")
                    continue  # Skip the entire batch if the first year fails the check

            # Process subsequent years if the first year passed the check
            for year_idx, year in enumerate(range(self.years_of_target[0], self.years_of_target[1] + 1)):
                if "recharge" in self.target_var:
                    var_recharge = f"recharge_{year}_250m"
                    recharge_data = f[var_recharge][min_x:max_x, min_y:max_y]
                    recharge_data[local_mask != 1] = 0  # Set no-data regions to 0

                var_ppt = f"ppt_{year}_250m"
                ppt_data = f[var_ppt][min_x:max_x, min_y:max_y]
                ppt_data[local_mask != 1] = 0  # Set no-data regions to 0
                ## divide ppt by 365
                ppt_data = ppt_data / 365

                if ppt_data.shape != (self.size, self.size):
                    print(f"Skipping ppt data for year {year} due to incorrect shape: {ppt_data.shape}")
                    valid_batch = False
                    break  # Skip the entire batch if this year fails the check

                batch_features[year_idx, 0, :, :] = torch.tensor(ppt_data)

                # Extract numerical and categorical features for each year
                for feature_idx, var in enumerate(self.all_feature_names):
                    data = f[var][min_x:max_x, min_y:max_y]
                    data[local_mask != 1] = 0  # Set no-data regions to 0

                    if data.shape != (self.size, self.size):
                        print(f"Skipping feature {var} due to incorrect shape: {data.shape}")
                        valid_batch = False
                        break  # Skip the entire batch if this feature fails the check

                    batch_features[year_idx, feature_idx + 1, :, :] = torch.tensor(data)

                # Save the valid recharge target
                batch_targets[year_idx, 0, :, :] = torch.tensor(recharge_data)

            if valid_batch:
                valid_features.append(batch_features)
                valid_targets.append(batch_targets)

        # Stack the valid batches into tensors
        if valid_features and valid_targets:
            all_features = torch.stack(valid_features, dim=0)
            all_targets = torch.stack(valid_targets, dim=0)
        else:
            all_features = torch.empty(0)  # Return empty tensors if no valid data found
            all_targets = torch.empty(0)

        return all_features, all_targets


    def process(self):
        if self.overwrite: 
            with h5py.File(self.dataset_path, 'r') as f:
                batch_region = f[f"{self.size}_{self.size}_batch_size"][:]  # Batch regions (self.sizexself.size blocks)
                mask = f['BaseRaster_250m'][:]  # Mask for valid regions
                unique_values = np.unique(batch_region)
                no_value = -999
                unique_values = unique_values[unique_values != no_value]  # Remove no_value entries

                train_size = int(0.7 * len(unique_values))
                val_size = int(0.2 * len(unique_values))
                test_size = len(unique_values) - train_size - val_size
                train_values, val_values, test_values = np.split(unique_values, [train_size, train_size + val_size])

                all_train_features, all_train_targets = self.data_loader(f, batch_region, mask, train_values)
                all_val_features, all_val_targets = self.data_loader(f, batch_region, mask, val_values)
                all_test_features, all_test_targets = self.data_loader(f, batch_region, mask, test_values)
            ## save the data in numpy 
            os.makedirs("ml_data", exist_ok=True)
            np.save("ml_data/all_train_features.npy", all_train_features)
            np.save("ml_data/all_train_targets.npy", all_train_targets)
            np.save("ml_data/all_val_features.npy", all_val_features)
            np.save("ml_data/all_val_targets.npy", all_val_targets)
            np.save("ml_data/all_test_features.npy", all_test_features)
            np.save("ml_data/all_test_targets.npy", all_test_targets)


        elif os.path.exists("ml_data/all_test_targets.npy"):
            all_train_features = np.load("ml_data/all_train_features.npy")
            all_train_targets = np.load("ml_data/all_train_targets.npy")
            all_val_features = np.load("ml_data/all_val_features.npy")
            all_val_targets = np.load("ml_data/all_val_targets.npy")
            all_test_features = np.load("ml_data/all_test_features.npy")
            all_test_targets = np.load("ml_data/all_test_targets.npy")

        print(f"Train features: {all_train_features.shape}, Train targets: {all_train_targets.shape}")
        print(f"Validation features: {all_val_features.shape}, Validation targets: {all_val_targets.shape}")
        print(f"Test features: {all_test_features.shape}, Test targets: {all_test_targets.shape}")
        

        return all_train_features, all_train_targets, all_val_features, all_val_targets, all_test_features, all_test_targets


def data_loader(config):
    data_loader = GeoTemporalDataLoader(config)
    return data_loader.process()



if __name__ == "__main__":
    config = {
        'overwrite': True,
        'target_array': "recharge",
        'database_path': f"/data/MyDataBase/HydroGeoDataset_ML_250.h5",
        'RESOLUTION': 250,
        'no_value': -999,
        'fig_path': 'figs',
        'start_year': 2004,
        'end_year': 2019,
        'batch_window': 256
    }

    all_train_features, all_train_targets, all_val_features, all_val_targets, all_test_features, all_test_targets = data_loader(config)
