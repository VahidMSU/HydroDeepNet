import os
import numpy as np
import sys
from mpi4py import MPI
import hdf5_mpi
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
from torch import nn, optim
import matplotlib.pyplot as plt
from collections import OrderedDict
from GeoNet.global_var import get_var_name
from GeoNet.local_models import CustomPyTorchModel
from GeoNet.import_data import import_simulated_data
from GeoNet.utils import write_performance
from torch.optim.lr_scheduler import OneCycleLR
from GeoNet.viz import plot_scatter_density
import contextlib

class DeepLearningGWEstimation:
    
    def __init__(self, config):
        self.config = config
        self.database_path = config['database_path']
        self.fig_path = config['fig_path']
        self.trained_model_path = os.path.join(self.fig_path, f'FFR_{config["target_array"]}.pth')
        self.figure_scatter_path = os.path.join(self.fig_path, f'FFR_{config["target_array"]}.png')
        self.performance_metrics_path = os.path.join(self.fig_path, f'FFR_performance_metrics_{config["target_array"]}.txt')
        self.numerical_arrays = get_var_name('numerical', config['target_array'], config['RESOLUTION'])
        self.categorical_arrays = get_var_name('categorical', config['target_array'], config['RESOLUTION'])
        self.train_loader = None
        self.test_loader = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None


    def get_device(self):
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def get_mask(self):
        import h5py
        with h5py.File(self.database_path, 'r') as f:
            DEM_ = f[f'BaseRaster_{self.config["RESOLUTION"]}m'][:]
            return DEM_ != -999

    def plot_grid(self, array, name):
        mask = self.get_mask()
        array[(~mask)] = -999
        print(f"Writing {name} to HDF5 file")
        print(f'array size: {array.shape}')
        print(f'name: {name}')
        print(f'resolution: {self.config["RESOLUTION"]}m')
        print(f'datatype: {array.dtype}')
        hdf5_mpi.hdf5_operations.write_hdf5_file("ML.h5",f"{self.config['RESOLUTION']}" ,name, array)
        print(f"Writing {name} to PNG file")
        array[~mask] = np.nan
        vmin, vmax = np.nanpercentile(array, [2.5, 97.5])
        plt.imshow(array, vmin=vmin, vmax=vmax)

        plt.colorbar()
        plt.title(name)
        plt.savefig(os.path.join(self.fig_path, f'{name}.png'), dpi=300)
        plt.close()

    def predict_grid(self):
        all_data, target, _ = import_simulated_data(
            self.database_path, self.config['target_array'], self.numerical_arrays, 
            self.categorical_arrays, num_samples=-1, return_grid=True, RESOLUTION=self.config['RESOLUTION'],
            huc8=config.get('huc8', None)
        )
        model = CustomPyTorchModel(input_dim=len(self.categorical_arrays) + len(self.numerical_arrays))
        state_dict = torch.load(self.trained_model_path)
        new_state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())
        model.load_state_dict(new_state_dict)
        model.eval()
        device = self.get_device()
        model.to(device)

        predictions = []
        with torch.no_grad():
            for i in range(all_data.shape[0]):
                inputs = torch.tensor(all_data[i].astype(np.float32)).to(device)
                outputs = model(inputs)
                predictions.extend(outputs.cpu().numpy().flatten())

        predictions = np.array(predictions).reshape(target.shape)
        write_performance(self.performance_metrics_path, target, predictions, f'Final_{self.config["target_array"]}')
        self.plot_grid(predictions, f'FFR_Predicted {self.config["target_array"]}')


def train_and_predict(config):
    torch.cuda.manual_seed_all(42)
    estimator = DeepLearningGWEstimation(config)
    estimator.predict_grid()

if __name__ == '__main__':
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    RESOLUTIONS = [250]
    fig_path = "figs"
    for RESOLUTION in RESOLUTIONS:
        database_path = f"HydroGeoDataset_ML_{RESOLUTION}.h5"
        os.makedirs(fig_path, exist_ok=True)
        ## check if the file exists otherwise raise an error
        if not os.path.exists(database_path):
            raise FileNotFoundError(f"File {database_path} not found")
        config = {
            'database_path': database_path,
            'fig_path': fig_path,
            'EPOCH': 10,
            'batch_size': 5000,
            'max_batch_size': 50000,
            'opt_lr': 0.01,
            'fold_size': 5,
            'RESOLUTION': RESOLUTION,
            'num_samples': 250000,
            'improvement_threshold': 0.1,
            'lr_reduction_patience': 25,
            'rate_of_batch_size_increase': 1.1,
            'early_stopping_patience': 25,
            'lr_reduction_factor': 0.1,
            'batch_size_increase_step': 50,
            'max_norm': 25,
        }

        target_arrays = [
        #    f'obs_H_COND_1_{RESOLUTION}m',
            f'obs_H_COND_2_{RESOLUTION}m',
            f'obs_SWL_{RESOLUTION}m',
        #    f'obs_V_COND_1_{RESOLUTION}m',
        #   f'obs_V_COND_2_{RESOLUTION}m',
        #    f'obs_TRANSMSV_1_{RESOLUTION}m',
        #    f'obs_TRANSMSV_2_{RESOLUTION}m',
        #    f'obs_AQ_THK_1_{RESOLUTION}m',
        #    f'obs_AQ_THK_2_{RESOLUTION}m'
        ]

        if rank < len(target_arrays):
            config['target_array'] = target_arrays[rank]
            train_and_predict(config)

    MPI.Finalize()
