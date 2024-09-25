import contextlib
import os
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
from torch import nn, optim
import matplotlib.pyplot as plt
from multiprocessing import Process
from collections import OrderedDict
from GeoNet.global_var import get_var_name
from GeoNet.local_models import select_model
from GeoNet.import_data import import_simulated_data, get_huc8_ranges
from GeoNet.utils import write_performance
from torch.optim.lr_scheduler import OneCycleLR
import shutil
from GeoNet.viz import plot_scatter_density
from GeoNet.plotting import plot_loss, plot_grid

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train_and_predict(config):
    torch.cuda.manual_seed_all(42)
    estimator = DeepLearningGWEstimation(config)
    estimator.delete_previous_model()
    estimator.train_model()
    estimator.predict_grid()


def check_directories():
    paths = ['models', 'figs', 'report']
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)




class DeepLearningGWEstimation:
    
    def __init__(self, config):
        self.config = config
        self.database_path = config['database_path']
        self.fig_path = config['fig_path']
        self.trained_model_path = os.path.join("models", f'FFR_{config["target_array"]}.pth')
        self.figure_scatter_path = os.path.join("figs", f'FFR_{config["target_array"]}.png')
        self.performance_metrics_path = os.path.join("report", f'FFR_performance_metrics_{config["target_array"]}.txt')
        self.numerical_arrays = get_var_name('numerical', config['target_array'], config['RESOLUTION'], config['year'])
        self.categorical_arrays = get_var_name('categorical', config['target_array'], config['RESOLUTION'], config['year'])
        self.train_loader = None
        self.test_loader = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def delete_previous_model(self):
        paths = [self.trained_model_path, self.performance_metrics_path, self.figure_scatter_path]
        for path in paths:
            if os.path.exists(path):
                os.remove(path)
        with contextlib.suppress(Exception):
            os.remove('report/feature_stat.txt')
        target_figs = [f for f in os.listdir(self.fig_path) if self.config["target_array"] in f]
        for fig in target_figs:
            os.remove(os.path.join(self.fig_path, fig))

    def get_device(self):
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    def predict_grid(self):
        all_data, target, _ = import_simulated_data(
            self.database_path, self.config['target_array'], self.numerical_arrays, 
            self.categorical_arrays, num_samples=-1, return_grid=True, RESOLUTION=self.config['RESOLUTION'],
            huc8=self.config['huc8'], config=self.config
        )
        model = select_model(self.config, input_dim=len(self.categorical_arrays) + len(self.numerical_arrays)).to(self.get_device())
        
        state_dict = torch.load(self.trained_model_path, weights_only=True)   
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
        plot_grid(self.config, predictions, f'FFR_Predicted_{self.config["target_array"]}')
        
    def model_selection(self):
        device = self.get_device()
        model = select_model(self.config, input_dim=len(self.categorical_arrays) + len(self.numerical_arrays)).to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        return model, device

    def data_loader(self):
        data, target, group = import_simulated_data(
            self.database_path, self.config['target_array'], self.numerical_arrays, 
            self.categorical_arrays, self.config['num_samples'], self.config['RESOLUTION'], 
            self.config['huc8'], self.config
        )
        groups = np.unique(group)
        np.random.shuffle(groups)
        groups = np.array_split(groups, self.config['fold_size'])
        train_groups = np.concatenate(groups[:-1])
        test_groups = groups[-1]
        train_mask = np.isin(group, train_groups)
        test_mask = np.isin(group, test_groups)
        self.X_train, self.X_test = data[train_mask], data[test_mask]
        self.y_train, self.y_test = target[train_mask], target[test_mask]
        self.init_data_loaders()

    def init_data_loaders(self):
        train_dataset = TensorDataset(
            torch.tensor(self.X_train.astype(np.float32)), 
            torch.tensor(self.y_train.astype(np.float32)).view(-1, 1)
        )
        test_dataset = TensorDataset(
            torch.tensor(self.X_test.astype(np.float32)), 
            torch.tensor(self.y_test.astype(np.float32)).view(-1, 1)
        )
        self.train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=50)
        self.test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=50)

    def epoch_loss(self, model, device, criterion, optimizer, scheduler):

        best_loss = float('inf')
        losses = []

        for epoch in range(self.config['EPOCH']):
            model.train()
            running_loss = 0.0

            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(self.train_loader.dataset)
            scheduler.step()

            if epoch_loss < best_loss:
                torch.save(model.state_dict(), self.trained_model_path)
                best_loss = epoch_loss

            print(
                f'Epoch {epoch+1}, Loss: {epoch_loss:.2f}, Best Loss: {best_loss:.2f}, '
                f'Learning rate: {optimizer.param_groups[0]["lr"]}'
            )
            losses.append(epoch_loss)
        
        return best_loss, losses


    def train_model(self):
        self.data_loader()
        model, device = self.model_selection()
        optimizer = optim.AdamW(model.parameters(), lr=self.config['opt_lr'], weight_decay=self.config['weight_decay'])
        #criterion = nn.HuberLoss(delta=1.5, reduction='mean')
        ### mse for criterion
        criterion = nn.MSELoss()

        scheduler = OneCycleLR(optimizer, max_lr=self.config['opt_lr'], steps_per_epoch=len(self.train_loader), epochs=self.config['EPOCH'])
        best_loss, losses = self.epoch_loss(model, device, criterion, optimizer, scheduler)
        plot_loss(losses, self.config)
        model.load_state_dict(torch.load(self.trained_model_path, weights_only=True))
        
        model.eval()
        predictions, actuals = [], []

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(inputs)
                predictions.extend(outputs.cpu().numpy().flatten())
                actuals.extend(targets.cpu().numpy().flatten())

        if actuals and predictions:
            write_performance(self.performance_metrics_path, np.array(actuals), np.array(predictions), f'Test_{self.config["target_array"]}')
            plot_scatter_density(np.array(actuals), np.array(predictions), self.fig_path, f'FFR_{self.config["target_array"]}')
        else:
            print('Predictions or actuals are empty')


if __name__ == '__main__':
    check_directories()
    RESOLUTIONS = [250]#,30]
    YEAR = 2016
    fig_path = "figs"
    for RESOLUTION in RESOLUTIONS:
        database_path = f"/data/MyDataBase/HydroGeoDataset_ML_{RESOLUTION}.h5"
        os.makedirs(fig_path, exist_ok=True)

        config = {
            
            'database_path': database_path,
            'fig_path': fig_path,
            'EPOCH': 1000,
            'batch_size': 1000,
            'max_batch_size': 50000,
            'opt_lr': 0.01,
            'fold_size': 5,
            'RESOLUTION': RESOLUTION,
            'num_samples': 45000,
            'improvement_threshold': 0.01,
            'lr_reduction_patience': 25,
            'rate_of_batch_size_increase': 1.1,
            'early_stopping_patience': 25,
            'lr_reduction_factor': 0.1,
            'batch_size_increase_step': 10,
            'max_norm': 100,
            'year': YEAR,
            'huc8': '4100013',
            'weight_decay': 0.005,
            'model': 'DeepResidualMLP',

        }

        target_arrays = [
                        f'obs_H_COND_1_{RESOLUTION}m',
                    #    f'obs_H_COND_2_{RESOLUTION}m', 
                    #    f'obs_SWL_{RESOLUTION}m',
                    #    f'obs_V_COND_1_{RESOLUTION}m', 
                    #    f'obs_V_COND_2_{RESOLUTION}m',
                    #    f'obs_TRANSMSV_1_{RESOLUTION}m', 
                    #    f'obs_TRANSMSV_2_{RESOLUTION}m', 
                    #    f'obs_AQ_THK_1_{RESOLUTION}m',
                    #    f'obs_AQ_THK_2_{RESOLUTION}m'
                    #     f'recharge_{YEAR}_{RESOLUTION}m',
                    ]

        processes = []

        for target_array in target_arrays:
            config['target_array'] = target_array
            p = Process(target=train_and_predict, args=(config,))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()
