import os
from requests import delete
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import DataParallel
from GeoCNN_LSTM.viz import plot_scatter_density, plot_grid, plot_loss_over_epochs
from torch.optim.lr_scheduler import OneCycleLR
from GeoCNN_LSTM.utils import write_performance, setup_gpu
from GeoCNN_LSTM.import_data import DataImporter
from GeoCNN_LSTM.local_models import CNN_LSTM as DEEP_LEARNING_MODEL
import h5py
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from GeoCNN_LSTM.obj_functions import criterion

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class GeoCNN:
    
    """ 
        aim of deep learning: predict recharge from annual rainfall data and terrain features static data.
    """

    def __init__(self, config):
        self.config = config
        self.database_path = config['database_path']
        self.model_path = config['model_path']
        self.performance_metrics_path = os.path.join("report", f'CNN_performance_metrics_{config["huc8"]}_recharge_predicted.txt')
        self.model = None
        self.fig_path = "figs"
        self.device = torch.device('cuda')
        self.model = None
        self.trained_model_path = f"{self.model_path}/CNN_{self.config['huc8']}_recharge.pth"
    def delete_model(self):
        current_dir = os.getcwd()
        if os.path.exists(os.path.join(current_dir,self.trained_model_path)):
            os.remove(os.path.join(current_dir,self.trained_model_path))    
            print(f"Deleted model at {self.trained_model_path}")
        else:
            print(f"No saved model found at {self.trained_model_path}")
    def load_best_model(self):
        model_path = self.trained_model_path
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            print(f"Loaded best model from {model_path}")
        else:
            print(f"No saved model found at {model_path}")

    def predict(self):
        self.load_best_model()
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(self.combined_input_test).squeeze(0)
            print(f'########## size of predictions: {predictions.shape}')  # torch.Size([7, 1, 553, 367])
        
        plot_grid(self.fig_path, np.where(self.mask.cpu().numpy() == 0, np.nan, predictions.cpu().numpy()), f"CNN_Prediction_{self.config['huc8']}_recharge")

        target_array = self.target_tensor_test.cpu().numpy().squeeze()
        predictions_array = predictions.cpu().numpy().squeeze()
        valid_indices = self.mask.cpu().numpy().squeeze() == 1
        target_array_valid = target_array[:, valid_indices]
        predictions_array_valid = predictions_array[:, valid_indices]
        write_performance(self.performance_metrics_path, target_array_valid, predictions_array_valid, 'Final')
        plot_scatter_density(target_array_valid, predictions_array_valid, self.fig_path, f"CNN_{self.config['huc8']}_recharge")

    def import_data(self):
        """ 
            Combined input shape (train): torch.Size([1, 9, 12, 553, 367])
            Target tensor shape (train): torch.Size([1, 9, 1, 553, 367])
            Combined input shape (test): torch.Size([1, 7, 12, 553, 367])
            Target tensor shape (test): torch.Size([1, 7, 1, 553, 367])
            Mask shape: torch.Size([553, 367])
            Note: The recharge can be -999, but we have static and rainfall data and we need to predict recharge.
            where mask is 1, we need to predict recharge and where mask is 0, we do not need any prediction.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        importer = DataImporter(self.config, device)
        self.combined_input_train, self.target_tensor_train, self.combined_input_test, self.target_tensor_test, self.dataset_train, self.mask = importer.recharge_rainfall_ds()
        ### update model parameter
        print(f"Combined input shape (train): {self.combined_input_train.shape}")
        import time 
        #time.sleep(50)
        
        self.model = DEEP_LEARNING_MODEL(number_of_channels=self.combined_input_train.size(2), height=self.combined_input_train.size(3), width=self.combined_input_train.size(4))
        self.model = self.model.to(self.device)
    def epoch_train(self):
        losses = []
        best_loss = float('inf')
        early_stopping_counter = 0

        dataloader = DataLoader(self.dataset_train, shuffle=True)

        for epoch in range(self.config['EPOCH']):
            self.model.train()
            running_loss = 0.0

            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)

                loss = criterion(self.mask, outputs, targets)
                loss.backward()
                # Apply gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_norm'])
                self.optimizer.step()
                self.optimizer.zero_grad()
                running_loss += loss.item()

            epoch_loss = running_loss / len(dataloader)
            self.scheduler.step()
            print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.2f}', "lr:", f"{self.scheduler.get_last_lr()[0]:.5f}", "early stopping counter:", early_stopping_counter)
            losses.append(epoch_loss)
            if epoch_loss < best_loss:
                torch.save(self.model.state_dict(), self.trained_model_path)
                best_loss = epoch_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= self.config['early_stopping_patience'] and abs(epoch_loss - best_loss) < self.config['early_stopping_threshold']:
                print('###################################################')
                print(f'##### Early stopping at epoch {epoch + 1} ###########')
                print('###################################################')
                break
        return losses

    def train_model(self):
        self.optimizer = AdamW(self.model.parameters(), lr=self.config['opt_lr'], weight_decay=1e-5)
        self.scheduler = OneCycleLR(self.optimizer, max_lr=10 * self.config['opt_lr'], steps_per_epoch=len(self.dataset_train), epochs=self.config['EPOCH'])
        
        losses = self.epoch_train()

        write_performance(
            self.performance_metrics_path,
            self.target_tensor_train.detach().cpu().numpy().squeeze(),
            self.model(self.combined_input_train).detach().squeeze().cpu().numpy(),
            'Train'
        )
        plot_loss_over_epochs(losses, self.config['huc8'], self.fig_path)

def main(config):
    setup_gpu(config['gpu_index'])
    geo_cnn = GeoCNN(config)
    geo_cnn.delete_model()
    geo_cnn.import_data()
    geo_cnn.train_model()
    geo_cnn.predict()

if __name__ == '__main__':
    main(
        {
            'gpu_index': 0,
            'database_path': '/data/MyDataBase/HydroGeoDataset_ML_250.h5',
            'RESOLUTION': 250,
            'EPOCH': 200,
            'opt_lr': 0.001,
            'start_training_year': 2004,
            'end_training_year': 2012,
            'start_testing_year': 2013,
            'end_testing_year': 2019,
            'early_stopping_patience': 20,
            'early_stopping_threshold': 0.01,
            'max_norm': 10,
            'batch_size': 1,
            'model_path': "models",
            'huc8': '4060105',
            'LSTM': True,

            'snow': True,
            'geoloc': True,
            'groundwater': True,
           # 'population_array': True,
            'landfire': True,
            'geology': True,
            'NHDPlus': True,
            'plot': True,


        }
    )
