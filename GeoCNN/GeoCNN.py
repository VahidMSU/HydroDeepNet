import os
import time
import logging
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from GeoCNN.viz import plot_loss_over_epochs
from torch.nn import MSELoss
from GeoCNN.utils import setup_path, clean_input_figs, setup_gpu
from multiprocessing import Pool
from GeoCNN.utils import EarlyStopping
import torch
from GeoCNN.select_models import select_model
import logging
from GeoCNN.global_var import huc8_list_loader, county_list_loader
from GeoCNN.prediction import store_and_evaluate_predictions
from GeoCNN.import_data import data_loader
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 2, 3"
logging.basicConfig(level=logging.INFO)

class GeoClassCNN:
    def __init__(self, config, train_input=None, train_target=None, val_input=None, val_target=None, test_input=None, test_target=None, no_value=-999):
        
        # Initialize the model
        self.best_loss = float('inf')
        self.no_value = no_value
        self.config = config
        self.database_path = config['database_path']
        self.target_array = config['target_array']
        self.fig_path = "figs"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_model_path = f'models/best_model_{self.target_array}.pth'
        
        # Convert numpy arrays to PyTorch tensors
        try:
            self.train_input = train_input.clone().detach().float().to(self.device)
            self.train_target = train_target.clone().detach().float().squeeze().to(self.device)
            self.val_input = val_input.clone().detach().float().to(self.device)
            self.val_target = val_target.clone().detach().float().squeeze().to(self.device)
            self.test_input = test_input.clone().detach().float().to(self.device)
            self.test_target = test_target.clone().detach().float().squeeze().to(self.device)
        except:
            ### try without clone
            self.train_input = torch.tensor(train_input).float().to(self.device)
            self.train_target = torch.tensor(train_target).float().squeeze().to(self.device)
            self.val_input = torch.tensor(val_input).float().to(self.device)
            self.val_target = torch.tensor(val_target).float().squeeze().to(self.device)
            self.test_input = torch.tensor(test_input).float().to(self.device)
            self.test_target = torch.tensor(test_target).float().squeeze().to(self.device)
            
            
        # TensorDatasets
        self.train_set = TensorDataset(self.train_input, self.train_target)
        self.val_set = TensorDataset(self.val_input, self.val_target)
        self.test_set = TensorDataset(self.test_input, self.test_target)
        
        # Logging shapes
        self.log_shapes()
    def log_shapes(self):
        print(f"========================================")
        logging.info(f"Training input shape: {self.train_input.shape}, Training target shape: {self.train_target.shape}")
        logging.info(f"Validation input shape: {self.val_input.shape}, Validation target shape: {self.val_target.shape}")
        logging.info(f"Test input shape: {self.test_input.shape}, Test target shape: {self.test_target.shape}")
        print(f"========================================")
    def batch_loss_evaluation(self, data_batch, criterion=None, optimizer=None, mode='train'):
        """ 
        Evaluates the loss for the batch of data. 
        """
        batch_losses = 0.0
        outputs = []
        nse_vals, mse_vals, rmse_vals = [], [], []

        for batch_idx, (inputs, targets) in enumerate(data_batch):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            if mode == 'train':
                optimizer.zero_grad()
            # Create mask to exclude no_value and zero values
            mask = (targets != self.no_value) & (targets != 0)
            mask = mask.float().squeeze()  # Ensure mask is a float tensor

            valid_preds, valid_targets, preds = self._forward_pass(inputs, targets, mode, mask)

            if mode in ['train', 'val']:
                assert criterion is not None, "Criterion must be provided for training/validation."
                # Apply the mask before calculating the loss
                #print(f"Valid preds shape: {valid_preds.shape}")
                #print(f"Valid targets shape: {valid_targets.shape}")
                loss = criterion(valid_preds, valid_targets.float())
                if mode == 'train':
                    self._backpropagation(loss, optimizer)
                batch_losses += loss.item()

            if mode == 'predict':
                
                store_and_evaluate_predictions(self.config, valid_preds, valid_targets, preds, outputs, nse_vals, mse_vals, rmse_vals, batch_idx, mask)

        if mode in ['train', 'val']:
            return batch_losses / len(data_batch)
        elif mode == 'predict':
            return outputs, np.mean(nse_vals), np.mean(mse_vals), np.mean(rmse_vals)


    def _forward_pass(self, inputs, targets, mode, mask):

        with torch.set_grad_enabled(mode == 'train'):
            preds = self.model(inputs)
            #print(f"Predictions shape in forward pass: {preds.shape}")
            #time.sleep(100)
            # Filter valid outputs and targets
            valid_preds = preds.squeeze() * mask
            valid_targets = targets.squeeze() * mask

        return valid_preds, valid_targets, preds

    def _backpropagation(self, loss, optimizer):        
        loss.backward()
        clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        optimizer.step()

    def validate(self, epoch, val_loader, criterion):
        
        """
        Validates the model on the validation dataset.
        """

        self.model.eval()  # Switch to evaluation mode
        with torch.no_grad():
            val_loss = self.batch_loss_evaluation(val_loader, criterion, mode='val')
            
        return val_loss

    def train_model(self):
        """
        Trains the model and performs validation using early stopping.
        """
        # Initialize model
        self.model = select_model(self.config, number_of_channels=self.train_input.shape[2], device=self.device)
        # Define optimizer and scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['opt_lr'], weight_decay=self.config['weight_decay'])
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=self.config['threshold_patience'])
        ## use cosine annealing
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001)
        ## 
        #from torch.optim.lr_scheduler import ExponentialLR
        #scheduler = ExponentialLR(optimizer, gamma=0.9)  # gamma is the decay factor

        # DataLoaders for training and validation
        train_loader = DataLoader(self.train_set, batch_size=self.config['batch_size'], shuffle=False)
        val_loader = DataLoader(self.val_set, batch_size=self.config['batch_size'], shuffle=False)
        # Loss function
        criterion = MSELoss()
        # Early stopping mechanism
        early_stopping = EarlyStopping(patience=self.config['early_stopping_patience'])

        losses = []
        self.val_losses = []

        for epoch in range(self.config['EPOCH']):
            self.model.train()  # Switch to training mode

            # Training step
            epoch_loss = self.batch_loss_evaluation(train_loader, criterion, optimizer, mode='train')
            #scheduler.step(epoch_loss)
            scheduler.step(metrics=epoch_loss)

            losses.append(epoch_loss)

            # Validation step
            val_loss = self.validate(epoch, val_loader, criterion)
            self.val_losses.append(val_loss)

            # Early stopping check
            should_stop, is_best_model = early_stopping.check_early_stopping(val_loss)

            if is_best_model:
                self.best_loss = epoch_loss
                torch.save(self.model.state_dict(), self.best_model_path)  # Save best model

            if should_stop:
                logging.info(f"Early stopping triggered at epoch {epoch+1}")
                break

            if epoch % 5 == 0:
                logging.info(f'Epoch {epoch+1}/{self.config["EPOCH"]}, train loss: {epoch_loss:.2f}, val loss: {val_loss:.2f}, LR: {optimizer.param_groups[0]["lr"]}')

        # Plot loss over epochs
        plot_loss_over_epochs(losses, self.val_losses, self.target_array, self.fig_path)

    def predict(self):
        """
        Prediction function to generate predictions on the test dataset, batch by batch.
        """
        test_loader = DataLoader(self.test_set, batch_size=1, shuffle=False)

        # Load the final saved model
        self.model.load_state_dict(torch.load(self.best_model_path))

        # Perform batch evaluation for prediction
        outputs, nse_vals, mse_vals, rmse_vals = self.batch_loss_evaluation(test_loader, mode='predict')

        print(f"Average Prediction NSE: {nse_vals:.2f}, MSE: {mse_vals:.2f}, RMSE: {rmse_vals:.2f}")


def main(config):
    no_value = -999
    clean_input_figs()
    setup_path()
    setup_gpu(config['gpu_index'])
    train_input, train_target, val_input, val_target, test_input, test_target = data_loader(config)

    geo_cnn = GeoClassCNN(config, train_input, train_target, val_input, val_target, test_input, test_target, no_value=no_value)
    geo_cnn.train_model()
    geo_cnn.predict()






if __name__ == '__main__':

    RESOLUTION = 250
    TARGET_ARRAYS = ['recharge']  #f'obs_SWL_{RESOLUTION}m', f'kriging_output_SWL_{RESOLUTION}m',
    
    args_list = [

        {   'overwrite': True,
            'batch_window': 256,
            'embed_size': 2**10,
            'plotting': False,  # Set to True to plot the data
            'batch_size': 4,
            'target_array': target_array,
            'gpu_index': i,
            'database_path': f"/data/MyDataBase/HydroGeoDataset_ML_{RESOLUTION}.h5",
            'RESOLUTION': RESOLUTION,
            'list_of_regions': county_list_loader(),
            'EPOCH': 500,
            'opt_lr': 0.01,
            'early_stopping_patience': 200,
            'threshold_patience': 10,
            'weight_decay': 0.1,
            "num_heads": 8,
            "num_layers": 4,
            "forward_expansion": 4,
            "verbose": False,
            'model': 'CNNTransformerRegressor', #'CNNTransformerRegressor',
            'no_value': -999,
            'fig_path': 'figs',
            'dropout': 0.1,
            'start_year': 2004,
            'end_year': 2019,
            # 'SimpleCNN', 'AdvancedCNN', 'ModifiedResNet', 'ModifiedResNetUNet', 
            # 'TransformerCNN',AdvancedRegressorCNN
            # 'CNNTransformerRegressor'
        }

        for i, target_array in enumerate(TARGET_ARRAYS)
    ]
    

    parallel = False

    if parallel:
        with Pool(processes=1) as pool:
            pool.map(main, args_list)
    else:
        for args in args_list:
            main(args)
