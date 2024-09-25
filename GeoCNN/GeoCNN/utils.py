
import os 
import torch
import logging
import numpy as np


def setup_path():
    os.makedirs('models', exist_ok=True)
    os.makedirs('figs', exist_ok=True)

def clean_input_figs():
    os.system('rm -rf figs/*')
    os.system('rm -rf models/*')
    os.system('rm -rf input_figs/*')
    os.system('rm -rf figs_scatters/*')

def setup_gpu(gpu_index):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_index)
    if torch.cuda.is_available():
        logging.info(f"Process on GPU: {torch.cuda.current_device()} with index {gpu_index}")
    else:
        logging.info("CUDA is not available.")


def calculate_metrics(target_array, predictions_array):
    def nse(obs, sim):
        return 1 - np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)
    def mse(obs, sim):
        return np.mean((obs - sim) ** 2)
    def rmse(obs, sim):
        return np.sqrt(np.mean((obs - sim) ** 2))
    
    nse_val = nse(target_array.flatten(), predictions_array.flatten())
    mse_val = mse(target_array.flatten(), predictions_array.flatten())
    rmse_val = rmse(target_array.flatten(), predictions_array.flatten())
    return nse_val, mse_val, rmse_val



class EarlyStopping:
    """
    Implements early stopping based on validation loss.
    Stops training if the validation loss doesn't improve for a number of epochs (patience).
    """
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
    

    def check_early_stopping(self, val_loss):
        """
        Checks if training should stop based on validation loss.
        
        Args:
            val_loss (float): The current validation loss.
        
        Returns:
            stop (bool): Whether to stop training.
            is_best (bool): Whether this is the best model so far.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False, True  # Don't stop, this is the best model
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True, False  # Stop training
        return False, False  # Continue training, not the best model

