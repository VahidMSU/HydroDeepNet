import os
import logging
from sympy import E
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
from scipy.stats import binned_statistic
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import DataParallel, CrossEntropyLoss
from torch.optim import AdamW
from multiprocessing import Pool
from GeoClassCNN.import_data import import_simulated_data
from GeoClassCNN.global_var import get_var_name
from GeoClassCNN.local_models import ClassSparseFCN as FCN
from GeoClassCNN.viz import plot_grid_class, plot_scatter, plot_grid_class_viridis, plot_scatter_class, plot_loss_over_epochs


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
logging.basicConfig(level=logging.INFO)

os.environ['TORCH_USE_CUDA_DSA'] = "1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def setup_gpu(gpu_index):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_index)
    if torch.cuda.is_available():
        logging.info(f"Process on GPU: {torch.cuda.current_device()} with index {gpu_index}")
    else:
        logging.info("CUDA is not available.")

def discretize_feature(feature, name, num_classes):
    valid_feature = feature[feature != -999]
    quantiles = np.percentile(valid_feature, np.linspace(0, 100, num_classes))
    digitized = np.digitize(feature, quantiles, right=True)
    digitized[feature == -999] = 0
    digitized = digitized.astype(np.int8)
    
    logging.info(f'{name} #Feature min-max: {feature.min()} - {feature.max()}')
    logging.info(f'{name} #Bins: {quantiles}')
    logging.info(f'{name} #Unique bins: {np.unique(digitized)}')
    logging.info(f'{name} # datatype: {digitized.dtype}')
    

    return digitized

def discretize_feature_target(feature, name, num_classes):
    valid_feature = feature[feature > 0]
    
    quantiles = np.percentile(valid_feature, np.linspace(0, 100, num_classes))
    digitized = np.digitize(feature, quantiles, right=True)
    digitized[feature == -999] = 0
    digitized = digitized.astype(np.int8)
    
    logging.info(f'{name} #Feature min-max: {feature.min()} - {feature.max()}')
    logging.info(f'{name} #Bins: {quantiles}')
    logging.info(f'{name} #Unique bins: {np.unique(digitized)}')
    logging.info(f'{name} # datatype: {digitized.dtype}')
    

    return digitized, quantiles

class GeoClassCNN:
    def __init__(self, config):
        self.config = config
        self.database_path = config['database_path']
        self.target_array = config['target_array']
        self.model = FCN(num_classes=config['num_classes'])
        self.fig_path = "figs"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DataParallel(self.model).to(self.device) if torch.cuda.device_count() > 1 else self.model.to(self.device)
        self.scaler = GradScaler()
        self.best_model_path = f'models/best_model_{self.target_array}.pth'

    def preprocess_data(self, numerical_data, categorical_data):
        numerical_tensors = [torch.tensor(discretize_feature(nd, "numerical data", self.config['num_classes']), dtype=torch.float32) for nd in numerical_data]
        categorical_tensors = [torch.tensor(cd, dtype=torch.float32) for cd in categorical_data]
        
        combined_input = torch.stack(numerical_tensors + categorical_tensors, dim=0).unsqueeze(0)
        return combined_input.to(self.device)


    def predict(self):
        with torch.no_grad():
            predictions = self.model(self.combined_input).squeeze(0)
        predictions_array = torch.softmax(predictions, dim=0).cpu().numpy()
        predicted_labels = np.argmax(predictions_array, axis=0)
        predicted_labels[predicted_labels == 0] = -999
        logging.info(f'Shape of predicted labels: {predicted_labels.shape}')
        logging.info(f'predicted_labels unique values: {np.unique(predicted_labels)}')

        plot_grid_class_viridis(self.fig_path, predicted_labels, f'ClassCNN_Predictions_{self.target_array}', num_classes=self.config['num_classes'], target_quantiles = self.target_quantiles)
        
        target_array = self.target_tensor.cpu().numpy().squeeze()
        if target_array.ndim > 1:
            target_array[target_array == 0] = -999
            logging.info(f'Shape of target array: {target_array.shape}')
            plot_scatter_class(target_array.flatten(), predicted_labels.flatten(), self.fig_path, self.target_array)

    def update_batch_size(self, epoch):
        if epoch % self.config['rate_of_batch_size_increase'] == 0 and self.config['batch_size'] < self.config['max_batch_size']:
            self.config['batch_size'] = min(self.config['max_batch_size'], self.config['batch_size'] * 2)
            logging.info(f"Batch size updated to {self.config['batch_size']}")

    def import_data(self):
        numerical_arrays = get_var_name("numerical", self.target_array, self.config['RESOLUTION'])
        categorical_arrays = get_var_name("categorical", self.target_array, self.config['RESOLUTION'])
        
        target, numerical_data, categorical_data, _ = import_simulated_data(self.database_path, self.target_array, numerical_arrays, categorical_arrays, self.config['RESOLUTION'], huc8=self.config['huc8'])
        
        self.combined_input = self.preprocess_data(numerical_data, categorical_data)
        
        target_labels, self.target_quantiles = discretize_feature_target(target, f'## target_{self.target_array}', self.config['num_classes'])

        logging.info("Unique target labels: %s", np.unique(target_labels))
        
        assert target_labels.min() >= 0 and target_labels.max() < self.config['num_classes'], "Invalid target labels"
        
        target_labels_np = np.array([target_labels])
        self.target_tensor = torch.tensor(target_labels_np, dtype=torch.long).to(self.device)
        
        self.dataset = TensorDataset(self.combined_input, self.target_tensor)
        self.train_loader = DataLoader(self.dataset, batch_size=self.config['batch_size'], shuffle=True)
        torch.cuda.empty_cache()
        
    def train_model(self):
        criterion = CrossEntropyLoss(ignore_index=0)
        optimizer = AdamW(self.model.parameters(), lr=self.config['opt_lr'], amsgrad=True)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=self.config['threshold_patience'], threshold=self.config['improvment_threshold'])
        best_loss = float('inf')
        early_stopping_counter = 0
        losses = []
        assert self.combined_input.dtype == torch.float32, "Input data should be float32"
        assert self.target_tensor.dtype == torch.long, "Target data should be long"
        
        for epoch in range(self.config['EPOCH']):
            self.model.train()
            running_loss = 0.0

            self.update_batch_size(epoch)
            self.train_loader = DataLoader(self.dataset, batch_size=self.config['batch_size'], shuffle=True)

            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                with autocast():
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                if torch.isnan(loss):
                    logging.warning(f"NaN loss encountered at epoch {epoch+1}. Skipping this batch.")
                    continue
                self.scaler.scale(loss).backward()
                clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
                running_loss += loss.item()

            epoch_loss = running_loss / len(self.train_loader)
            scheduler.step(epoch_loss)
            logging.info(f'Epoch {epoch+1}, Loss: {epoch_loss:.2f}, learning rate: {optimizer.param_groups[0]["lr"]}, batch size: {self.config["batch_size"]}')
            losses.append(epoch_loss)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                early_stopping_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= self.config['early_stopping_patience'] and abs(epoch_loss - best_loss) < self.config['early_stopping_threshold']:
                logging.info('###################################################')
                logging.info(f'##### Early stopping at epoch {epoch+1} ##########')
                logging.info('###################################################')
                break

        plot_loss_over_epochs(losses, self.target_array, self.fig_path)

def main(config):
    setup_gpu(config['gpu_index'])
    geo_cnn = GeoClassCNN(config)
    geo_cnn.import_data()
    geo_cnn.train_model()
    geo_cnn.predict()

if __name__ == '__main__':
    RESOLUTION = 250
    EPOCH = 100
    BATCH_SIZE = 500
    NUM_CLASSES = 6
    OPT_LR = 0.001
    MAX_BATCH_SIZE = 25000
    IMPROVEMENT_THRESHOLD = 0.01
    THRESHOLD_PATIENCE = 25
    RATE_OF_BATCH_SIZE_INCREASE = EPOCH // 10

    logging.info('rate of batch size increase: %s', RATE_OF_BATCH_SIZE_INCREASE)
    
    TARGET_ARRAYS = [
        f'obs_H_COND_1_{RESOLUTION}m',
     #   f'obs_H_COND_2_{RESOLUTION}m', 
     #   f'obs_SWL_{RESOLUTION}m',
     #   f'obs_V_COND_1_{RESOLUTION}m', 
     #   f'obs_V_COND_2_{RESOLUTION}m',
     #   f'obs_TRANSMSV_1_{RESOLUTION}m', 
     #   f'obs_TRANSMSV_2_{RESOLUTION}m', 
     #   f'obs_AQ_THK_1_{RESOLUTION}m',
     #   f'obs_AQ_THK_2_{RESOLUTION}m'
     #   f'recharge_2016_{RESOLUTION}m',
    ]
    
    args_list = [
        {
            'target_array': target_array,
            'gpu_index': i,
            'database_path': f"/data/MyDataBase/HydroGeoDataset_ML_{RESOLUTION}.h5",
            'RESOLUTION': RESOLUTION,
            'EPOCH': EPOCH,
            'batch_size': BATCH_SIZE,
            'num_classes': NUM_CLASSES,
            'opt_lr': OPT_LR,
            'max_batch_size': MAX_BATCH_SIZE,
            'improvment_threshold': IMPROVEMENT_THRESHOLD,
            'threshold_patience': THRESHOLD_PATIENCE,
            'rate_of_batch_size_increase': RATE_OF_BATCH_SIZE_INCREASE,
            'early_stopping_patience': 50,
            'early_stopping_threshold': 0.01,
            'huc8': None,
        }
        
        for i, target_array in enumerate(TARGET_ARRAYS)
    ]
    parallel = False
    if parallel:
        with Pool(processes=len(args_list)) as pool:
            pool.map(main, args_list)
    else:
        for args in args_list:
            main(args)
