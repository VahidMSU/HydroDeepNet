import os
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from torch.cuda.amp import GradScaler, autocast
from torch import nn, optim

import matplotlib.pyplot as plt

features = ['Aquifer_Characteristics_Of_Glacial_Drift_250m', 'average_temperature_raster_250m',
		'BaseRaster_250m', 'COUNTY_250m', 'DEM_250m',
		'geomorphons_250m_250Dis', 'Glacial_Landsystems_250m',
		'gSURRGO_swat_250m', 'HUC12_250m', 'HUC8_250m', 'kriging_output_AQ_THK_1_250m',
		'kriging_output_AQ_THK_2_250m', 'kriging_output_H_COND_1_250m',
		'kriging_output_H_COND_2_250m', 'kriging_output_SWL_250m', 'kriging_output_TRANSMSV_1_250m', 
		'kriging_output_TRANSMSV_2_250m', 'kriging_output_V_COND_1_250m', 'kriging_output_V_COND_2_250m',
		'kriging_stderr_AQ_THK_1_250m', 'kriging_stderr_AQ_THK_2_250m', 'kriging_stderr_H_COND_1_250m',
		'kriging_stderr_H_COND_2_250m', 'kriging_stderr_SWL_250m', 'kriging_stderr_TRANSMSV_1_250m', 
		'kriging_stderr_TRANSMSV_2_250m', 'kriging_stderr_V_COND_1_250m', 'kriging_stderr_V_COND_2_250m', 
		'landforms_250m_250Dis', 'landuse_250m', 'melt_rate_raster_250m', 'MI_geol_poly_250m', 
		'NHDPlusID_250m', 'non_snow_accumulation_raster_250m', 'obs_AQ_THK_1_250m', 
		'obs_AQ_THK_2_250m', 'obs_H_COND_1_250m', 'obs_H_COND_2_250m', 'obs_SWL_250m', 
		'obs_TRANSMSV_1_250m', 'obs_TRANSMSV_2_250m', 'obs_V_COND_1_250m', 'obs_V_COND_2_250m', 
		'snowpack_sublimation_rate_raster_250m', 'snow_accumulation_raster_250m',
		'snow_layer_thickness_raster_250m', 'snow_water_equivalent_raster_250m', 'Soil_STATSGO_250m']




class CustomPyTorchModel(nn.Module):
    def __init__(self, input_dim):
        super(CustomPyTorchModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.output(x)
        return x

class DeepLearningGWEstimation:
	def __init__(self, DIC):
		self.DIC = DIC
		self.trained_model_path = os.path.join(self.DIC, 'custom_model.pth')
		self.figure_scatter_path = os.path.join(self.DIC, 'scatter_plot.png')
		self.performance_metrics_path = os.path.join(self.DIC, 'performance_metrics.txt')
		self.delete_previous_plots()
		self.epoch = 25
		self.numerical_arrays = ['kriging_output_SWL_250m', 'non_snow_accumulation_raster_250m', 'kriging_output_AQ_THK_1_250m', 'kriging_output_AQ_THK_2_250m']
		self.categorical_arrays = ['geomorphons_250m_250Dis', 'MI_geol_poly_250m', 'Glacial_Landsystems_250m', 'landuse_250m', 'landforms_250m_250Dis','Aquifer_Characteristics_Of_Glacial_Drift_250m']
		self.target_array = 'obs_SWL_250m'

	def delete_previous_model(self):
		if os.path.exists(self.trained_model_path):
			os.remove(self.trained_model_path)
		if os.path.exists(self.performance_metrics_path):
			os.remove(self.performance_metrics_path)
		if os.path.exists(self.figure_scatter_path):
			os.remove(self.figure_scatter_path)
	def get_device(self):
		return 'cuda' if torch.cuda.is_available() else 'cpu'
	def write_performance(self, y_true, y_pred, stage):
		## flatten if 2d
		if len(y_true.shape) > 1:
			y_true = y_true.flatten()
			y_pred = y_pred.flatten()
		mask = y_true != -999
		y_true = y_true[mask]
		y_pred = y_pred[mask]
		mse = mean_squared_error(y_true, y_pred)
		nse = 1 - mse / np.var(y_true)
		rmse = np.sqrt(mse)
		r2 = np.corrcoef(y_true, y_pred)[0, 1] ** 2
		mpe = np.mean((y_true - y_pred) / y_true) * 100
		performance = f'STAGE: {stage}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, NSE: {nse:.2f}, R2: {r2:.2f}, MPE: {mpe:.2f}'
		with open(self.performance_metrics_path, 'w') as f:
			f.write(performance)
		print(performance)
	def import_simulated_data(self, num_samples=1000, return_grid=False):
		if self.target_array is None:
			self.target_array = 'obs_SWL_250m'
		if self.numerical_arrays is None:
			self.numerical_arrays = []
		if self.categorical_arrays is None:
			self.categorical_arrays = []
		with h5py.File(os.path.join(self.DIC, 'HydroGeoDataset_MLm.h5'), 'r') as f:
			numerical_data = [f[array_name][:] for array_name in self.numerical_arrays]
			categorical_data = [f[array_name][:] for array_name in self.categorical_arrays]
			target = f[self.target_array][:]

		if not return_grid:
			numerical_data_flat = [array.flatten() for array in numerical_data]
			categorical_data_flat = [array.flatten() for array in categorical_data]
			target_flat = target.flatten()
			valid_indices = target_flat != -999

			numerical_data_valid = [array[valid_indices][:, np.newaxis] for array in numerical_data_flat]
			categorical_data_valid = [array[valid_indices][:, np.newaxis] for array in categorical_data_flat]
			target_valid = target_flat[valid_indices]
			
			# Check if there's only one array and avoid concatenation if so
			if len(numerical_data_valid + categorical_data_valid) > 1:
				data_valid = np.concatenate(numerical_data_valid + categorical_data_valid, axis=1)
			else:
				data_valid = numerical_data_valid[0] if numerical_data_valid else categorical_data_valid[0]
				
			data_valid = data_valid[:num_samples, :]
			target_valid = target_valid[:num_samples]
			print(f'mldata has shape: {data_valid.shape}')
			print(f'target has shape: {target_valid.shape}')
		else:
			# Ensure each array is at least 3D before concatenation
			numerical_data = [array[:, :, np.newaxis] for array in numerical_data]
			categorical_data = [array[:, :, np.newaxis] for array in categorical_data]
			data_valid = np.concatenate(numerical_data + categorical_data, axis=2)
			target_valid = target

		return data_valid, target_valid

	def scatter_plot(self, y_true, y_pred):
		plt.scatter(y_true, y_pred)
		plt.xlabel('True Values')
		plt.ylabel('Predictions')
		plt.title('True vs Predicted Values')
		plt.savefig(self.figure_scatter_path)
		plt.close()

	def print_shape_size(self, data, name):
		print(f'{name} shape: {data.shape}, size: {data.size}')

	def train_model(self, num_samples=100000):
		self.delete_previous_model()

		data, target = self.import_simulated_data(num_samples)
		
		# Data preprocessing remains unchanged
		
		X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
		
		train_dataset = TensorDataset(torch.tensor(X_train.astype(np.float32)),
										torch.tensor(y_train.astype(np.float32)).view(-1, 1))
		test_dataset = TensorDataset(torch.tensor(X_test.astype(np.float32)),
										torch.tensor(y_test.astype(np.float32)).view(-1, 1))
		
		train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
		test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
		
		device = self.get_device()
		model = CustomPyTorchModel(input_dim=X_train.shape[1]).to(device)
		optimizer = optim.AdamW(model.parameters(), lr=0.001)
		criterion = nn.MSELoss()
		scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
		scaler = GradScaler()

		best_loss = float('inf')
		for epoch in range(self.epoch):
			model.train()
			running_loss = 0.0
			for inputs, targets in train_loader:
				inputs, targets = inputs.to(device), targets.to(device)
				optimizer.zero_grad()

				with autocast():
					outputs = model(inputs)
					loss = criterion(outputs, targets)
				
				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()

				running_loss += loss.item() * inputs.size(0)
			epoch_loss = running_loss / len(train_loader.dataset)
			scheduler.step()
			
			if epoch_loss < best_loss:
				best_loss = epoch_loss
				torch.save(model.state_dict(), self.trained_model_path)
			
			print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}')
		
		print(f'Best Model saved with loss: {best_loss:.4f}')
		
		# Model evaluation using the best model
		model.load_state_dict(torch.load(self.trained_model_path))
		model.eval()
		predictions, actuals = [], []
		with torch.no_grad():
			for inputs, targets in test_loader:
				inputs, targets = inputs.to(device), targets.to(device)
				with autocast():
					outputs = model(inputs)
				predictions.extend(outputs.cpu().numpy().flatten())
				actuals.extend(targets.cpu().numpy().flatten())

		mse = mean_squared_error(actuals, predictions)
		print(f'Mean Squared Error on Test Set: {mse:.4f}')
		self.write_performance(np.array(actuals), np.array(predictions), 'Test')
		self.scatter_plot(np.array(actuals), np.array(predictions))

	# Additional methods for write_performance, scatter_plot, etc., remain unchanged.

	def get_mask(self):
		with h5py.File(os.path.join(self.DIC, 'HydroGeoDataset_MLm.h5'), 'r') as f:
			huc8 = f['HUC8_250m'][:]
			## get the 2d mask
			mask = huc8 != -999	
		return mask
	def delete_previous_plots(self):
		for file in os.listdir(self.DIC):
			if file.endswith('.png'):
				os.remove(os.path.join(self.DIC, file))
	def print_statistics(self, array, name):
		## flatten if 2d
		if len(array.shape) > 1:
			array = array.flatten()
		mask = array != -999
		array = array[mask]
		percentiles = np.nanpercentile(array, [0, 25, 50, 75, 100])
		range_ = percentiles[-1] - percentiles[0]
		mean = np.nanmean(array)
		std = np.nanstd(array)
		print(f'{name} Statistics:')
		print(f'Min: {percentiles[0]:.2f}, 25th Percentile: {percentiles[1]:.2f}, Median: {percentiles[2]:.2f}, 75th Percentile: {percentiles[3]:.2f}, Max: {percentiles[4]:.2f}')
	def plot_grid(self, array, name):
		mask = self.get_mask()
		array[~mask] = np.nan
		
		plt.imshow(array, vmin=-10, vmax=100)
		plt.colorbar()
		plt.title(name)
		plt.show()
		self.print_statistics(array, name)
		plt.savefig(os.path.join(self.DIC, f'{name}.png'))
		plt.close()
	def predict_grid(self):
		## we need to get the data first
		all_data, target = self.import_simulated_data(return_grid=True)
		## then load the trained_model_path 
		model = CustomPyTorchModel(input_dim=len(self.categorical_arrays) + len(self.numerical_arrays))
		model.load_state_dict(torch.load(self.trained_model_path))
		print(f'Model loaded from {self.trained_model_path}')
		print(f'all data shape: {all_data.shape}')	
		print(f'target shape: {target.shape}')
		model.eval()
		predictions = []
		device = self.get_device()
		model = model.to(device)
		print(f'Using device: {device}')
		with torch.no_grad():
			for i in range(all_data.shape[0]):
				inputs = torch.tensor(all_data[i].astype(np.float32)).to(device)
				outputs = model(inputs)
				predictions.extend(outputs.cpu().numpy().flatten())


		predictions = np.array(predictions).reshape(target.shape)	
		self.write_performance(target, predictions, 'Final')
		
		self.plot_grid(predictions, 'Predicted GW Levels')
		self.print_statistics(predictions, 'Predicted GW Levels')
		self.print_statistics(target, 'True GW Levels')
if __name__ == '__main__':
	DIC = "/home/rafieiva/MyDataBase/codes/SWAT-CONUS/gw_machine_learning/"
	
	estimator = DeepLearningGWEstimation(DIC)
	estimator.train_model()
	estimator.predict_grid()
	