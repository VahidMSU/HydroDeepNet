from requests import get
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
from GeoClassCNN.viz import plot_grid_class, plot_scatter, plot_grid_class_viridis
from matplotlib import pyplot as plt
from GeoClassCNN.import_data import get_mask, import_simulated_data
from GeoClassCNN.global_var import get_var_name
from GeoClassCNN.local_models import SparseFCN as FCN
from GeoClassCNN.obj_functions import masked_mse_loss
from GeoClassCNN.viz import plot_grid, plot_scatter_density
from GeoClassCNN.viz import plot_grid
from GeoClassCNN.viz import plot_scatter_density

def setup_ddp(rank, world_size):
	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = '12345'
	dist.init_process_group('nccl', rank=rank, world_size=world_size)
	torch.cuda.set_device(rank)

class GeoClassCNN:
	def __init__(self, database_path, target_array, RESOLUTION, EPOCH, batch_size, rank):
		self.rank = rank
		self.database_path = database_path
		self.target_array = target_array
		self.model = FCN()
		self.EPOCH = EPOCH
		self.RESOLUTION = RESOLUTION
		self.batch_size = batch_size
		self.name = target_array
		self.fig_path = "figs"
		self.early_stopping_patience = 10
		self.early_stopping_threshold = 0.01
		self.max_batch_size = 1024*2*2*2*2*2

		if torch.cuda.is_available():
			self.device = torch.device('cuda', rank)
			self.model.cuda(rank)
			self.model = DDP(self.model, device_ids=[rank])
		else:
			self.device = torch.device('cpu')

	def preprocess_data(self, numerical_data, categorical_data):
		numerical_tensors = [torch.tensor(nd, dtype=torch.float32) for nd in numerical_data]
		categorical_tensors = [torch.tensor(cd, dtype=torch.long) for cd in categorical_data]
		print(f'Numerical tensors: {len(numerical_tensors)}')
		print(f'Categorical tensors: {len(categorical_tensors)}')

		# Normalize numerical data and stack tensors
		for numerical_tensor_ in numerical_tensors:
			numerical_tensor = numerical_tensor_
			valid_mask = numerical_tensor != -999
			mean_val = torch.mean(numerical_tensor[valid_mask])
			std_val = torch.std(numerical_tensor[valid_mask])
			numerical_tensor_[valid_mask] = (
				numerical_tensor[valid_mask] - mean_val
			) / std_val

		# Combine and batch data
		combined_input = torch.cat(numerical_tensors + categorical_tensors, dim=0)
		combined_input = combined_input.transpose(0, 1)  # Transpose to put channel dimension in correct place if needed

		# Ensure the shape is [batch_size, channels, height, width]
		# Assuming 'height' is 1849 and 'width' is 1458, correct shape should be:
		combined_input = combined_input.reshape([1, 11, 1848, 1457])  # Adjust '11' to match the actual channel count

		return combined_input.to(self.device)

	def predict(self):
		with torch.no_grad():
			predictions = self.model(self.combined_input).squeeze(0)
		predictions_array = predictions.cpu().numpy()[0]
		predictions_array[predictions_array == -999] = np.nan
		plot_grid(self.fig_path, predictions_array, f'CNN_DDP_Predictions_{self.name}')

		target_array = self.target_tensor.cpu().numpy().squeeze()
		predictions_array = predictions.cpu().numpy().squeeze()
		mask = get_mask(target_array, predictions_array)
		filtered_targets = target_array[mask]
		filtered_predictions = predictions_array[mask]
		plot_scatter_density(filtered_targets, filtered_predictions, self.fig_path, f"CNN_DDP_{self.name}")


	def import_data(self):
		numerical_arrays = get_var_name("numerical", self.target_array, self.RESOLUTION)
		categorical_arrays = get_var_name("categorical", self.target_array, self.RESOLUTION)
		target, numerical_data, categorical_data, groups = import_simulated_data(self.database_path, self.target_array, numerical_arrays, categorical_arrays, self.RESOLUTION)

		self.combined_input = self.preprocess_data(numerical_data, categorical_data)
		self.target_tensor = torch.tensor(target, dtype=torch.float32).to(self.device)
		self.group_tensor = torch.tensor(groups, dtype=torch.long).to(self.device)
		self.weights = (self.target_tensor != -999).float()

		# Correct dimensions
		if self.combined_input.dim() == 3:
			self.combined_input = self.combined_input.unsqueeze(0)  # Ensure batch dimension is consistent
		self.target_tensor = self.target_tensor.unsqueeze(0)  # Ensure target tensor has a batch dimension
		self.weights = self.weights.unsqueeze(0)  # Ensure weights have a batch dimension
		self.group_tensor = self.group_tensor.unsqueeze(0)  # Ensure group tensor has a batch dimension

		# Debug output to check shapes before creating TensorDataset
		print(f"Combined input shape: {self.combined_input.shape}")
		print(f"Target tensor shape: {self.target_tensor.shape}")
		print(f"Weights shape: {self.weights.shape}")
		print(f"Group tensor shape: {self.group_tensor.shape}")

		# Create the TensorDataset
		self.dataset = TensorDataset(self.combined_input, self.target_tensor, self.weights, self.group_tensor)
		self.sampler = DistributedSampler(self.dataset, num_replicas=dist.get_world_size(), rank=self.rank)
		self.train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, sampler=self.sampler)


	def train_model(self):
		criterion = masked_mse_loss
		optimizer = Adam(self.model.parameters(), lr=0.0001, amsgrad=True)
		scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, threshold=0.001, threshold_mode='rel')
		best_loss = float('inf')
		losses = []

		for epoch in range(self.EPOCH):
			self.model.train()
			running_loss = 0.0
			self.sampler.set_epoch(epoch)

			for inputs, targets, weights, groups in self.train_loader:
				inputs, targets, weights, groups = inputs.to(self.device), targets.to(self.device), weights.to(self.device), groups.to(self.device)
				optimizer.zero_grad() # Zero the gradients for each batch to avoid accumulation	
				outputs = self.model(inputs) # Forward pass to get the model outputs
				loss = criterion(outputs, targets, weights, groups) # Calculate the loss
				loss.backward() # Backward pass to calculate the gradients
				optimizer.step() # Update the model parameters
				running_loss += loss.item() #

			epoch_loss = running_loss / len(self.train_loader)
			scheduler.step(epoch_loss)
			if self.rank == 0:
				print(f'Epoch {epoch+1}, Loss: {epoch_loss}')
				losses.append(epoch_loss)  # Only main process handles output

			# Check if the current loss is the best so far
			if epoch_loss < best_loss:
				best_loss = epoch_loss
				early_stopping_counter = 0
			else:
				early_stopping_counter += 1

			# Check if early stopping criteria is met
			if early_stopping_counter >= self.early_stopping_patience and abs(epoch_loss - best_loss) < self.early_stopping_threshold:
				print('###################################################')
				print(f'##### Early stopping at epoch {epoch+1} ###########')
				print('###################################################')
				break

		# Plot and save the loss over epochs
		plt.plot(range(1, len(losses)+1), losses)
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.title('Loss over Epochs')
		plt.grid(True)
		plt.savefig(f"{self.fig_path}/CNN_DDP_loss_over_epochs_{self.name}.png", dpi=300)
		plt.close()

def main(rank, world_size, args):
	setup_ddp(rank, world_size)
	RESOLUTION = 250
	EPOCH = 100
	target_array = f'obs_H_COND_1_{RESOLUTION}m'
	database_path = f"/data/MyDataBase/HydroGeoDataset_ML_{RESOLUTION}.h5"
	geo_cnn = GeoClassCNN(database_path, target_array, RESOLUTION, EPOCH=EPOCH, batch_size=5000, rank=rank)
	geo_cnn.import_data()
	geo_cnn.train_model()
	if rank == 0:
		geo_cnn.predict()

if __name__ == '__main__':
	world_size = 4
	torch.multiprocessing.spawn(main, args=(world_size, ()), nprocs=world_size, join=True)