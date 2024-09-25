import numpy as np
import os
from sklearn.metrics import mean_squared_error


def write_performance(performance_metrics_path, y_true, y_pred, stage):
	## flatten if 2d
	print(f'size of y_true: {y_true.shape}')
	print(f'size of y_pred: {y_pred.shape}')

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
	mpe = np.mean((y_true - y_pred) / (y_true+0.0001)) * 100
	performance = f'{stage}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, NSE: {nse:.2f}, R2: {r2:.2f}, MPE: {mpe:.2f}\n'
	with open(performance_metrics_path, 'a') as f:
		f.write(performance)
	print(performance)

def setup_gpu(gpu_index):
	import torch
	os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_index)
	if torch.cuda.is_available():
		print(f"Process on GPU: {torch.cuda.current_device()} with index {gpu_index}")
	else:
		print("CUDA is not available.")