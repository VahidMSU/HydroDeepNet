import os
import numpy as np
from sklearn.metrics import mean_squared_error
def write_performance(performance_metrics_path, y_true, y_pred, stage):
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
	performance = f'{stage}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, NSE: {nse:.2f}, R2: {r2:.2f}, MPE: {mpe:.2f}\n'
	with open(performance_metrics_path, 'a') as f:
		f.write(performance)
	print(performance)
def print_shape_size(data, name):
	print(f'{name} shape: {data.shape}, size: {data.size}')

def print_statistics(array, name):
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