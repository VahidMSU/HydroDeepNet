from GeoCNN.utils import calculate_metrics
from GeoCNN.viz import plot_grid, plot_scatter
import os
import torch
import logging
import numpy as np




def store_and_evaluate_predictions(config, valid_preds, valid_targets, preds, outputs, nse_vals, mse_vals, rmse_vals, batch_idx, mask):
    print("store_and_evaluate_predictions")
    print(f"shape of valid_preds: {valid_preds.shape}")
    print(f"shape of valid_targets: {valid_targets.shape}")
    print(f"shape of preds: {preds.shape}")

    fig_path = config['fig_path']
    no_value = config['no_value']

    # Loop through the time dimension (3 years in this case)
    time_steps = preds.shape[1]  # Assuming shape [batch_size, time_steps, channels, height, width]

    for t in range(time_steps):
        # Get predictions and targets for the current year (time step)
        preds_t = preds[:, t, :, :, :].squeeze()  # Remove batch and channel dimensions
        valid_preds_t = valid_preds[t]  # For each year, valid_preds will be indexed by time step
        valid_targets_t = valid_targets[t]

        print(f"Processing time step {t + 1}, shape of preds_t: {preds_t.shape}")
        print(f"Processing time step {t + 1}, shape of valid_preds_t: {valid_preds_t.shape}")
        print(f"Processing time step {t + 1}, shape of valid_targets_t: {valid_targets_t.shape}")

        # Store predictions for later metrics calculation
        outputs.append(valid_preds_t.cpu().numpy())

        # Calculate metrics for predictions
        nse_val, mse_val, rmse_val = calculate_metrics(valid_targets_t.cpu().numpy(), valid_preds_t.cpu().numpy())
        nse_vals.append(nse_val)
        mse_vals.append(mse_val)
        rmse_vals.append(rmse_val)

        # Detach predictions and targets for plotting
        preds_t = preds_t.cpu().detach().numpy()  # preds_t shape: [64, 64]
        targets_t = valid_targets_t.cpu().detach().numpy()  # valid_targets_t shape: [64, 64]


        # Apply mask (if necessary), mask is applied for each year
        # preds_t = preds_t * mask
        # targets_t = targets_t * mask

        # Plot grid and scatter for the current year (time step)
        plot_grid(fig_path, preds_t, f"preds_year_{t + 1}_batch_{batch_idx + 1}", no_value=0)
        plot_scatter(targets_t.flatten(), preds_t.flatten(),  # Pass year-specific targets and predictions
                     fig_path, f"scatter_year_{t + 1}_batch_{batch_idx + 1}",
                     0, nse_val, mse_val, rmse_val)

        logging.info(f'Batch {batch_idx + 1}, Year {t + 1} - NSE: {nse_val:.2f}, MSE: {mse_val:.2f}, RMSE: {rmse_val:.2f}')
