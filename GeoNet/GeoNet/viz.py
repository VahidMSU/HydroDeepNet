import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
def scatter_plot(y_true, y_pred, figure_scatter_path, target_array):
    plt.scatter(y_true, y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'True vs Predicted Values ({target_array})')
    plt.savefig(figure_scatter_path, dpi=300)
    plt.close()
def plot_scatter_density(x, y, fig_path, name):
    total_points = len(x)
    if total_points > 10000:  # Use subsampling if the dataset is very large
        # Subsample the data if it's too large
        subsample_ratio = 0.1  # Adjust this based on your needs
        subsample_size = int(total_points * subsample_ratio)
        indices = np.random.choice(total_points, subsample_size, replace=False)
        x, y = x[indices], y[indices]

    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    fig, ax = plt.subplots()
    scatter = ax.scatter(x, y, c=z, s=50, edgecolor='none', cmap='viridis')
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Density')

    # Add a 1:1 line
    min_val = min(np.min(x), np.min(y))
    max_val = max(np.max(x), np.max(y))
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

    # Calculate MSE, RMSE, MAPE, NSE, R2
    mse = np.mean((y - x) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y - x) / x)) * 100
    nse = 1 - np.sum((y - x) ** 2) / np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - mse / np.var(y)

    # Annotate the metrics
    ax.annotate(f"MSE: {mse:.2f}", xy=(0.05, 0.95), xycoords='axes fraction')
    ax.annotate(f"RMSE: {rmse:.2f}", xy=(0.05, 0.90), xycoords='axes fraction')
    ax.annotate(f"MAPE: {mape:.2f}%", xy=(0.05, 0.85), xycoords='axes fraction')
    ax.annotate(f"NSE: {nse:.2f}", xy=(0.05, 0.80), xycoords='axes fraction')
    ax.annotate(f"R2: {r2:.2f}", xy=(0.05, 0.75), xycoords='axes fraction')

    plt.xlabel('Target')
    plt.ylabel('Predictions')
    plt.title('Predictions vs Target with Point Density')
    plt.grid(True)
    plt.savefig(f"{fig_path}/{name}_predictions_vs_target_density.png", dpi=300)
    plt.close()
