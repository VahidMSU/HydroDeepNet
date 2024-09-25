from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import gaussian_kde
import h5py
def plot_loss_over_epochs(losses, fig_path, name):
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.grid(True)
    plt.savefig(f"{fig_path}/CNN_loss_over_epochs_{name}_recharge_predicted.png", dpi=300)
    plt.close()
def plot_grid(fig_path, array, name):
    print(f"size of array to plot: {array.shape}")  
    ## min max except -999
    vmin = np.nanmin(array)
    vmax = np.nanmax(array)
    print(f"min: {vmin}, max: {vmax}")
    # size of array to plot: (15, 1, 308, 339)
    if array.shape[0] > 1:
        for i in range(array.shape[0]):
            plt.imshow(array[i,0 ,:, :], cmap='jet', vmin=vmin, vmax=vmax)
            plt.colorbar()
            plt.title(f'{name}_{i}')
            # Save the plot as a PNG file
            plt.savefig(os.path.join(fig_path, f'{name}_{i}.png'), dpi=300)
            plt.close()


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

def plot_grid_class_viridis(fig_path, array, name, num_classes, target_quantiles):
    # Replace -999 with np.nan
    array = np.where(array == -999, np.nan, array)
    
    # Remove zero from target_quantiles
    target_quantiles = target_quantiles[1:]
    
    # Ensure the number of unique classes matches the number of quantiles
    unique_classes = np.unique(array[~np.isnan(array)])
    print(f"Unique classes: {unique_classes}")
    print(f"Number of unique quantiles: {target_quantiles}")
    
    if len(unique_classes) != len(target_quantiles):
        raise ValueError("The number of unique classes does not match the number of quantiles.")
    
    # Create a colormap and a normalization based on the unique classes
    cmap = plt.get_cmap('viridis', len(target_quantiles))
    norm = BoundaryNorm(boundaries=np.arange(len(target_quantiles) + 1) - 0.5, ncolors=len(target_quantiles))

    # Create the plot
    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed

    img = plt.imshow(array, cmap=cmap, norm=norm)
    
    # Map each unique class to its respective target quantile label
    quantile_labels = [target_quantiles[int(cls) - 1] for cls in unique_classes if int(cls) - 1 < len(target_quantiles)]
    
    # Create a colorbar with the appropriate labels
    cbar = plt.colorbar(img, ticks=np.arange(len(target_quantiles)))
    cbar.ax.set_yticklabels([f'{label:.1f}' for label in quantile_labels])  # Update colorbar labels
    
    plt.title(name)
    
    # Save the plot as a PNG file
    plt.savefig(os.path.join(fig_path, f'{name}.png'), dpi=300)
    plt.close()


def plot_grid_class(fig_path, array, name):
    # Replace -999 with np.nan
    array = np.where(array == -999, np.nan, array)
    
    # Define the number of unique categories (excluding NaN)
    unique_classes = np.unique(array[~np.isnan(array)])
    n_categories = len(unique_classes)
    
    # Create a discrete colormap
    cmap = plt.get_cmap('tab20', n_categories)  # 'tab20' is a good discrete colormap for up to 20 classes

    # Create the plot
    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
    img = plt.imshow(array, cmap=cmap, vmin=np.nanmin(unique_classes)-0.5, vmax=np.nanmax(unique_classes)+0.5)
    cbar = plt.colorbar(img, ticks=np.arange(np.nanmin(unique_classes), np.nanmax(unique_classes)+1))
    cbar.ax.set_yticklabels(unique_classes.astype(int))  # Update colorbar labels to show class indices
    
    plt.title(name)
    plt.show()
    
    # Save the plot as a PNG file
    plt.savefig(os.path.join(fig_path, f'{name}.png'), dpi=300)
    plt.close()


def visualize_predictions(predicted_labels):
    # Dummy visualization example, replace with actual visualization logic
    plt.figure()
    plt.hist(predicted_labels.numpy(), bins=range(self.model.num_classes + 1), align='left')
    plt.xlabel('Classes')
    plt.ylabel('Frequency')
    plt.title('Histogram of Predicted Classes')
    plt.show()

def plot_scatter(x, y, fig_path, name):
    # Convert to numpy arrays if not already, to facilitate boolean indexing
    x = np.array(x)
    y = np.array(y)

    # Create a mask to filter out entries where either x or y is -999
    valid_mask = (x != -999) & (y != -999)

    # Apply the mask to x and y
    filtered_x = x[valid_mask]
    filtered_y = y[valid_mask]

    # Create scatter plot with filtered data
    plt.scatter(filtered_x, filtered_y, alpha=0.5)
    plt.xlabel('Target')
    plt.ylabel('Predictions')
    plt.title(f'Predictions vs Target - {name}')
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(f"{fig_path}/ClassCNN_{name}_predictions_vs_target.png", dpi=300)
    plt.close()

import numpy as np
import matplotlib.pyplot as plt
def plot_loss_over_epochs(losses, name, fig_path):
    plt.plot(range(1, len(losses)+1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.grid(True)
    plt.savefig(f"{fig_path}/ClassCNN_loss_over_epochs_{name}.png", dpi=300)
    plt.close()
def plot_scatter_class(x, y, fig_path, name):
    # Convert to numpy arrays if not already, to facilitate boolean indexing
    x = np.array(x)
    y = np.array(y)

    # Create a mask to filter out entries where either x or y is -999
    valid_mask = (x != -999) & (y != -999)

    # Apply the mask to x and y
    filtered_x = x[valid_mask]
    filtered_y = y[valid_mask]

    # Calculate accuracy
    accuracy = np.mean(filtered_x == filtered_y)
    accuracy_percent = f"{accuracy * 100:.2f}%"

    # Create scatter plot with filtered data
    plt.scatter(filtered_x, filtered_y, alpha=0.5)
    plt.xlabel('Target')
    plt.ylabel('Predictions')
    plt.title(f'Predictions vs Target - {name}')
    
    # Annotate plot with accuracy
    plt.annotate(f'Accuracy: {accuracy_percent}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor='green', facecolor='white'))

    plt.grid(True)

    # Save the plot to a file
    plt.savefig(f"{fig_path}/ClassCNN_{name}_predictions_vs_target.png", dpi=300)
    plt.close()

def plot_scatter_density(x, y, fig_path, name):
    # Remove -999 values from x and y
    print(f"Shape of x before removing -999: {x.shape}")
    print(f"Shape of y before removing -999: {y.shape}")
    mask = (x != -999) & (y != -999)
    x = x[mask]
    y = y[mask]
    
    # Remove inf and nan values from x and y
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    
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

    plt.xlabel('Target')
    plt.ylabel('Predictions')
    plt.title('Predictions vs Target with Point Density')
    plt.grid(True)
    plt.savefig(f"{fig_path}/{name}_predictions_vs_target_density.png", dpi=300)
    plt.close()