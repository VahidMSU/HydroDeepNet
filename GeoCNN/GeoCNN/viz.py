import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import gaussian_kde
from matplotlib.colors import BoundaryNorm
import numpy as np



def plot_scatter(x, y, fig_path, name, no_value, nse, mse, rmse):
    # Convert to numpy arrays if not already, to facilitate boolean indexing
    x = np.array(x)
    y = np.array(y)

    # Create a mask to filter out entries where either x or y is no_value
    valid_mask = (x != no_value) & (y != no_value)

    # Apply the mask to x and y
    filtered_x = x[valid_mask]
    filtered_y = y[valid_mask]

    # Create scatter plot with filtered data
    plt.scatter(filtered_x, filtered_y, alpha=0.5)
    plt.xlabel('Target')
    plt.ylabel('Predictions')
    plt.title(f'Predictions vs Target - {name}')
    plt.grid(True)
    ### add metric with annotation
    plt.annotate(f'NSE: {nse:.2f}\nMSE: {mse:.2f}\nRMSE: {rmse:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10
                    ,bbox=dict(boxstyle="round,pad=0.3", edgecolor='green', facecolor='white'))
    # Save the plot to a file
    os.makedirs(f"{fig_path}_scatters", exist_ok=True)
    plt.savefig(f"{fig_path}_scatters/{name}.png", dpi=300)
    plt.close()
def plot_grid(fig_path, array, name, no_value):
    # Replace no_value with np.nan
    array = np.where(array == no_value, np.nan, array)
    array = np.where(array == 0, np.nan, array)

    # Optional: Adjust the color limits to enhance visual contrast, ignoring NaNs
    vmin = np.nanpercentile(array, 0.5)
    vmax = np.nanpercentile(array, 99.5)
    print(f"size of array to plot: {array.shape}")  

    # Check if the array is 3D or 4D
    if len(array.shape) == 3:  # Handle 3D array (like your case: (6, 1848, 1457))
        for i in range(array.shape[0]):
            plt.imshow(array[i, :, :], vmin=vmin, vmax=vmax)
            _extracted_from_plot_grid_15(name, i, fig_path)
    elif len(array.shape) == 4:  # Handle 4D array (if needed in other cases)
        for i in range(array.shape[0]):
            plt.imshow(array[i, 0, :, :], vmin=vmin, vmax=vmax)
            _extracted_from_plot_grid_15(name, i, fig_path)
    else:
        plt.imshow(array, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title(name)    
        # Save the plot as a PNG file
        os.makedirs(os.path.join(fig_path,'grids'), exist_ok=True)
        plt.savefig(os.path.join(fig_path,'grids' ,f'{name}.png'), dpi=300)  
        plt.close()


# TODO Rename this here and in `plot_grid`
def _extracted_from_plot_grid_15(name, i, fig_path):
    plt.colorbar()
    plt.title(f'{name}_{i}')
    # Save the plot as a PNG file
    plt.savefig(os.path.join(fig_path, f'{name}_{i}.png'), dpi=300)
    plt.close()



def plot_grid_class_viridis(fig_path, array, name, num_classes, target_quantiles):
    # Replace no_value with np.nan
    array = np.where(array == no_value, np.nan, array)
    
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


def plot_grid_class(fig_path, array, name, no_value):
    # Replace no_value with np.nan
    array = np.where(array == no_value, np.nan, array)
    
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


import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_gssurgo_grid(gssurgo_grid, column):
    # Replace -999 with np.nan for proper handling of missing data
    gssurgo_grid_clean = np.where(gssurgo_grid == -999, np.nan, gssurgo_grid)
    
    # Flatten the grid and exclude NaN values for percentile calculation
    valid_values = gssurgo_grid_clean[~np.isnan(gssurgo_grid_clean)].flatten()

    # Calculate the 2.5th and 97.5th percentiles for value limiting
    lower_percentile = np.percentile(valid_values, 0.001)
    upper_percentile = np.percentile(valid_values, 99.999)

    # Create a figure with two subplots: one for the grid and one for the histogram
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Plot the grid, limiting the color range between the 2.5th and 97.5th percentiles
    im = ax[0].imshow(gssurgo_grid_clean, cmap='viridis', vmin=lower_percentile, vmax=upper_percentile)
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title(f'gssurgo_grid_{column} (Limited to 2.5th - 97.5th Percentiles)')

    # Plot the histogram, limiting the range for display
    ax[1].hist(valid_values, bins=50, range=(lower_percentile, upper_percentile), edgecolor='black')
    ax[1].set_title(f'Histogram of {column} (Limited to 2.5th - 97.5th Percentiles)')
    ax[1].set_xlabel(f'{column} values')
    ax[1].set_ylabel('Frequency')

    # Save the combined plot
    plt.tight_layout()
    plt.savefig(f'database/gssurgo_grid_{column}_combined.png')
    plt.close()
def plot_loss_over_epochs(losses,val_losses, name, fig_path):
    
    plt.plot(range(1, len(losses)), losses[1:], color='blue', label='Training Loss')  
    plt.plot(range(1, len(val_losses)), val_losses[1:], color='red', label='Validation Loss')
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

    # Create a mask to filter out entries where either x or y is no_value
    valid_mask = (x != no_value) & (y != no_value)

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
    # Remove no_value values from x and y
    print(f"Shape of x before removing no_value: {x.shape}")
    print(f"Shape of y before removing no_value: {y.shape}")
    mask = (x != no_value) & (y != no_value)
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


import matplotlib.pyplot as plt
import numpy as np

def plot_feature(array2d, array_name, no_value):
    import logging
    logging.info(f"Plotting feature: {array_name}")
    #assert no_value not in array2d, "Array do not contains no_value values"
    # Replace no_value and 0 values with NaN for proper handling
    array2d = np.where(array2d == no_value, np.nan, array2d)
    
    # Get valid (non-NaN) values for the histogram
    valid_values = array2d[~np.isnan(array2d)]
    if len(valid_values) == 0:
        logging.warning(f"No valid values found in {array_name}")
        return
    # Create the figure and subplots (2 rows, 1 column)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the 2D heatmap on the first subplot
    im = axes[0].imshow(array2d, vmin=np.nanpercentile(array2d, 5), vmax=np.nanpercentile(array2d, 95), cmap='jet')
    axes[0].set_title(f"{array_name} (Valid Count: {np.sum(~np.isnan(array2d))})")
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)  # Add color bar next to the heatmap

    # Plot the histogram on the second subplot
    axes[1].hist(valid_values, bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[1].set_title(f"Histogram of {array_name}")
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Frequency")
    
    # Add statistical info (mean, median, std) below the histogram
    mean_val = np.nanmean(valid_values)
    median_val = np.nanmedian(valid_values)
    std_val = np.nanstd(valid_values)
    axes[1].text(0.95, 0.95, f"Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}",
                transform=axes[1].transAxes, fontsize=10, verticalalignment='top',
                horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.7))

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plot as a PNG image
    os.makedirs(os.path.dirname(f"input_figs/{array_name}.png"), exist_ok=True)
    plt.savefig(f"input_figs/{array_name}.png")
    plt.close()

