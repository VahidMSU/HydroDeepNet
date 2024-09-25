import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt

def plot_loss_curve(train_losses, val_losses, logger):
    # Function to plot training and validation loss curves
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs. Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('figs/loss_curve.png')
    plt.show()
    logger.info('Loss curves plotted and saved as loss_curve.png')
def plot_scatter(geodataframe, variable, logger, stage='None'):
    ### the aim of this function is to plot the scatter for comparing predictions vs observations
    assert variable in geodataframe.columns, f"{variable} not in columns"
    assert "geometry" in geodataframe.columns, "geometry not in columns"
    assert f"predicted_{variable}" in geodataframe.columns, f"predicted_{variable} not in columns"

    # Filter out negative values
    geodataframe = geodataframe[geodataframe[variable] > 0]
    logger.info(f"Shape after removing negative values: {geodataframe.shape}")

    # Plot the scatter
    fig, ax = plt.subplots()
    ax.scatter(geodataframe[variable], geodataframe[f"predicted_{variable}"], alpha=0.5)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='red', linestyle='--')
    ax.set_xlabel(f"Observed {variable}")
    ax.set_ylabel(f"Predicted {variable}")
    ax.set_title(f"Observed vs. Predicted {variable}")

    # Save the plot
    os.makedirs('figs', exist_ok=True)
    plt.savefig(f"figs/{stage}_{variable}_scatter.png", dpi=300)


def check_raster_resolution(raster):
    ## check if the raster resolution is not 250 resolution
    assert raster.shape[0] == 250, f"Raster resolution is not 250, the actual resolution is {raster.shape[0]}"
    assert raster.shape[1] == 250, f"Raster resolution is not 250, the actual resolution is {raster.shape[1]}"


        

def plot_variable(geodataframe, variable, logger, vmin=None, vmax=None, stage='None', raster_size=(250, 250)):
    assert variable in geodataframe.columns, f"{variable} not in columns"
    assert "geometry" in geodataframe.columns, "geometry not in columns"
    
    # Filter out negative values
    geodataframe = geodataframe[geodataframe[variable] > 0]
    logger.info(f"Shape after removing negative values: {geodataframe.shape}")
    
    # Get bounds for rasterization
    bounds = geodataframe.total_bounds  # (minx, miny, maxx, maxy)
    
    # Define the transform and create an empty raster array
    transform = from_bounds(*bounds, raster_size[1], raster_size[0])
    raster = np.zeros(raster_size, dtype='float32')
    
    # Rasterize the geometries based on the variable values
    shapes = ((geom, value) for geom, value in zip(geodataframe.geometry, geodataframe[variable]))
    rasterized = rasterize(shapes=shapes, out_shape=raster.shape, transform=transform, fill=np.nan)
    
    check_raster_resolution(rasterized)


    # Plot the raster
    fig, ax = plt.subplots()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cax = ax.imshow(rasterized, cmap='viridis', norm=norm, extent=bounds)
    
    plt.colorbar(cax, ax=ax, label=variable)
    plt.title(f"{variable}")
    
    # Save the plot
    os.makedirs('figs', exist_ok=True)
    plt.savefig(f"figs/{stage}_{variable}_raster.png", dpi=300)
    
    logger.info(f"Raster plot saved as figs/{stage}_{variable}_raster.png")
    plt.show()
