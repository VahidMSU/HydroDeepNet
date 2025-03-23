"""
Utility functions for cleaning up and validating data in MODGenX.
These functions should be used to ensure data integrity throughout the workflow.
"""

from typing import Optional, Tuple, Union, Callable
import numpy as np
import os
import matplotlib.pyplot as plt
from MODGenX.logger_singleton import get_logger
from numpy.typing import NDArray, ArrayLike

logger = get_logger()

def clean_nan_values(
    array: NDArray, 
    fill_strategy: str = 'mean', 
    fill_value: Optional[float] = None, 
    allow_zeros: bool = True, 
    name: str = "array"
) -> NDArray:
    """
    Clean NaN values in a numpy array.
    
    Parameters:
    -----------
    array : numpy.ndarray
        Array to clean
    fill_strategy : str, optional
        Strategy for filling NaN values: 'mean', 'median', 'zero', 'value'
    fill_value : float, optional
        Value to use if fill_strategy is 'value'
    allow_zeros : bool, optional
        Whether to allow zeros in the array
    name : str, optional
        Name of the array for logging purposes
    
    Returns:
    --------
    numpy.ndarray
        Cleaned array
    """
    # Check if there are any NaN values
    if not np.isnan(array).any():
        return array
    
    # Create a copy to avoid modifying the original
    cleaned = array.copy()
    
    # Identify NaN values
    nan_mask = np.isnan(cleaned)
    nan_count = np.sum(nan_mask)
    logger.info(f"Found {nan_count} NaN values in {name} with shape {array.shape}")
    
    # Fill NaN values based on strategy
    if fill_strategy == 'mean':
        if allow_zeros:
            fill_val = np.nanmean(cleaned)
        else:
            valid_vals = cleaned[(~nan_mask) & (cleaned > 0)]
            fill_val = np.mean(valid_vals) if len(valid_vals) > 0 else 0
    elif fill_strategy == 'median':
        if allow_zeros:
            fill_val = np.nanmedian(cleaned)
        else:
            valid_vals = cleaned[(~nan_mask) & (cleaned > 0)]
            fill_val = np.median(valid_vals) if len(valid_vals) > 0 else 0
    elif fill_strategy == 'zero':
        fill_val = 0
    elif fill_strategy == 'value' and fill_value is not None:
        fill_val = fill_value
    else:
        logger.warning(f"Invalid fill_strategy: {fill_strategy}, using mean")
        fill_val = np.nanmean(cleaned)
    
    logger.info(f"Filling NaN values in {name} with {fill_val} using strategy '{fill_strategy}'")
    cleaned[nan_mask] = fill_val
    
    return cleaned

def ensure_directory(directory_path, clean=False):
    """
    Ensure a directory exists and optionally clean it.
    
    Parameters:
    -----------
    directory_path : str
        Path to directory
    clean : bool, optional
        Whether to delete and recreate the directory
        
    Returns:
    --------
    bool
        True if operation was successful, False otherwise
    """
    try:
        if clean and os.path.exists(directory_path):
            import shutil
            shutil.rmtree(directory_path)
            logger.info(f"Cleaned directory: {directory_path}")
        
        os.makedirs(directory_path, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory_path}")
        return True
    except Exception as e:
        logger.error(f"Error ensuring directory {directory_path}: {str(e)}")
        return False

def validate_raster(array, name="array", min_threshold=None, max_threshold=None, allow_constant=False):
    """
    Validate a raster array for common issues.
    
    Parameters:
    -----------
    array : numpy.ndarray
        Array to validate
    name : str, optional
        Name of the array for logging purposes
    min_threshold : float, optional
        Minimum acceptable value threshold
    max_threshold : float, optional
        Maximum acceptable value threshold
    allow_constant : bool, optional
        Whether to allow constant values across the array
        
    Returns:
    --------
    bool
        True if validation passes, False otherwise
    """
    # Check for NaN values
    if np.isnan(array).any():
        nan_count = np.sum(np.isnan(array))
        logger.warning(f"{name} contains {nan_count} NaN values")
        return False
    
    # Check for constant values
    if not allow_constant and np.min(array) == np.max(array):
        logger.warning(f"{name} has constant value {np.min(array)}")
        return False
    
    # Check for value range
    if min_threshold is not None and np.min(array) < min_threshold:
        logger.warning(f"{name} minimum value {np.min(array)} is below threshold {min_threshold}")
        return False
        
    if max_threshold is not None and np.max(array) > max_threshold:
        logger.warning(f"{name} maximum value {np.max(array)} is above threshold {max_threshold}")
        return False
    
    return True

def plot_diagnostic(array, title, output_path, cmap='viridis', vmin=None, vmax=None):
    """
    Create a diagnostic plot of an array.
    
    Parameters:
    -----------
    array : numpy.ndarray
        Array to plot
    title : str
        Plot title
    output_path : str
        Path to save the plot
    cmap : str, optional
        Colormap to use
    vmin : float, optional
        Minimum value for color scale
    vmax : float, optional
        Maximum value for color scale
        
    Returns:
    --------
    bool
        True if plot was created successfully, False otherwise
    """
    try:
        # Create figure with specified dimensions
        plt.figure(figsize=(10, 8))
        
        # Plot the array data
        im = plt.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, label=title)
        plt.title(title)
        
        # Set axes grid lines at 50 interval spacing starting from 0
        rows, cols = array.shape
        
        # Calculate appropriate tick intervals
        x_interval = max(1, cols // 50)
        y_interval = max(1, rows // 50)
        
        # Set ticks at exact intervals of 50 starting from 0
        x_ticks = np.arange(0, cols, x_interval)
        y_ticks = np.arange(0, rows, y_interval)
        
        plt.xticks(x_ticks)
        plt.yticks(y_ticks)
        
        # Add grid based on these ticks
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add axes labels
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        
        # Save with tight layout and high resolution
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved diagnostic plot to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating diagnostic plot: {str(e)}")
        return False
