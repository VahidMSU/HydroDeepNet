"""
Plot utilities for safe matplotlib usage in multiprocessing environments.

This module provides helper functions to ensure matplotlib figures are properly
created and closed, preventing memory leaks and corruption in parallel processing.
"""
import matplotlib.pyplot as plt
from contextlib import contextmanager
from typing import Optional, Tuple, Any
import logging

# Configure logger
logger = logging.getLogger(__name__)

@contextmanager
def safe_figure(figsize: Optional[Tuple[float, float]] = None, 
               dpi: Optional[int] = None, 
               facecolor: Optional[str] = None,
               **fig_kwargs) -> plt.Figure:
    """
    Context manager for safely creating and closing matplotlib figures.
    
    Args:
        figsize: Optional figure size as (width, height) in inches
        dpi: Optional dots per inch for the figure
        facecolor: Optional background color
        **fig_kwargs: Additional keyword arguments for plt.figure()
        
    Yields:
        Matplotlib Figure object
        
    Example:
        with safe_figure(figsize=(10, 6)) as fig:
            ax = fig.add_subplot(111)
            ax.plot(x, y)
            fig.savefig('output.png')
    """
    try:
        # Create figure with specified parameters
        kwargs = {}
        if figsize is not None:
            kwargs['figsize'] = figsize
        if dpi is not None:
            kwargs['dpi'] = dpi
        if facecolor is not None:
            kwargs['facecolor'] = facecolor
            
        # Add any additional kwargs
        kwargs.update(fig_kwargs)
        
        # Create the figure
        fig = plt.figure(**kwargs)
        yield fig
    finally:
        # Always close the figure, even if an exception occurs
        plt.close(fig)

def save_figure(fig: plt.Figure, 
               output_path: str, 
               dpi: Optional[int] = 300, 
               bbox_inches: str = 'tight', 
               **save_kwargs) -> bool:
    """
    Safely save a matplotlib figure and close it.
    
    Args:
        fig: Matplotlib Figure object
        output_path: Path where to save the figure
        dpi: Resolution in dots per inch
        bbox_inches: Bounding box in inches ('tight' adjusts to content)
        **save_kwargs: Additional keyword arguments for fig.savefig()
        
    Returns:
        True if successful, False otherwise
        
    Example:
        fig, ax = plt.subplots()
        ax.plot(x, y)
        save_figure(fig, 'output.png')
    """
    try:
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Set default kwargs
        kwargs = {'dpi': dpi, 'bbox_inches': bbox_inches}
        
        # Add any additional kwargs
        kwargs.update(save_kwargs)
        
        # Save the figure
        fig.savefig(output_path, **kwargs)
        
        # Close the figure
        plt.close(fig)
        return True
    except Exception as e:
        logger.error(f"Error saving figure to {output_path}: {e}")
        # Make sure figure is closed even if saving fails
        try:
            plt.close(fig)
        except:
            pass
        return False

def close_all_figures():
    """
    Close all open matplotlib figures to prevent memory leaks and figure conflicts.
    This should be called after generating reports to ensure clean state.
    
    Returns:
        None
    """
    try:
        # Get the number of open figures
        n_figs = len(plt.get_fignums())
        
        # Close all figures
        plt.close('all')
        
        if n_figs > 0:
            logger.info(f"Closed {n_figs} matplotlib figures")
    except Exception as e:
        logger.warning(f"Error closing matplotlib figures: {e}")
