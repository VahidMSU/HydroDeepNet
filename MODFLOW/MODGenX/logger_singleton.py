"""
Singleton logger module for the entire application.
This ensures consistent logging across all modules.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from MODGenX.Logger import Logger

# Global logger instance
_global_logger = None

def get_logger(name="MODFLOWGenXLogger", path_handler=None, config=None, 
               report_path=None, verbose=True, log_level=logging.INFO,
               max_bytes=10485760, backup_count=5):
    """
    Get a singleton logger instance.
    
    Parameters:
    -----------
    name : str, optional
        Logger name
    path_handler : PathHandler, optional
        Path handler instance
    config : MODFLOWGenXPaths, optional
        Configuration object
    report_path : str, optional
        Custom report path
    verbose : bool, optional
        Whether to print to console
    log_level : int, optional
        Logging level (default: logging.INFO)
    max_bytes : int, optional
        Maximum log file size before rotation
    backup_count : int, optional
        Number of backup files to keep
        
    Returns:
    --------
    Logger
        Singleton logger instance
    """
    global _global_logger
    
    if _global_logger is None:
        _global_logger = Logger(
            name=name,
            path_handler=path_handler,
            config=config,
            report_path=report_path,
            verbose=verbose
        )
        
        # Configure the logger with additional options
        for handler in _global_logger.logger.handlers:
            handler.setLevel(log_level)
            
            # Convert file handlers to RotatingFileHandler
            if isinstance(handler, logging.FileHandler) and not isinstance(handler, RotatingFileHandler):
                file_path = handler.baseFilename
                _global_logger.logger.removeHandler(handler)
                
                # Create rotating handler instead
                rotating_handler = RotatingFileHandler(
                    file_path, 
                    maxBytes=max_bytes, 
                    backupCount=backup_count
                )
                rotating_handler.setFormatter(handler.formatter)
                rotating_handler.setLevel(log_level)
                _global_logger.logger.addHandler(rotating_handler)
    
    return _global_logger
