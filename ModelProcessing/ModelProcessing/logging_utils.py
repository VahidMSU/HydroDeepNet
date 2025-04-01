"""
Central logging module for ModelProcessing package.

This module provides a standardized logging setup that can be used
throughout the ModelProcessing package to ensure consistent
log handling and formatting.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
import datetime
import sys

# Create global logger dictionary to store loggers by name
LOGGERS = {}

# Constants for logging
DEFAULT_LOG_FILE = '/data/SWATGenXApp/codes/ModelProcessing/logs/ModelProcessing.log'
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
DEFAULT_MAX_BYTES = 10485760  # 10MB
DEFAULT_BACKUP_COUNT = 5

def setup_logger(
    name='ModelProcessing',
    log_file=DEFAULT_LOG_FILE,
    level=DEFAULT_LOG_LEVEL,
    console_output=True,
    file_output=True,
    format_string=DEFAULT_FORMAT,
    date_format=DEFAULT_DATE_FORMAT,
    max_bytes=DEFAULT_MAX_BYTES,
    backup_count=DEFAULT_BACKUP_COUNT
):
    """
    Set up a logger with consistent formatting and handlers.
    
    Args:
        name (str): Logger name, used for hierarchical logging
        log_file (str): Path to log file (if None, no file logging)
        level (int): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output (bool): Whether to output logs to console
        file_output (bool): Whether to output logs to file
        format_string (str): Format string for log messages
        date_format (str): Format string for dates in log messages
        max_bytes (int): Maximum size for log files before rotation
        backup_count (int): Number of backup log files to keep
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Check if we already created this logger
    if name in LOGGERS:
        return LOGGERS[name]
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Avoid duplicate logs
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(format_string, date_format)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if requested and log_file provided
    if file_output and log_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Use rotating file handler to manage log size
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Store the logger
    LOGGERS[name] = logger
    
    return logger

def get_logger(name):
    """
    Creates and returns a logger with the given name.
    
    Args:
        name: Name for the logger, typically __name__ of the calling module
        
    Returns:
        A configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only configure if handlers haven't been added yet
    if not logger.handlers:
        # Set the log level
        logger.setLevel(logging.INFO)
        
        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                                    datefmt='%Y-%m-%d %H:%M:%S')
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
        
        # Prevent propagation to the root logger to avoid duplicate messages
        logger.propagate = False
    
    return logger

def log_to_file(message, file_path):
    """
    Write a log message directly to a specified file.
    This is used for model-specific logs that need to be in specific locations.
    
    Args:
        message (str): Message to write to the log file
        file_path (str): Path to the log file
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Format timestamp
        timestamp = datetime.datetime.now().strftime(DEFAULT_DATE_FORMAT)
        
        # Open file in append mode
        with open(file_path, 'a') as f:
            f.write(f"{timestamp} - {message}\n")
    except Exception as e:
        # Log the error to the main logger
        logger = get_logger()
        logger.error(f"Failed to write log to {file_path}: {str(e)}")
