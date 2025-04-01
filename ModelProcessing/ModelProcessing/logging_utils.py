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

def setup_logger(
    name='ModelProcessing',
    log_file=None,
    level=logging.INFO,
    console_output=True,
    file_output=True,
    format_string='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    date_format='%Y-%m-%d %H:%M:%S',
    max_bytes=10485760,  # 10MB
    backup_count=5
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

def get_logger(name='ModelProcessing'):
    """
    Get a logger by name. If it doesn't exist, returns the root ModelProcessing logger.
    
    Args:
        name (str): Logger name
        
    Returns:
        logging.Logger: The requested logger
    """
    if name in LOGGERS:
        return LOGGERS[name]
    
    # If the specific logger doesn't exist, create a child logger
    parent_logger_name = 'ModelProcessing'
    
    # If parent logger doesn't exist, set it up
    if parent_logger_name not in LOGGERS:
        setup_logger(parent_logger_name)
    
    # Create a child logger
    child_logger = logging.getLogger(f"{parent_logger_name}.{name}")
    LOGGERS[name] = child_logger
    
    return child_logger

def log_to_file(message, filepath, include_timestamp=True):
    """
    Append a message to a specific log file.
    
    Args:
        message (str): Message to log
        filepath (str): Path to the log file
        include_timestamp (bool): Whether to include timestamp
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if include_timestamp:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"{timestamp} - {message}"
    
    with open(filepath, 'a') as file:
        file.write(f"{message}\n")