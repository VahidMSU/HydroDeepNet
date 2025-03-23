"""
Singleton logger module for MODGenX.
This ensures we have a consistent logger throughout the application.
"""

import os
import logging
import tempfile
from MODGenX.Logger import Logger
from MODGenX.config import MODFLOWGenXPaths
from MODGenX.path_handler import PathHandler

# Global logger instance
_logger = None

def initialize_logger(verbose=True, config=None, path_handler=None, name="MODFLOWGenXLogger"):
    """
    Initialize the global logger.
    
    Args:
        verbose (bool): Whether to print logs to console.
        config (MODFLOWGenXPaths): Configuration object.
        path_handler (PathHandler): Path handler instance.
        name (str): Logger name.
        
    Returns:
        Logger: The initialized logger.
    """
    global _logger
    
    if _logger is None:
        if not path_handler and not config:
            # Create a temporary directory for logs that doesn't require special permissions
            temp_log_dir = tempfile.mkdtemp(prefix="modgenx_logs_")
            
            # Create default config with temporary paths that don't require elevated permissions
            config = MODFLOWGenXPaths(
                username="temp_user",
                BASE_PATH=temp_log_dir,
                MODFLOW_MODEL_NAME="default_model",
                LEVEL="default",
                VPUID="default",
                NAME="default",
                RESOLUTION=250,
                report_path=temp_log_dir  # Set report path to temp directory
            )
            
            # Don't validate the config as it's just for temporary logging
            # config.validate()
            
            # Create special fallback logger directly using Python's logging
            fallback_logger = create_fallback_logger(name, temp_log_dir)
            return FallbackLogger(fallback_logger)
        
        try:
            # Normal initialization with provided config or path_handler
            _logger = Logger(verbose=verbose, name=name, config=config, path_handler=path_handler)
        except Exception as e:
            print(f"Warning: Failed to initialize custom logger: {str(e)}")
            print("Falling back to standard logging")
            
            # Create a basic logger as fallback
            fallback_logger = create_fallback_logger(name)
            _logger = FallbackLogger(fallback_logger)
    
    return _logger

def create_fallback_logger(name, log_dir=None):
    """Create a standard Python logger as fallback"""
    fallback_logger = logging.getLogger(name)
    fallback_logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    if fallback_logger.handlers:
        fallback_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    fallback_logger.addHandler(console_handler)
    
    # Add file handler if log_dir is provided
    if log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"{name}.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            fallback_logger.addHandler(file_handler)
            fallback_logger.info(f"Logging to file: {log_file}")
        except Exception as e:
            print(f"Warning: Failed to create log file: {str(e)}")
    
    return fallback_logger

class FallbackLogger:
    """Wrapper class that matches our Logger interface"""
    def __init__(self, logger):
        self.logger = logger
    
    def info(self, message):
        self.logger.info(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def error(self, message):
        self.logger.error(message)

def get_logger():
    """
    Get the global logger instance, initializing it if necessary.
    
    Returns:
        Logger: The global logger instance.
    """
    global _logger
    
    if _logger is None:
        _logger = initialize_logger()
    
    return _logger
