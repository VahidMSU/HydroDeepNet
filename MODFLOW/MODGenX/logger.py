import logging
import os
import sys

class MODGenXLogger:
    """
    Centralized logging for the MODGenX module.
    Provides consistent logging across all components.
    """
    
    _instance = None
    LOGS_DIR = "/data/SWATGenXApp/codes/MODFLOW/logs"
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure only one logger instance exists"""
        if cls._instance is None:
            cls._instance = super(MODGenXLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config=None, log_level=logging.INFO, log_to_file=True):
        """
        Initialize the logger.
        
        Parameters:
        - config: ConfigHandler object with path information
        - log_level: Logging level (default: INFO)
        - log_to_file: Whether to log to a file (default: True)
        """
        if self._initialized:
            return
            
        self.logger = logging.getLogger('MODGenX')
        self.logger.setLevel(log_level)
        self.logger.handlers = []  # Clear any existing handlers
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Create file handler if requested
        if log_to_file and config:
            try:
                # Use fixed path for logs directory
                log_dir = self.LOGS_DIR
                os.makedirs(log_dir, exist_ok=True)
                
                # Create a unique log filename based on the model configuration
                model_identifier = f"{config.VPUID}_{config.LEVEL}_{config.NAME}_{config.MODFLOW_MODEL_NAME}"
                log_file = os.path.join(log_dir, f"{model_identifier}.log")
                
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(log_level)
                file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)
                self.logger.info(f"Log file created at: {log_file}")
            except Exception as e:
                self.logger.warning(f"Could not create log file: {str(e)}")
        
        self._initialized = True
    
    def set_level(self, level):
        """Set the logging level"""
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)
    
    def debug(self, message):
        """Log a debug message"""
        self.logger.debug(message)
    
    def info(self, message):
        """Log an info message"""
        self.logger.info(message)
    
    def warning(self, message):
        """Log a warning message"""
        self.logger.warning(message)
    
    def error(self, message):
        """Log an error message"""
        self.logger.error(message)
    
    def critical(self, message):
        """Log a critical message"""
        self.logger.critical(message)
    
    def exception(self, message):
        """Log an exception with traceback"""
        self.logger.exception(message)
