import logging
import os
from MODGenX.path_handler import PathHandler

class Logger:
    """
    A simple logger class that logs messages to a file and console with timestamps.
    """
    def __init__(self, report_path=None, verbose=True, rewrite=False, name="MODFLOWGenXLogger", config=None, path_handler=None):
        """
        Initialize the Logger class.

        Args:
            report_path (str): Directory where log file is stored.
            verbose (bool): Whether to print logs to console.
            rewrite (bool): Whether to rewrite the log file if it exists.
            name (str): Logger name.
            config (MODFLOWGenXPaths): Configuration object.
            path_handler (PathHandler): Path handler instance.
        """
        # Use path_handler if provided
        if path_handler:
            self.report_path = os.path.dirname(path_handler.get_log_path(name))
            self.log_file = path_handler.get_log_path(name)
        elif config:
            # Create a temporary path handler if config is provided
            # Import here to avoid circular imports
            try:
                from MODGenX.path_handler import PathHandler
                temp_handler = PathHandler(config)
                self.report_path = os.path.dirname(temp_handler.get_log_path(name))
                self.log_file = temp_handler.get_log_path(name)
            except ImportError:
                # Fall back to legacy mode if PathHandler can't be imported
                self.report_path = report_path or "/data/SWATGenXApp/codes/MODFLOW/logs"
                self.log_file = os.path.join(self.report_path, f"{name}.log")
        else:
            # Legacy mode
            self.report_path = report_path or "/data/SWATGenXApp/codes/MODFLOW/logs"
            self.log_file = os.path.join(self.report_path, f"{name}.log")
        
        # Create the directory if it doesn't exist
        os.makedirs(self.report_path, exist_ok=True)
        
        self.verbose = verbose
        self.rewrite = rewrite
        self.logger = self._setup_logger(name)
    
    def _setup_logger(self, name):
        """
        Set up the logger to log messages to a file and console.

        Returns:
            logging.Logger: Configured logger.
        """
        if os.path.exists(self.log_file) and self.rewrite:
            os.remove(self.log_file)

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File Handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

        # Console Handler
        if self.verbose:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(log_format)
            logger.addHandler(console_handler)
        
        logger.info(f"Logging initialized at {self.log_file}")
        return logger
    
    def info(self, message):
        """ Log an info message. """
        self.logger.info(message)
    
    def error(self, message):
        """ Log an error message. """
        self.logger.error(message)
    
    def warning(self, message):
        """ Log a warning message. """
        self.logger.warning(message)


if __name__ == "__main__":
    # Test the logger
    logger = Logger(verbose=True)
    logger.info("This is a test info message")
    logger.warning("This is a test warning message")
    logger.error("This is a test error message")
    
    print(f"Log file created at: {logger.log_file}")
    print("Check both console output above and the log file for timestamps")