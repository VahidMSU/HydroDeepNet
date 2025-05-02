import logging
import os
import tempfile
from MODGenX.path_handler import PathHandler

class Logger:
    """
    A simple logger class that logs messages to a file and console with timestamps.
    """
    def __init__(self, verbose=True, rewrite=False, name="MODFLOWGenXLogger", config=None, path_handler=None):
        """
        Initialize the Logger class.

        Args:
            verbose (bool): Whether to print logs to console.
            rewrite (bool): Whether to rewrite the log file if it exists.
            name (str): Logger name.
            config (MODFLOWGenXPaths): Configuration object.
            path_handler (PathHandler): Path handler instance.
        """
        # Ensure either path_handler or config is provided
        if not path_handler and not config:
            raise ValueError("Either path_handler or config must be provided to Logger")
            
        # Use path_handler if provided
        if path_handler:
            try:
                self.report_path = os.path.dirname(path_handler.get_log_path(name))
                self.log_file = path_handler.get_log_path(name)
            except Exception as e:
                # Fall back to a temporary log file if path_handler fails
                self._setup_temp_logging(name, str(e))
        elif config:
            try:
                # Create a temporary path handler if config is provided
                temp_handler = PathHandler(config)
                self.report_path = os.path.dirname(temp_handler.get_log_path(name))
                self.log_file = temp_handler.get_log_path(name)
            except Exception as e:
                # Fall back to a temporary log file if PathHandler fails
                self._setup_temp_logging(name, str(e))

        # Create the directory if it doesn't exist
        try:
            os.makedirs(self.report_path, exist_ok=True)
        except PermissionError:
            # Fall back to a temporary directory if we can't create the report path
            self._setup_temp_logging(name, f"Permission denied when creating {self.report_path}")
        
        self.verbose = verbose
        self.rewrite = rewrite
        self.logger = self._setup_logger(name)
    
    def _setup_temp_logging(self, name, error_msg):
        """Set up temporary logging when normal paths fail"""
        print(f"Warning: Using temporary log file. Error: {error_msg}")
        temp_dir = tempfile.gettempdir()
        self.report_path = temp_dir
        self.log_file = os.path.join(temp_dir, f"{name}.log")
    
    def _setup_logger(self, name):
        """
        Set up the logger to log messages to a file and console.

        Returns:
            logging.Logger: Configured logger.
        """
        if os.path.exists(self.log_file) and self.rewrite:
            try:
                os.remove(self.log_file)
            except (PermissionError, OSError) as e:
                print(f"Warning: Could not remove existing log file: {str(e)}")

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File Handler
        try:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(log_format)
            logger.addHandler(file_handler)
        except (PermissionError, OSError) as e:
            print(f"Warning: Could not create log file: {str(e)}")
            # Do not add file handler if we can't create the log file
        
        # Console Handler (always works)
        if self.verbose:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(log_format)
            logger.addHandler(console_handler)
        
        # Only log to console if we couldn't create the log file
        if len(logger.handlers) > 0:
            logger.info(f"Logging initialized at {self.log_file}")
        else:
            # Make sure we have at least one handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(log_format)
            logger.addHandler(console_handler)
            logger.info("Logging initialized (console only, could not create log file)")
        
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