import logging
import os


class LoggerSetup:
    """
    Class to set up a logger for logging messages to a file and optionally to the console.

    Attributes:
        report_path (str): Path to the directory where the log file will be saved.
        logger (logging.Logger): Configured logger.
        verbose (bool): Whether to print logs to console.
    """
    
    def __init__(self, report_path=None, rewrite=False, verbose=False):
        """
        Initialize the LoggerSetup class.

        Args:
            report_path (str): Path to the directory where the log file will be saved.
            rewrite (bool): Whether to overwrite existing log file. Defaults to False.
            verbose (bool): Whether to print logs to console. Defaults to False.
        """
        self.report_path = "/data/SWATGenXApp/codes/AI_agent/logs"
        self.logger = None
        self.rewrite = rewrite
        self.verbose = verbose

    def setup_logger(self, name="AI_AgentLogger"):
        """
        Set up the logger to log messages to a file and optionally to the console.

        Returns:
            logging.Logger: Configured logger.
        """
        if not self.logger:
            # Define the path for the log file
            path = os.path.join(self.report_path, f"{name}.log")
            if self.rewrite and os.path.exists(path):
                os.remove(path)
            # Create a logger
            self.logger = logging.getLogger(name)
            self.logger.setLevel(logging.INFO)  # Set the logging level

            # FileHandler for logging to a file
            file_handler = logging.FileHandler(path)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)

            # Add console handler only if verbose is True
            if self.verbose:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                self.logger.addHandler(console_handler)

            self.logger.info(f"Logging to {path}")

        return self.logger
        
    def set_verbose(self, verbose):
        """
        Set the verbose mode for logging.
        
        Args:
            verbose (bool): Whether to print logs to console.
        """
        self.verbose = verbose
        
        # If logger already exists, update handlers
        if self.logger:
            # Remove any existing console handlers
            for handler in self.logger.handlers[:]:
                if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                    self.logger.removeHandler(handler)
            
            # Add a new console handler if verbose is True
            if verbose:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                self.logger.addHandler(console_handler)

    def error(self, message, time_stamp=True):
        """
        Log an error message.

        Args:
            message (str): The error message to log.
            time_stamp (bool): Whether to include a timestamp in the log.
        """
        self.info(message, level="error", time_stamp=time_stamp)

    def warning(self, message, time_stamp=True):
        """
        Log a warning message.

        Args:
            message (str): The warning message to log.
            time_stamp (bool): Whether to include a timestamp in the log.
        """
        self.info(message, level="warning", time_stamp=time_stamp)

    def info(self, message, level="info", time_stamp=True):
        """
        Log a message with or without a timestamp.

        Args:
            message (str): The message to log.
            level (str): The logging level (e.g., "info", "error").
            time_stamp (bool): Whether to include a timestamp in the log.
        """
        # Create a temporary logger with the desired format
        temp_logger = logging.getLogger("TempLogger")
        temp_logger.setLevel(self.logger.level)

        # Remove existing handlers to avoid duplicates
        temp_logger.handlers.clear()

        # Define the log format based on the time_stamp flag
        log_format = '%(asctime)s - %(levelname)s - %(message)s' if time_stamp else '%(levelname)s - %(message)s'

        # Add file handler
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                new_file_handler = logging.FileHandler(handler.baseFilename)
                new_file_handler.setFormatter(logging.Formatter(log_format))
                temp_logger.addHandler(new_file_handler)

        # Add console handler only if verbose is True
        if self.verbose:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(log_format))
            temp_logger.addHandler(console_handler)

        # Log the message at the specified level
        log_methods = {
            "info": temp_logger.info,
            "error": temp_logger.error,
            "warning": temp_logger.warning,
            "debug": temp_logger.debug
        }
        log_method = log_methods.get(level.lower(), temp_logger.info)
        log_method(message)


