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

        # Configure additional logging settings
        self.configure_logging()
        
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
        if not self.logger:
            return
        self.logger.error(message)

    def warning(self, message, time_stamp=True):
        """
        Log a warning message.

        Args:
            message (str): The warning message to log.
            time_stamp (bool): Whether to include a timestamp in the log.
        """
        if not self.logger:
            return
        self.logger.warning(message)

    def info(self, message, level="info", time_stamp=True):
        """
        Log a message with the specified level.

        Args:
            message (str): The message to log.
            level (str): The logging level (e.g., "info", "error").
            time_stamp (bool): Whether to include a timestamp in the log.
        """
        if not self.logger:
            return
            
        log_methods = {
            "info": self.logger.info,
            "error": self.logger.error,
            "warning": self.logger.warning,
            "debug": self.logger.debug
        }
        log_method = log_methods.get(level.lower(), self.logger.info)
        log_method(message)




    def configure_logging(self):
        """Configure logging for both our code and Agno's logging."""
        try:
           
            
            log_file = "/data/SWATGenXApp/codes/AI_agent/logs/AI_AgentLogger.log"
            
            # Create a file handler that writes directly to our log file
            # rather than going through our logger (which would cause recursion)
            if log_file and os.path.exists(os.path.dirname(log_file)):
                file_handler = logging.FileHandler(log_file)
                formatter = logging.Formatter('%(asctime)s - AGNO - %(levelname)s - %(message)s')
                file_handler.setFormatter(formatter)
                file_handler.setLevel(logging.WARNING)  # Only capture warnings and errors
                
                # Capture all existing loggers
                root_logger = logging.getLogger()
                
                # Remove any existing handlers to prevent double logging
                for handler in list(root_logger.handlers):
                    if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                        # Keep file handlers but remove console handlers
                        root_logger.removeHandler(handler)
                
                # Add our direct file handler to the root logger
                root_logger.addHandler(file_handler)
                
                # Set level on the root logger
                root_logger.setLevel(logging.WARNING)
                
                # Silence specific noisy loggers
                for logger_name in ['openai', 'httpx', 'urllib3', 'asyncio', 'agno']:
                    logging.getLogger(logger_name).setLevel(logging.WARNING)
                
                self.logger.info("Configured Agno logging to write directly to log file")
            else:
                # Fallback: just silence everything to avoid console output
                logging.getLogger().setLevel(logging.ERROR)
                for logger_name in ['openai', 'httpx', 'urllib3', 'asyncio', 'agno']:
                    logging.getLogger(logger_name).setLevel(logging.ERROR)
                self.logger.info("Configured Agno logging with minimal output (no log file found)")
            
        except Exception as e:
            self.logger.error(f"Error configuring logging: {str(e)}")
