import logging
import os


class LoggerSetup:
	def __init__(self, report_path="/data/SWATGenXApp/codes/assistant/logs/", name="AIAssistantLogger", verbose=True, rewrite=False):
		"""
		Initialize the LoggerSetup class.

		Args:
			report_path (str): Path to the directory where the log file will be saved.
			verbose (bool): Whether to print logs to console. Defaults to True.
		"""
		self.report_path = report_path
		self.logger = None
		self.verbose = verbose
		self.rewrite = rewrite
		self.name = name

		# Check if logger with this name already exists to prevent handler duplication
		existing_logger = logging.getLogger(name)
		if existing_logger.handlers:
			# Logger already exists, use it
			self.logger = existing_logger
		else:
			# Create a new logger
			self._setup_logger()

	def _setup_logger(self):
		"""Set up a new logger with file and optional console handlers"""
		# Define the path for the log file
		path = os.path.join(self.report_path, f"{self.name}.log")
		if self.rewrite and os.path.exists(path):
			os.remove(path)
			
		# Create a logger
		self.logger = logging.getLogger(self.name)
		self.logger.setLevel(logging.INFO)  # Set the logging level
		
		# Prevent propagation to the root logger to avoid duplicate logs
		self.logger.propagate = False

		# Create formatters
		timestamp_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
		simple_format = logging.Formatter('%(levelname)s - %(message)s')

		# FileHandler for logging to a file
		file_handler = logging.FileHandler(path)
		file_handler.setLevel(logging.INFO)
		file_handler.setFormatter(timestamp_format)
		self.logger.addHandler(file_handler)

		# Conditionally add console handler based on verbose flag
		if self.verbose:
			console_handler = logging.StreamHandler()
			console_handler.setLevel(logging.INFO)
			console_handler.setFormatter(timestamp_format)
			self.logger.addHandler(console_handler)

		self.logger.info(f"Logging to {path}")

	def error(self, message, time_stamp=True):
		"""
		Log an error message.

		Args:
			message (str): The error message to log.
			time_stamp (bool): Whether to include a timestamp in the log.
		"""
		self.logger.error(message)

	def warning(self, message, time_stamp=True):
		"""
		Log a warning message.

		Args:
			message (str): The warning message to log.
			time_stamp (bool): Whether to include a timestamp in the log.
		"""
		self.logger.warning(message)

	def info(self, message, level="info", time_stamp=True):
		"""
		Log a message.

		Args:
			message (str): The message to log.
			level (str): The logging level (e.g., "info", "error").
			time_stamp (bool): Whether to include a timestamp in the log.
		"""
		# Use the appropriate logging method based on level
		log_methods = {
			"info": self.logger.info,
			"error": self.logger.error,
			"warning": self.logger.warning,
			"debug": self.logger.debug
		}
		log_method = log_methods.get(level.lower(), self.logger.info)
		log_method(message)

	def debug(self, message, time_stamp=True):
		"""
		Log a debug message.
		"""
		self.logger.debug(message)


