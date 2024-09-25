
import logging

def setup_logger(log_file_path):    
    # Clear the log file at the start
    with open(log_file_path, 'w') as f:
        f.write("")

    # Create a custom logger
    logger = logging.getLogger(log_file_path)

    # Set the logging level
    logger.setLevel(logging.INFO)

    # Create handlers
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatters and add them to handlers
    formatter = logging.Formatter('%(message)s')  # Only log the message itself
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger