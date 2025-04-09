"""
Centralized logging configuration for the AI agent.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Define log directory
LOG_DIR = "/data/SWATGenXApp/codes/web_application/logs"

def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Optional specific log file name
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler
    if log_file is None:
        log_file = f"{name.lower().replace('.', '_')}.log"
    
    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, log_file),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

# Create specific loggers
def get_ai_logger():
    """Get logger for AI agent operations."""
    return setup_logger('AI_Agent', 'ai_agent.log')

def get_chat_logger():
    """Get logger for chat interactions."""
    return setup_logger('Chat', 'chat.log')

def get_rag_logger():
    """Get logger for RAG operations."""
    return setup_logger('RAG', 'rag.log')

def get_report_logger():
    """Get logger for report generation."""
    return setup_logger('Report', 'report_generation.log') 