import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
import sys
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

class Logger:
    """Advanced logging system for the RAG application."""
    
    def __init__(self,
                 log_dir: str = "logs",
                 app_name: str = "rag_system",
                 log_level: int = logging.INFO,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 console_output: bool = True,
                 json_format: bool = False):
        """
        Initialize the logger.
        
        Args:
            log_dir: Directory to store log files
            app_name: Name of the application
            log_level: Logging level
            max_file_size: Maximum size of each log file in bytes
            backup_count: Number of backup files to keep
            console_output: Whether to output logs to console
            json_format: Whether to use JSON format for logs
        """
        self.log_dir = Path(log_dir)
        self.app_name = app_name
        self.log_level = log_level
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.console_output = console_output
        self.json_format = json_format
        
        # Create logger
        self.logger = logging.getLogger(app_name)
        self.logger.setLevel(log_level)
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize handlers
        self.handlers = self._setup_handlers()
        
        # Set propagate to False to avoid duplicate logs
        self.logger.propagate = False
    
    def _setup_handlers(self) -> Dict[str, logging.Handler]:
        """Set up logging handlers."""
        handlers = {}
        
        # Create formatters
        if self.json_format:
            formatter = self._create_json_formatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Main log file handler (size-based rotation)
        main_log_path = self.log_dir / f"{self.app_name}.log"
        main_handler = RotatingFileHandler(
            main_log_path,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        main_handler.setFormatter(formatter)
        main_handler.setLevel(self.log_level)
        self.logger.addHandler(main_handler)
        handlers['main'] = main_handler
        
        # Error log file handler (daily rotation)
        error_log_path = self.log_dir / f"{self.app_name}_error.log"
        error_handler = TimedRotatingFileHandler(
            error_log_path,
            when='midnight',
            interval=1,
            backupCount=self.backup_count
        )
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        self.logger.addHandler(error_handler)
        handlers['error'] = error_handler
        
        # Debug log file handler (size-based rotation)
        debug_log_path = self.log_dir / f"{self.app_name}_debug.log"
        debug_handler = RotatingFileHandler(
            debug_log_path,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        debug_handler.setFormatter(formatter)
        debug_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(debug_handler)
        handlers['debug'] = debug_handler
        
        # Console handler
        if self.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(self.log_level)
            self.logger.addHandler(console_handler)
            handlers['console'] = console_handler
        
        return handlers
    
    def _create_json_formatter(self) -> logging.Formatter:
        """Create a JSON formatter for structured logging."""
        def json_format(record):
            extra = {
                key: value
                for key, value in record.__dict__.items()
                if key not in {
                    'args', 'asctime', 'created', 'exc_info', 'exc_text',
                    'filename', 'funcName', 'levelname', 'levelno', 'lineno',
                    'module', 'msecs', 'msg', 'name', 'pathname', 'process',
                    'processName', 'relativeCreated', 'stack_info', 'thread',
                    'threadName'
                }
            }
            
            log_data = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'logger': record.name,
                'level': record.levelname,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            
            if extra:
                log_data['extra'] = extra
            if record.exc_info:
                log_data['exception'] = self._format_exception(record.exc_info)
            
            return json.dumps(log_data)
        
        return logging.Formatter(json_format)
    
    def _format_exception(self, exc_info) -> Dict[str, str]:
        """Format exception information."""
        if not exc_info:
            return {}
            
        return {
            'type': exc_info[0].__name__,
            'message': str(exc_info[1]),
            'traceback': self._format_traceback(exc_info[2])
        }
    
    def _format_traceback(self, tb) -> str:
        """Format traceback information."""
        import traceback
        return ''.join(traceback.format_tb(tb))
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug message."""
        self.logger.debug(message, extra=extra)
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message."""
        self.logger.info(message, extra=extra)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message."""
        self.logger.warning(message, extra=extra)
    
    def error(self, message: str, exc_info: bool = True, extra: Optional[Dict[str, Any]] = None):
        """Log error message."""
        self.logger.error(message, exc_info=exc_info, extra=extra)
    
    def critical(self, message: str, exc_info: bool = True, extra: Optional[Dict[str, Any]] = None):
        """Log critical message."""
        self.logger.critical(message, exc_info=exc_info, extra=extra)
    
    def log_metric(self, metric_name: str, value: Any, extra: Optional[Dict[str, Any]] = None):
        """Log a metric value."""
        metric_data = {
            'metric_name': metric_name,
            'value': value,
            'timestamp': datetime.now().isoformat()
        }
        if extra:
            metric_data.update(extra)
        
        self.info(f"Metric: {metric_name}", extra=metric_data)
    
    def get_logs(self, 
                 log_type: str = 'main',
                 num_lines: int = 100,
                 level: Optional[str] = None) -> List[str]:
        """
        Get recent log entries.
        
        Args:
            log_type: Type of log to retrieve ('main', 'error', 'debug')
            num_lines: Number of lines to retrieve
            level: Optional level filter
            
        Returns:
            List of log entries
        """
        log_file = self.log_dir / f"{self.app_name}{'_' + log_type if log_type != 'main' else ''}.log"
        
        if not log_file.exists():
            return []
        
        with open(log_file, 'r') as f:
            # Read last n lines
            lines = f.readlines()[-num_lines:]
            
            # Filter by level if specified
            if level:
                level = level.upper()
                lines = [line for line in lines if f" - {level} - " in line]
            
            return lines
    
    def set_level(self, level: int):
        """Set logging level."""
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)
    
    def add_handler(self, handler: logging.Handler):
        """Add a custom handler."""
        if self.json_format:
            handler.setFormatter(self._create_json_formatter())
        else:
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
        self.logger.addHandler(handler)
    
    def remove_handler(self, handler_name: str):
        """Remove a handler by name."""
        if handler_name in self.handlers:
            self.logger.removeHandler(self.handlers[handler_name])
            del self.handlers[handler_name]
    
    def cleanup_old_logs(self, days: int = 30):
        """Clean up log files older than specified days."""
        from datetime import timedelta
        
        cutoff = datetime.now() - timedelta(days=days)
        
        for log_file in self.log_dir.glob(f"{self.app_name}*.log*"):
            if log_file.stat().st_mtime < cutoff.timestamp():
                try:
                    log_file.unlink()
                except Exception as e:
                    self.error(f"Error deleting old log file {log_file}: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        stats = {
            'log_dir': str(self.log_dir),
            'app_name': self.app_name,
            'log_level': logging.getLevelName(self.log_level),
            'handlers': list(self.handlers.keys()),
            'log_files': {}
        }
        
        # Collect stats for each log file
        for log_file in self.log_dir.glob(f"{self.app_name}*.log"):
            try:
                file_stats = log_file.stat()
                stats['log_files'][log_file.name] = {
                    'size': file_stats.st_size,
                    'last_modified': datetime.fromtimestamp(file_stats.st_mtime).isoformat()
                }
            except Exception as e:
                self.error(f"Error getting stats for {log_file}: {str(e)}")
        
        return stats
