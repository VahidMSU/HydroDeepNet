import time
import json
import os
import logging
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'debug.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('SWATGenX')

def log_query_info(query_info):
    """Log structured information about a query."""
    logger.info(f"Query Analysis Result:")
    logger.info(f"  County: {query_info.get('county')}")
    logger.info(f"  State: {query_info.get('state')}")
    logger.info(f"  Years: {query_info.get('years')}")
    logger.info(f"  Analysis Type: {query_info.get('analysis_type')}")
    logger.info(f"  Focus: {query_info.get('focus')}")

def log_data_structure(data):
    """Log the structure of retrieved data."""
    if not data:
        logger.info("Data retrieval result: No data")
        return
        
    logger.info("Data retrieval result:")
    if 'config' in data:
        logger.info(f"  Config: {json.dumps(data['config'])}")
    if 'climate' in data:
        climate = data['climate']
        if climate:
            logger.info(f"  Climate data: {type(climate)} with shape {[len(x) if x is not None else 0 for x in climate]}")
    if 'landcover' in data:
        landcover = data['landcover']
        if landcover:
            logger.info(f"  Landcover data: {len(landcover)} years")
            logger.info(f"  Landcover years: {list(landcover.keys())}")

def timing_log(step_name, start_time, end_time):
    """Log the timing of a processing step."""
    elapsed = end_time - start_time
    logger.info(f"{step_name} completed in {elapsed:.2f} seconds")

def timed_function(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.info(f"{func.__name__} executed in {elapsed_time:.2f} seconds")
        return result
    return wrapper

class PerformanceTracker:
    """Track performance metrics across the application."""
    
    def __init__(self):
        self.metrics = {
            'query_parsing': [],
            'data_retrieval': [],
            'analysis': [],
            'synthesis': [],
            'total_response': [],
            'api_calls': 0,
            'errors': 0
        }
    
    def record(self, metric, value):
        """Record a timing metric."""
        if metric in self.metrics and isinstance(self.metrics[metric], list):
            self.metrics[metric].append(value)
    
    def increment(self, metric):
        """Increment a counter metric."""
        if metric in self.metrics and isinstance(self.metrics[metric], (int, float)):
            self.metrics[metric] += 1
    
    def get_stats(self):
        """Get statistics summary."""
        stats = {}
        for key, values in self.metrics.items():
            if isinstance(values, list) and values:
                stats[f"{key}_avg"] = sum(values) / len(values)
                stats[f"{key}_min"] = min(values)
                stats[f"{key}_max"] = max(values)
                stats[f"{key}_count"] = len(values)
            else:
                stats[key] = values
        return stats
    
    def reset(self):
        """Reset all metrics."""
        for key in self.metrics:
            if isinstance(self.metrics[key], list):
                self.metrics[key] = []
            else:
                self.metrics[key] = 0

# Create singleton instance
performance_tracker = PerformanceTracker()

def log_error(error_msg, exception=None):
    """Log errors with detailed traceback."""
    if exception:
        logger.error(f"{error_msg}: {str(exception)}", exc_info=True)
    else:
        logger.error(error_msg)
    performance_tracker.increment('errors')

def get_performance_stats():
    """Get current performance statistics."""
    return performance_tracker.get_stats()
