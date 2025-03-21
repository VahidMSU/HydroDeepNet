import os
import sys
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='/data/SWATGenXApp/codes/web_application/logs/celery_worker_startup.log'
)
logger = logging.getLogger('celery_worker')

# Log startup information
logger.info("Starting Celery worker...")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Python path: {sys.path}")
logger.info(f"FLASK_ENV: {os.environ.get('FLASK_ENV', 'Not set')}")

# Add the project directory to Python path (one level up from this file)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
logger.info(f"Added to Python path: {base_dir}")

# Explicitly add the web_application directory to the path
web_app_dir = os.path.dirname(os.path.abspath(__file__))
if web_app_dir not in sys.path:
    sys.path.insert(0, web_app_dir)
    logger.info(f"Added to Python path: {web_app_dir}")

# Import Redis for pre-flight checks
from redis import Redis, exceptions

# Function to check if Redis is ready
def wait_for_redis_ready(max_retries=10, initial_delay=1):
    """Wait for Redis to be ready, handling loading state."""
    delay = initial_delay
    redis_urls = [
        'redis://localhost:6379/0',
        'redis://127.0.0.1:6379/0',
        'redis://redis:6379/0'
    ]
    
    logger.info(f"Testing Redis readiness with URLs: {redis_urls}")
    
    for attempt in range(max_retries):
        for url in redis_urls:
            try:
                logger.info(f"Testing Redis readiness at {url} (attempt {attempt+1}/{max_retries})")
                client = Redis.from_url(url, socket_timeout=3, socket_connect_timeout=3)
                client.ping()
                logger.info(f"✅ Redis is ready at {url}")
                return True, url
            except exceptions.BusyLoadingError:
                logger.warning(f"Redis at {url} is loading dataset. Waiting {delay}s before retry...")
                time.sleep(delay)
                delay = min(delay * 1.5, 10)  # Increase delay but cap it
            except Exception as e:
                logger.error(f"Error connecting to Redis at {url}: {e}")
    
    logger.error(f"Redis not ready after {max_retries} attempts on any URL")
    return False, None

# Check Redis readiness before importing Celery
is_ready, working_url = wait_for_redis_ready(max_retries=10)
if is_ready and working_url:
    # Set environment variable for celery_app to use
    os.environ['REDIS_URL'] = working_url
    logger.info(f"Set REDIS_URL environment variable to {working_url}")

# Import and configure Celery instance
try:
    # Import the celery instance directly from the celery_app module
    from celery_app import celery
    logger.info(f"Celery instance imported successfully from celery_app")
    logger.info(f"Broker URL: {celery.conf.broker_url}")
    logger.info(f"Backend URL: {celery.conf.result_backend}")
except Exception as e:
    logger.error(f"Error importing Celery instance: {e}")
    # Print traceback for debugging
    import traceback
    logger.error(f"Traceback: {traceback.format_exc()}")
    raise

# This module acts as an entry point for the Celery worker
# The celery -A celery_worker command will import this file and use the "celery" instance

if __name__ == '__main__':
    logger.info("Starting Celery worker from __main__")
    celery.start()