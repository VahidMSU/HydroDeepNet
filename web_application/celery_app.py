from celery import Celery
import os
from redis import Redis, ConnectionError, exceptions
import socket
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_working_redis_url():
    """
    Try multiple Redis connections to find a working one.
    Returns a working Redis URL or the default if none work.
    """
    default_url = 'redis://localhost:6379/0'
    alternative_urls = [
        'redis://127.0.0.1:6379/0',  # Try explicit IP
        'redis://redis:6379/0',       # Try service name if using Docker
        # Add potential production-specific Redis URL if different
        'redis://localhost:6379/0?socket_timeout=5'
    ]
    
    # Add retry logic to handle temporary Redis unavailability
    max_retries = 5
    retry_delay = 1  # seconds
    loading_retry_delay = 2  # seconds for loading dataset, needs more time
    
    # Log environment variables for debugging
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    logger.info(f"PATH: {os.environ.get('PATH', 'Not set')}")
    logger.info(f"FLASK_ENV: {os.environ.get('FLASK_ENV', 'Not set')}")
    
    for attempt in range(max_retries):
        # First try the default
        try:
            logger.info(f"Testing Redis connection to {default_url} (attempt {attempt+1}/{max_retries})")
            client = Redis.from_url(default_url, socket_timeout=3, socket_connect_timeout=3)
            client.ping()
            logger.info("✅ Successfully connected to Redis using default URL")
            return default_url
        except exceptions.BusyLoadingError as e:
            # This is a temporary error - Redis is starting up and loading dataset
            logger.warning(f"Redis at {default_url} is currently loading dataset. Waiting {loading_retry_delay}s...")
            time.sleep(loading_retry_delay)
            loading_retry_delay = min(loading_retry_delay * 1.5, 10)  # Increase delay but cap it
            continue  # Try again with the same URL
        except (ConnectionError, socket.error) as e:
            logger.warning(f"Could not connect to Redis at {default_url}: {e}")
        
        # Try alternatives
        for url in alternative_urls:
            try:
                logger.info(f"Testing Redis connection to {url} (attempt {attempt+1}/{max_retries})")
                client = Redis.from_url(url, socket_timeout=3, socket_connect_timeout=3)
                client.ping()
                logger.info(f"✅ Successfully connected to Redis using alternative URL: {url}")
                return url
            except exceptions.BusyLoadingError as e:
                # This is a temporary error - Redis is starting up and loading dataset
                logger.warning(f"Redis at {url} is currently loading dataset. Waiting {loading_retry_delay}s...")
                time.sleep(loading_retry_delay)
                loading_retry_delay = min(loading_retry_delay * 1.5, 10)
                # Don't continue to next URL - try this one again
                break
            except (ConnectionError, socket.error) as e:
                logger.warning(f"Could not connect to Redis at {url}: {e}")
        
        # Wait before retrying
        if attempt < max_retries - 1:
            logger.warning(f"Retrying Redis connection in {retry_delay} seconds...")
            time.sleep(retry_delay)
            # Increase delay for next attempt (exponential backoff)
            retry_delay = min(retry_delay * 2, 10)
    
    # If nothing works after all retries, return default and let Celery handle retries
    logger.warning("⚠️ No working Redis connection found after retries, using default with retry")
    return default_url

def make_celery(app=None):
    """Create a Celery instance"""
    # Get the working Redis URL - with more graceful handling
    try:
        redis_url = get_working_redis_url()
        logger.info(f"Initializing Celery with Redis URL: {redis_url}")
    except Exception as e:
        logger.error(f"Error finding working Redis URL: {e}")
        # Fall back to default URL and let Celery handle reconnection
        redis_url = 'redis://localhost:6379/0'
        logger.info(f"Falling back to default Redis URL: {redis_url}")
    
    # Create a more robust Celery configuration
    celery = Celery(
        'web_application',
        broker=redis_url,
        backend=redis_url,
        include=['app.tasks']
    )
    
    # Configure broker_connection_retry_on_startup to address deprecation warning
    celery.conf.broker_connection_retry_on_startup = True
    
    # Set additional Celery configuration for better reliability
    celery.conf.broker_connection_retry = True
    celery.conf.broker_connection_max_retries = 10
    celery.conf.broker_transport_options = {
        'socket_timeout': 10,
        'socket_connect_timeout': 10,
        'visibility_timeout': 43200,  # 12 hours
        'retry_on_timeout': True
    }
    
    # Save configuration to file for debugging
    try:
        config_log = os.path.join('/data/SWATGenXApp/codes/web_application/logs', 'celery_config.log')
        with open(config_log, 'w') as f:
            f.write(f"Broker URL: {redis_url}\n")
            f.write(f"Result Backend: {redis_url}\n")
            f.write(f"Include: {celery.conf.include}\n")
            f.write(f"Broker retry on startup: {celery.conf.broker_connection_retry_on_startup}\n")
            f.write(f"Environment: {os.environ.get('FLASK_ENV', 'Not set')}\n")
        logger.info(f"Saved Celery configuration to {config_log}")
    except Exception as e:
        logger.error(f"Error saving Celery configuration: {e}")
    
    # Optional: Configure Celery from Flask app config
    if app:
        celery.conf.update(app.config)
        
        class ContextTask(celery.Task):
            def __call__(self, *args, **kwargs):
                with app.app_context():
                    return self.run(*args, **kwargs)
                    
        celery.Task = ContextTask
        
    return celery

# Create the celery instance - but do this more safely
try:
    celery = make_celery()
    logger.info("Successfully created Celery instance")
except Exception as e:
    # Log but don't crash, let the actual user of this module handle it
    logger.error(f"Error creating Celery instance: {e}")
    # Create a minimal Celery instance that will be reconfigured later
    celery = Celery('web_application')
    celery.conf.broker_connection_retry = True
    celery.conf.broker_connection_retry_on_startup = True