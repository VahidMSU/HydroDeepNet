import os
import sys
import logging
import time
import traceback

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

# Function to check and clean corrupted tasks in Redis
def check_redis_queue_corruption(redis_url, max_items=100):
    """
    Check Redis queues for corrupted tasks and log warnings.
    This helps catch the 'KeyError: properties' issue early.
    """
    try:
        logger.info("Checking Redis queues for corruption...")
        client = Redis.from_url(redis_url, socket_timeout=5, socket_connect_timeout=5)
        
        # Check main queue for Celery model tasks
        queue_name = "model_creation"
        queue_length = client.llen(queue_name)
        logger.info(f"Queue '{queue_name}' has {queue_length} messages")
        
        if queue_length == 0:
            logger.info("Queue is empty, no corruption check needed")
            return
        
        # Limit the number of items to check
        items_to_check = min(queue_length, max_items)
        corrupt_items = 0
        
        for i in range(items_to_check):
            try:
                # Get item without removing it
                item = client.lindex(queue_name, i)
                if not item:
                    continue
                
                # Try to decode and validate
                message = item.decode('utf-8')
                data = __import__('json').loads(message)
                
                # Check for the essential 'properties' field that causes crashes
                if 'properties' not in data:
                    corrupt_items += 1
                    logger.warning(f"Corrupted Celery message found at position {i}: Missing 'properties' field")
                    
                    # Log the first 3 corrupt items in detail for debugging
                    if corrupt_items <= 3:
                        logger.warning(f"Corrupt message sample: {str(data)[:200]}...")
            except Exception as e:
                logger.warning(f"Error checking message at position {i}: {e}")
        
        if corrupt_items > 0:
            logger.error(f"⚠️ Found {corrupt_items} corrupted messages in queue '{queue_name}'")
            logger.error(f"Run cleanup script to fix: python /data/SWATGenXApp/codes/scripts/cleanup_corrupted_tasks.py")
        else:
            logger.info(f"No corrupted messages detected in first {items_to_check} items of queue '{queue_name}'")
            
    except Exception as e:
        logger.error(f"Error checking Redis queue corruption: {e}")
        logger.error(traceback.format_exc())

# Check Redis readiness before importing Celery
is_ready, working_url = wait_for_redis_ready(max_retries=10)
if is_ready and working_url:
    # Set environment variable for celery_app to use
    os.environ['REDIS_URL'] = working_url
    logger.info(f"Set REDIS_URL environment variable to {working_url}")
    
    # Check for corrupted messages that might crash workers
    check_redis_queue_corruption(working_url)

# Apply patch for the KeyError: 'properties' issue
# This monkey patches the Message.__init__ method to handle missing properties
try:
    logger.info("Applying safer message deserialization patch...")
    from kombu.transport.virtual.base import Message
    
    # Store the original __init__ method
    original_init = Message.__init__
    
    # Define a safer __init__ method that handles missing properties
    def safer_init(self, *args, **kwargs):
        try:
            # Try the original initialization
            original_init(self, *args, **kwargs)
        except KeyError as e:
            if str(e) == "'properties'" and args and isinstance(args[0], dict):
                # Fix the missing properties in the payload
                payload = args[0]
                logger.warning(f"Fixing missing 'properties' in message: {payload.get('task', 'unknown')}")
                
                # Add minimal properties to prevent the crash
                payload['properties'] = {
                    'correlation_id': '',
                    'reply_to': '',
                    'delivery_mode': 2,
                    'delivery_info': {'exchange': '', 'routing_key': 'celery'},
                    'priority': 0,
                    'body_encoding': 'base64',
                    'delivery_tag': str(int(time.time()))
                }
                
                # Now try initialization again
                original_init(self, *args, **kwargs)
            else:
                # For other KeyError issues, re-raise
                raise
    
    # Apply the patch
    Message.__init__ = safer_init
    logger.info("Message deserialization patch applied successfully")
except Exception as e:
    logger.error(f"Error applying message deserialization patch: {e}")
    logger.error(traceback.format_exc())

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
    logger.error(f"Traceback: {traceback.format_exc()}")
    raise

# This module acts as an entry point for the Celery worker
# The celery -A celery_worker command will import this file and use the "celery" instance

if __name__ == '__main__':
    logger.info("Starting Celery worker from __main__")
    celery.start()