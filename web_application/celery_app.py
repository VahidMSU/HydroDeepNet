from celery import Celery
import os
from redis import Redis, ConnectionError, exceptions
import socket
import logging
import time
import json
from datetime import datetime

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/data/SWATGenXApp/codes/web_application/logs/celery_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Log startup information
logger.info(f"Starting celery_app module initialization at {datetime.now().isoformat()}")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")

def get_working_redis_url():
    """
    Try multiple Redis connections to find a working one with improved error handling.
    Returns a working Redis URL or the default if none work.
    """
    default_url = 'redis://localhost:6379/0'
    
    # Use ordered list of URLs to try, with the most likely to work first
    redis_urls = [
        os.environ.get('REDIS_URL', ''),  # First try environment variable if set
        'redis://127.0.0.1:6379/0',       # Explicit localhost IP
        'redis://localhost:6379/0',       # Standard localhost name
        'redis://redis:6379/0',           # Docker service name
    ]
    
    # Filter out empty URLs
    redis_urls = [url for url in redis_urls if url]
    
    # If no URLs, use default
    if not redis_urls:
        redis_urls = [default_url]
    
    # Add retry logic with exponential backoff
    max_retries = 6  # More retries for production environment
    retry_delay = 1  # Initial delay in seconds
    
    logger.info(f"Testing {len(redis_urls)} Redis URLs with {max_retries} max retries")
    
    for attempt in range(max_retries):
        for url in redis_urls:
            try:
                logger.info(f"Testing Redis connection to {url} (attempt {attempt+1}/{max_retries})")
                redis_options = {
                    'socket_timeout': 5,
                    'socket_connect_timeout': 5,
                    'health_check_interval': 10
                }
                client = Redis.from_url(url, **redis_options)
                
                # Ping with timeout for better reliability
                ping_result = client.ping()
                
                # Additional validation - try a simple set/get operation
                test_key = f"celery_app_test_{datetime.now().timestamp()}"
                client.set(test_key, "1", ex=60)  # Set with 60s expiry
                test_value = client.get(test_key)
                client.delete(test_key)  # Clean up
                
                if ping_result and test_value:
                    logger.info(f"✅ Successfully connected to Redis at {url} (ping: {ping_result}, test: {test_value})")
                    return url
                else:
                    logger.warning(f"Redis at {url} responded but validation failed: ping={ping_result}, test={test_value}")
                    
            except exceptions.BusyLoadingError:
                # Redis is starting up, wait longer
                loading_wait = min(5 * (attempt + 1), 30)  # Increase wait time with attempt number
                logger.warning(f"Redis at {url} is loading dataset. Waiting {loading_wait}s...")
                time.sleep(loading_wait)
                # Don't move to next URL, try this one again
                break
                
            except (ConnectionError, socket.error, exceptions.ConnectionError) as e:
                # Connection failed, try next URL
                logger.warning(f"Could not connect to Redis at {url}: {e}")
                
            except Exception as e:
                # Unexpected error
                logger.error(f"Unexpected error connecting to Redis at {url}: {str(e)}")
                
        # If we've tried all URLs and none worked, wait before next attempt
        if attempt < max_retries - 1:  
            backoff_time = retry_delay * (2 ** attempt)  # Exponential backoff
            backoff_time = min(backoff_time, 60)  # Cap at 60 seconds
            logger.warning(f"All Redis URLs failed. Retrying in {backoff_time} seconds...")
            time.sleep(backoff_time)
    
    # If we get here, all retries failed
    logger.error(f"⚠️ No working Redis connection found after {max_retries} attempts, using default")
    return default_url

def make_celery(app=None):
    """Create a Celery instance with robust configuration for production"""
    # Get the Redis URL with graceful fallback
    try:
        redis_url = get_working_redis_url()
        logger.info(f"Initializing Celery with Redis URL: {redis_url}")
    except Exception as e:
        logger.error(f"Critical error finding working Redis URL: {str(e)}")
        # Log full stacktrace for debugging
        import traceback
        logger.error(traceback.format_exc())
        
        # Fall back to default URL
        redis_url = 'redis://localhost:6379/0'
        logger.warning(f"Falling back to default Redis URL: {redis_url}")
    
    # Store the working URL in environment
    os.environ['REDIS_URL'] = redis_url
    
    # Create Celery instance with optimized configuration
    celery = Celery(
        'web_application',
        broker=redis_url,
        backend=redis_url,
        include=[
            'app.swatgenx_tasks'
        ]
    )
    
    # Pull configuration from environment variables via Config class if app is provided
    if app:
        # Get config values from app, which gets them from environment
        worker_prefetch_multiplier = app.config.get('CELERY_WORKER_PREFETCH_MULTIPLIER', 8)
        task_soft_time_limit = app.config.get('CELERY_TASK_SOFT_TIME_LIMIT', 43200)
        task_time_limit = app.config.get('CELERY_TASK_TIME_LIMIT', 86400)
        disable_rate_limits = app.config.get('CELERY_DISABLE_RATE_LIMITS', True)
        task_default_rate_limit = app.config.get('CELERY_TASK_DEFAULT_RATE_LIMIT', None)
        broker_connection_retry = app.config.get('CELERY_BROKER_CONNECTION_RETRY', True)
        broker_connection_max_retries = app.config.get('CELERY_BROKER_CONNECTION_MAX_RETRIES', 20)
        redis_max_connections = app.config.get('CELERY_REDIS_MAX_CONNECTIONS', 500)
        max_tasks_per_child = app.config.get('CELERY_MAX_TASKS_PER_CHILD', 100)
        max_memory_per_child_mb = app.config.get('CELERY_MAX_MEMORY_PER_CHILD_MB', 8192)
    else:
        # Get from environment directly if no app
        worker_prefetch_multiplier = int(os.environ.get('CELERY_WORKER_PREFETCH_MULTIPLIER', '8'))
        task_soft_time_limit = int(os.environ.get('CELERY_TASK_SOFT_TIME_LIMIT', '43200'))
        task_time_limit = int(os.environ.get('CELERY_TASK_TIME_LIMIT', '86400'))
        disable_rate_limits = os.environ.get('CELERY_DISABLE_RATE_LIMITS', 'true').lower() == 'true'
        task_default_rate_limit = os.environ.get('CELERY_TASK_DEFAULT_RATE_LIMIT')
        broker_connection_retry = os.environ.get('CELERY_BROKER_CONNECTION_RETRY', 'true').lower() == 'true'
        broker_connection_max_retries = int(os.environ.get('CELERY_BROKER_CONNECTION_MAX_RETRIES', '20'))
        redis_max_connections = int(os.environ.get('CELERY_REDIS_MAX_CONNECTIONS', '500'))
        max_tasks_per_child = int(os.environ.get('CELERY_MAX_TASKS_PER_CHILD', '100'))
        max_memory_per_child_mb = int(os.environ.get('CELERY_MAX_MEMORY_PER_CHILD_MB', '8192'))

    # Log the configuration values being used
    logger.info(f"Using Celery configuration from environment:")
    logger.info(f"worker_prefetch_multiplier: {worker_prefetch_multiplier}")
    logger.info(f"disable_rate_limits: {disable_rate_limits}")
    logger.info(f"task_default_rate_limit: {task_default_rate_limit}")
    logger.info(f"max_tasks_per_child: {max_tasks_per_child}")
    
    # Configure Celery for high concurrency and resource utilization
    celery_config = {
        # Essential connection settings
        'broker_connection_retry': broker_connection_retry,
        'broker_connection_retry_on_startup': True,
        'broker_connection_max_retries': broker_connection_max_retries,
        'broker_connection_timeout': 10,
        
        # Transport options for Redis
        'broker_transport_options': {
            'socket_timeout': 10,
            'socket_connect_timeout': 10,
            'visibility_timeout': 43200,  # 12 hours (in seconds)
            'retry_on_timeout': True,
            'max_connections': redis_max_connections,
            'health_check_interval': 30  # Check broker connection every 30s
        },
        
        # Task routing
        'task_routes': {
            'app.swatgenx_tasks.create_model_task': {'queue': 'model_creation'},
        },
        
        # Task execution settings for high throughput
        'task_acks_late': True,  # Tasks acknowledged after execution
        'task_reject_on_worker_lost': True,  # Requeue tasks if worker crashes
        'task_default_rate_limit': task_default_rate_limit,  # Configure rate limiting from environment
        'worker_prefetch_multiplier': worker_prefetch_multiplier,  # Allow each worker to prefetch multiple tasks
        'worker_disable_rate_limits': disable_rate_limits,  # Configure rate limiting from environment
        'task_track_started': True,  # Track when tasks are started
        'task_send_sent_event': True,  # Enable sent events for task tracking
        'worker_send_task_events': True,  # Enable task events for monitoring
        
        # Result settings
        'result_backend': redis_url,
        'result_expires': 60 * 60 * 24 * 7,  # Results expire in 7 days
        'result_extended': True,  # Store extended task metadata
        
        # Task time limits - extended for larger workloads
        'task_soft_time_limit': task_soft_time_limit,  # Configurable soft time limit
        'task_time_limit': task_time_limit,  # Configurable hard time limit
        
        # Security settings
        'accept_content': ['json', 'pickle'],
        'task_serializer': 'json',
        'result_serializer': 'json',
        
        # Resource management settings optimized for high concurrency
        'worker_max_tasks_per_child': max_tasks_per_child,  # Process more tasks before worker restart
        'worker_max_memory_per_child': max_memory_per_child_mb * 1024 * 1024,  # Configurable memory limit
        
        # Pool settings for better resource utilization
        'worker_pool': 'prefork',
        'worker_pool_restarts': True,
    }
    
    # Apply configuration
    celery.conf.update(celery_config)
    
    # Log full configuration for troubleshooting
    try:
        config_log = os.path.join('/data/SWATGenXApp/codes/web_application/logs', 'celery_config.log')
        with open(config_log, 'w') as f:
            # Convert config to JSON for better readability
            config_json = {k: (str(v) if not isinstance(v, (dict, list, int, float, bool)) else v) 
                           for k, v in celery_config.items()}
            f.write(json.dumps(config_json, indent=2))
            f.write("\n\nEnvironment Variables:\n")
            for key, value in os.environ.items():
                if key.startswith('CELERY') or key.startswith('REDIS') or key.startswith('FLASK'):
                    f.write(f"{key}={value}\n")
        logger.info(f"Saved Celery configuration to {config_log}")
    except Exception as e:
        logger.error(f"Error saving Celery configuration: {str(e)}")
    
    # Configure Celery from Flask app if provided
    if app:
        celery.conf.update(app.config)
        
        class ContextTask(celery.Task):
            def __call__(self, *args, **kwargs):
                with app.app_context():
                    return self.run(*args, **kwargs)
        
        celery.Task = ContextTask
    
    return celery

# Create the celery instance safely
try:
    logger.info("Creating Celery instance...")
    celery = make_celery()
    logger.info("Successfully created Celery instance")
except Exception as e:
    logger.error(f"Error creating Celery instance: {str(e)}")
    import traceback
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Create a minimal Celery instance that will work in degraded mode
    logger.warning("Creating minimal fallback Celery instance")
    celery = Celery('web_application')
    celery.conf.broker_connection_retry = True
    celery.conf.broker_connection_retry_on_startup = True
    
# Export the celery instance for workers to use
logger.info("celery_app module initialization complete")