"""
Redis utilities for health checks and connection validation
"""
import logging
import socket
import time
from redis import Redis, ConnectionError, exceptions
from flask import current_app

logger = logging.getLogger(__name__)

def check_redis_health():
    """
    Check if Redis is available and healthy.
    Returns a dict with status information.
    """
    redis_urls = [
        current_app.config.get('REDIS_URL', 'redis://localhost:6379/0'),
        'redis://127.0.0.1:6379/0',  # Try explicit IP
        'redis://localhost:6379/0'   # Try explicit hostname
    ]
    
    # Try each URL until one works, with special handling for loading state
    for url in redis_urls:
        try:
            logger.info(f"Testing Redis health: {url}")
            client = Redis.from_url(
                url, 
                socket_timeout=2,
                socket_connect_timeout=2
            )
            
            # Special handling for Redis loading state
            max_loading_retries = 3
            for loading_attempt in range(max_loading_retries):
                try:
                    response = client.ping()
                    if response:
                        logger.info(f"Redis health check successful: {url}")
                        return {
                            'healthy': True,
                            'message': f'Redis is available at {url}',
                            'working_url': url
                        }
                except exceptions.BusyLoadingError:
                    logger.warning(f"Redis at {url} is loading dataset (attempt {loading_attempt+1}/{max_loading_retries})")
                    if loading_attempt < max_loading_retries - 1:
                        # Wait for Redis to finish loading, increasing delay each time
                        time.sleep(2 * (loading_attempt + 1))
                    else:
                        # Last attempt failed
                        return {
                            'healthy': False,
                            'message': 'Redis is still loading the dataset. Please try again later.',
                            'temporary': True
                        }
                
        except ConnectionError as e:
            logger.warning(f"Redis connection error at {url}: {e}")
        except socket.error as e:
            logger.warning(f"Socket error connecting to Redis at {url}: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error checking Redis at {url}: {e}")
    
    # None of the URLs worked
    logger.error("All Redis connection attempts failed")
    return {
        'healthy': False,
        'message': 'Could not connect to Redis server. The service may be down or unreachable.'
    }

def get_working_redis_connection():
    """
    Try to establish a working Redis connection.
    Returns a Redis client or None if no connection works.
    """
    health_check = check_redis_health()
    if health_check['healthy']:
        try:
            return Redis.from_url(
                health_check['working_url'],
                socket_timeout=5,
                socket_connect_timeout=5
            )
        except Exception as e:
            logger.error(f"Error creating Redis client: {e}")
    
    return None
