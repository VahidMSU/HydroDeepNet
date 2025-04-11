import redis
import pickle
import logging
from pathlib import Path
from Logger import Logger
from datetime import datetime, timedelta

class RedisTools:
    """A utility class for Redis operations used in document processing."""
    
    def __init__(self, 
                 redis_url="redis://localhost:6379/0", 
                 session_id=None,
                 log_dir: str = "logs",
                 log_level: int = logging.INFO):
        """
        Initialize Redis tools.
        
        Args:
            redis_url: URL for Redis connection
            session_id: Unique session identifier
            log_dir: Directory for log files
            log_level: Logging level
        """
        self.session_id = session_id
        
        # Set up logging
        self.logger = Logger(
            log_dir=log_dir,
            app_name="redis_tools",
            log_level=log_level
        )
        
        self.logger.info("Initializing RedisTools")
        
        try:
            self.redis = redis.Redis.from_url(redis_url)
            self.logger.info(f"Connected to Redis at {redis_url}")
            self.has_redis = True
        except Exception as e:
            self.logger.warning(f"Could not connect to Redis: {str(e)}. Using in-memory storage instead.")
            self.has_redis = False
            
        # Memory storage for fallback when Redis is unavailable
        self.memory_storage = {}
        self.expiry_tracking = {}
        
        self.logger.info("RedisTools initialization completed")
    
    def save_to_redis(self, key, value, expire_seconds=86400, prefix="doc_reader"):
        """
        Save a value to Redis with expiration time (default 24 hours).
        
        Args:
            key: Key to store the value under
            value: Value to store (will be pickled)
            expire_seconds: Expiration time in seconds
            prefix: Key prefix for namespace
            
        Returns:
            bool: Success status
        """
        full_key = f"{prefix}:{self.session_id}:{key}"
        
        if not self.has_redis:
            self.logger.debug(f"Redis unavailable, storing {full_key} in memory")
            try:
                # Store in memory as fallback
                self.memory_storage[full_key] = value
                # Track expiration time
                self.expiry_tracking[full_key] = {
                    "expire_at": datetime.now() + timedelta(seconds=expire_seconds)
                }
                return True
            except Exception as e:
                self.logger.error(f"Error storing in memory: {str(e)}")
                return False
        
        try:
            self.logger.debug(f"Saving data to Redis key: {full_key}")
            serialized = pickle.dumps(value)
            result = self.redis.set(full_key, serialized, ex=expire_seconds)
            if result:
                self.logger.debug(f"Successfully saved data to Redis key: {full_key}")
            else:
                self.logger.warning(f"Failed to save data to Redis key: {full_key}")
            return result
        except Exception as e:
            self.logger.error(f"Error saving to Redis: {str(e)}")
            return False
    
    def load_from_redis(self, key, prefix="doc_reader"):
        """
        Load a value from Redis.
        
        Args:
            key: Key to retrieve
            prefix: Key prefix for namespace
            
        Returns:
            The stored value or None if not found
        """
        full_key = f"{prefix}:{self.session_id}:{key}"
        
        if not self.has_redis:
            self.logger.debug(f"Redis unavailable, loading {full_key} from memory")
            # Check if key exists in memory and not expired
            if full_key in self.memory_storage:
                if full_key in self.expiry_tracking:
                    if datetime.now() > self.expiry_tracking[full_key]["expire_at"]:
                        self.logger.debug(f"Key {full_key} has expired in memory storage")
                        # Remove expired key
                        del self.memory_storage[full_key]
                        del self.expiry_tracking[full_key]
                        return None
                return self.memory_storage[full_key]
            return None
        
        try:
            self.logger.debug(f"Loading data from Redis key: {full_key}")
            serialized = self.redis.get(full_key)
            if serialized:
                self.logger.debug(f"Found data for Redis key: {full_key}")
                return pickle.loads(serialized)
            self.logger.debug(f"No data found for Redis key: {full_key}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading from Redis: {str(e)}")
            return None
    
    def get_cached_analysis(self, cache_key, prefix="doc_reader"):
        """
        Get cached analysis results from Redis.
        
        Args:
            cache_key: The key for the cached analysis
            prefix: Key prefix for namespace
            
        Returns:
            The cached analysis or None if not found
        """
        self.logger.debug(f"Getting cached analysis for: {cache_key}")
        return self.load_from_redis(f"analysis:{cache_key}", prefix)
    
    def clear_redis_session(self, prefix="doc_reader"):
        """
        Clear all keys for the current session.
        
        Args:
            prefix: Key prefix for namespace
        """
        self.logger.info(f"Clearing Redis session for prefix: {prefix}, session: {self.session_id}")
        
        if not self.has_redis:
            # Clear memory storage
            pattern = f"{prefix}:{self.session_id}:"
            keys_to_delete = [k for k in self.memory_storage.keys() if k.startswith(pattern)]
            for key in keys_to_delete:
                del self.memory_storage[key]
                if key in self.expiry_tracking:
                    del self.expiry_tracking[key]
            self.logger.info(f"Cleared {len(keys_to_delete)} keys from memory storage")
            return
        
        try:
            pattern = f"{prefix}:{self.session_id}:*"
            keys = list(self.redis.scan_iter(match=pattern))
            if keys:
                self.redis.delete(*keys)
                self.logger.info(f"Cleared {len(keys)} keys from Redis")
            else:
                self.logger.info(f"No keys found to clear in Redis")
        except Exception as e:
            self.logger.error(f"Error clearing Redis session: {str(e)}")
    
    def sync_data(self, data_dict, keys_to_sync, prefix="doc_reader"):
        """
        Sync multiple data items with Redis.
        
        Args:
            data_dict: Dictionary containing the data to sync
            keys_to_sync: List of keys in data_dict to sync
            prefix: Key prefix for namespace
            
        Returns:
            dict: Updated data dictionary
        """
        self.logger.debug(f"Syncing {len(keys_to_sync)} keys with storage")
        
        if not self.has_redis:
            self.logger.debug("Redis unavailable, syncing with memory storage")
            return data_dict
            
        updated_dict = data_dict.copy()
        
        for key in keys_to_sync:
            if key in data_dict:
                # Try to load from Redis first
                redis_data = self.load_from_redis(key, prefix)
                if redis_data:
                    updated_dict[key] = redis_data
                    self.logger.debug(f"Updated key {key} from Redis")
                else:
                    # Save current data
                    self.save_to_redis(key, data_dict[key], prefix=prefix)
                    self.logger.debug(f"Saved key {key} to Redis")
                    
        return updated_dict
        
    def get_stats(self):
        """Get statistics about Redis usage."""
        stats = {
            "connected": self.has_redis,
            "memory_storage_keys": len(self.memory_storage),
            "expiring_keys": len(self.expiry_tracking)
        }
        
        if self.has_redis:
            try:
                info = self.redis.info()
                stats.update({
                    "redis_version": info.get("redis_version"),
                    "connected_clients": info.get("connected_clients"),
                    "used_memory_human": info.get("used_memory_human"),
                    "total_commands_processed": info.get("total_commands_processed")
                })
            except Exception as e:
                self.logger.error(f"Error getting Redis stats: {str(e)}")
        
        return stats
