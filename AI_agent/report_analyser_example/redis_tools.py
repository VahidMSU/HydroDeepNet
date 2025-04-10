import redis
import pickle
import logging
from Logger import LoggerSetup

# Get logger


class RedisTools:
    """A utility class for Redis operations used in document processing."""
    
    def __init__(self, redis_url="redis://localhost:6379/0", session_id=None):
        """
        Initialize Redis tools.
        
        Args:
            redis_url: URL for Redis connection
            session_id: Unique session identifier
        """
        self.session_id = session_id
        self.logger = LoggerSetup(verbose=False, rewrite=False)
        self.logger = self.logger.setup_logger("AI_AgentLogger")
        try:
            self.redis = redis.Redis.from_url(redis_url)
            self.logger.info(f"Connected to Redis at {redis_url}")
            self.has_redis = True
        except Exception as e:
            self.logger.warning(f"Could not connect to Redis: {str(e)}. Using in-memory storage instead.")
            self.has_redis = False
    
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
        if not self.has_redis:
            return False
        
        try:
            full_key = f"{prefix}:{self.session_id}:{key}"
            serialized = pickle.dumps(value)
            self.redis.set(full_key, serialized, ex=expire_seconds)
            return True
        except Exception as e:
            self.logger.error(f"save_to_redis Error saving to Redis: {str(e)}")
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
        if not self.has_redis:
            return None
        
        try:
            full_key = f"{prefix}:{self.session_id}:{key}"
            serialized = self.redis.get(full_key)
            if serialized:
                return pickle.loads(serialized)
            return None
        except Exception as e:
            self.logger.error(f"load_from_redis Error loading from Redis: {str(e)}")
            return None
    
    def clear_redis_session(self, prefix="doc_reader"):
        """
        Clear all keys for the current session.
        
        Args:
            prefix: Key prefix for namespace
        """
        if not self.has_redis:
            return
        
        try:
            pattern = f"{prefix}:{self.session_id}:*"
            for key in self.redis.scan_iter(match=pattern):
                self.redis.delete(key)
        except Exception as e:
            self.logger.error(f"clear_redis_session Error clearing Redis session: {str(e)}")
    
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
        if not self.has_redis:
            return data_dict
            
        updated_dict = data_dict.copy()
        
        for key in keys_to_sync:
            if key in data_dict:
                # Try to load from Redis first
                redis_data = self.load_from_redis(key, prefix)
                if redis_data:
                    updated_dict[key] = redis_data
                else:
                    # Save current data
                    self.save_to_redis(key, data_dict[key], prefix=prefix)
                    
        return updated_dict
