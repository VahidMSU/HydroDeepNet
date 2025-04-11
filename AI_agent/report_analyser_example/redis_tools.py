import redis
import pickle
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import json
import os
from pathlib import Path

class RedisTools:
    """Redis operations manager with comprehensive logging."""
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379/0",
                 session_id: Optional[str] = None,
                 log_dir: str = "logs",
                 log_level: int = logging.INFO):
        """
        Initialize Redis tools with logging.
        
        Args:
            redis_url: URL for Redis connection
            session_id: Unique session identifier
            log_dir: Directory for log files
            log_level: Logging level
        """
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up logging
        self._setup_logging(log_dir, log_level)
        
        # Initialize Redis connection
        try:
            self.redis = redis.Redis.from_url(redis_url)
            self.logger.info(f"Connected to Redis at {redis_url}")
            self.has_redis = True
            
            # Test connection
            self.redis.ping()
            self.logger.info("Redis connection test successful")
            
        except Exception as e:
            self.logger.error(f"Could not connect to Redis: {str(e)}. Using in-memory storage instead.")
            self.has_redis = False
            self._setup_memory_storage()
    
    def _setup_logging(self, log_dir: str, log_level: int):
        """Set up logging configuration."""
        # Create logger
        self.logger = logging.getLogger(f"RedisTools_{self.session_id}")
        self.logger.setLevel(log_level)
        
        # Create log directory
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # File handler for Redis operations
        redis_handler = logging.FileHandler(
            log_path / f"redis_operations_{self.session_id}.log"
        )
        redis_handler.setLevel(log_level)
        redis_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(redis_handler)
        
        # File handler for errors
        error_handler = logging.FileHandler(
            log_path / f"redis_errors_{self.session_id}.log"
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
            'Exception: %(exc_info)s'
        ))
        self.logger.addHandler(error_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(console_handler)
    
    def _setup_memory_storage(self):
        """Set up in-memory storage when Redis is unavailable."""
        self.memory_storage = {}
        self.memory_expiry = {}
        self.logger.info("Initialized in-memory storage")
    
    def save_to_redis(self, 
                     key: str, 
                     value: Any, 
                     expire_seconds: int = 86400, 
                     prefix: str = "doc_reader") -> bool:
        """
        Save a value to Redis with expiration time.
        
        Args:
            key: Key to store the value under
            value: Value to store (will be pickled)
            expire_seconds: Expiration time in seconds
            prefix: Key prefix for namespace
            
        Returns:
            bool: Success status
        """
        try:
            full_key = f"{prefix}:{self.session_id}:{key}"
            
            # Log the operation
            self.logger.debug(f"Saving data to key: {full_key}")
            self.logger.debug(f"Data type: {type(value)}")
            
            # Serialize the value
            try:
                serialized = pickle.dumps(value)
            except Exception as e:
                self.logger.error(f"Error serializing value: {str(e)}")
                return False
            
            if self.has_redis:
                # Save to Redis
                success = self.redis.set(
                    full_key,
                    serialized,
                    ex=expire_seconds
                )
                
                if success:
                    self.logger.info(f"Successfully saved data to Redis key: {full_key}")
                    self.logger.debug(f"Expiration set to {expire_seconds} seconds")
                else:
                    self.logger.error(f"Failed to save data to Redis key: {full_key}")
                
                return bool(success)
            else:
                # Save to memory
                self.memory_storage[full_key] = serialized
                self.memory_expiry[full_key] = datetime.now().timestamp() + expire_seconds
                self.logger.info(f"Saved data to memory storage: {full_key}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error saving to storage: {str(e)}", exc_info=True)
            return False
    
    def load_from_redis(self, 
                       key: str, 
                       prefix: str = "doc_reader") -> Optional[Any]:
        """
        Load a value from storage.
        
        Args:
            key: Key to retrieve
            prefix: Key prefix for namespace
            
        Returns:
            The stored value or None if not found
        """
        try:
            full_key = f"{prefix}:{self.session_id}:{key}"
            self.logger.debug(f"Loading data from key: {full_key}")
            
            if self.has_redis:
                # Load from Redis
                serialized = self.redis.get(full_key)
            else:
                # Load from memory
                if full_key in self.memory_storage:
                    # Check expiration
                    if datetime.now().timestamp() > self.memory_expiry[full_key]:
                        self.logger.debug(f"Key expired: {full_key}")
                        del self.memory_storage[full_key]
                        del self.memory_expiry[full_key]
                        return None
                    serialized = self.memory_storage[full_key]
                else:
                    self.logger.debug(f"Key not found: {full_key}")
                    return None
            
            if serialized:
                try:
                    value = pickle.loads(serialized)
                    self.logger.info(f"Successfully loaded data from key: {full_key}")
                    self.logger.debug(f"Loaded data type: {type(value)}")
                    return value
                except Exception as e:
                    self.logger.error(f"Error deserializing value: {str(e)}")
                    return None
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error loading from storage: {str(e)}", exc_info=True)
            return None
    
    def get_cached_analysis(self, 
                          cache_key: str, 
                          prefix: str = "doc_reader") -> Optional[Any]:
        """
        Get cached analysis results.
        
        Args:
            cache_key: The key for the cached analysis
            prefix: Key prefix for namespace
            
        Returns:
            The cached analysis or None if not found
        """
        return self.load_from_redis(f"analysis:{cache_key}", prefix)
    
    def clear_redis_session(self, prefix: str = "doc_reader"):
        """
        Clear all keys for the current session.
        
        Args:
            prefix: Key prefix for namespace
        """
        try:
            pattern = f"{prefix}:{self.session_id}:*"
            self.logger.info(f"Clearing session data matching pattern: {pattern}")
            
            if self.has_redis:
                # Clear from Redis
                keys = list(self.redis.scan_iter(match=pattern))
                if keys:
                    self.redis.delete(*keys)
                    self.logger.info(f"Cleared {len(keys)} keys from Redis")
            else:
                # Clear from memory
                keys_to_delete = [
                    k for k in self.memory_storage.keys()
                    if k.startswith(f"{prefix}:{self.session_id}:")
                ]
                for k in keys_to_delete:
                    del self.memory_storage[k]
                    del self.memory_expiry[k]
                self.logger.info(f"Cleared {len(keys_to_delete)} keys from memory storage")
                
        except Exception as e:
            self.logger.error(f"Error clearing session data: {str(e)}", exc_info=True)
    
    def sync_data(self, 
                 data_dict: Dict[str, Any],
                 keys_to_sync: List[str],
                 prefix: str = "doc_reader") -> Dict[str, Any]:
        """
        Sync multiple data items with storage.
        
        Args:
            data_dict: Dictionary containing the data to sync
            keys_to_sync: List of keys in data_dict to sync
            prefix: Key prefix for namespace
            
        Returns:
            dict: Updated data dictionary
        """
        if not self.has_redis:
            return data_dict
            
        try:
            updated_dict = data_dict.copy()
            self.logger.debug(f"Syncing {len(keys_to_sync)} keys")
            
            for key in keys_to_sync:
                if key in data_dict:
                    # Try to load from storage first
                    stored_data = self.load_from_redis(key, prefix)
                    if stored_data:
                        self.logger.debug(f"Found stored data for key: {key}")
                        updated_dict[key] = stored_data
                    else:
                        # Save current data
                        self.logger.debug(f"Saving current data for key: {key}")
                        self.save_to_redis(key, data_dict[key], prefix=prefix)
            
            return updated_dict
            
        except Exception as e:
            self.logger.error(f"Error syncing data: {str(e)}", exc_info=True)
            return data_dict
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about Redis operations."""
        try:
            stats = {
                "session_id": self.session_id,
                "has_redis": self.has_redis,
                "timestamp": datetime.now().isoformat()
            }
            
            if self.has_redis:
                # Get Redis stats
                info = self.redis.info()
                stats.update({
                    "redis_version": info.get("redis_version"),
                    "connected_clients": info.get("connected_clients"),
                    "used_memory_human": info.get("used_memory_human"),
                    "total_connections_received": info.get("total_connections_received")
                })
            else:
                # Get memory storage stats
                stats.update({
                    "memory_keys": len(self.memory_storage),
                    "memory_size": sum(len(pickle.dumps(v)) for v in self.memory_storage.values())
                })
            
            self.logger.info(f"Retrieved storage stats: {json.dumps(stats, indent=2)}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting stats: {str(e)}", exc_info=True)
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
