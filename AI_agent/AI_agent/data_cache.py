import os
import json
import time
import hashlib
import pickle
from AI_agent.debug_utils import logger

class DataCache:
    """
    Cache system for storing and retrieving data to avoid repeated API calls
    and data processing operations.
    """
    
    def __init__(self, cache_dir=None, max_age=7*24*60*60):  # Default: 1 week cache
        """
        Initialize the cache system.
        
        Args:
            cache_dir (str): Directory to store cache files
            max_age (int): Maximum age of cache entries in seconds
        """
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')
        
        self.cache_dir = cache_dir
        self.max_age = max_age
        self.metadata_file = os.path.join(self.cache_dir, 'cache_metadata.json')
        self.metadata = {}
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load metadata if exists
        self._load_metadata()
    
    def _load_metadata(self):
        """Load cache metadata from file."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.error(f"Error loading cache metadata: {e}")
                self.metadata = {}
    
    def _save_metadata(self):
        """Save cache metadata to file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f)
        except Exception as e:
            logger.error(f"Error saving cache metadata: {e}")
    
    def _generate_key(self, cache_type, params):
        """Generate a unique cache key based on parameters."""
        # Convert params to a sorted, stringified representation for consistent hashing
        if isinstance(params, dict):
            # Sort the dictionary by keys
            sorted_items = sorted(params.items())
            params_str = json.dumps(sorted_items)
        else:
            params_str = str(params)
            
        # Create a hash of the parameters
        key = hashlib.md5((cache_type + params_str).encode()).hexdigest()
        return key
    
    def get(self, cache_type, params):
        """
        Retrieve data from cache.
        
        Args:
            cache_type (str): Type of data ('climate', 'landcover', etc.)
            params (dict): Parameters used to generate the data
            
        Returns:
            The cached data or None if not found or expired
        """
        key = self._generate_key(cache_type, params)
        
        # Check if key exists in metadata
        if key not in self.metadata:
            return None
            
        # Check if cache entry has expired
        entry = self.metadata[key]
        if time.time() - entry['timestamp'] > self.max_age:
            logger.info(f"Cache entry {key} has expired")
            return None
            
        # Try to load the cache file
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        if not os.path.exists(cache_file):
            logger.warning(f"Cache file {cache_file} not found despite metadata entry")
            return None
            
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Retrieved {cache_type} data from cache for {entry['description']}")
            return data
        except Exception as e:
            logger.error(f"Error loading cache file {cache_file}: {e}")
            return None
    
    def put(self, cache_type, params, data, description=None):
        """
        Store data in cache.
        
        Args:
            cache_type (str): Type of data ('climate', 'landcover', etc.)
            params (dict): Parameters used to generate the data
            data: The data to cache
            description (str): Optional description of the cached data
        """
        if data is None:
            return  # Don't cache None values
            
        key = self._generate_key(cache_type, params)
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        
        # Create entry in metadata
        self.metadata[key] = {
            'type': cache_type,
            'timestamp': time.time(),
            'description': description or f"{cache_type} data for {params}"
        }
        
        # Save the data
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            self._save_metadata()
            logger.info(f"Cached {cache_type} data for {description or params}")
        except Exception as e:
            logger.error(f"Error caching data: {e}")
    
    def invalidate(self, cache_type=None, params=None):
        """
        Invalidate cache entries.
        
        Args:
            cache_type (str): Type of data to invalidate (None for all)
            params (dict): Parameters to match (None for all of the specified type)
        """
        if cache_type is None and params is None:
            # Clear entire cache
            for key in list(self.metadata.keys()):
                cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
                if os.path.exists(cache_file):
                    os.remove(cache_file)
            self.metadata = {}
            self._save_metadata()
            logger.info("Cleared entire cache")
            return
            
        if params is not None:
            # Invalidate specific entry
            key = self._generate_key(cache_type, params)
            if key in self.metadata:
                cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                del self.metadata[key]
                self._save_metadata()
                logger.info(f"Invalidated cache entry for {cache_type} with params {params}")
        elif cache_type is not None:
            # Invalidate all entries of a specific type
            keys_to_remove = []
            for key, entry in self.metadata.items():
                if entry['type'] == cache_type:
                    cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
                    if os.path.exists(cache_file):
                        os.remove(cache_file)
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.metadata[key]
            
            self._save_metadata()
            logger.info(f"Invalidated all cache entries of type {cache_type}")
    
    def cleanup(self):
        """Remove expired cache entries."""
        current_time = time.time()
        keys_to_remove = []
        
        for key, entry in self.metadata.items():
            if current_time - entry['timestamp'] > self.max_age:
                cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.metadata[key]
        
        self._save_metadata()
        logger.info(f"Cleaned up {len(keys_to_remove)} expired cache entries")
    
    def get_stats(self):
        """Get statistics about the cache."""
        stats = {
            'total_entries': len(self.metadata),
            'types': {},
            'total_size_kb': 0
        }
        
        # Count entries by type
        for entry in self.metadata.values():
            cache_type = entry['type']
            if cache_type not in stats['types']:
                stats['types'][cache_type] = 0
            stats['types'][cache_type] += 1
        
        # Calculate total size
        for key in self.metadata:
            cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
            if os.path.exists(cache_file):
                stats['total_size_kb'] += os.path.getsize(cache_file) / 1024
        
        return stats

# Create a singleton instance
data_cache = DataCache()

# Convenience functions
def get_cached_data(cache_type, params):
    """Get data from cache."""
    return data_cache.get(cache_type, params)

def cache_data(cache_type, params, data, description=None):
    """Store data in cache."""
    data_cache.put(cache_type, params, data, description)
