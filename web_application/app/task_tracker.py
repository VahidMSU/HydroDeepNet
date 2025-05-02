import logging
import os
import json
import time
from datetime import datetime
from redis import Redis, ConnectionError, exceptions
import socket
import traceback

# Set up logging
log_dir = "/data/SWATGenXApp/codes/web_application/logs"
os.makedirs(log_dir, exist_ok=True)  # Ensure log directory exists

logger = logging.getLogger(__name__)
handler = logging.FileHandler(os.path.join(log_dir, "task_tracker.log"))
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class TaskTracker:
    """
    Tracks the status and progress of Celery tasks using Redis for storage
    and provides real-time status updates with improved reliability.
    """
    
    # Task status constants
    STATUS_PENDING = "PENDING"
    STATUS_STARTED = "STARTED"
    STATUS_SUCCESS = "SUCCESS"
    STATUS_FAILURE = "FAILURE"
    STATUS_REVOKED = "REVOKED"
    STATUS_RETRY = "RETRY"
    STATUS_RECEIVED = "RECEIVED"
    
    def __init__(self, redis_url=None):
        """Initialize with Redis connection"""
        self.redis_url = redis_url or os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
        self._redis = None
        self._redis_last_error = None
        self._redis_error_time = 0
        self._fallback_active = False
        
        # Local cache for task data when Redis is unavailable
        self._local_task_cache = {}
        self._key_prefix = "task_tracker:"
        
        self.log_file = os.path.join(log_dir, "model_tasks.log")
        logger.info(f"TaskTracker initialized with Redis URL: {self.redis_url}")
        
        # Ensure log file exists
        if not os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'w') as f:
                    f.write(f"{datetime.now().isoformat()} - TaskTracker initialized\n")
            except Exception as e:
                logger.error(f"Failed to create log file {self.log_file}: {e}")
        
    @property
    def redis(self):
        """Lazy Redis connection creation with error handling and recovery"""
        # If Redis previously failed, wait before retrying
        if self._redis_last_error and time.time() - self._redis_error_time < 10:
            raise ConnectionError(f"Redis connection recently failed: {self._redis_last_error}")
            
        # Create new connection if needed
        if self._redis is None:
            try:
                self._redis = Redis.from_url(
                    self.redis_url, 
                    socket_timeout=5, 
                    socket_connect_timeout=5,
                    health_check_interval=30,
                    retry_on_timeout=True
                )
                # Test connection with ping
                self._redis.ping()
                if self._fallback_active:
                    logger.info("Redis connection restored, switching from fallback mode")
                    self._fallback_active = False
                return self._redis
            except Exception as e:
                self._redis = None
                self._redis_last_error = str(e)
                self._redis_error_time = time.time()
                self._fallback_active = True
                logger.error(f"Redis connection error: {e}")
                logger.info("Switching to local storage fallback")
                raise
        return self._redis
    
    def _with_redis_fallback(self, redis_func, fallback_func, *args, **kwargs):
        """Execute a function with Redis, falling back to local storage if Redis fails"""
        try:
            return redis_func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Redis operation failed, using fallback: {str(e)}")
            return fallback_func(*args, **kwargs)
        
    def register_task(self, task_id, username, site_no, additional_info=None):
        """Register a new task with initial information"""
        task_key = f"task:{task_id}"
        task_info = {
            "task_id": task_id,
            "username": username,
            "site_no": site_no,
            "status": self.STATUS_PENDING,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "progress": 0,
            "info": additional_info or {}
        }
        
        def redis_register():
            # Store in Redis
            self.redis.set(task_key, json.dumps(task_info))
            
            # Add to sorted set for ordered listing
            self.redis.zadd("tasks:by_time", {task_id: time.time()})
            
            # Add to user's task list
            self.redis.sadd(f"user:{username}:tasks", task_id)
            return True
            
        def fallback_register():
            # Store in local cache
            self._local_task_cache[task_key] = task_info
            # Also add to username lookup
            user_key = f"user:{username}:tasks"
            if user_key not in self._local_task_cache:
                self._local_task_cache[user_key] = set()
            self._local_task_cache[user_key].add(task_id)
            return True
            
        try:
            success = self._with_redis_fallback(redis_register, fallback_register)
            # Always log to file for permanent record
            self._append_to_log(task_info)
            logger.info(f"Registered task {task_id} for user {username}, site {site_no}")
            return success
        except Exception as e:
            logger.error(f"Error registering task {task_id}: {e}")
            logger.error(traceback.format_exc())
            # Still try to log to file even if redis/local storage failed
            try:
                self._append_to_log(task_info)
            except Exception:
                pass
            return False
    
    def update_task_status(self, task_id, status, progress=None, info=None):
        """Update task status and progress with fallback handling"""
        task_key = f"task:{task_id}"
        
        def redis_update():
            task_json = self.redis.get(task_key)
            if not task_json:
                logger.warning(f"Attempted to update non-existent task in Redis: {task_id}")
                return False
                
            task_info = json.loads(task_json)
            task_info["status"] = status
            task_info["updated_at"] = datetime.now().isoformat()
            
            if progress is not None:
                task_info["progress"] = progress
                
            if info:
                if "info" not in task_info:
                    task_info["info"] = {}
                task_info["info"].update(info)
            
            # Update in Redis
            self.redis.set(task_key, json.dumps(task_info))
            return task_info
            
        def fallback_update():
            if task_key not in self._local_task_cache:
                logger.warning(f"Attempted to update non-existent task in local cache: {task_id}")
                return False
                
            task_info = self._local_task_cache[task_key]
            task_info["status"] = status
            task_info["updated_at"] = datetime.now().isoformat()
            
            if progress is not None:
                task_info["progress"] = progress
                
            if info:
                if "info" not in task_info:
                    task_info["info"] = {}
                task_info["info"].update(info)
                
            # Update local cache
            self._local_task_cache[task_key] = task_info
            return task_info
            
        try:
            task_info = self._with_redis_fallback(redis_update, fallback_update)
            if task_info:
                # Log to file for permanent record
                self._append_to_log(task_info)
                logger.info(f"Updated task {task_id} status to {status}, progress: {progress}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error updating task {task_id}: {e}")
            return False
    
    def get_task_status(self, task_id):
        """Get the current status of a task with fallback support"""
        task_key = f"task:{task_id}"
        
        def redis_get():
            task_json = self.redis.get(task_key)
            if not task_json:
                return None
            return json.loads(task_json)
            
        def fallback_get():
            return self._local_task_cache.get(task_key)
            
        try:
            return self._with_redis_fallback(redis_get, fallback_get)
        except Exception as e:
            logger.error(f"Error getting task status for {task_id}: {e}")
            return None
    
    def get_user_tasks(self, username, limit=500):  # Increased default limit to 500
        """Get all tasks for a specific user with fallback support"""
        def redis_get():
            # Get task IDs for user
            task_ids = self.redis.smembers(f"user:{username}:tasks")
            
            if not task_ids:
                return []
            
            # Get task details with higher limits
            tasks = []
            for task_id in task_ids:
                task_id = task_id.decode() if isinstance(task_id, bytes) else task_id
                task_info = self.get_task_status(task_id)
                if task_info:
                    tasks.append(task_info)
            
            return tasks
            
        def fallback_get():
            user_key = f"user:{username}:tasks"
            task_ids = self._local_task_cache.get(user_key, set())
            
            tasks = []
            for task_id in task_ids:
                task_info = self._local_task_cache.get(f"task:{task_id}")
                if task_info:
                    tasks.append(task_info)
                    
            return tasks
            
        try:
            tasks = self._with_redis_fallback(redis_get, fallback_get)
            # Sort by creation time (newest first) and apply limit
            tasks.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            return tasks[:limit] if limit else tasks  # Return all if limit is None
        except Exception as e:
            logger.error(f"Error getting tasks for user {username}: {e}")
            return []
    
    def get_active_tasks(self, limit=1000):  # Increased default limit to 1000
        """Get all active tasks with fallback support"""
        def redis_get():
            # Get all task IDs from the sorted set
            task_ids = self.redis.zrevrange("tasks:by_time", 0, limit-1)
            
            # Get task details
            tasks = []
            for task_id in task_ids:
                task_id = task_id.decode() if isinstance(task_id, bytes) else task_id
                task_info = self.get_task_status(task_id)
                if task_info and task_info.get("status") not in [self.STATUS_SUCCESS, self.STATUS_FAILURE, self.STATUS_REVOKED]:
                    tasks.append(task_info)
            
            return tasks
            
        def fallback_get():
            # In fallback mode, check all tasks in local cache
            active_tasks = []
            for key, task_info in self._local_task_cache.items():
                if key.startswith("task:") and isinstance(task_info, dict):
                    if task_info.get("status") not in [self.STATUS_SUCCESS, self.STATUS_FAILURE, self.STATUS_REVOKED]:
                        active_tasks.append(task_info)
            
            # Sort by creation time (newest first)
            active_tasks.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            return active_tasks[:limit]
            
        try:
            return self._with_redis_fallback(redis_get, fallback_get)
        except Exception as e:
            logger.error(f"Error getting active tasks: {e}")
            return []
    
    def _append_to_log(self, task_info):
        """Append task update to log file for permanent record"""
        try:
            # Ensure the log directory exists
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            
            with open(self.log_file, "a") as f:
                f.write(f"{datetime.now().isoformat()} - {json.dumps(task_info)}\n")
        except Exception as e:
            logger.error(f"Error writing to task log file: {e}")

# Create a singleton instance
task_tracker = TaskTracker()