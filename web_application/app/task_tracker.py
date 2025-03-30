import logging
import os
import json
import time
from datetime import datetime
from flask import current_app
from redis import Redis

# Set up logging
log_dir = "/data/SWATGenXApp/codes/web_application/logs"
logger = logging.getLogger(__name__)
handler = logging.FileHandler(os.path.join(log_dir, "task_tracker.log"))
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class TaskTracker:
    """
    Tracks the status and progress of Celery tasks using Redis for storage
    and provides real-time status updates.
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
        self.log_file = os.path.join(log_dir, "model_tasks.log")
        logger.info(f"TaskTracker initialized with Redis URL: {self.redis_url}")
        
    @property
    def redis(self):
        """Lazy Redis connection creation"""
        if self._redis is None:
            try:
                self._redis = Redis.from_url(self.redis_url, socket_timeout=5, socket_connect_timeout=5)
            except Exception as e:
                logger.error(f"Error connecting to Redis: {e}")
                raise
        return self._redis
        
    def register_task(self, task_id, username, site_no, additional_info=None):
        """
        Register a new task with initial information
        
        Args:
            task_id (str): Celery task ID
            username (str): Username who initiated the task
            site_no (str): Site number for model creation
            additional_info (dict): Any additional metadata about the task
            
        Returns:
            bool: Whether the operation was successful
        """
        try:
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
            
            # Store in Redis
            self.redis.set(task_key, json.dumps(task_info))
            
            # Add to sorted set for ordered listing
            self.redis.zadd("tasks:by_time", {task_id: time.time()})
            
            # Add to user's task list
            self.redis.sadd(f"user:{username}:tasks", task_id)
            
            # Log to file for permanent record
            self._append_to_log(task_info)
            
            logger.info(f"Registered task {task_id} for user {username}, site {site_no}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering task {task_id}: {e}")
            return False
    
    def update_task_status(self, task_id, status, progress=None, info=None):
        """
        Update task status and progress
        
        Args:
            task_id (str): Celery task ID
            status (str): New status
            progress (int, optional): Progress percentage (0-100)
            info (dict, optional): Additional status information
            
        Returns:
            bool: Whether the operation was successful
        """
        try:
            task_key = f"task:{task_id}"
            task_json = self.redis.get(task_key)
            
            if not task_json:
                logger.warning(f"Attempted to update non-existent task: {task_id}")
                return False
                
            task_info = json.loads(task_json)
            task_info["status"] = status
            task_info["updated_at"] = datetime.now().isoformat()
            
            if progress is not None:
                task_info["progress"] = progress
                
            if info:
                task_info["info"].update(info)
            
            # Update in Redis
            self.redis.set(task_key, json.dumps(task_info))
            
            # Log to file for permanent record
            self._append_to_log(task_info)
            
            logger.info(f"Updated task {task_id} status to {status}, progress: {progress}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating task {task_id}: {e}")
            return False
    
    def get_task_status(self, task_id):
        """
        Get the current status of a task
        
        Args:
            task_id (str): Celery task ID
            
        Returns:
            dict: Task information or None if not found
        """
        try:
            task_key = f"task:{task_id}"
            task_json = self.redis.get(task_key)
            
            if not task_json:
                return None
                
            return json.loads(task_json)
            
        except Exception as e:
            logger.error(f"Error getting task status for {task_id}: {e}")
            return None
    
    def get_user_tasks(self, username, limit=50):
        """
        Get all tasks for a specific user, ordered by creation time
        
        Args:
            username (str): Username to get tasks for
            limit (int): Maximum number of tasks to return
            
        Returns:
            list: List of task information dictionaries
        """
        try:
            # Get task IDs for user
            task_ids = self.redis.smembers(f"user:{username}:tasks")
            
            if not task_ids:
                return []
            
            # Get task details
            tasks = []
            for task_id in task_ids:
                task_id = task_id.decode() if isinstance(task_id, bytes) else task_id
                task_info = self.get_task_status(task_id)
                if task_info:
                    tasks.append(task_info)
            
            # Sort by creation time (newest first) and limit
            tasks.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            return tasks[:limit]
            
        except Exception as e:
            logger.error(f"Error getting tasks for user {username}: {e}")
            return []
    
    def get_active_tasks(self, limit=50):
        """
        Get all active tasks (not completed or failed)
        
        Args:
            limit (int): Maximum number of tasks to return
            
        Returns:
            list: List of active task information dictionaries
        """
        try:
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
            
        except Exception as e:
            logger.error(f"Error getting active tasks: {e}")
            return []
    
    def _append_to_log(self, task_info):
        """Append task update to log file for permanent record"""
        try:
            with open(self.log_file, "a") as f:
                f.write(f"{datetime.now().isoformat()} - {json.dumps(task_info)}\n")
        except Exception as e:
            logger.error(f"Error writing to task log file: {e}")

# Create a singleton instance
task_tracker = TaskTracker()