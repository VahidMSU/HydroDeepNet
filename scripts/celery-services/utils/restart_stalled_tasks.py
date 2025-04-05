#!/usr/bin/env python3
"""
Redis and Celery Task Restart Utility

This script identifies and restarts stalled Celery tasks:
- Find tasks that have been inactive for more than N hours
- Create new duplicate tasks with the same parameters
- Log task restart operations for tracking

Use with caution in production environments!
"""

import os
import sys
import json
import time
import argparse
import redis
import datetime
import uuid
import logging
from tabulate import tabulate

# Add application path for imports
sys.path.insert(0, '/data/SWATGenXApp/codes')
sys.path.insert(0, '/data/SWATGenXApp/codes/web_application')

# Configure output colors
HEADER = '\033[95m'
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'

# Configuration
REDIS_URL = 'redis://localhost:6379/0'
TASK_LOG_FILE = '/data/SWATGenXApp/codes/web_application/logs/model_tasks.log'
BACKUP_DIR = '/data/SWATGenXApp/codes/web_application/logs/backups'
USE_COLORS = True
MAX_KEYS_TO_PROCESS = 10000  # Safety limit for large Redis instances

def print_colored(text, color=None, bold=False):
    """Print text with optional color formatting"""
    if USE_COLORS and color:
        bold_code = BOLD if bold else ''
        print(f"{color}{bold_code}{text}{ENDC}")
    else:
        print(text)

def connect_to_redis(retry=3):
    """Connect to Redis with retry logic"""
    urls = [
        REDIS_URL,
        'redis://127.0.0.1:6379/0',
        'redis://localhost:6379/0',
        'redis://redis:6379/0'
    ]
    
    for attempt in range(retry):
        for url in urls:
            try:
                print_colored(f"Connecting to Redis at {url} (attempt {attempt+1}/{retry})...", BLUE)
                client = redis.Redis.from_url(
                    url, 
                    socket_timeout=5,
                    socket_connect_timeout=5,
                    decode_responses=False  # Keep as bytes for better handling
                )
                client.ping()  # Test connection
                print_colored(f"Successfully connected to Redis at {url}", GREEN)
                return client
            except redis.RedisError as e:
                print_colored(f"Failed to connect to Redis at {url}: {str(e)}", YELLOW)
            except Exception as e:
                print_colored(f"Unexpected error connecting to Redis at {url}: {str(e)}", YELLOW)
        
        # Wait before retry
        if attempt < retry - 1:
            delay = 2 ** attempt  # Exponential backoff
            print_colored(f"Retrying in {delay} seconds...", BLUE)
            time.sleep(delay)
            
    print_colored("Failed to connect to Redis after multiple attempts", RED, bold=True)
    return None

def identify_stalled_tasks(client, stale_hours=1, max_tasks=100, username_filter=None, site_filter=None):
    """Identify tasks that have not been updated for more than stale_hours"""
    print_colored(f"\nIDENTIFYING STALLED TASKS (Inactive > {stale_hours} hours)", HEADER, bold=True)
    
    try:
        # Get current timestamp for comparison
        current_time = time.time()
        stale_seconds = stale_hours * 60 * 60
        oldest_acceptable_time = current_time - stale_seconds
        
        # Search for celery task metadata records
        task_meta_keys = []
        task_pattern = "celery-task-meta-*"
        
        print_colored(f"Scanning Redis for task metadata matching '{task_pattern}'...", BLUE)
        
        # Scan for task metadata keys
        for key in client.scan_iter(match=task_pattern, count=1000):
            task_meta_keys.append(key)
            if len(task_meta_keys) >= MAX_KEYS_TO_PROCESS:
                print_colored(f"Reached maximum key limit ({MAX_KEYS_TO_PROCESS}). Limiting scan.", YELLOW)
                break
        
        print_colored(f"Found {len(task_meta_keys)} task metadata records", GREEN)
        
        # Also look for task status in task tracking records
        task_status_keys = []
        for key in client.scan_iter(match="task:*", count=1000):
            task_status_keys.append(key)
            if len(task_status_keys) >= MAX_KEYS_TO_PROCESS:
                print_colored(f"Reached maximum key limit for task status keys", YELLOW)
                break
                
        print_colored(f"Found {len(task_status_keys)} task status records", GREEN)
        
        # Collect stalled tasks by checking both metadata and status
        stalled_tasks = []
        checked_count = 0
        
        # Helper function to decode task data
        def decode_data(data):
            try:
                if isinstance(data, bytes):
                    return json.loads(data.decode('utf-8'))
                return data
            except:
                return None
                
        # Check task metadata keys first
        for key in task_meta_keys:
            try:
                checked_count += 1
                task_id = key.decode('utf-8').replace('celery-task-meta-', '')
                data = client.get(key)
                
                if not data:
                    continue
                    
                task_data = decode_data(data)
                if not task_data:
                    continue
                
                # Check if the task is in a non-terminal state and hasn't been updated
                status = task_data.get('status')
                if status not in ['PENDING', 'STARTED', 'RECEIVED', 'RETRY']:
                    continue
                    
                # Look for date_done or last update time
                date_done = task_data.get('date_done')
                if date_done:
                    # Convert to timestamp if it's a string
                    if isinstance(date_done, str):
                        try:
                            # Try ISO format first
                            date_done = datetime.datetime.fromisoformat(date_done).timestamp()
                        except ValueError:
                            try:
                                # Try other common formats
                                date_done = datetime.datetime.strptime(date_done, '%Y-%m-%d %H:%M:%S.%f').timestamp()
                            except ValueError:
                                continue
                    
                    # Check if it's stale
                    if date_done < oldest_acceptable_time:
                        task_meta = {
                            'task_id': task_id,
                            'status': status,
                            'last_update': date_done,
                            'hours_stalled': (current_time - date_done) / 3600,
                            'source': 'metadata'
                        }
                        
                        # Try to get additional task information from task: records
                        task_info_key = f"task:{task_id}"
                        task_info_data = client.get(task_info_key.encode('utf-8'))
                        if task_info_data:
                            task_info = decode_data(task_info_data)
                            if task_info:
                                # Add user and site info if available
                                task_meta['username'] = task_info.get('username')
                                task_meta['site_no'] = task_info.get('site_no')
                                task_meta['info'] = task_info.get('info')
                        
                        # Apply filters if provided
                        if username_filter and task_meta.get('username') != username_filter:
                            continue
                        if site_filter and task_meta.get('site_no') != site_filter:
                            continue
                            
                        stalled_tasks.append(task_meta)
            except Exception as e:
                print_colored(f"Error processing task metadata key {key}: {str(e)}", YELLOW)
        
        # Now check task status keys for additional information
        for key in task_status_keys:
            try:
                task_id = key.decode('utf-8').replace('task:', '')
                
                # Skip if we already found this task
                if any(t['task_id'] == task_id for t in stalled_tasks):
                    continue
                    
                data = client.get(key)
                if not data:
                    continue
                    
                task_data = decode_data(data)
                if not task_data:
                    continue
                
                # Check status
                status = task_data.get('status')
                if status not in ['PENDING', 'STARTED', 'RECEIVED', 'RETRY']:
                    continue
                
                # Apply username filter if provided
                if username_filter and task_data.get('username') != username_filter:
                    continue
                    
                # Apply site filter if provided
                if site_filter and task_data.get('site_no') != site_filter:
                    continue
                
                # Check updated_at or other timestamp fields
                updated_at = task_data.get('updated_at')
                if not updated_at:
                    # Try other possible fields
                    updated_at = task_data.get('created_at')
                    
                if updated_at:
                    # Convert to timestamp if it's a string
                    if isinstance(updated_at, str):
                        try:
                            updated_at = datetime.datetime.fromisoformat(updated_at).timestamp()
                        except ValueError:
                            try:
                                updated_at = datetime.datetime.strptime(updated_at, '%Y-%m-%d %H:%M:%S.%f').timestamp()
                            except ValueError:
                                continue
                    
                    # Check if it's stale
                    if updated_at < oldest_acceptable_time:
                        stalled_tasks.append({
                            'task_id': task_id,
                            'status': status,
                            'last_update': updated_at,
                            'hours_stalled': (current_time - updated_at) / 3600,
                            'source': 'status',
                            'username': task_data.get('username'),
                            'site_no': task_data.get('site_no'),
                            'info': task_data.get('info')
                        })
            except Exception as e:
                print_colored(f"Error processing task status key {key}: {str(e)}", YELLOW)
        
        # Sort by staleness (most stale first)
        stalled_tasks.sort(key=lambda x: x.get('hours_stalled', 0), reverse=True)
        
        # Limit to max_tasks
        if len(stalled_tasks) > max_tasks:
            print_colored(f"Limiting to {max_tasks} most stale tasks out of {len(stalled_tasks)} found", YELLOW)
            stalled_tasks = stalled_tasks[:max_tasks]
            
        # Display identified stalled tasks
        if stalled_tasks:
            print_colored(f"Found {len(stalled_tasks)} stalled tasks (inactive > {stale_hours} hours):", BLUE, bold=True)
            
            # Create a formatted table
            headers = ["Task ID", "Status", "Hours Stalled", "Username", "Site No"]
            table_data = []
            
            for task in stalled_tasks:
                hours = task.get('hours_stalled', 0)
                hours_display = f"{hours:.1f}"
                
                row = [
                    task['task_id'][:12] + "...",  # Truncate ID for display
                    task.get('status', 'UNKNOWN'),
                    hours_display,
                    task.get('username', 'unknown'),
                    task.get('site_no', 'unknown')
                ]
                table_data.append(row)
                
            # Print table
            table = tabulate(table_data, headers=headers, tablefmt="grid")
            print(table)
            
            return stalled_tasks
        else:
            print_colored(f"No stalled tasks found (inactive > {stale_hours} hours)", GREEN)
            return []
            
    except Exception as e:
        print_colored(f"Error identifying stalled tasks: {str(e)}", RED)
        import traceback
        print_colored(traceback.format_exc(), RED)
        return []

def restart_task(client, task_data, task_log_file, dry_run=True, remove_original=False):
    """Restart a stalled task by creating a new task with the same parameters"""
    task_id = task_data.get('task_id')
    username = task_data.get('username')
    site_no = task_data.get('site_no')
    info = task_data.get('info', {})
    
    # We need these parameters to restart the task
    if not all([username, site_no]):
        print_colored(f"Cannot restart task {task_id}: missing required parameters", YELLOW)
        return False
    
    # Extract resolution parameters from info
    ls_resolution = info.get('ls_resolution', 250)
    dem_resolution = info.get('dem_resolution', 30)
    
    # In dry run mode, just report what would be done
    if dry_run:
        print_colored(f"Would restart task for user={username}, site_no={site_no}, " + 
                     f"ls_res={ls_resolution}, dem_res={dem_resolution}", BLUE)
        if remove_original:
            print_colored(f"Would remove original task {task_id} after restart", BLUE)
        else:
            print_colored(f"Would mark original task {task_id} as RESTARTED", BLUE)
        return True
    
    try:
        # First, record that we're restarting this task
        current_time = datetime.datetime.now().isoformat()
        log_entry = {
            "task_id": task_id,
            "username": username,
            "site_no": site_no,
            "status": "RESTARTED",
            "created_at": current_time,
            "updated_at": current_time,
            "progress": 0,
            "info": {
                "ls_resolution": ls_resolution,
                "dem_resolution": dem_resolution,
                "start_time": time.time(),
                "message": f"Task restarted by restart script. Original task ID: {task_id}"
            }
        }
        
        # Write to log file
        with open(task_log_file, 'a') as f:
            f.write(f"{current_time} - {json.dumps(log_entry)}\n")
        
        # Create a new task JSON message for the queue
        new_task_id = str(uuid.uuid4())
        
        # Prepare task message with all required Celery fields to avoid 'properties' error
        task_body = {
            "id": new_task_id,
            "task": "app.swatgenx_tasks.create_model_task",
            "args": [username, site_no, ls_resolution, dem_resolution],
            "kwargs": {},
            "retries": 0,
            "eta": None,
            "expires": None,
            "utc": True,
            "callbacks": None,
            "errbacks": None,
            "timelimit": [None, None],
            "taskset": None,
            "chord": None,
        }
        
        # Build a complete Celery task message with all required fields including properties
        current_timestamp = int(time.time())
        celery_message = {
            "body": json.dumps(task_body),
            "content-encoding": "utf-8",
            "content-type": "application/json",
            "headers": {
                "lang": "py",
                "task": "app.swatgenx_tasks.create_model_task",
                "id": new_task_id,
                "shadow": None,
                "eta": None,
                "expires": None,
                "group": None,
                "group_index": None,
                "retries": 0,
                "timelimit": [None, None],
                "root_id": new_task_id,
                "parent_id": None,
                "argsrepr": repr([username, site_no, ls_resolution, dem_resolution]),
                "kwargsrepr": "{}",
                "origin": "gen" + str(current_timestamp)
            },
            "properties": {
                "correlation_id": new_task_id,
                "reply_to": "",
                "delivery_mode": 2,
                "delivery_info": {
                    "exchange": "",
                    "routing_key": "model_creation"
                },
                "priority": 0,
                "body_encoding": "base64",
                "delivery_tag": str(current_timestamp)
            }
        }
        
        # Serialize the task for Celery
        task_message = json.dumps(celery_message)
        
        # Push to appropriate queue (model_creation queue for model tasks)
        queue_name = "model_creation"
        client.rpush(queue_name, task_message)
        
        # Create a simple task record for the new task
        new_task_record = {
            "task_id": new_task_id,
            "username": username, 
            "site_no": site_no,
            "status": "PENDING",
            "created_at": current_time,
            "updated_at": current_time,
            "progress": 0,
            "info": {
                "ls_resolution": ls_resolution,
                "dem_resolution": dem_resolution,
                "start_time": time.time(),
                "restarted_from": task_id,
                "message": "Task created by restart operation"
            }
        }
        
        # Store the task record in Redis
        client.set(f"task:{new_task_id}", json.dumps(new_task_record))
        
        # Log the restart in the task log
        with open(task_log_file, 'a') as f:
            f.write(f"{current_time} - {json.dumps(new_task_record)}\n")
            
        # Now handle the original task - either update its status or remove it
        if remove_original:
            # Remove the original task from Redis
            meta_key = f"celery-task-meta-{task_id}"
            status_key = f"task:{task_id}"
            
            deleted_count = 0
            if client.exists(meta_key.encode('utf-8')):
                client.delete(meta_key.encode('utf-8'))
                deleted_count += 1
                
            if client.exists(status_key.encode('utf-8')):
                client.delete(status_key.encode('utf-8'))
                deleted_count += 1
                
            print_colored(f"Removed {deleted_count} Redis keys for original task {task_id}", GREEN)
        else:
            # Update the original task status instead of removing it
            # This prevents the same task from being restarted multiple times
            status_key = f"task:{task_id}"
            if client.exists(status_key.encode('utf-8')):
                try:
                    # Get current task data
                    task_json = client.get(status_key.encode('utf-8'))
                    if task_json:
                        original_task = json.loads(task_json)
                        
                        # Update the status and add reference to new task
                        original_task['status'] = "RESTARTED"
                        original_task['updated_at'] = current_time
                        if 'info' not in original_task:
                            original_task['info'] = {}
                        original_task['info']['restarted_as'] = new_task_id
                        original_task['info']['restart_time'] = current_time
                        
                        # Save the updated task status
                        client.set(status_key, json.dumps(original_task))
                        print_colored(f"Updated original task {task_id} status to RESTARTED", GREEN)
                except Exception as e:
                    print_colored(f"Error updating original task status: {str(e)}", YELLOW)
            
            # Also try to update the task metadata if it exists
            meta_key = f"celery-task-meta-{task_id}"
            if client.exists(meta_key.encode('utf-8')):
                try:
                    # Get current metadata
                    meta_json = client.get(meta_key.encode('utf-8'))
                    if meta_json:
                        meta_data = json.loads(meta_json)
                        
                        # Update status and add info about restart
                        meta_data['status'] = "SUCCESS"  # Mark as SUCCESS to prevent restart detection
                        meta_data['result'] = {
                            "status": "RESTARTED",
                            "restarted_as": new_task_id,
                            "restart_time": current_time
                        }
                        
                        # Save the updated metadata
                        client.set(meta_key, json.dumps(meta_data))
                        print_colored(f"Updated task metadata for {task_id}", GREEN)
                except Exception as e:
                    print_colored(f"Error updating task metadata: {str(e)}", YELLOW)
            
        print_colored(f"Successfully restarted task as new task ID: {new_task_id}", GREEN)
        return True
        
    except Exception as e:
        print_colored(f"Error restarting task {task_id}: {str(e)}", RED)
        import traceback
        print_colored(traceback.format_exc(), RED)
        return False

def restart_stalled_tasks(client, stale_hours=1, max_restarts=10, username_filter=None, 
                         site_filter=None, dry_run=True, no_confirm=False, remove_original=False):
    """Identify and restart stalled tasks"""
    print_colored("\nRESTARTING STALLED TASKS", HEADER, bold=True)
    
    # First identify stalled tasks
    stalled_tasks = identify_stalled_tasks(client, stale_hours, max_restarts, 
                                          username_filter, site_filter)
    
    if not stalled_tasks:
        return True
        
    if dry_run:
        action_desc = "remove original and restart" if remove_original else "restart"
        print_colored(f"DRY RUN: Would {action_desc} {len(stalled_tasks)} stalled tasks", YELLOW, bold=True)
        return True
    
    # Get confirmation before proceeding
    if not no_confirm:
        action_desc = "remove original and restart" if remove_original else "restart"
        confirm = input(f"Ready to {action_desc} {len(stalled_tasks)} stalled tasks? (yes/no): ")
        if confirm.lower() != "yes":
            print_colored("Task restart cancelled", BLUE)
            return False
    
    # Restart each task
    success_count = 0
    for task in stalled_tasks:
        if restart_task(client, task, TASK_LOG_FILE, dry_run=False, remove_original=remove_original):
            success_count += 1
    
    print_colored(f"Successfully restarted {success_count}/{len(stalled_tasks)} tasks", 
                 GREEN if success_count == len(stalled_tasks) else YELLOW)
    return success_count > 0

def clear_stalled_task(client, task_id, dry_run=True):
    """Clear a stalled task from Redis without restarting it"""
    print_colored(f"Clearing stalled task: {task_id}", BLUE)
    
    try:
        # Check if task exists
        meta_key = f"celery-task-meta-{task_id}"
        status_key = f"task:{task_id}"
        
        meta_exists = client.exists(meta_key.encode('utf-8'))
        status_exists = client.exists(status_key.encode('utf-8'))
        
        if not meta_exists and not status_exists:
            print_colored(f"Task {task_id} not found in Redis", YELLOW)
            return False
        
        if dry_run:
            if meta_exists:
                print_colored(f"Would delete task metadata: {meta_key}", YELLOW)
            if status_exists:
                print_colored(f"Would delete task status: {status_key}", YELLOW)
            return True
            
        # Delete the task
        deleted = 0
        if meta_exists:
            client.delete(meta_key.encode('utf-8'))
            deleted += 1
        if status_exists:
            client.delete(status_key.encode('utf-8'))
            deleted += 1
            
        print_colored(f"Successfully deleted {deleted} Redis keys for task {task_id}", GREEN)
        return True
        
    except Exception as e:
        print_colored(f"Error clearing task {task_id}: {str(e)}", RED)
        return False

def setup_logging(log_file=None):
    """Set up logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_level = logging.INFO
    
    # Configure the root logger
    logging.basicConfig(level=log_level, format=log_format)
    
    # Create a logger for this script
    logger = logging.getLogger('restart_stalled_tasks')
    
    # Add a file handler if a log file is specified
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(log_format))
            logger.addHandler(file_handler)
            print_colored(f"Logging to file: {log_file}", BLUE)
        except Exception as e:
            print_colored(f"Error setting up log file: {str(e)}", RED)
    
    return logger

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Redis and Celery Task Restart Utility')
    
    # Main options
    parser.add_argument('--stale-hours', type=float, default=1.0,
                       help='Number of hours before a task is considered stalled (default: 1)')
    parser.add_argument('--max-restarts', type=int, default=10,
                       help='Maximum number of tasks to restart in one run (default: 10)')
    parser.add_argument('--dry-run', '-d', action='store_true',
                       help='Show what would be done without making changes')
    parser.add_argument('--yes', '-y', action='store_true', 
                       help='Skip all confirmation prompts')
    parser.add_argument('--clean', action='store_true',
                       help='Disable colors in output')
    
    # Filtering options
    parser.add_argument('--username', type=str,
                       help='Filter tasks by username')
    parser.add_argument('--site', type=str,
                       help='Filter tasks by site number')
    parser.add_argument('--task-id', type=str,
                       help='Work with a specific task ID')
    
    # Action options
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument('--list-only', action='store_true',
                            help='Only list stalled tasks without restarting')
    action_group.add_argument('--clear-task', action='store_true',
                            help='Clear task instead of restarting (requires --task-id)')
    action_group.add_argument('--fix-corrupted', action='store_true',
                            help='Scan and fix corrupted tasks in Redis queues')
    
    # Additional options
    parser.add_argument('--log-file', type=str,
                       help='Log all actions to the specified file')
    parser.add_argument('--remove-original', action='store_true',
                       help='Remove original stalled tasks after restart instead of just updating status')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.clear_task and not args.task_id:
        parser.error("--clear-task requires --task-id to be specified")
    
    # Configure global settings
    global USE_COLORS
    USE_COLORS = not args.clean
    
    return args

def main():
    """Main function"""
    args = parse_args()
    
    # Set up logging
    logger = setup_logging(args.log_file)
    
    # Connect to Redis
    client = connect_to_redis()
    if not client:
        logger.error("Failed to connect to Redis")
        sys.exit(1)
    
    # Add correct path for cleanup_corrupted_tasks import
    sys.path.insert(0, '/data/SWATGenXApp/codes/scripts/celery-services/services')

    # Now import the cleanup script correctly
    try:
        import cleanup_corrupted_tasks
    except ImportError:
        print_colored("Error: cleanup_corrupted_tasks.py script not found. Please ensure it's available in the services directory.", RED, bold=True)
        sys.exit(1)

    try:
        # Handle specific task ID operations
        if args.task_id:
            if args.clear_task:
                clear_stalled_task(client, args.task_id, args.dry_run)
            else:
                # For a specific task ID, get task data and restart it
                task_data = None
                
                # Check task metadata key
                meta_key = f"celery-task-meta-{args.task_id}"
                meta_data = client.get(meta_key.encode('utf-8'))
                
                # Check task status key
                status_key = f"task:{args.task_id}"
                status_data = client.get(status_key.encode('utf-8'))
                
                if meta_data or status_data:
                    # Construct task data from available information
                    task_data = {'task_id': args.task_id}
                    
                    if status_data:
                        try:
                            status_info = json.loads(status_data.decode('utf-8'))
                            task_data.update({
                                'username': status_info.get('username'),
                                'site_no': status_info.get('site_no'),
                                'info': status_info.get('info', {})
                            })
                        except:
                            pass
                    
                    # Restart the specific task
                    if 'username' in task_data and 'site_no' in task_data:
                        restart_task(client, task_data, TASK_LOG_FILE, args.dry_run, args.remove_original)
                    else:
                        print_colored(f"Cannot restart task {args.task_id}: missing required data", RED)
                else:
                    print_colored(f"Task {args.task_id} not found in Redis", RED)
        
        # Handle fixing corrupted tasks
        elif args.fix_corrupted:
            if args.dry_run:
                print_colored("Scanning for corrupted tasks (dry run):", BLUE, bold=True)
                cleanup_corrupted_tasks.scan_all_celery_queues(client, dry_run=True, repair=True)
            else:
                if not args.yes:
                    confirm = input("Ready to remove corrupted tasks from Redis queues? (yes/no): ")
                    if confirm.lower() != "yes":
                        print_colored("Operation cancelled", BLUE)
                        return
                    
                print_colored("Scanning and fixing corrupted tasks:", GREEN, bold=True)
                cleanup_corrupted_tasks.scan_all_celery_queues(client, dry_run=False, repair=True)
            
        # Otherwise perform bulk operations
        elif args.list_only:
            identify_stalled_tasks(client, args.stale_hours, args.max_restarts, 
                                  args.username, args.site)
        else:
            restart_stalled_tasks(client, args.stale_hours, args.max_restarts, 
                                 args.username, args.site, args.dry_run, args.yes, args.remove_original)
            
    except KeyboardInterrupt:
        print_colored("\nOperation cancelled by user", YELLOW)
        logger.warning("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print_colored(f"\nUnexpected error: {str(e)}", RED)
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
