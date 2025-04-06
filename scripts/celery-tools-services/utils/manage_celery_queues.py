#!/usr/bin/env python3
"""
SWATGenX Celery Queue Management Tool
Diagnose and manage Celery queues, inspect tasks, and reset queues when needed
"""

import os
import sys
import json
import time
import argparse
import logging
import redis
from datetime import datetime
from tabulate import tabulate

# Add application path for imports
sys.path.insert(0, '/data/SWATGenXApp/codes')
sys.path.insert(0, '/data/SWATGenXApp/codes/web_application')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('queue_manager')

# Configure output formats
HEADER = '\033[95m'
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'

# Global variables
USE_COLORS = True
REDIS_URL = 'redis://localhost:6379/0'

def print_colored(text, color, bold=False):
    """Print colored text if colors are enabled"""
    if USE_COLORS:
        bold_code = BOLD if bold else ''
        print(f"{color}{bold_code}{text}{ENDC}")
    else:
        print(text)

def connect_to_redis(url=None):
    """Connect to Redis with error handling"""
    try:
        redis_url = url or REDIS_URL
        client = redis.Redis.from_url(redis_url, socket_timeout=5, socket_connect_timeout=5)
        client.ping()  # Test connection
        return client
    except redis.RedisError as e:
        print_colored(f"Error connecting to Redis at {redis_url}: {str(e)}", RED)
        return None

def list_queues(client):
    """List all Celery queues in Redis"""
    print_colored("\nCELERY QUEUES", HEADER, bold=True)
    
    queues = []
    # Celery queue naming pattern
    known_queues = ['celery', 'model_creation']
    
    # Check queue lengths and details
    for queue in known_queues:
        try:
            queue_len = client.llen(queue)
            queues.append({
                'name': queue,
                'tasks': queue_len,
                'is_empty': queue_len == 0
            })
        except redis.RedisError as e:
            print_colored(f"Error checking queue {queue}: {str(e)}", RED)
    
    # Display queue information
    queue_table = []
    for q in queues:
        if USE_COLORS:
            status = f"{GREEN}Empty{ENDC}" if q['is_empty'] else f"{YELLOW}{q['tasks']} tasks{ENDC}"
        else:
            status = "Empty" if q['is_empty'] else f"{q['tasks']} tasks"
        
        queue_table.append([q['name'], q['tasks'], status])
    
    if queue_table:
        print(tabulate(
            queue_table,
            headers=["Queue Name", "Tasks Waiting", "Status"],
            tablefmt="pretty"
        ))
    else:
        print_colored("No queues found", YELLOW)
    
    return queues

def inspect_queue_tasks(client, queue_name, limit=10, full=False):
    """Inspect tasks in a specific queue"""
    print_colored(f"\nINSPECTING QUEUE: {queue_name}", HEADER, bold=True)
    
    try:
        # Get queue length
        queue_len = client.llen(queue_name)
        print(f"Queue length: {queue_len} tasks")
        
        if queue_len == 0:
            print_colored("Queue is empty", GREEN)
            return []
        
        # Get sample of tasks from the queue
        # Note: We're using LRANGE to peek at the tasks without removing them
        end_idx = min(limit - 1, queue_len - 1)
        tasks_data = client.lrange(queue_name, 0, end_idx)
        
        if not tasks_data:
            print_colored("Couldn't retrieve task data", YELLOW)
            return []
        
        # Parse and display task information
        tasks = []
        task_table = []
        
        for i, task_bytes in enumerate(tasks_data):
            try:
                # Celery queue items are JSON strings
                task_str = task_bytes.decode('utf-8')
                
                # Extract task information - different handling for different queue types
                if task_str.startswith('{"'):
                    # Try JSON parsing
                    task_data = json.loads(task_str)
                    
                    # Extract task details
                    if 'headers' in task_data and 'task' in task_data['headers']:
                        task_name = task_data['headers'].get('task', 'unknown')
                        task_id = task_data['headers'].get('id', 'no-id')
                        args = task_data.get('args', [])
                        kwargs = task_data.get('kwargs', {})
                        
                        # Try to extract more details if available
                        created_at = kwargs.get('created_at', 'unknown')
                        username = kwargs.get('username', 'unknown')
                        site_no = kwargs.get('site_no', 'unknown')
                        
                        # Store task information
                        task = {
                            'id': task_id,
                            'name': task_name,
                            'args': args,
                            'kwargs': kwargs,
                            'created_at': created_at,
                            'username': username,
                            'site_no': site_no
                        }
                        tasks.append(task)
                        
                        # Create table row
                        if full:
                            # In full mode, show all details
                            created_display = created_at if isinstance(created_at, str) else 'unknown'
                            task_table.append([
                                i+1, 
                                task_name.split('.')[-1],  # Just the last part of the task name
                                task_id[:10] + "...",      # Truncated ID
                                username,
                                site_no,
                                created_display
                            ])
                        else:
                            # In summary mode, show fewer columns
                            task_table.append([
                                i+1,
                                task_name.split('.')[-1],
                                username,
                                site_no
                            ])
                else:
                    # Non-JSON format - just display raw data
                    print(f"Task {i+1}: (raw data) {task_str[:50]}...")
            except Exception as e:
                print_colored(f"Error parsing task {i+1}: {str(e)}", RED)
        
        # Display task table if we have tasks
        if task_table:
            if full:
                print(tabulate(
                    task_table,
                    headers=["#", "Task Type", "Task ID", "Username", "Site No", "Created At"],
                    tablefmt="pretty"
                ))
            else:
                print(tabulate(
                    task_table,
                    headers=["#", "Task Type", "Username", "Site No"],
                    tablefmt="pretty"
                ))
                
            if limit < queue_len:
                print_colored(f"(Showing {limit} of {queue_len} tasks)", YELLOW)
        else:
            print_colored("No tasks found or couldn't parse tasks", YELLOW)
            
        return tasks
            
    except redis.RedisError as e:
        print_colored(f"Redis error: {str(e)}", RED)
        return []
        
def flush_queue(client, queue_name, confirm=True):
    """Remove all tasks from a queue with confirmation"""
    if confirm:
        print_colored(f"\nWARNING: You are about to delete all tasks in queue '{queue_name}'!", RED, bold=True)
        print_colored("This action cannot be undone and may affect running processes.", YELLOW)
        
        confirmation = input(f"Type '{queue_name}' to confirm deletion: ")
        if confirmation != queue_name:
            print_colored("Deletion cancelled.", BLUE)
            return False
    
    try:
        # Get queue length before deletion
        queue_len = client.llen(queue_name)
        
        if queue_len == 0:
            print_colored(f"Queue '{queue_name}' is already empty.", GREEN)
            return True
            
        # Delete the queue
        client.delete(queue_name)
        
        # Check if deletion was successful
        new_len = client.llen(queue_name)
        if new_len == 0:
            print_colored(f"Successfully removed {queue_len} tasks from queue '{queue_name}'.", GREEN)
            return True
        else:
            print_colored(f"Something went wrong. Queue still has {new_len} tasks.", RED)
            return False
            
    except redis.RedisError as e:
        print_colored(f"Redis error: {str(e)}", RED)
        return False

def backup_queue(client, queue_name, backup_dir='/data/SWATGenXApp/codes/web_application/logs'):
    """Backup all tasks in a queue to a JSON file"""
    print_colored(f"\nBACKING UP QUEUE: {queue_name}", HEADER, bold=True)
    
    try:
        # Ensure backup directory exists
        os.makedirs(backup_dir, exist_ok=True)
        
        # Get queue length
        queue_len = client.llen(queue_name)
        if queue_len == 0:
            print_colored(f"Queue '{queue_name}' is empty. Nothing to backup.", YELLOW)
            return None
            
        # Get all tasks from the queue
        tasks_data = client.lrange(queue_name, 0, -1)
        
        # Parse the tasks
        tasks = []
        for task_bytes in tasks_data:
            try:
                task_str = task_bytes.decode('utf-8')
                tasks.append(task_str)
            except Exception:
                # If decoding fails, store the raw bytes as hex
                tasks.append("HEX:" + task_bytes.hex())
        
        # Create a timestamped backup file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = os.path.join(backup_dir, f"queue_backup_{queue_name}_{timestamp}.json")
        
        # Write the backup file
        with open(backup_file, 'w') as f:
            json.dump({
                'queue': queue_name,
                'timestamp': datetime.now().isoformat(),
                'task_count': queue_len,
                'tasks': tasks
            }, f, indent=2)
        
        print_colored(f"Successfully backed up {queue_len} tasks to {backup_file}", GREEN)
        return backup_file
        
    except Exception as e:
        print_colored(f"Error backing up queue: {str(e)}", RED)
        return None

def filter_queue_by_task_type(client, queue_name, task_types_to_remove, backup_first=True):
    """
    Remove specific task types from a queue while preserving others.
    
    Args:
        client: Redis client
        queue_name: Name of the queue to filter
        task_types_to_remove: List of task type names to remove
        backup_first: Whether to backup the queue before modification
    
    Returns:
        Tuple of (success, removed_count, remaining_count)
    """
    print_colored(f"\nREMOVING SPECIFIC TASK TYPES FROM QUEUE: {queue_name}", HEADER, bold=True)
    print(f"Task types to remove: {', '.join(task_types_to_remove)}")
    
    # Create backup first if requested
    if backup_first:
        backup_file = backup_queue(client, queue_name)
        if not backup_file:
            print_colored("Backup failed, aborting task removal for safety", RED)
            return False, 0, 0
    
    try:
        # Get all tasks from the queue
        queue_len = client.llen(queue_name)
        if queue_len == 0:
            print_colored(f"Queue '{queue_name}' is empty. Nothing to remove.", YELLOW)
            return True, 0, 0
        
        # Get all current tasks
        tasks_data = client.lrange(queue_name, 0, -1)
        
        # Create a temporary queue with only the tasks we want to keep
        temp_queue_name = f"{queue_name}_temp_{int(time.time())}"
        kept_count = 0
        removed_count = 0
        
        for task_bytes in tasks_data:
            try:
                # Parse the task
                task_str = task_bytes.decode('utf-8')
                
                # Only process JSON-formatted tasks
                if task_str.startswith('{'):
                    task_data = json.loads(task_str)
                    
                    # Extract task name from headers if available
                    task_name = "unknown"
                    if 'headers' in task_data and 'task' in task_data['headers']:
                        task_name = task_data['headers'].get('task', 'unknown')
                    
                    # Check if this task type should be removed
                    should_remove = False
                    for task_type in task_types_to_remove:
                        if task_type in task_name:
                            should_remove = True
                            break
                    
                    if should_remove:
                        removed_count += 1
                    else:
                        # Keep this task by pushing to temp queue
                        client.rpush(temp_queue_name, task_bytes)
                        kept_count += 1
                else:
                    # Non-JSON task, keep it to be safe
                    client.rpush(temp_queue_name, task_bytes)
                    kept_count += 1
                    
            except Exception as e:
                # On error, keep the task to be safe
                print_colored(f"Error parsing task: {str(e)}", YELLOW)
                client.rpush(temp_queue_name, task_bytes)
                kept_count += 1
        
        # Now we have separated tasks, atomically replace the original queue
        # Use Redis transaction to make this operation atomic
        pipe = client.pipeline()
        pipe.delete(queue_name)  # Delete original queue
        
        # If we have tasks to keep, rename temp queue to original
        if kept_count > 0:
            pipe.rename(temp_queue_name, queue_name)
        else:
            pipe.delete(temp_queue_name)  # Just clean up the temp queue
            
        pipe.execute()
        
        # Verify results
        new_queue_len = client.llen(queue_name)
        
        print_colored(f"Successfully removed {removed_count} tasks of types: {', '.join(task_types_to_remove)}", GREEN)
        print_colored(f"Kept {kept_count} tasks of other types", GREEN)
        print_colored(f"New queue length: {new_queue_len}", GREEN)
        
        return True, removed_count, kept_count
        
    except Exception as e:
        print_colored(f"Error filtering queue: {str(e)}", RED)
        # Try to clean up temp queue if it exists
        try:
            client.delete(temp_queue_name)
        except:
            pass
        return False, 0, 0

def is_valid_site_number(site_no):
    """
    Check if a site number appears to be valid.
    
    Valid formats:
    - 8-digit USGS site numbers (e.g., "03604000")
    - 9-digit USGS site numbers (e.g., "041590774")
    
    Invalid formats:
    - Anything with square brackets (e.g., "[2025-04-0")
    - Anything that's not primarily numeric
    """
    if not site_no or not isinstance(site_no, str):
        return False
        
    # Square brackets indicate malformed data
    if '[' in site_no or ']' in site_no:
        return False
        
    # Site numbers should be mostly numeric (allow some special chars)
    numeric_chars = sum(c.isdigit() for c in site_no)
    if numeric_chars < len(site_no) * 0.7:  # At least 70% digits
        return False
        
    # Valid USGS site numbers typically have 8-9 digits
    digits_only = ''.join(c for c in site_no if c.isdigit())
    if not (7 <= len(digits_only) <= 10):
        return False
        
    return True

def filter_tasks_by_site_number(client, queue_name, remove_invalid_site_numbers=True, backup_first=True):
    """
    Filter tasks based on site number validity.
    
    Args:
        client: Redis client
        queue_name: Name of the queue to filter
        remove_invalid_site_numbers: If True, remove tasks with invalid site numbers
        backup_first: Whether to backup the queue before modification
    
    Returns:
        Tuple of (success, removed_count, remaining_count)
    """
    print_colored(f"\nFILTERING TASKS BY SITE NUMBER IN QUEUE: {queue_name}", HEADER, bold=True)
    
    # Create backup first if requested
    if backup_first:
        backup_file = backup_queue(client, queue_name)
        if not backup_file:
            print_colored("Backup failed, aborting task filtering for safety", RED)
            return False, 0, 0
    
    try:
        # Get all tasks from the queue
        queue_len = client.llen(queue_name)
        if queue_len == 0:
            print_colored(f"Queue '{queue_name}' is empty. Nothing to filter.", YELLOW)
            return True, 0, 0
        
        # Get all current tasks
        tasks_data = client.lrange(queue_name, 0, -1)
        
        # Create a temporary queue with only the tasks we want to keep
        temp_queue_name = f"{queue_name}_temp_{int(time.time())}"
        kept_count = 0
        removed_count = 0
        invalid_site_numbers = set()
        
        for task_bytes in tasks_data:
            try:
                # Parse the task
                task_str = task_bytes.decode('utf-8')
                
                # Only process JSON-formatted tasks
                if task_str.startswith('{'):
                    task_data = json.loads(task_str)
                    
                    # Extract site number from kwargs if available
                    site_no = None
                    if 'kwargs' in task_data:
                        site_no = task_data.get('kwargs', {}).get('site_no')
                    
                    # Check if this task has an invalid site number
                    if site_no is not None and not is_valid_site_number(str(site_no)):
                        invalid_site_numbers.add(str(site_no))
                        if remove_invalid_site_numbers:
                            removed_count += 1
                            continue  # Skip this task - don't add to temp queue
                    
                # Keep this task
                client.rpush(temp_queue_name, task_bytes)
                kept_count += 1
                    
            except Exception as e:
                # On error, keep the task to be safe
                print_colored(f"Error parsing task: {str(e)}", YELLOW)
                client.rpush(temp_queue_name, task_bytes)
                kept_count += 1
        
        # If we're actually removing tasks
        if remove_invalid_site_numbers and removed_count > 0:
            # Now we have separated tasks, atomically replace the original queue
            # Use Redis transaction to make this operation atomic
            pipe = client.pipeline()
            pipe.delete(queue_name)  # Delete original queue
            
            # If we have tasks to keep, rename temp queue to original
            if kept_count > 0:
                pipe.rename(temp_queue_name, queue_name)
            else:
                pipe.delete(temp_queue_name)  # Just clean up the temp queue
                
            pipe.execute()
            
            # Verify results
            new_queue_len = client.llen(queue_name)
            
            print_colored(f"Found {len(invalid_site_numbers)} invalid site number patterns:", YELLOW)
            for site in sorted(invalid_site_numbers):
                print(f"  - '{site}'")
            
            print_colored(f"Successfully removed {removed_count} tasks with invalid site numbers", GREEN)
            print_colored(f"Kept {kept_count} tasks with valid site numbers", GREEN)
            print_colored(f"New queue length: {new_queue_len}", GREEN)
        else:
            # Just reporting, not removing
            client.delete(temp_queue_name)  # Clean up temp queue
            
            print_colored(f"Found {len(invalid_site_numbers)} invalid site number patterns:", YELLOW)
            for site in sorted(invalid_site_numbers):
                print(f"  - '{site}'")
            
            print_colored(f"Would remove {removed_count} tasks with invalid site numbers", BLUE)
            print_colored(f"Would keep {kept_count} tasks with valid site numbers", BLUE)
            print_colored("No changes were made (dry run)", BLUE)
            
        return True, removed_count, kept_count
        
    except Exception as e:
        print_colored(f"Error filtering tasks by site number: {str(e)}", RED)
        # Try to clean up temp queue if it exists
        try:
            client.delete(temp_queue_name)
        except:
            pass
        return False, 0, 0

def main():
    """Main function to run the script"""
    parser = argparse.ArgumentParser(description='SWATGenX Celery Queue Management Tool')
    
    # Command-line arguments
    parser.add_argument('--list', '-l', action='store_true', help='List all Celery queues')
    parser.add_argument('--inspect', '-i', type=str, help='Inspect tasks in a specific queue')
    parser.add_argument('--flush', '-f', type=str, help='Remove all tasks from a queue')
    parser.add_argument('--backup', '-b', type=str, help='Backup tasks in a queue to a file')
    parser.add_argument('--limit', type=int, default=20, help='Limit the number of tasks to inspect')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation for destructive operations')
    parser.add_argument('--full', action='store_true', help='Show full task details when inspecting')
    parser.add_argument('--clean', action='store_true', help='Disable colors in output')
    parser.add_argument('--remove-tasks', type=str, help='Remove specific task types from queue (comma-separated)')
    parser.add_argument('--queue', '-q', type=str, help='Queue to operate on with --remove-tasks')
    parser.add_argument('--fix-site-numbers', type=str, help='Remove tasks with invalid site numbers from specified queue')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be removed without making changes')
    
    args = parser.parse_args()
    
    # Configure color output
    global USE_COLORS
    USE_COLORS = not args.clean
    
    # Connect to Redis
    redis_client = connect_to_redis()
    if not redis_client:
        sys.exit(1)
    
    # Execute requested operations
    if args.list:
        list_queues(redis_client)
    
    if args.inspect:
        inspect_queue_tasks(redis_client, args.inspect, args.limit, args.full)
    
    if args.backup:
        backup_queue(redis_client, args.backup)
    
    if args.flush:
        backup_file = backup_queue(redis_client, args.flush)
        if backup_file or args.yes:
            flush_queue(redis_client, args.flush, not args.yes)
    
    # New functionality to remove specific task types
    if args.remove_tasks and args.queue:
        task_types = [t.strip() for t in args.remove_tasks.split(',')]
        print_colored(f"Preparing to remove tasks of types: {', '.join(task_types)}", YELLOW)
        
        if not args.yes:
            confirmation = input(f"Type '{args.queue}' to confirm removal of specified task types: ")
            if confirmation != args.queue:
                print_colored("Task removal cancelled.", BLUE)
                sys.exit(0)
                
        filter_queue_by_task_type(redis_client, args.queue, task_types)
    
    # New functionality to filter tasks by site number
    if args.fix_site_numbers:
        print_colored(f"Checking for tasks with invalid site numbers in queue: {args.fix_site_numbers}", YELLOW)
        
        if not args.dry_run and not args.yes:
            confirmation = input(f"Type '{args.fix_site_numbers}' to confirm removal of tasks with invalid site numbers: ")
            if confirmation != args.fix_site_numbers:
                print_colored("Task removal cancelled.", BLUE)
                sys.exit(0)
        
        filter_tasks_by_site_number(redis_client, args.fix_site_numbers, not args.dry_run)
    
    # If no operation specified, show help
    if not any([args.list, args.inspect, args.backup, args.flush, 
               (args.remove_tasks and args.queue), args.fix_site_numbers]):
        parser.print_help()

if __name__ == "__main__":
    main()
