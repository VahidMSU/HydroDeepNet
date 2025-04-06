#!/usr/bin/env python3
"""
SWATGenX Task Repair Utility
Fix stuck tasks and reset task state in Redis
"""

import os
import sys
import json
import argparse
import redis
import re
import threading
import signal
import time
from datetime import datetime, timedelta
from tabulate import tabulate

# Configure output colors
HEADER = '\033[95m'
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'

# Add application path for imports
sys.path.insert(0, '/data/SWATGenXApp/codes')
sys.path.insert(0, '/data/SWATGenXApp/codes/web_application')

# Configuration
REDIS_URL = 'redis://localhost:6379/0'
USE_COLORS = True
MAX_TASKS_TO_SHOW = 20

def print_colored(text, color=None, bold=False):
    """Print text with optional color formatting"""
    if USE_COLORS and color:
        bold_code = BOLD if bold else ''
        print(f"{color}{bold_code}{text}{ENDC}")
    else:
        print(text)

def connect_to_redis():
    """Connect to Redis with error handling"""
    try:
        client = redis.Redis.from_url(REDIS_URL, socket_timeout=5, socket_connect_timeout=5)
        client.ping()  # Test connection
        return client
    except redis.RedisError as e:
        print_colored(f"Error connecting to Redis: {str(e)}", RED)
        return None

def is_valid_site_number(site_no):
    """Check if site number format is valid"""
    if not site_no or not isinstance(site_no, str):
        return False
        
    # Square brackets indicate malformed data
    if '[' in site_no or ']' in site_no:
        return False
        
    # Site numbers should be mostly numeric
    numeric_chars = sum(c.isdigit() for c in site_no)
    if numeric_chars < len(site_no) * 0.7:  # At least 70% digits
        return False
        
    # Valid USGS site numbers typically have 8-9 digits
    digits_only = ''.join(c for c in site_no if c.isdigit())
    if not (7 <= len(digits_only) <= 10):
        return False
        
    return True

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Operation timed out")

def with_timeout(func, args=(), kwargs={}, timeout_secs=10, default=None):
    """Run a function with a timeout"""
    result = [default]
    exception = [None]
    
    def worker():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e
    
    # Start the worker thread
    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()
    
    # Wait for the thread to complete or timeout
    thread.join(timeout_secs)
    if thread.is_alive():
        return default, TimeoutError(f"Function {func.__name__} timed out after {timeout_secs} seconds")
    
    # Return the result or exception
    if exception[0]:
        return default, exception[0]
    return result[0], None

def safe_redis_get(client, key, timeout=5):
    """Safely get a value from Redis with timeout"""
    try:
        # Use timeout function to avoid hanging
        value, error = with_timeout(client.get, args=(key,), timeout_secs=timeout)
        if error:
            print_colored(f"Error retrieving key {key}: {error}", YELLOW)
            return None
        return value
    except Exception as e:
        print_colored(f"Error retrieving key {key}: {e}", YELLOW)
        return None

def safe_json_loads(json_str, default=None):
    """Safely parse JSON with additional error handling"""
    if not json_str:
        return default
    
    # If input is bytes, decode first
    if isinstance(json_str, bytes):
        try:
            json_str = json_str.decode('utf-8')
        except UnicodeDecodeError:
            # Try latin-1 as fallback for binary data
            try:
                json_str = json_str.decode('latin-1')
            except Exception:
                return default
    
    try:
        # Parse the JSON
        return json.loads(json_str)
    except json.JSONDecodeError:
        # If it fails, try to sanitize the string
        try:
            # Remove control characters that might break JSON parsing
            clean_str = ''.join(ch for ch in json_str if ch.isprintable())
            return json.loads(clean_str)
        except:
            return default

def list_active_tasks(redis_client, include_all=False, max_keys_to_process=5000):
    """List all active tasks stored in Redis"""
    print_colored("\nACTIVE TASKS IN REDIS", HEADER, bold=True)
    
    active_tasks = []
    task_keys = []
    
    # Get all keys that might be task-related
    try:
        # Use a safer approach with scanning instead of keys
        print("Scanning Redis for task-related keys (this may take a moment)...")
        all_keys = []
        cursor = '0'
        scan_count = 0
        pattern_matches = 0
        
        # Define patterns to match task keys
        patterns = ["celery-task-meta-*", "task:*"]
        
        # Use scan to iterate through keys safely
        while cursor != 0 and scan_count < 10:  # Limit scan operations
            cursor, keys = redis_client.scan(cursor=cursor, match="*", count=1000)
            scan_count += 1
            
            # Process this batch of keys
            if keys:
                decoded_keys = [k.decode('utf-8') for k in keys]
                for key in decoded_keys:
                    for pattern in patterns:
                        # Do simple pattern matching
                        pattern_base = pattern.rstrip('*')
                        if key.startswith(pattern_base):
                            all_keys.append(key)
                            pattern_matches += 1
                            break
            
            # Limit the number of keys to process to avoid overload
            if len(all_keys) >= max_keys_to_process:
                print_colored(f"Warning: Limiting to {max_keys_to_process} keys to avoid overload", YELLOW)
                break
        
        task_keys = all_keys
    except redis.RedisError as e:
        print_colored(f"Error scanning Redis keys: {str(e)}", RED)
        # Fallback to keys command with small patterns
        for pattern in ["celery-task-meta-*", "task:*"]:
            try:
                pattern_keys = redis_client.keys(pattern)
                pattern_keys = [k.decode('utf-8') for k in pattern_keys]
                print(f"Found {len(pattern_keys)} keys matching {pattern}")
                task_keys.extend(pattern_keys[:2500])  # Limit each pattern
            except redis.RedisError as e:
                print_colored(f"Error retrieving keys with pattern {pattern}: {str(e)}", RED)
    
    print(f"Found {len(task_keys)} task-related keys in Redis")
    
    # Add a progress indicator for processing
    print("Processing task data...")
    progress_interval = max(1, len(task_keys) // 20)  # Show progress at 5% intervals
    
    # Extract task information
    processed = 0
    for i, key in enumerate(task_keys):
        # Show progress indicator
        if i % progress_interval == 0:
            progress = (i / len(task_keys)) * 100
            print(f"Processing... {progress:.1f}% ({i}/{len(task_keys)})", end='\r')
        
        try:
            # Use safer get with timeout
            task_data_bytes = safe_redis_get(redis_client, key)
            if not task_data_bytes:
                continue
                
            # Parse task data safely
            task_data = safe_json_loads(task_data_bytes)
            if not task_data:
                continue
                
            # Check if task is active based on status
            status = task_data.get('status')
            if status and (include_all or status in ['PENDING', 'STARTED', 'RECEIVED', 'RETRY']):
                # Extract important task details with validation
                task_id = task_data.get('task_id')
                if not task_id and '-' in key:
                    # Try to extract from key
                    task_id = key.split('-')[-1]
                
                # Basic validations to avoid corrupt data
                if not isinstance(task_id, str):
                    task_id = str(task_id) if task_id else 'unknown-' + key
                
                username = task_data.get('username', 'unknown')
                if not isinstance(username, str):
                    username = str(username) if username else 'unknown'
                
                site_no = task_data.get('site_no', 'unknown')
                if not isinstance(site_no, str):
                    site_no = str(site_no) if site_no is not None else 'unknown'
                
                created_at = task_data.get('created_at', 'unknown')
                if not isinstance(created_at, str):
                    created_at = str(created_at) if created_at else 'unknown'
                
                # Format time string
                if isinstance(created_at, str):
                    try:
                        created_dt = datetime.fromisoformat(created_at)
                        created_at = created_dt.strftime('%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        pass
                
                # Check if task is stale
                is_stale = False
                if isinstance(created_at, str) and created_at != 'unknown':
                    try:
                        created_dt = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
                        # If task created more than 1 day ago and still active, mark as stale
                        if datetime.now() - created_dt > timedelta(days=1):
                            is_stale = True
                    except ValueError:
                        pass
                
                # Add valid site number flag
                invalid_site = False
                if isinstance(site_no, str) and not is_valid_site_number(site_no):
                    invalid_site = True
                
                # Store task info
                active_tasks.append({
                    'task_id': task_id,
                    'username': username,
                    'site_no': site_no,
                    'status': status,
                    'created_at': created_at,
                    'redis_key': key,
                    'is_stale': is_stale,
                    'invalid_site': invalid_site
                })
                
                processed += 1
                
        except Exception as e:
            print_colored(f"Error processing key {key}: {str(e)}", YELLOW)
    
    # Clear progress indicator
    print(" " * 80, end='\r')
    print(f"Successfully processed {processed} tasks out of {len(task_keys)} keys")
    
    # Display tasks
    if active_tasks:
        # Sort by creation time (newest first)
        sorted_tasks = sorted(
            active_tasks, 
            key=lambda x: x.get('created_at', '0000-00-00') if x.get('created_at', 'unknown') != 'unknown' else '0000-00-00',
            reverse=True
        )
        
        # Create display table
        task_table = []
        for i, task in enumerate(sorted_tasks[:MAX_TASKS_TO_SHOW]):
            # Format status with color
            if USE_COLORS:
                if task['is_stale']:
                    status_display = f"{RED}{task['status']} (STALE){ENDC}"
                else:
                    status_display = f"{YELLOW}{task['status']}{ENDC}"
            else:
                status_display = f"{task['status']}" + (" (STALE)" if task['is_stale'] else "")
            
            # Format site number with color if invalid
            site_no = task['site_no']
            if USE_COLORS and task['invalid_site']:
                site_display = f"{RED}{site_no}{ENDC}"
            else:
                site_display = site_no
                
            task_table.append([
                i+1,
                task['username'][:15],
                site_display,
                status_display,
                task['created_at'],
                task['task_id'][:10] + '...' if len(task['task_id']) > 10 else task['task_id']
            ])
        
        print(tabulate(
            task_table,
            headers=["#", "Username", "Site No", "Status", "Created", "Task ID"],
            tablefmt="pretty"
        ))
        
        if len(sorted_tasks) > MAX_TASKS_TO_SHOW:
            print(f"(Showing {MAX_TASKS_TO_SHOW} of {len(sorted_tasks)} tasks)")
            
        # Return statistics
        stale_count = sum(1 for t in active_tasks if t['is_stale'])
        invalid_site_count = sum(1 for t in active_tasks if t['invalid_site'])
        return {
            'total': len(active_tasks),
            'stale': stale_count,
            'invalid_site': invalid_site_count,
            'tasks': sorted_tasks
        }
    else:
        print_colored("No active tasks found in Redis", YELLOW)
        return {'total': 0, 'stale': 0, 'invalid_site': 0, 'tasks': []}

def clear_stuck_tasks(redis_client, criteria, dry_run=True):
    """
    Clear stuck tasks that match specific criteria with improved error handling
    """
    print_colored("\nCLEARING STUCK TASKS", HEADER, bold=True)
    
    # First get all tasks
    task_info = list_active_tasks(redis_client, include_all=False)
    if task_info['total'] == 0:
        print_colored("No active tasks to clear", YELLOW)
        return
    
    # Filter tasks based on criteria
    tasks_to_clear = []
    
    if criteria.get('all_active', False):
        tasks_to_clear = task_info['tasks']
        reason = "all active tasks"
    else:
        for task in task_info['tasks']:
            should_clear = False
            
            if criteria.get('stale', False) and task['is_stale']:
                should_clear = True
                reason = "stale tasks"
                
            if criteria.get('invalid_site', False) and task['invalid_site']:
                should_clear = True
                reason = f"tasks with invalid site numbers"
                
            if should_clear:
                tasks_to_clear.append(task)
    
    # Report what will be cleared
    if not tasks_to_clear:
        print_colored("No tasks match the clearing criteria", YELLOW)
        return
        
    print_colored(f"Found {len(tasks_to_clear)} tasks to clear ({reason})", YELLOW, bold=True)
    
    # In dry run mode, just show affected tasks
    if dry_run:
        print_colored("\nDRY RUN - No changes will be made", BLUE, bold=True)
        
        # Show sample of tasks that would be cleared
        task_table = []
        for i, task in enumerate(tasks_to_clear[:10]):
            task_table.append([
                i+1,
                task['username'][:15],
                task['site_no'],
                task['status'],
                task['created_at'],
                task['redis_key']
            ])
            
        print(tabulate(
            task_table,
            headers=["#", "Username", "Site No", "Status", "Created", "Redis Key"],
            tablefmt="pretty"
        ))
        
        if len(tasks_to_clear) > 10:
            print(f"(Showing 10 of {len(tasks_to_clear)} tasks that would be cleared)")
    else:
        print_colored("\nCLEARING TASKS - THIS CANNOT BE UNDONE", RED, bold=True)
        
        # Actually delete the tasks
        success_count = 0
        error_count = 0
        
        for task in tasks_to_clear:
            try:
                redis_key = task['redis_key']
                redis_client.delete(redis_key)
                success_count += 1
                
                # Also try to clean up related keys that might exist
                task_id = task['task_id']
                for related_key_pattern in [f"celery-task-meta-{task_id}", f"task:{task_id}"]:
                    try:
                        redis_client.delete(related_key_pattern)
                    except:
                        pass
            except Exception as e:
                print_colored(f"Error clearing task {task['task_id']}: {str(e)}", RED)
                error_count += 1
        
        print_colored(f"Successfully cleared {success_count} tasks", GREEN, bold=True)
        if error_count > 0:
            print_colored(f"Failed to clear {error_count} tasks", RED)
            
    # Offer advice for next steps
    print_colored("\nRECOMMENDED NEXT STEPS:", BLUE, bold=True)
    print("1. Restart Celery workers to ensure clean state")
    print("   sudo systemctl restart celery-worker.service")
    print("2. Verify queue status")
    print("   python /data/SWATGenXApp/codes/scripts/monitor_celery_status.py --queues")
    print("3. Check if workers are now processing tasks correctly")
    print("   python /data/SWATGenXApp/codes/scripts/monitor_celery_status.py --workers")

def repair_task_log():
    """Clean up the task log file to remove invalid/corrupt entries"""
    print_colored("\nREPAIRING TASK LOG FILE", HEADER, bold=True)
    
    log_file = '/data/SWATGenXApp/codes/web_application/logs/model_tasks.log'
    if not os.path.exists(log_file):
        print_colored(f"Task log file not found: {log_file}", RED)
        return False
    
    # Create backup of original file
    backup_file = f"{log_file}.backup_{int(time.time())}"
    try:
        import shutil
        shutil.copy2(log_file, backup_file)
        print_colored(f"Created backup of task log at: {backup_file}", GREEN)
    except Exception as e:
        print_colored(f"Error creating backup: {str(e)}", RED)
        return False
    
    # Read and filter the log entries
    valid_entries = []
    invalid_entries = []
    
    try:
        with open(log_file, 'r', errors='replace') as f:
            lines = f.readlines()
        
        for line in lines:
            if " - {" not in line:
                valid_entries.append(line)  # Not a JSON entry
                continue
            
            try:
                # Parse log entry
                timestamp, task_json = line.split(" - ", 1)
                task_data = json.loads(task_json)
                
                # Validate key fields
                if not isinstance(task_data.get('task_id'), str):
                    invalid_entries.append(line)
                    continue
                
                # Make sure site_no is a string
                if 'site_no' in task_data and not isinstance(task_data['site_no'], str):
                    task_data['site_no'] = str(task_data['site_no'])
                
                # Regenerate valid entry
                valid_entries.append(f"{timestamp} - {json.dumps(task_data)}\n")
                
            except Exception:
                invalid_entries.append(line)
        
        # Write cleaned file
        with open(log_file, 'w') as f:
            f.writelines(valid_entries)
        
        # Save invalid entries for inspection
        if invalid_entries:
            invalid_file = f"{log_file}.invalid_{int(time.time())}"
            with open(invalid_file, 'w') as f:
                f.writelines(invalid_entries)
            print_colored(f"Removed {len(invalid_entries)} invalid entries from log file", YELLOW)
            print_colored(f"Invalid entries saved to: {invalid_file}", YELLOW)
        
        print_colored(f"Task log repaired successfully. Kept {len(valid_entries)} valid entries.", GREEN)
        return True
    
    except Exception as e:
        print_colored(f"Error repairing task log: {str(e)}", RED)
        return False

def emergency_reset():
    """Perform an emergency reset to clear stuck tasks and reset Celery"""
    print_colored("\nEMERGENCY RESET", HEADER, bold=True)
    print_colored("This will remove ALL task-related data from Redis and reset the system", RED, bold=True)
    
    confirmation = input("Type 'RESET' to confirm emergency reset: ")
    if confirmation != "RESET":
        print_colored("Emergency reset cancelled", BLUE)
        return
    
    try:
        # Connect to Redis
        redis_client = connect_to_redis()
        if not redis_client:
            print_colored("Failed to connect to Redis", RED)
            return
        
        # 1. Backup all data first
        print_colored("1. Creating backups...", BLUE)
        
        # Backup task log
        log_file = '/data/SWATGenXApp/codes/web_application/logs/model_tasks.log'
        if os.path.exists(log_file):
            backup_log = f"{log_file}.emergency_backup_{int(time.time())}"
            try:
                import shutil
                shutil.copy2(log_file, backup_log)
                print_colored(f"  - Task log backed up to: {backup_log}", GREEN)
            except Exception as e:
                print_colored(f"  - Error backing up task log: {str(e)}", RED)
        
        # Backup Redis task keys
        task_keys = []
        for pattern in ["celery-task-meta-*", "task:*"]:
            try:
                keys = redis_client.keys(pattern)
                task_keys.extend([k.decode('utf-8') for k in keys])
            except Exception as e:
                print_colored(f"  - Error getting Redis keys for {pattern}: {str(e)}", RED)
        
        backup_data = {}
        for key in task_keys:
            try:
                value = redis_client.get(key)
                if value:
                    backup_data[key] = value.decode('utf-8', errors='replace')
            except Exception:
                pass
        
        redis_backup = f"/data/SWATGenXApp/codes/web_application/logs/redis_tasks_backup_{int(time.time())}.json"
        try:
            with open(redis_backup, 'w') as f:
                json.dump(backup_data, f, indent=2)
            print_colored(f"  - Redis tasks backed up to: {redis_backup}", GREEN)
        except Exception as e:
            print_colored(f"  - Error backing up Redis data: {str(e)}", RED)
        
        # 2. Clear all Redis task data
        print_colored("2. Removing task data from Redis...", BLUE)
        removed_count = 0
        for key in task_keys:
            try:
                redis_client.delete(key)
                removed_count += 1
            except Exception:
                pass
        print_colored(f"  - Removed {removed_count} keys from Redis", GREEN)
        
        # 3. Clear queues
        print_colored("3. Clearing Celery queues...", BLUE)
        for queue in ['celery', 'model_creation']:
            try:
                queue_len = redis_client.llen(queue)
                redis_client.delete(queue)
                print_colored(f"  - Cleared queue '{queue}' ({queue_len} tasks)", GREEN)
            except Exception as e:
                print_colored(f"  - Error clearing queue {queue}: {str(e)}", RED)
        
        # 4. Repair task log
        print_colored("4. Repairing task log...", BLUE)
        repair_task_log()
        
        # 5. Suggest next steps
        print_colored("\nEMERGENCY RESET COMPLETED", GREEN, bold=True)
        print_colored("\nRECOMMENDED NEXT STEPS:", BLUE, bold=True)
        print("1. Restart Celery workers:")
        print("   sudo systemctl restart celery-worker.service")
        print("2. Restart the Flask application:")
        print("   sudo systemctl restart flask-app.service")
        print("3. Monitor the system:")
        print("   python /data/SWATGenXApp/codes/scripts/monitor_celery_status.py")
        
    except Exception as e:
        print_colored(f"Error during emergency reset: {str(e)}", RED)

def main():
    parser = argparse.ArgumentParser(description='SWATGenX Task Repair Utility')
    parser.add_argument('--list', '-l', action='store_true', help='List active tasks in Redis')
    parser.add_argument('--list-all', '-a', action='store_true', help='List all tasks in Redis (not just active)')
    parser.add_argument('--max-keys', type=int, default=5000, help='Maximum number of Redis keys to process')
    parser.add_argument('--clear-stale', action='store_true', help='Clear stale tasks (older than 1 day)')
    parser.add_argument('--clear-invalid-site', action='store_true', help='Clear tasks with invalid site numbers')
    parser.add_argument('--clear-all-active', action='store_true', help='Clear all active tasks')
    parser.add_argument('--dry-run', '-d', action='store_true', help='Show what would be cleared without making changes')
    parser.add_argument('--repair-log', action='store_true', help='Repair the task log file by removing invalid entries')
    parser.add_argument('--emergency-reset', action='store_true', help='Perform emergency reset of all task data (use with extreme caution)')
    parser.add_argument('--clean', action='store_true', help='Disable colors in output')
    args = parser.parse_args()
    
    # Configure color output
    global USE_COLORS
    USE_COLORS = not args.clean
    
    # Handle emergency reset separately
    if args.emergency_reset:
        emergency_reset()
        return
    
    # Handle log repair separately
    if args.repair_log:
        repair_task_log()
        return
    
    # Connect to Redis
    redis_client = connect_to_redis()
    if not redis_client:
        sys.exit(1)
    
    # Default action if none specified
    if not any([args.list, args.list_all, args.clear_stale, args.clear_invalid_site, args.clear_all_active]):
        args.list = True
    
    # Perform requested operations
    if args.list or args.list_all:
        list_active_tasks(redis_client, include_all=args.list_all, max_keys_to_process=args.max_keys)
    
    # Handle clearing operations
    if args.clear_stale or args.clear_invalid_site or args.clear_all_active:
        criteria = {
            'stale': args.clear_stale,
            'invalid_site': args.clear_invalid_site,
            'all_active': args.clear_all_active
        }
        clear_stuck_tasks(redis_client, criteria, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
