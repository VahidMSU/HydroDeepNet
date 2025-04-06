#!/usr/bin/env python3
"""
Redis and Celery Complete Cleanup Utility

This script provides comprehensive cleanup for Redis and Celery data:
- Clear specific or all Celery queues
- Remove task metadata from Redis
- Clean up task history
- Reset stuck tasks
- Perform emergency system reset

Use with caution in production environments!
"""

import os
import sys
import json
import time
import argparse
import redis
import re
import datetime
import glob
import shutil
import subprocess
from collections import defaultdict

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

def ensure_backup_dir():
    """Ensure backup directory exists"""
    if not os.path.exists(BACKUP_DIR):
        try:
            os.makedirs(BACKUP_DIR)
            print_colored(f"Created backup directory: {BACKUP_DIR}", GREEN)
        except Exception as e:
            print_colored(f"Error creating backup directory: {str(e)}", RED)
            return False
    return True

def backup_redis_keys(client, pattern="*", max_keys=MAX_KEYS_TO_PROCESS):
    """Backup Redis keys matching pattern to a JSON file"""
    if not ensure_backup_dir():
        return None
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(BACKUP_DIR, f"redis_backup_{timestamp}.json")
    
    try:
        print_colored(f"Scanning Redis for keys matching '{pattern}'...", BLUE)
        
        # Use scan_iter for memory-efficient iteration
        keys = []
        scan_count = 0
        for key in client.scan_iter(match=pattern, count=1000):
            keys.append(key)
            scan_count += 1
            
            # Safety limit
            if len(keys) >= max_keys:
                print_colored(f"Reached maximum key limit ({max_keys}). Stopping scan.", YELLOW)
                break
                
            # Progress update
            if scan_count % 1000 == 0:
                print(f"Scanned {scan_count} keys, found {len(keys)} matching...")
        
        print_colored(f"Found {len(keys)} keys matching pattern '{pattern}'", GREEN)
        
        # Backup key values to JSON file
        backup_data = {}
        processed = 0
        
        for i, key in enumerate(keys):
            try:
                # Update progress periodically
                if i % 100 == 0:
                    print(f"Backing up keys: {i}/{len(keys)}...", end="\r")
                
                # Get different Redis data types appropriately
                key_type = client.type(key).decode('utf-8')
                
                if key_type == 'string':
                    val = client.get(key)
                    try:
                        # Try to decode JSON data
                        val_str = val.decode('utf-8')
                        try:
                            # Store as parsed JSON if possible
                            backup_data[key.decode('utf-8')] = {'type': key_type, 'value': json.loads(val_str)}
                        except:
                            # Store as string if not JSON
                            backup_data[key.decode('utf-8')] = {'type': key_type, 'value': val_str}
                    except:
                        # Store as base64 if binary
                        import base64
                        backup_data[key.decode('utf-8')] = {
                            'type': key_type, 
                            'value': base64.b64encode(val).decode('ascii'),
                            'encoding': 'base64'
                        }
                elif key_type == 'list':
                    vals = client.lrange(key, 0, -1)
                    backup_data[key.decode('utf-8')] = {
                        'type': key_type, 
                        'value': [v.decode('utf-8', errors='replace') for v in vals]
                    }
                elif key_type == 'set':
                    vals = client.smembers(key)
                    backup_data[key.decode('utf-8')] = {
                        'type': key_type, 
                        'value': [v.decode('utf-8', errors='replace') for v in vals]
                    }
                elif key_type == 'hash':
                    val_dict = client.hgetall(key)
                    backup_data[key.decode('utf-8')] = {
                        'type': key_type, 
                        'value': {k.decode('utf-8', errors='replace'): v.decode('utf-8', errors='replace') 
                                 for k, v in val_dict.items()}
                    }
                elif key_type == 'zset':
                    vals = client.zrange(key, 0, -1, withscores=True)
                    backup_data[key.decode('utf-8')] = {
                        'type': key_type, 
                        'value': {v[0].decode('utf-8', errors='replace'): v[1] for v in vals}
                    }
                else:
                    backup_data[key.decode('utf-8')] = {'type': key_type, 'value': 'unsupported-type'}
                
                processed += 1
                
            except Exception as e:
                print_colored(f"\nError backing up key {key}: {str(e)}", RED)
        
        # Write backup to file
        with open(backup_file, 'w') as f:
            json.dump(backup_data, f, indent=2)
        
        print_colored(f"\nSuccessfully backed up {processed} keys to {backup_file}", GREEN)
        return backup_file
        
    except Exception as e:
        print_colored(f"Error backing up Redis keys: {str(e)}", RED)
        return None

def backup_log_file(log_file):
    """Create a backup of a log file"""
    if not ensure_backup_dir() or not os.path.exists(log_file):
        return None
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(BACKUP_DIR, f"{os.path.basename(log_file)}.{timestamp}")
    
    try:
        shutil.copy2(log_file, backup_file)
        print_colored(f"Created backup of {log_file} at {backup_file}", GREEN)
        return backup_file
    except Exception as e:
        print_colored(f"Error backing up log file: {str(e)}", RED)
        return None

def clear_redis_keys(client, pattern, dry_run=True, backup=True):
    """Clear Redis keys matching pattern"""
    print_colored(f"\nCLEARING REDIS KEYS: {pattern}", HEADER, bold=True)
    
    if backup:
        backup_file = backup_redis_keys(client, pattern)
        if not backup_file and not dry_run:
            print_colored("Backup failed. Add --skip-backup to proceed without backup.", RED)
            return False
    
    try:
        # Scan for keys matching pattern
        keys = []
        for key in client.scan_iter(match=pattern, count=1000):
            keys.append(key)
            if len(keys) >= MAX_KEYS_TO_PROCESS:
                print_colored(f"Reached maximum key limit ({MAX_KEYS_TO_PROCESS}). Stopping scan.", YELLOW)
                break
        
        if not keys:
            print_colored(f"No keys found matching pattern '{pattern}'", YELLOW)
            return True
            
        print_colored(f"Found {len(keys)} keys matching pattern '{pattern}'", BLUE)
        
        # Sample of keys for verification
        if len(keys) > 0:
            print_colored("Sample keys:", BLUE)
            for key in keys[:5]:
                print(f"  - {key.decode('utf-8')}")
            if len(keys) > 5:
                print(f"  - ... and {len(keys) - 5} more")
        
        if dry_run:
            print_colored(f"DRY RUN: Would delete {len(keys)} keys", YELLOW, bold=True)
            return True
        
        # Delete keys in batches using pipeline for efficiency
        batch_size = 1000
        deleted = 0
        
        for i in range(0, len(keys), batch_size):
            batch = keys[i:i+batch_size]
            if batch:
                pipe = client.pipeline()
                for key in batch:
                    pipe.delete(key)
                result = pipe.execute()
                deleted += sum(result)
                print(f"Deleted {deleted}/{len(keys)} keys...", end="\r")
        
        print_colored(f"\nSuccessfully deleted {deleted} keys matching '{pattern}'", GREEN)
        return True
        
    except Exception as e:
        print_colored(f"Error clearing Redis keys: {str(e)}", RED)
        return False

def clear_celery_queue(client, queue_name, dry_run=True, backup=True):
    """Clear a specific Celery queue"""
    print_colored(f"\nCLEARING CELERY QUEUE: {queue_name}", HEADER, bold=True)
    
    try:
        # Check if queue exists
        queue_length = client.llen(queue_name)
        
        if queue_length == 0:
            print_colored(f"Queue '{queue_name}' is empty", YELLOW)
            return True
            
        print_colored(f"Queue '{queue_name}' contains {queue_length} tasks", BLUE)
        
        if backup:
            # Backup queue contents
            backup_file = backup_redis_keys(client, queue_name)
            if not backup_file and not dry_run:
                print_colored("Backup failed. Add --skip-backup to proceed without backup.", RED)
                return False
        
        if dry_run:
            print_colored(f"DRY RUN: Would delete all {queue_length} tasks from queue '{queue_name}'", YELLOW, bold=True)
            return True
        
        # Delete the queue
        client.delete(queue_name)
        print_colored(f"Successfully cleared queue '{queue_name}'", GREEN)
        
        # Verify queue is now empty
        new_length = client.llen(queue_name)
        if new_length == 0:
            print_colored(f"Verified queue '{queue_name}' is now empty", GREEN)
        else:
            print_colored(f"Warning: Queue '{queue_name}' still contains {new_length} tasks", YELLOW)
            
        return True
        
    except Exception as e:
        print_colored(f"Error clearing Celery queue: {str(e)}", RED)
        return False

def clear_celery_task_metadata(client, dry_run=True, backup=True):
    """Clear Celery task metadata from Redis"""
    print_colored("\nCLEARING CELERY TASK METADATA", HEADER, bold=True)
    
    try:
        # Backup task metadata if requested
        if backup:
            backup_file = backup_redis_keys(client, "celery-task-meta-*")
            if not backup_file and not dry_run:
                print_colored("Backup failed. Add --skip-backup to proceed without backup.", RED)
                return False
        
        # Clear task metadata
        patterns = ["celery-task-meta-*", "task:*"]
        success = True
        
        for pattern in patterns:
            if not clear_redis_keys(client, pattern, dry_run, backup=False):  # Already backed up
                success = False
        
        return success
        
    except Exception as e:
        print_colored(f"Error clearing Celery task metadata: {str(e)}", RED)
        return False

def clear_kombu_bindings(client, dry_run=True, backup=True):
    """Clear Celery Kombu bindings and queues"""
    print_colored("\nCLEARING CELERY KOMBU BINDINGS", HEADER, bold=True)
    
    try:
        # Backup Kombu data if requested
        if backup:
            backup_file = backup_redis_keys(client, "_kombu.*")
            if not backup_file and not dry_run:
                print_colored("Backup failed. Add --skip-backup to proceed without backup.", RED)
                return False
                
        patterns = ["_kombu.*"]
        success = True
        
        for pattern in patterns:
            if not clear_redis_keys(client, pattern, dry_run, backup=False):  # Already backed up
                success = False
                
        return success
        
    except Exception as e:
        print_colored(f"Error clearing Kombu bindings: {str(e)}", RED)
        return False

def clear_task_history_log(dry_run=True, backup=True):
    """Clean up and reset the task history log file"""
    print_colored("\nCLEARING TASK HISTORY LOG", HEADER, bold=True)
    
    if not os.path.exists(TASK_LOG_FILE):
        print_colored(f"Task log file not found: {TASK_LOG_FILE}", YELLOW)
        return True
    
    try:
        # Get file size for reporting
        file_size = os.path.getsize(TASK_LOG_FILE)
        file_size_mb = file_size / (1024 * 1024)
        
        print_colored(f"Current task log size: {file_size_mb:.2f} MB", BLUE)
        
        # Backup the log file
        if backup:
            backup_file = backup_log_file(TASK_LOG_FILE)
            if not backup_file and not dry_run:
                print_colored("Backup failed. Add --skip-backup to proceed without backup.", RED)
                return False
                
        if dry_run:
            print_colored(f"DRY RUN: Would clear task history log (saving {file_size_mb:.2f} MB)", YELLOW, bold=True)
            return True
            
        # Create empty log file (or truncate existing)
        with open(TASK_LOG_FILE, 'w') as f:
            f.write("")
            
        print_colored(f"Successfully cleared task history log, freeing {file_size_mb:.2f} MB", GREEN)
        return True
        
    except Exception as e:
        print_colored(f"Error clearing task history log: {str(e)}", RED)
        return False

def restart_services(service_list, dry_run=True):
    """Restart systemd services"""
    print_colored("\nRESTARTING SERVICES", HEADER, bold=True)
    
    if dry_run:
        print_colored(f"DRY RUN: Would restart services: {', '.join(service_list)}", YELLOW, bold=True)
        return True
    
    for service in service_list:
        try:
            print_colored(f"Restarting {service}...", BLUE)
            result = subprocess.run(['sudo', 'systemctl', 'restart', service], 
                                   check=True, 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE)
            print_colored(f"Successfully restarted {service}", GREEN)
        except subprocess.CalledProcessError as e:
            print_colored(f"Error restarting {service}: {e.stderr.decode('utf-8')}", RED)
        except Exception as e:
            print_colored(f"Error restarting {service}: {str(e)}", RED)
    
    # Check status of restarted services
    for service in service_list:
        try:
            result = subprocess.run(['systemctl', 'is-active', service],
                                   check=False,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
            if result.stdout.decode('utf-8').strip() == 'active':
                print_colored(f"{service} is active", GREEN)
            else:
                print_colored(f"{service} is not active", RED)
        except Exception as e:
            print_colored(f"Error checking status of {service}: {str(e)}", RED)
            
    return True

def perform_emergency_reset(client, dry_run=True):
    """Perform emergency reset of Redis and Celery"""
    print_colored("\nEMERGENCY RESET", HEADER, bold=True)
    print_colored("This will completely reset the Celery task system", RED, bold=True)
    
    if not dry_run:
        confirm = input("Type 'RESET' to confirm emergency reset: ")
        if confirm != "RESET":
            print_colored("Emergency reset cancelled", BLUE)
            return False
    
    steps = [
        # First backup everything
        (backup_redis_keys, [client, "*"], "Backing up all Redis data"),
        (backup_log_file, [TASK_LOG_FILE], "Backing up task log"),
        
        # Clear Redis keys in specific order
        (clear_celery_queue, [client, "celery", dry_run, False], "Clearing default Celery queue"),
        (clear_celery_queue, [client, "model_creation", dry_run, False], "Clearing model_creation queue"),
        (clear_celery_task_metadata, [client, dry_run, False], "Clearing Celery task metadata"),
        (clear_kombu_bindings, [client, dry_run, False], "Clearing Kombu bindings"),
        
        # Clear task history log
        (clear_task_history_log, [dry_run, False], "Clearing task history log"),
        
        # Restart services
        (restart_services, [["celery-worker.service", "flask-app.service"], dry_run], "Restarting services")
    ]
    
    success_count = 0
    for func, args, description in steps:
        try:
            print_colored(f"\n{description}...", BLUE)
            result = func(*args)
            if result:
                success_count += 1
            else:
                print_colored(f"Step failed: {description}", RED)
        except Exception as e:
            print_colored(f"Error in step '{description}': {str(e)}", RED)
    
    # Report overall status
    if success_count == len(steps):
        print_colored("\nEmergency reset completed successfully!", GREEN, bold=True)
    else:
        print_colored(f"\nEmergency reset completed with {len(steps) - success_count} errors", YELLOW, bold=True)
    
    print_colored("\nNext Steps:", BLUE, bold=True)
    print("1. Verify services are running properly:")
    print("   systemctl status celery-worker.service")
    print("   systemctl status flask-app.service")
    print("2. Monitor system status:")
    print("   python /data/SWATGenXApp/codes/scripts/monitor_celery_status.py")
    
    return success_count == len(steps)

def clear_old_backups(max_age_days=30, dry_run=True):
    """Clean up old backup files to save disk space"""
    print_colored("\nCLEARING OLD BACKUPS", HEADER, bold=True)
    
    if not os.path.exists(BACKUP_DIR):
        print_colored(f"Backup directory not found: {BACKUP_DIR}", YELLOW)
        return True
    
    # Find backup files older than max_age_days
    cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
    old_backups = []
    
    for file_path in glob.glob(os.path.join(BACKUP_DIR, "*")):
        if os.path.isfile(file_path):
            file_time = os.path.getmtime(file_path)
            if file_time < cutoff_time:
                file_age_days = (time.time() - file_time) / (24 * 60 * 60)
                old_backups.append((file_path, file_age_days))
    
    if not old_backups:
        print_colored(f"No backups older than {max_age_days} days found", GREEN)
        return True
    
    total_size = sum(os.path.getsize(file_path) for file_path, _ in old_backups)
    total_size_mb = total_size / (1024 * 1024)
    
    print_colored(f"Found {len(old_backups)} backup files older than {max_age_days} days " + 
                 f"(total size: {total_size_mb:.2f} MB)", BLUE)
    
    if dry_run:
        print_colored(f"DRY RUN: Would delete {len(old_backups)} old backup files", YELLOW, bold=True)
        return True
    
    # Delete old backups
    deleted_count = 0
    for file_path, age_days in old_backups:
        try:
            os.remove(file_path)
            deleted_count += 1
        except Exception as e:
            print_colored(f"Error deleting backup {file_path}: {str(e)}", RED)
    
    print_colored(f"Successfully deleted {deleted_count}/{len(old_backups)} old backup files, " + 
                 f"freeing {total_size_mb:.2f} MB", GREEN)
    return True

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Redis and Celery Cleanup Utility')
    
    # Main action groups
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument('--emergency-reset', action='store_true', 
                             help='Perform emergency reset of Redis and Celery')
    action_group.add_argument('--clear-queues', action='store_true',
                             help='Clear specific or all Celery queues')
    action_group.add_argument('--clear-task-metadata', action='store_true',
                             help='Clear Celery task metadata from Redis')
    action_group.add_argument('--clear-kombu', action='store_true',
                             help='Clear Celery Kombu bindings and exchanges')
    action_group.add_argument('--clear-task-log', action='store_true',
                             help='Clear task history log file')
    action_group.add_argument('--clear-redis-keys', metavar='PATTERN',
                             help='Clear Redis keys matching pattern')
    action_group.add_argument('--backup-redis', metavar='PATTERN',
                             help='Backup Redis keys matching pattern')
    action_group.add_argument('--clear-old-backups', action='store_true',
                             help='Clean up old backup files')
    
    # Target options
    parser.add_argument('--queue', metavar='QUEUE', 
                       help='Target specific queue for --clear-queues')
    
    # Common options
    parser.add_argument('--dry-run', '-d', action='store_true', 
                       help='Show what would be done without making changes')
    parser.add_argument('--skip-backup', action='store_true',
                       help='Skip creating backups before clearing data')
    parser.add_argument('--max-age-days', type=int, default=30,
                       help='Maximum age in days for --clear-old-backups')
    parser.add_argument('--yes', '-y', action='store_true',
                       help='Skip all confirmation prompts')
    parser.add_argument('--clean', action='store_true',
                       help='Disable colors in output')
    
    args = parser.parse_args()
    
    # Configure global settings
    global USE_COLORS
    USE_COLORS = not args.clean
    
    return args

def print_task_restart_info():
    """Print information about the task restart utility"""
    print_colored("\nNOTE: Task restart functionality has been moved to a dedicated script", BLUE, bold=True)
    print("To restart stalled tasks, use the restart_stalled_tasks.py script instead:")
    print("  python /data/SWATGenXApp/codes/scripts/restart_stalled_tasks.py --stale-hours 1")
    print("  python /data/SWATGenXApp/codes/scripts/restart_stalled_tasks.py --help")

def main():
    """Main function"""
    args = parse_args()
    
    # Connect to Redis
    client = connect_to_redis()
    if not client and not args.clear_task_log and not args.clear_old_backups:
        sys.exit(1)
    
    try:
        # Handle command options
        if args.emergency_reset:
            perform_emergency_reset(client, args.dry_run)
            
        elif args.clear_queues:
            if args.queue:
                clear_celery_queue(client, args.queue, args.dry_run, not args.skip_backup)
            else:
                # Clear all known queues
                queue_names = ['celery', 'model_creation']
                for queue in queue_names:
                    clear_celery_queue(client, queue, args.dry_run, not args.skip_backup)
                    
        elif args.clear_task_metadata:
            clear_celery_task_metadata(client, args.dry_run, not args.skip_backup)
            
        elif args.clear_kombu:
            clear_kombu_bindings(client, args.dry_run, not args.skip_backup)
            
        elif args.clear_task_log:
            clear_task_history_log(args.dry_run, not args.skip_backup)
            
        elif args.clear_redis_keys:
            clear_redis_keys(client, args.clear_redis_keys, args.dry_run, not args.skip_backup)
            
        elif args.backup_redis:
            backup_redis_keys(client, args.backup_redis)
            
        elif args.clear_old_backups:
            clear_old_backups(args.max_age_days, args.dry_run)
        
        # Inform about task restart functionality
        print_task_restart_info()
            
    except KeyboardInterrupt:
        print_colored("\nOperation cancelled by user", YELLOW)
        sys.exit(1)
    except Exception as e:
        print_colored(f"\nUnexpected error: {str(e)}", RED)
        sys.exit(1)

if __name__ == "__main__":
    main()
