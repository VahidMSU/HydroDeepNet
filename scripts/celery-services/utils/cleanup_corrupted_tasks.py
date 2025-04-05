#!/usr/bin/env python3
"""
Redis Queue Corruption Cleanup Tool for SWATGenX

This script finds and removes corrupted Celery task messages in Redis queues
that are causing worker crashes with 'KeyError: properties'.
"""

import sys
import os
import json
import redis
import argparse
import time
from datetime import datetime

# Add application path for imports
sys.path.insert(0, '/data/SWATGenXApp/codes')
sys.path.insert(0, '/data/SWATGenXApp/codes/web_application')

# Configure output formats
HEADER = '\033[95m'
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'

# Global variables for configuration
USE_COLORS = True
BACKUP_DIR = "/data/SWATGenXApp/codes/web_application/logs/queue_backups"
LOG_FILE = "/data/SWATGenXApp/codes/web_application/logs/corrupted_tasks.log"

def print_colored(text, color=None, bold=False):
    """Print text with optional color and emphasis"""
    if USE_COLORS and color:
        prefix = BOLD + color if bold else color
        print(f"{prefix}{text}{ENDC}")
    else:
        print(text)

def log_message(message, level="INFO"):
    """Log a message to the log file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"{timestamp} - {level} - {message}\n")

def connect_to_redis(host='localhost', port=6379, db=0):
    """Connect to Redis with error handling"""
    try:
        client = redis.Redis(host=host, port=port, db=db, 
                            socket_connect_timeout=5,
                            socket_timeout=5)
        # Test connection
        client.ping()
        print_colored("Connected to Redis successfully", GREEN)
        return client
    except redis.RedisError as e:
        print_colored(f"Error connecting to Redis: {e}", RED, bold=True)
        return None
    except Exception as e:
        print_colored(f"Unexpected error connecting to Redis: {e}", RED, bold=True)
        return None

def backup_queue(client, queue_name):
    """Create a backup of a queue before modifying it"""
    try:
        if not os.path.exists(BACKUP_DIR):
            os.makedirs(BACKUP_DIR)
            
        # Get all items in the queue
        queue_items = client.lrange(queue_name, 0, -1)
        
        if not queue_items:
            print_colored(f"Queue {queue_name} is empty, no backup needed", YELLOW)
            return True
            
        # Create a timestamped backup file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(BACKUP_DIR, f"{queue_name}_backup_{timestamp}.json")
        
        # Save items to the backup file
        with open(backup_file, 'w') as f:
            items_data = []
            for idx, item in enumerate(queue_items):
                try:
                    # Try to decode and parse as JSON for readability
                    decoded_item = item.decode('utf-8')
                    parsed_item = json.loads(decoded_item)
                    items_data.append(parsed_item)
                except:
                    # If parsing fails, store raw bytes as string
                    items_data.append(str(item))
            
            json.dump(items_data, f, indent=2)
        
        print_colored(f"Created backup of queue '{queue_name}' with {len(queue_items)} items at: {backup_file}", GREEN)
        log_message(f"Created backup of queue '{queue_name}' with {len(queue_items)} items at: {backup_file}")
        return True
        
    except Exception as e:
        print_colored(f"Error backing up queue {queue_name}: {e}", RED)
        log_message(f"Error backing up queue {queue_name}: {e}", "ERROR")
        return False

def verify_task_message(message_bytes):
    """
    Verify if a message is a valid Celery task by checking for required fields.
    Returns (is_valid, parsed_message, error_details)
    """
    try:
        # Try to decode the message
        decoded = message_bytes.decode('utf-8')
        parsed = json.loads(decoded)
        
        # Check for the required 'properties' field in a Celery task
        if 'properties' not in parsed:
            return False, parsed, "Missing 'properties' field"
            
        # Check for other essential Celery task fields
        required_fields = ['body', 'content-encoding', 'content-type', 'headers']
        missing_fields = [field for field in required_fields if field not in parsed]
        
        if missing_fields:
            return False, parsed, f"Missing fields: {', '.join(missing_fields)}"
            
        # Message appears to be valid
        return True, parsed, None
        
    except UnicodeDecodeError:
        return False, None, "Message is not valid UTF-8"
    except json.JSONDecodeError:
        return False, None, "Message is not valid JSON"
    except Exception as e:
        return False, None, f"Unexpected error: {str(e)}"

def scan_queue_for_corruption(client, queue_name, dry_run=True, repair=False):
    """
    Scan a queue for corrupted messages that are missing the 'properties' field
    If repair=True, attempt to reconstruct the missing properties
    """
    try:
        # Get the length of the queue
        queue_length = client.llen(queue_name)
        print_colored(f"Scanning queue '{queue_name}' with {queue_length} items", BLUE)
        
        if queue_length == 0:
            print_colored(f"Queue {queue_name} is empty, nothing to scan", YELLOW)
            return 0
            
        # Create a backup first
        if not dry_run:
            backup_queue(client, queue_name)
        
        # Examine each item in the queue
        corrupted_items = []
        repaired_items = []
        
        for idx in range(queue_length):
            item = client.lindex(queue_name, idx)
            is_valid, parsed, error = verify_task_message(item)
            
            if not is_valid:
                print_colored(f"Found corrupted message at position {idx}: {error}", RED)
                
                if repair and parsed is not None:
                    # Try to repair the message by adding the missing 'properties' field
                    if error == "Missing 'properties' field":
                        try:
                            # Add a minimal valid 'properties' field
                            parsed['properties'] = {
                                'correlation_id': '',
                                'reply_to': '',
                                'delivery_mode': 2,  # persistent
                                'delivery_info': {'exchange': '', 'routing_key': queue_name},
                                'priority': 0,
                                'body_encoding': 'base64',
                                'delivery_tag': str(int(time.time()))
                            }
                            
                            # If we can reconstruct the task, mark it for repair
                            repaired_items.append((idx, json.dumps(parsed).encode('utf-8')))
                            print_colored(f"  → Message at position {idx} can be repaired", GREEN)
                        except Exception as e:
                            print_colored(f"  → Cannot repair message: {e}", YELLOW)
                
                # Record the corrupted item for removal
                corrupted_items.append(idx)
        
        # Report results
        if corrupted_items:
            print_colored(f"Found {len(corrupted_items)} corrupted messages in queue '{queue_name}'", 
                         YELLOW if dry_run else RED, bold=True)
            
            if repair:
                print_colored(f"Of those, {len(repaired_items)} can be repaired", GREEN)
                
            if not dry_run:
                # Actually remove or repair the corrupted items
                if repair and repaired_items:
                    # Repair the messages
                    for idx, repaired_data in repaired_items:
                        # Replace the corrupted message with the repaired one
                        client.lset(queue_name, idx, repaired_data)
                        print_colored(f"Repaired message at position {idx}", GREEN)
                    
                    # Remove any that couldn't be repaired
                    for idx in sorted(set(corrupted_items) - set([x[0] for x in repaired_items]), reverse=True):
                        # Create a temporary value to identify the item (Redis doesn't support direct index removal)
                        temp_value = f"CORRUPTED_ITEM_{int(time.time())}_{idx}"
                        client.lset(queue_name, idx, temp_value)
                        # Remove all occurrences of this value
                        client.lrem(queue_name, 0, temp_value)
                        print_colored(f"Removed corrupted message at position {idx}", YELLOW)
                else:
                    # Just remove all corrupted messages
                    for idx in sorted(corrupted_items, reverse=True):
                        # Using the same technique - set then remove
                        temp_value = f"CORRUPTED_ITEM_{int(time.time())}_{idx}"
                        client.lset(queue_name, idx, temp_value)
                        client.lrem(queue_name, 0, temp_value)
                        print_colored(f"Removed corrupted message at position {idx}", YELLOW)
                
                # Log the cleanup action
                log_message(f"Cleaned up {len(corrupted_items)} corrupted messages from queue '{queue_name}'. " +
                           f"Repaired: {len(repaired_items)}, Removed: {len(corrupted_items) - len(repaired_items)}")
        else:
            print_colored(f"No corrupted messages found in queue '{queue_name}'", GREEN)
        
        return len(corrupted_items)
        
    except Exception as e:
        print_colored(f"Error scanning queue {queue_name}: {e}", RED, bold=True)
        log_message(f"Error scanning queue {queue_name}: {e}", "ERROR")
        return 0

def scan_redis_for_wrongtype_keys(client, pattern="celery-task-meta-*", dry_run=True):
    """
    Find and fix Redis keys with WRONGTYPE errors (typically corrupted task metadata)
    """
    try:
        print_colored("\nSCANNING FOR WRONGTYPE ERRORS IN REDIS KEYS", HEADER, bold=True)
        
        # Use Redis SCAN to find keys matching the pattern
        cursor = '0'
        found_keys = []
        wrongtype_keys = []
        
        while True:
            cursor, keys = client.scan(cursor=cursor, match=pattern, count=1000)
            
            if keys:
                found_keys.extend(keys)
                
                # Check each key for WRONGTYPE errors
                for key in keys:
                    key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                    try:
                        # Try to check the key type - this will succeed even if it's the wrong type
                        key_type = client.type(key).decode('utf-8')
                        
                        # For celery task metadata, the type should be 'string' (JSON string)
                        # If it's not, we have a problem
                        if key_str.startswith('celery-task-meta-') and key_type != 'string':
                            print_colored(f"Found WRONGTYPE key: {key_str} (type: {key_type})", YELLOW)
                            wrongtype_keys.append(key)
                    except redis.exceptions.ResponseError as e:
                        if "WRONGTYPE" in str(e):
                            print_colored(f"WRONGTYPE error on key: {key_str}", RED)
                            wrongtype_keys.append(key)
                        else:
                            # Other Redis errors
                            print_colored(f"Error checking key {key_str}: {e}", RED)
            
            # Exit the loop when we've processed all keys
            if cursor == '0':
                break
        
        # Report results
        if found_keys:
            print_colored(f"Scanned {len(found_keys)} keys matching pattern '{pattern}'", BLUE)
            
            if wrongtype_keys:
                print_colored(f"Found {len(wrongtype_keys)} keys with WRONGTYPE issues", 
                             YELLOW if dry_run else RED, bold=True)
                
                # Sample of problematic keys
                print_colored("Sample of problematic keys:", BLUE)
                for key in wrongtype_keys[:5]:
                    key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                    print(f"  - {key_str}")
                
                if len(wrongtype_keys) > 5:
                    print(f"  ... and {len(wrongtype_keys) - 5} more keys")
                    
                if not dry_run:
                    # Fix the problematic keys by deleting them (they're likely corrupted)
                    print_colored(f"\nFixing {len(wrongtype_keys)} WRONGTYPE keys...", GREEN, bold=True)
                    
                    deleted_count = 0
                    error_count = 0
                    
                    for key in wrongtype_keys:
                        try:
                            client.delete(key)
                            deleted_count += 1
                            
                            # Log progress every 100 keys
                            if deleted_count % 100 == 0:
                                print_colored(f"  Deleted {deleted_count} of {len(wrongtype_keys)} keys", GREEN)
                        except Exception as e:
                            error_count += 1
                            print_colored(f"Error deleting key: {e}", RED)
                    
                    # Final status
                    print_colored(f"\nSuccessfully deleted {deleted_count} keys with WRONGTYPE issues", GREEN, bold=True)
                    if error_count > 0:
                        print_colored(f"Failed to delete {error_count} keys", YELLOW)
                    
                    # Log the action
                    log_message(f"Deleted {deleted_count} keys with WRONGTYPE issues. " +
                               f"Errors: {error_count}. Pattern: {pattern}")
                else:
                    # In dry-run mode
                    print_colored(f"\nWould delete {len(wrongtype_keys)} keys with WRONGTYPE issues",
                                 BLUE, bold=True)
                    print_colored("Run without --dry-run to fix these issues", BLUE)
            else:
                print_colored("No keys with WRONGTYPE issues found", GREEN)
        else:
            print_colored(f"No keys found matching pattern '{pattern}'", YELLOW)
            
        return len(wrongtype_keys)
        
    except Exception as e:
        print_colored(f"Error scanning for WRONGTYPE keys: {e}", RED, bold=True)
        log_message(f"Error scanning for WRONGTYPE keys: {e}", "ERROR")
        return 0

def scan_all_celery_queues(client, dry_run=True, repair=False, queues=None):
    """Scan all Celery queues for corrupted messages"""
    try:
        print_colored("\nSCANNING CELERY QUEUES FOR CORRUPTION", HEADER, bold=True)
        
        # Get all the queues if not specified
        if not queues:
            # Default queues to check
            queues = ['celery', 'model_creation']
            
            # Look for additional queues - they're stored as keys with prefixes
            for key in client.keys('*'):
                key_str = key.decode('utf-8')
                # Celery also uses keys with celery in them
                if key_str not in queues and ':' not in key_str and '.' not in key_str:
                    queues.append(key_str)
        
        total_corrupted = 0
        for queue in queues:
            corrupted = scan_queue_for_corruption(client, queue, dry_run, repair)
            total_corrupted += corrupted
            
        if total_corrupted > 0:
            action = "would be " if dry_run else "were "
            print_colored(f"\nTotal of {total_corrupted} corrupted messages {action}found and {'repaired/removed' if repair else 'identified'}", 
                         YELLOW if dry_run else GREEN, bold=True)
        else:
            print_colored("\nNo corrupted messages found in any queue", GREEN)
        
        # Additionally scan for WRONGTYPE keys in Redis, which is a common source of worker crashes
        total_wrongtype = scan_redis_for_wrongtype_keys(client, pattern="celery-task-meta-*", dry_run=dry_run)
            
        return total_corrupted + total_wrongtype
            
    except Exception as e:
        print_colored(f"Error scanning Celery queues: {e}", RED, bold=True)
        log_message(f"Error scanning Celery queues: {e}", "ERROR")
        return 0

def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description='Fix corrupted Celery tasks in Redis queues')
    parser.add_argument('--dry-run', '-d', action='store_true', 
                      help='Identify corrupted messages without removing them')
    parser.add_argument('--repair', '-r', action='store_true',
                      help='Try to repair corrupted messages instead of just removing them')
    parser.add_argument('--queue', '-q', type=str, nargs='+',
                      help='Specific queue(s) to check (default: all Celery queues)')
    parser.add_argument('--redis-host', type=str, default='localhost',
                      help='Redis host (default: localhost)')
    parser.add_argument('--redis-port', type=int, default=6379,
                      help='Redis port (default: 6379)')
    parser.add_argument('--redis-db', type=int, default=0,
                      help='Redis database (default: 0)')
    parser.add_argument('--yes', '-y', action='store_true',
                      help='Skip confirmation prompts')
    parser.add_argument('--clean', action='store_true',
                      help='Disable colored output')
    parser.add_argument('--fix-wrongtype', action='store_true',
                      help='Scan and fix WRONGTYPE Redis keys only')
    parser.add_argument('--pattern', type=str, default='celery-task-meta-*',
                      help='Redis key pattern to scan for WRONGTYPE issues (default: celery-task-meta-*)')
    
    args = parser.parse_args()
    
    # Configure global settings
    global USE_COLORS
    USE_COLORS = not args.clean
    
    # Connect to Redis
    client = connect_to_redis(args.redis_host, args.redis_port, args.redis_db)
    if not client:
        sys.exit(1)
    
    # Handle WRONGTYPE-only mode
    if args.fix_wrongtype:
        total_wrongtype = scan_redis_for_wrongtype_keys(client, pattern=args.pattern, dry_run=args.dry_run)
        if total_wrongtype > 0 and not args.dry_run:
            print_colored("\nDone! You should restart Celery workers now.", GREEN, bold=True)
            print_colored("Run: sudo systemctl restart celery-worker.service", BLUE)
        return
    
    # In dry run mode, just report problems
    if args.dry_run:
        scan_all_celery_queues(client, dry_run=True, repair=args.repair, queues=args.queue)
        return
    
    # For actual modification, get confirmation unless --yes was specified
    if not args.yes:
        print_colored("\nWARNING: This will scan Redis queues and remove corrupted messages.", YELLOW, bold=True)
        if args.repair:
            print_colored("The script will attempt to repair corrupted messages when possible.", BLUE)
        print_colored("A backup of each queue will be created before modification.", BLUE)
        confirm = input("\nProceed? (yes/no): ")
        if confirm.lower() != "yes":
            print_colored("Operation cancelled.", BLUE)
            return
    
    # Proceed with the actual scan and cleanup
    scan_all_celery_queues(client, dry_run=False, repair=args.repair, queues=args.queue)
    
    print_colored("\nDone! You should restart Celery workers now.", GREEN, bold=True)
    print_colored("Run: sudo systemctl restart celery-worker.service", BLUE)

if __name__ == "__main__":
    main()
