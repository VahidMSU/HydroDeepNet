#!/data/SWATGenXApp/codes/.venv/bin/python
"""
Redis WRONGTYPE Error Batch Repair Tool

For handling large numbers of corrupted Redis keys that cause "WRONGTYPE Operation against
a key holding the wrong kind of value" errors, which can crash Celery workers.

This version uses batching to prevent getting stuck on large sets of corrupted keys.
"""

import sys
import os
import redis
import argparse
import time
import json
import signal
from datetime import datetime
import logging
from multiprocessing import Pool, cpu_count

# Add application path for imports
sys.path.insert(0, '/data/SWATGenXApp/codes')
sys.path.insert(0, '/data/SWATGenXApp/codes/web_application')

# Configure output formats for console display
HEADER = '\033[95m'
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'

# Global variables for configuration
USE_COLORS = True
LOG_FILE = "/data/SWATGenXApp/codes/web_application/logs/redis_wrongtype_batch_fix.log"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('redis_batch_fix')

def print_colored(text, color=None, bold=False):
    """Print text with optional color and emphasis"""
    if USE_COLORS and color:
        prefix = BOLD + color if bold else color
        print(f"{prefix}{text}{ENDC}")
    else:
        print(text)

def connect_to_redis(host='localhost', port=6379, db=0):
    """Connect to Redis with error handling"""
    try:
        client = redis.Redis(host=host, port=port, db=db, 
                            socket_timeout=5,
                            socket_connect_timeout=5)
        # Test connection
        client.ping()
        print_colored("Connected to Redis successfully", GREEN)
        logger.info("Connected to Redis successfully")
        return client
    except redis.RedisError as e:
        print_colored(f"Error connecting to Redis: {e}", RED, bold=True)
        logger.error(f"Redis connection error: {e}")
        return None
    except Exception as e:
        print_colored(f"Unexpected error connecting to Redis: {e}", RED, bold=True)
        logger.error(f"Unexpected Redis connection error: {e}")
        return None

def find_problematic_keys(client, pattern="celery-task-meta-*", batch_size=100):
    """
    Find keys that match the pattern and have WRONGTYPE errors
    Returns a list of problematic keys
    
    Uses batch scanning to avoid blocking Redis for too long
    """
    problematic_keys = []
    scanned_count = 0
    cursor = '0'
    
    print_colored(f"Scanning Redis for keys matching pattern: {pattern}", BLUE)
    logger.info(f"Starting scan for pattern: {pattern}")
    
    try:
        start_time = time.time()
        last_report_time = start_time
        
        while cursor != 0:
            cursor, keys = client.scan(cursor=cursor, match=pattern, count=batch_size)
            
            if not keys:
                continue
                
            scanned_count += len(keys)
            batch_problems = []
            
            # Check each key for WRONGTYPE errors
            for key in keys:
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                
                try:
                    # Check key type - this succeeds even for wrong types
                    key_type = client.type(key).decode('utf-8')
                    
                    # For celery-task-meta-* keys, we expect JSON strings
                    # If it's not a string, it's likely corrupted
                    if key_str.startswith('celery-task-meta-') and key_type != 'string':
                        batch_problems.append(key)
                        logger.info(f"Found problematic key: {key_str} (type: {key_type})")
                except redis.exceptions.ResponseError as e:
                    if "WRONGTYPE" in str(e):
                        batch_problems.append(key)
                        logger.warning(f"WRONGTYPE error for key: {key_str}")
            
            # Add the batch of problematic keys
            problematic_keys.extend(batch_problems)
            
            # Show progress every 5 seconds
            current_time = time.time()
            if current_time - last_report_time >= 5:
                elapsed = current_time - start_time
                keys_per_second = scanned_count / elapsed if elapsed > 0 else 0
                print_colored(f"Scanned {scanned_count} keys, found {len(problematic_keys)} problematic keys "
                             f"({elapsed:.1f} seconds, {keys_per_second:.1f} keys/sec)", YELLOW)
                last_report_time = current_time
    
        elapsed = time.time() - start_time
        keys_per_second = scanned_count / elapsed if elapsed > 0 else 0
        print_colored(f"Scan completed in {elapsed:.1f} seconds ({keys_per_second:.1f} keys/sec)", GREEN)
        print_colored(f"Scanned {scanned_count} keys total", BLUE)
        print_colored(f"Found {len(problematic_keys)} problematic keys", 
                     GREEN if len(problematic_keys) == 0 else YELLOW if len(problematic_keys) < 10 else RED, 
                     bold=True)
        
        return problematic_keys
        
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print_colored("\nScan interrupted by user!", YELLOW, bold=True)
        print_colored(f"Partial results: Scanned {scanned_count} keys, found {len(problematic_keys)} problematic keys", YELLOW)
        logger.warning(f"Scan interrupted by user after {elapsed:.1f} seconds")
        return problematic_keys
    except Exception as e:
        print_colored(f"Error scanning keys: {e}", RED)
        logger.error(f"Error during key scan: {e}", exc_info=True)
        return problematic_keys

def fix_keys_in_batches(client, problematic_keys, batch_size=50, backup=True):
    """
    Fix problematic keys in batches to prevent long-running operations
    Returns the number of successfully fixed keys
    """
    if not problematic_keys:
        print_colored("No problematic keys to fix", GREEN)
        return 0
        
    key_count = len(problematic_keys)
    print_colored(f"Preparing to fix {key_count} problematic keys in batches of {batch_size}", YELLOW, bold=True)
    
    # Show some examples of problematic keys
    if key_count > 0:
        print_colored("\nSample of problematic keys:", BLUE)
        for key in problematic_keys[:5]:
            key_str = key.decode('utf-8') if isinstance(key, bytes) else key
            print(f"  - {key_str}")
        
        if key_count > 5:
            print(f"  ... and {key_count - 5} more keys")
    
    # Create a backup of keys if requested
    if backup:
        try:
            backup_dir = "/data/SWATGenXApp/codes/web_application/logs/redis_backups"
            os.makedirs(backup_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = os.path.join(backup_dir, f"redis_wrongtype_backup_{timestamp}.json")
            
            # Create a minimal backup without values for performance
            backup_data = {key.decode('utf-8') if isinstance(key, bytes) else key: True 
                           for key in problematic_keys}
            
            with open(backup_file, 'w') as f:
                json.dump({
                    "timestamp": timestamp,
                    "count": len(problematic_keys),
                    "keys": list(backup_data.keys())
                }, f)
                
            print_colored(f"Created key list backup: {backup_file}", GREEN)
            logger.info(f"Created backup of {len(problematic_keys)} key names at {backup_file}")
        except Exception as e:
            print_colored(f"Error creating backup: {e}", RED)
            logger.error(f"Error creating backup: {e}")
    
    # Process keys in batches
    total_batches = (key_count + batch_size - 1) // batch_size
    success_count = 0
    error_count = 0
    start_time = time.time()
    
    for batch_num in range(total_batches):
        batch_start = batch_num * batch_size
        batch_end = min(batch_start + batch_size, key_count)
        batch_keys = problematic_keys[batch_start:batch_end]
        
        print_colored(f"\nProcessing batch {batch_num+1}/{total_batches} ({batch_end-batch_start} keys)...", BLUE)
        
        batch_success = 0
        batch_errors = 0
        
        # Process this batch
        for key in batch_keys:
            try:
                client.delete(key)
                batch_success += 1
            except Exception as e:
                batch_errors += 1
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                logger.error(f"Error deleting key {key_str}: {e}")
        
        # Update totals
        success_count += batch_success
        error_count += batch_errors
        
        # Show batch results
        if batch_errors > 0:
            print_colored(f"  Batch result: {batch_success} deleted, {batch_errors} failed", YELLOW)
        else:
            print_colored(f"  Batch result: {batch_success} deleted successfully", GREEN)
            
        # Show overall progress
        progress = (batch_num + 1) / total_batches * 100
        elapsed = time.time() - start_time
        keys_per_sec = (batch_start + batch_end) / elapsed if elapsed > 0 else 0
        
        est_remaining = (elapsed / (batch_num + 1)) * (total_batches - batch_num - 1) if batch_num > 0 else 0
        
        print_colored(f"  Overall progress: {progress:.1f}% ({success_count}/{key_count}) - "
                     f"{keys_per_sec:.1f} keys/sec - "
                     f"Est. remaining: {est_remaining:.1f} seconds", BLUE)
        
        # Small delay between batches to prevent Redis overload
        if batch_num < total_batches - 1:
            time.sleep(0.1)
    
    # Show final results
    print_colored(f"\nOperation complete: {success_count} of {key_count} keys fixed", 
                 GREEN if success_count == key_count else YELLOW, bold=True)
    
    if error_count > 0:
        print_colored(f"Failed to fix {error_count} keys", RED)
        
    logger.info(f"Fixed {success_count} of {key_count} problematic keys. Errors: {error_count}")
    
    return success_count

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Fix Redis WRONGTYPE errors in batches')
    parser.add_argument('--pattern', '-p', type=str, default='celery-task-meta-*',
                       help='Redis key pattern to scan (default: celery-task-meta-*)')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Redis host (default: localhost)')
    parser.add_argument('--port', type=int, default=6379,
                       help='Redis port (default: 6379)')
    parser.add_argument('--db', type=int, default=0,
                       help='Redis database (default: 0)')
    parser.add_argument('--dry-run', '-d', action='store_true',
                       help='Show what would be done without making changes')
    parser.add_argument('--batch-size', '-b', type=int, default=50,
                       help='Number of keys to process in each batch (default: 50)')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip creating backups (faster but less safe)')
    parser.add_argument('--no-color', action='store_true',
                       help='Disable colored output')
    parser.add_argument('--scan-batch-size', type=int, default=100,
                       help='Batch size for scanning Redis keys (default: 100)')
    parser.add_argument('--timeout', type=int, default=3600,
                       help='Timeout in seconds for the entire operation (default: 3600)')
    parser.add_argument('--yes', '-y', action='store_true',
                       help='Skip confirmation prompts')
    
    args = parser.parse_args()
    
    # Set global variables
    global USE_COLORS
    USE_COLORS = not args.no_color
    
    # Set up timeout handler
    def timeout_handler(signum, frame):
        print_colored("\nOperation timed out!", RED, bold=True)
        logger.error(f"Operation timed out after {args.timeout} seconds")
        sys.exit(1)
    
    # Register the timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.timeout)
    
    print_colored("Redis WRONGTYPE Batch Fix Tool", HEADER, bold=True)
    print_colored("===============================", HEADER, bold=True)
    
    try:
        # Connect to Redis
        client = connect_to_redis(args.host, args.port, args.db)
        if not client:
            sys.exit(1)
        
        # Scan for problematic keys
        print_colored(f"\nScanning for problematic keys matching pattern: {args.pattern}", BLUE, bold=True)
        problematic_keys = find_problematic_keys(client, args.pattern, batch_size=args.scan_batch_size)
        
        if not problematic_keys:
            print_colored("\nNo problematic keys found!", GREEN, bold=True)
            sys.exit(0)
        
        # In dry run mode, just report what would be done
        if args.dry_run:
            print_colored(f"\nWould fix {len(problematic_keys)} problematic keys (dry run mode)", YELLOW)
            sys.exit(0)
        
        # Confirm before proceeding in non-dry-run mode
        if not args.yes:
            print_colored(f"\nWARNING: About to delete {len(problematic_keys)} Redis keys. This cannot be undone.", 
                         YELLOW, bold=True)
            if not args.no_backup:
                print_colored("A backup of key names will be created first.", BLUE)
            confirm = input("\nProceed with deletion? (yes/no): ")
            if confirm.lower() != "yes":
                print_colored("Operation cancelled.", BLUE)
                sys.exit(0)
        
        # Fix the problematic keys in batches
        fixed_count = fix_keys_in_batches(client, problematic_keys, batch_size=args.batch_size, 
                                          backup=not args.no_backup)
        
        if fixed_count > 0:
            print_colored("\nYou should restart Celery workers to ensure proper operation:", GREEN)
            print_colored("sudo systemctl restart celery-worker.service", BLUE)
    
    except KeyboardInterrupt:
        print_colored("\nOperation cancelled by user", YELLOW, bold=True)
        logger.warning("Operation cancelled by user")
    except Exception as e:
        print_colored(f"\nUnexpected error: {e}", RED, bold=True)
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        # Disable the alarm
        signal.alarm(0)

if __name__ == "__main__":
    main()
