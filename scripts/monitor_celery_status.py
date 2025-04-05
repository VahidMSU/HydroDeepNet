#!/usr/bin/env python3
"""
Celery Monitoring Script for SWATGenX
Check active workers, queue status, and analyze user model tasks
"""

import os
import sys
import json
import time
import datetime
import argparse
import redis
from collections import Counter, defaultdict
from tabulate import tabulate
import subprocess
import re
import signal

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

# Add new global variables for script configuration
USE_COLORS = True
COMMAND_TIMEOUT = 30  # seconds
MAX_OUTPUT_WIDTH = 120  # characters

def strip_ansi_codes(text):
    """Remove ANSI color codes from text"""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def print_header(text):
    """Print section header"""
    if USE_COLORS:
        print(f"\n{HEADER}{BOLD}{'=' * 80}{ENDC}")
        print(f"{HEADER}{BOLD} {text} {ENDC}")
        print(f"{HEADER}{BOLD}{'=' * 80}{ENDC}")
    else:
        print(f"\n{'=' * 80}")
        print(f" {text} ")
        print(f"{'=' * 80}")

def print_subheader(text):
    """Print subsection header"""
    if USE_COLORS:
        print(f"\n{BLUE}{BOLD} {text} {ENDC}")
        print(f"{BLUE}{BOLD} {'-' * len(text)} {ENDC}")
    else:
        print(f"\n {text}")
        print(f" {'-' * len(text)}")

def run_command_with_timeout(cmd, timeout=COMMAND_TIMEOUT):
    """Run a shell command with timeout protection"""
    try:
        proc = subprocess.Popen(
            cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Set up timeout handling
        def timeout_handler(signum, frame):
            proc.kill()
            raise TimeoutError(f"Command timed out after {timeout} seconds")
        
        # Set the signal handler and alarm
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        stdout, stderr = proc.communicate()
        
        # Cancel the alarm
        signal.alarm(0)
        
        return {
            'returncode': proc.returncode,
            'stdout': stdout,
            'stderr': stderr,
            'success': proc.returncode == 0
        }
    except Exception as e:
        return {
            'returncode': -1,
            'stdout': '',
            'stderr': str(e),
            'success': False
        }

def get_celery_workers_status():
    """Get information about active Celery workers using celery inspect"""
    print_header("CELERY WORKER STATUS")
    
    try:
        # Get active workers information
        inspect_cmd = "cd /data/SWATGenXApp/codes/web_application && "\
                     "/data/SWATGenXApp/codes/.venv/bin/celery -A celery_worker inspect stats"
        result = run_command_with_timeout(inspect_cmd)
        
        if not result['success'] or not result['stdout']:
            status_msg = f"Error running celery inspect: {result['stderr']}"
            print(f"{RED}{status_msg}{ENDC}" if USE_COLORS else status_msg)
            return None

        # Parse and display worker stats
        worker_stats = {}
        current_worker = None
        lines = result['stdout'].split('\n')
        
        workers_found = 0
        for line in lines:
            if "celery@" in line and ":" in line:
                current_worker = line.strip().rstrip(':')
                worker_stats[current_worker] = {"tasks": {}, "processes": 0}
                workers_found += 1
            elif current_worker and "->" in line:
                key, value = line.split('->', 1)
                key = key.strip()
                value = value.strip()
                
                # Extract key metrics
                if key == 'processes':
                    worker_stats[current_worker]["processes"] = int(value)
                elif key == 'pool':
                    worker_stats[current_worker]["pool"] = value
                elif key == 'prefetch_count':
                    worker_stats[current_worker]["prefetch"] = int(value)
                elif key == 'broker':
                    worker_stats[current_worker]["broker"] = value
        
        # Display active workers table
        if workers_found > 0:
            status_msg = f"Found {workers_found} active Celery workers"
            print(f"{GREEN}{BOLD}{status_msg}{ENDC}" if USE_COLORS else status_msg)
            workers_table = []
            for worker, stats in worker_stats.items():
                workers_table.append([
                    worker,
                    stats.get("processes", "N/A"),
                    stats.get("pool", "N/A"),
                    stats.get("prefetch", "N/A"),
                ])
            
            table = tabulate(
                workers_table,
                headers=["Worker", "Processes", "Pool Type", "Prefetch Count"],
                tablefmt="pretty"
            )
            print(table)
        else:
            status_msg = "No active Celery workers found!"
            print(f"{RED}{status_msg}{ENDC}" if USE_COLORS else status_msg)
        
        # Get active tasks by worker
        print_subheader("ACTIVE TASKS BY WORKER")
        active_cmd = "cd /data/SWATGenXApp/codes/web_application && "\
                     "/data/SWATGenXApp/codes/.venv/bin/celery -A celery_worker inspect active"
        active_result = run_command_with_timeout(active_cmd)
        
        if active_result['success'] and active_result['stdout']:
            # Check if any workers are running active tasks
            if "empty" in active_result['stdout'] and "No active tasks" in active_result['stdout']:
                status_msg = "No active tasks currently being processed by workers"
                print(f"{YELLOW}{status_msg}{ENDC}" if USE_COLORS else status_msg)
            else:
                # Display the active tasks by worker
                print(active_result['stdout'][:MAX_OUTPUT_WIDTH])
                if len(active_result['stdout']) > MAX_OUTPUT_WIDTH:
                    print("... (output truncated) ...")
        else:
            status_msg = f"Error getting active tasks: {active_result['stderr']}"
            print(f"{RED}{status_msg}{ENDC}" if USE_COLORS else status_msg)
        
        # Get registered tasks
        print_subheader("REGISTERED TASK TYPES")
        registered_cmd = "cd /data/SWATGenXApp/codes/web_application && "\
                         "/data/SWATGenXApp/codes/.venv/bin/celery -A celery_worker inspect registered"
        registered_result = run_command_with_timeout(registered_cmd)
        
        if registered_result['success'] and registered_result['stdout']:
            # Extract and display task names more cleanly
            task_types = set()
            for line in registered_result['stdout'].split('\n'):
                if line.startswith('  '):
                    task_name = line.strip()
                    if task_name:
                        task_types.add(task_name)
            
            if task_types:
                status_msg = "Registered task types:"
                print(f"{GREEN}{status_msg}{ENDC}" if USE_COLORS else status_msg)
                # Limit the number of task types shown to keep output clean
                task_list = sorted(task_types)
                max_tasks_to_show = 15
                for task in task_list[:max_tasks_to_show]:
                    print(f"  - {task}")
                if len(task_list) > max_tasks_to_show:
                    print(f"  ... and {len(task_list) - max_tasks_to_show} more tasks")
            else:
                status_msg = "No registered task types found"
                print(f"{YELLOW}{status_msg}{ENDC}" if USE_COLORS else status_msg)
        else:
            status_msg = f"Error getting registered tasks: {registered_result['stderr']}"
            print(f"{RED}{status_msg}{ENDC}" if USE_COLORS else status_msg)
            
        return workers_found
        
    except Exception as e:
        status_msg = f"Error checking Celery worker status: {str(e)}"
        print(f"{RED}{status_msg}{ENDC}" if USE_COLORS else status_msg)
        return None

def get_redis_queue_status():
    """Check Redis queues and get queue information"""
    print_header("REDIS QUEUE STATUS")
    
    try:
        # Connect to Redis with timeout
        redis_client = redis.Redis.from_url('redis://localhost:6379/0', 
                                          socket_connect_timeout=2,
                                          socket_timeout=5)
        try:
            redis_info = redis_client.info()
        except redis.RedisError as e:
            status_msg = f"Failed to get Redis info: {str(e)}"
            print(f"{RED}{status_msg}{ENDC}" if USE_COLORS else status_msg)
            return None
        
        # Check Redis info
        print_subheader("REDIS SERVER INFO")
        print(f"Redis version: {redis_info.get('redis_version', 'unknown')}")
        print(f"Connected clients: {redis_info.get('connected_clients', 'unknown')}")
        print(f"Memory used: {redis_info.get('used_memory_human', 'unknown')}")
        print(f"Total connections received: {redis_info.get('total_connections_received', 'unknown')}")
        print(f"Total commands processed: {redis_info.get('total_commands_processed', 'unknown')}")
        
        # Get queue sizes with error handling for each queue
        print_subheader("QUEUE LENGTHS")
        queues = {}
        queue_names = ['model_creation', 'celery']
        
        for queue in queue_names:
            try:
                queues[queue] = redis_client.llen(queue)
            except redis.RedisError:
                queues[queue] = "Error"
        
        queue_table = []
        for queue_name, length in queues.items():
            if length == "Error":
                status = f"{RED}Error{ENDC}" if USE_COLORS else "Error"
            else:
                if USE_COLORS:
                    status = f"{GREEN}Empty{ENDC}" if length == 0 else f"{YELLOW}{length} tasks{ENDC}"
                else:
                    status = "Empty" if length == 0 else f"{length} tasks"
            queue_table.append([queue_name, length if length != "Error" else "N/A", status])
        
        print(tabulate(
            queue_table,
            headers=["Queue Name", "Tasks Waiting", "Status"],
            tablefmt="pretty"
        ))
        
        # Get delayed tasks (if any)
        print_subheader("SCHEDULED TASKS")
        try:
            scheduled_count = redis_client.zcard('_kombu.binding.schedule')
            if scheduled_count > 0:
                status_msg = f"Found {scheduled_count} scheduled (delayed) tasks"
                print(f"{YELLOW}{status_msg}{ENDC}" if USE_COLORS else status_msg)
            else:
                status_msg = "No scheduled tasks found"
                print(f"{GREEN}{status_msg}{ENDC}" if USE_COLORS else status_msg)
        except redis.RedisError as e:
            status_msg = f"Error checking scheduled tasks: {str(e)}"
            print(f"{RED}{status_msg}{ENDC}" if USE_COLORS else status_msg)
            
        return queues
        
    except redis.RedisError as e:
        status_msg = f"Redis connection error: {str(e)}"
        print(f"{RED}{status_msg}{ENDC}" if USE_COLORS else status_msg)
        return None
    except Exception as e:
        status_msg = f"Error checking Redis queues: {str(e)}"
        print(f"{RED}{status_msg}{ENDC}" if USE_COLORS else status_msg)
        return None

def analyze_user_tasks():
    """Analyze tasks by user from the task tracker log"""
    print_header("USER TASK ANALYSIS")
    
    try:
        # Path to the task tracker log
        log_file = '/data/SWATGenXApp/codes/web_application/logs/model_tasks.log'
        
        if not os.path.exists(log_file):
            status_msg = f"Task log file not found: {log_file}"
            print(f"{RED}{status_msg}{ENDC}" if USE_COLORS else status_msg)
            return None
            
        # Parse the log file to extract task information
        tasks = {}
        users = defaultdict(list)
        
        # Use file size limitation to prevent loading too much data
        max_file_size = 10 * 1024 * 1024  # 10 MB
        file_size = os.path.getsize(log_file)
        
        if file_size > max_file_size:
            status_msg = f"Log file is very large ({file_size / 1024 / 1024:.1f} MB). Only loading the last 10 MB."
            print(f"{YELLOW}{status_msg}{ENDC}" if USE_COLORS else status_msg)
            
            # Only read the last part of the file
            with open(log_file, 'rb') as f:
                f.seek(max(0, file_size - max_file_size))
                # Skip potentially incomplete first line
                f.readline()
                lines = f.read().decode('utf-8', errors='replace').splitlines()
        else:
            with open(log_file, 'r', errors='replace') as f:
                lines = f.readlines()
        
        line_count = 0
        parse_errors = 0
        
        for line in lines:
            line_count += 1
            if " - {" not in line:
                continue
                
            try:
                timestamp, task_json = line.split(" - ", 1)
                task_data = json.loads(task_json)
                task_id = task_data.get('task_id')
                
                if not task_id:
                    continue
                    
                # Store task data
                tasks[task_id] = task_data
                
                # Group by username
                username = task_data.get('username')
                if username:
                    users[username].append(task_id)
            except Exception as e:
                parse_errors += 1
                continue
        
        if parse_errors > 0:
            status_msg = f"Warning: {parse_errors} log entries couldn't be parsed"
            print(f"{YELLOW}{status_msg}{ENDC}" if USE_COLORS else status_msg)
        
        # Analyze tasks by status
        status_counts = Counter()
        for task in tasks.values():
            status = task.get('status', 'UNKNOWN')
            status_counts[status] += 1
        
        # Display overall statistics
        print_subheader("OVERALL TASK STATISTICS")
        print(f"Total tasks tracked: {len(tasks)}")
        
        status_table = []
        for status, count in sorted(status_counts.items()):
            if USE_COLORS:
                status_color = GREEN if status == 'SUCCESS' else YELLOW if status == 'STARTED' else RED if status == 'FAILURE' else BLUE
                status_display = f"{status_color}{status}{ENDC}"
            else:
                status_display = status
                
            percentage = count/len(tasks)*100 if len(tasks) > 0 else 0
            status_table.append([status_display, count, f"{percentage:.1f}%"])
        
        print(tabulate(
            status_table,
            headers=["Status", "Count", "Percentage"],
            tablefmt="pretty"
        ))
        
        # Calculate ongoing/queued tasks
        active_statuses = ['PENDING', 'STARTED', 'RECEIVED', 'RETRY']
        active_tasks = [t for t in tasks.values() if t.get('status') in active_statuses]
        
        # Display per-user statistics
        print_subheader("USER TASK BREAKDOWN")
        
        if not users:
            status_msg = "No user task data found"
            print(f"{YELLOW}{status_msg}{ENDC}" if USE_COLORS else status_msg)
            return None
        
        user_table = []
        # Limit to top 20 users to avoid cluttering the display
        max_users_to_show = 20
        
        for i, (username, task_ids) in enumerate(sorted(users.items(), key=lambda x: len(x[1]), reverse=True)):
            if i >= max_users_to_show:
                break
                
            # Count by status for this user
            user_statuses = Counter()
            for task_id in task_ids:
                task = tasks.get(task_id, {})
                status = task.get('status', 'UNKNOWN')
                user_statuses[status] += 1
            
            # Get active task count for this user
            active_count = sum(user_statuses[s] for s in active_statuses if s in user_statuses)
            success_count = user_statuses.get('SUCCESS', 0)
            failed_count = user_statuses.get('FAILURE', 0)
            
            user_table.append([
                username[:20] if len(username) > 20 else username,  # Truncate long usernames
                len(task_ids),
                active_count,
                success_count,
                failed_count
            ])
        
        print(tabulate(
            user_table,
            headers=["Username", "Total Tasks", "Active/Queued", "Completed", "Failed"],
            tablefmt="pretty"
        ))
        
        if len(users) > max_users_to_show:
            status_msg = f"(Showing top {max_users_to_show} of {len(users)} users)"
            print(f"{YELLOW}{status_msg}{ENDC}" if USE_COLORS else status_msg)
        
        # Show currently active tasks in detail
        if active_tasks:
            print_subheader("CURRENTLY ACTIVE TASKS")
            active_table = []
            max_active_to_show = 10  # Reduce from 20 to 10 to make output more compact
            
            for task in sorted(active_tasks, key=lambda x: x.get('created_at', ''), reverse=True)[:max_active_to_show]:
                username = task.get('username', 'unknown')
                status = task.get('status', 'UNKNOWN')
                site_no = task.get('site_no', 'unknown')
                created = task.get('created_at', 'unknown')
                progress = task.get('progress', 0)
                
                # Format created time to be more readable
                try:
                    created_dt = datetime.datetime.fromisoformat(created)
                    created_str = created_dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    created_str = created
                
                task_id = task.get('task_id', 'unknown')
                
                # Prepare status display with or without color
                if USE_COLORS:
                    status_display = f"{YELLOW}{status} ({progress}%){ENDC}"
                else:
                    status_display = f"{status} ({progress}%)"
                    
                active_table.append([
                    username[:15] if len(username) > 15 else username,  # Truncate long usernames
                    site_no[:10] if len(str(site_no)) > 10 else site_no,  # Truncate long site numbers
                    status_display,
                    created_str,
                    task_id[:8] + '...'  # Truncate task ID for display
                ])
            
            print(tabulate(
                active_table,
                headers=["User", "Site No", "Status", "Created", "Task ID"],
                tablefmt="pretty"
            ))
            
            if len(active_tasks) > max_active_to_show:
                status_msg = f"(Showing {max_active_to_show} of {len(active_tasks)} active tasks)"
                print(f"{YELLOW}{status_msg}{ENDC}" if USE_COLORS else status_msg)
        
        return users
            
    except Exception as e:
        status_msg = f"Error analyzing user tasks: {str(e)}"
        print(f"{RED}{status_msg}{ENDC}" if USE_COLORS else status_msg)
        return None

def check_system_resources():
    """Check system resources (CPU, memory, disk)"""
    print_header("SYSTEM RESOURCES")
    
    try:
        # Check CPU load
        print_subheader("CPU LOAD")
        try:
            with open('/proc/loadavg', 'r') as f:
                load = f.read().strip().split()
                load_1min, load_5min, load_15min = load[0:3]
                
                # Get CPU count for context
                cpu_count = os.cpu_count() or 1
                
                # Format with color based on load
                def format_load(load_str, cpu_count):
                    load_val = float(load_str)
                    if not USE_COLORS:
                        return load_str
                        
                    if load_val > cpu_count * 0.8:
                        return f"{RED}{load_str}{ENDC}"
                    elif load_val > cpu_count * 0.5:
                        return f"{YELLOW}{load_str}{ENDC}"
                    else:
                        return f"{GREEN}{load_str}{ENDC}"
                
                print(f"Load average: {format_load(load_1min, cpu_count)} (1m), "
                      f"{format_load(load_5min, cpu_count)} (5m), "
                      f"{format_load(load_15min, cpu_count)} (15m)")
                print(f"Number of CPU cores: {cpu_count}")
        except Exception as e:
            status_msg = f"Error reading CPU load: {str(e)}"
            print(f"{RED}{status_msg}{ENDC}" if USE_COLORS else status_msg)
        
        # Check memory usage
        print_subheader("MEMORY USAGE")
        mem_result = run_command_with_timeout("free -h")
        if mem_result['success']:
            print(mem_result['stdout'])
        else:
            status_msg = f"Error getting memory info: {mem_result['stderr']}"
            print(f"{RED}{status_msg}{ENDC}" if USE_COLORS else status_msg)
        
        # Check disk usage
        print_subheader("DISK USAGE")
        disk_result = run_command_with_timeout("df -h /data")
        if disk_result['success']:
            print(disk_result['stdout'])
        else:
            status_msg = f"Error getting disk info: {disk_result['stderr']}"
            print(f"{RED}{status_msg}{ENDC}" if USE_COLORS else status_msg)
            
    except Exception as e:
        status_msg = f"Error checking system resources: {str(e)}"
        print(f"{RED}{status_msg}{ENDC}" if USE_COLORS else status_msg)

def main():
    """Main function to run the monitoring script"""
    parser = argparse.ArgumentParser(description='Monitor Celery, Redis, and task status')
    parser.add_argument('--workers', action='store_true', help='Show worker status only')
    parser.add_argument('--queues', action='store_true', help='Show queue status only')
    parser.add_argument('--tasks', action='store_true', help='Show user tasks only')
    parser.add_argument('--resources', action='store_true', help='Show system resources only')
    parser.add_argument('--output', '-o', type=str, help='Write report to file')
    parser.add_argument('--clean', action='store_true', help='Disable colors and use simpler output')
    parser.add_argument('--timeout', type=int, default=COMMAND_TIMEOUT, 
                       help=f'Command timeout in seconds (default: {COMMAND_TIMEOUT})')
    args = parser.parse_args()
    
    # Configure global settings
    global USE_COLORS
    USE_COLORS = not args.clean and args.output is None
    
    # Update timeout without using global
    if args.timeout != COMMAND_TIMEOUT:
        # We'll pass the new timeout directly to functions that need it
        timeout = args.timeout
    else:
        timeout = COMMAND_TIMEOUT
    
    # Store original stdout for later restoration
    original_stdout = sys.stdout
    if args.output:
        try:
            sys.stdout = open(args.output, 'w')
            print(f"SWATGenX Celery Monitoring Report - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        except Exception as e:
            sys.stdout = original_stdout
            status_msg = f"Error opening output file: {str(e)}"
            print(f"{RED}{status_msg}{ENDC}" if USE_COLORS else status_msg)
            return
    
    try:
        # Run all checks if no specific ones are requested
        run_all = not (args.workers or args.queues or args.tasks or args.resources)
        
        if run_all or args.resources:
            check_system_resources()
        
        if run_all or args.workers:
            worker_count = get_celery_workers_status()
        
        if run_all or args.queues:
            queue_info = get_redis_queue_status()
        
        if run_all or args.tasks:
            user_tasks = analyze_user_tasks()
            
        # Display summary if running all checks
        if run_all:
            print_header("SUMMARY")
            print(f"Report generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Celery workers active: {worker_count if worker_count is not None else 'Error'}")
            
            if queue_info:
                waiting_tasks = sum(v for v in queue_info.values() if isinstance(v, int))
                print(f"Tasks waiting in queues: {waiting_tasks}")
                
            if user_tasks:
                print(f"Total users with models: {len(user_tasks)}")
                
                # Get list of users with active tasks - using more efficient approach
                active_statuses = ['PENDING', 'STARTED', 'RECEIVED', 'RETRY']
                users_with_active = 0
                
                # Connect to Redis once instead of for each task
                try:
                    r = redis.Redis.from_url('redis://localhost:6379/0', socket_connect_timeout=2)
                    # Check a sample of tasks for each user (max 5 per user) to improve performance
                    for username, task_ids in user_tasks.items():
                        has_active = False
                        for task_id in task_ids[:5]:  # Check only first 5 tasks per user
                            try:
                                task_json = r.get(f"task:{task_id}")
                                if task_json:
                                    task = json.loads(task_json)
                                    if task.get('status') in active_statuses:
                                        has_active = True
                                        break
                            except:
                                pass
                        
                        if has_active:
                            users_with_active += 1
                except Exception as e:
                    status_msg = f"Error checking active users: {str(e)}"
                    print(f"{YELLOW}{status_msg}{ENDC}" if USE_COLORS else status_msg)
                
                print(f"Users with active models: {users_with_active}")
            
    except Exception as e:
        status_msg = f"Unexpected error: {str(e)}"
        print(f"{RED}{status_msg}{ENDC}" if USE_COLORS else status_msg)
    finally:
        # Restore original stdout if we redirected it
        if args.output:
            output_file = args.output
            sys.stdout.close()
            sys.stdout = original_stdout
            print(f"Report written to {output_file}")
            
            # If we're also writing to a file but still want colors in terminal,
            # we need to convert the file to strip ANSI codes
            if not args.clean:
                try:
                    with open(output_file, 'r') as f:
                        content = f.read()
                    
                    # Strip ANSI color codes for the file output
                    with open(output_file, 'w') as f:
                        f.write(strip_ansi_codes(content))
                except Exception as e:
                    print(f"Warning: Failed to strip color codes from output file: {str(e)}")

if __name__ == "__main__":
    main()
