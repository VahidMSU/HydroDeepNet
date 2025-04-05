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

def print_header(text):
    """Print section header"""
    print(f"\n{HEADER}{BOLD}{'=' * 80}{ENDC}")
    print(f"{HEADER}{BOLD} {text} {ENDC}")
    print(f"{HEADER}{BOLD}{'=' * 80}{ENDC}")

def print_subheader(text):
    """Print subsection header"""
    print(f"\n{BLUE}{BOLD} {text} {ENDC}")
    print(f"{BLUE}{BOLD} {'-' * len(text)} {ENDC}")

def get_celery_workers_status():
    """Get information about active Celery workers using celery inspect"""
    print_header("CELERY WORKER STATUS")
    
    try:
        # Get activew workers information
        inspect_cmd = "cd /data/SWATGenXApp/codes/web_application && "\
                     "/data/SWATGenXApp/codes/.venv/bin/celery -A celery_worker inspect stats"
        result = subprocess.run(inspect_cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0 or not result.stdout:
            print(f"{RED}Error running celery inspect: {result.stderr}{ENDC}")
            return None

        # Parse and display worker stats
        worker_stats = {}
        current_worker = None
        lines = result.stdout.split('\n')
        
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
            print(f"{GREEN}{BOLD}Found {workers_found} active Celery workers{ENDC}")
            workers_table = []
            for worker, stats in worker_stats.items():
                workers_table.append([
                    worker,
                    stats.get("processes", "N/A"),
                    stats.get("pool", "N/A"),
                    stats.get("prefetch", "N/A"),
                ])
            
            print(tabulate(
                workers_table,
                headers=["Worker", "Processes", "Pool Type", "Prefetch Count"],
                tablefmt="pretty"
            ))
        else:
            print(f"{RED}No active Celery workers found!{ENDC}")
        
        # Get active tasks by worker
        print_subheader("ACTIVE TASKS BY WORKER")
        active_cmd = "cd /data/SWATGenXApp/codes/web_application && "\
                     "/data/SWATGenXApp/codes/.venv/bin/celery -A celery_worker inspect active"
        active_result = subprocess.run(active_cmd, shell=True, capture_output=True, text=True)
        
        if active_result.returncode == 0 and active_result.stdout:
            # Check if any workers are running active tasks
            if "empty" in active_result.stdout and "No active tasks" in active_result.stdout:
                print(f"{YELLOW}No active tasks currently being processed by workers{ENDC}")
            else:
                # Display the active tasks by worker
                print(active_result.stdout)
        else:
            print(f"{RED}Error getting active tasks: {active_result.stderr}{ENDC}")
        
        # Get registered tasks
        print_subheader("REGISTERED TASK TYPES")
        registered_cmd = "cd /data/SWATGenXApp/codes/web_application && "\
                         "/data/SWATGenXApp/codes/.venv/bin/celery -A celery_worker inspect registered"
        registered_result = subprocess.run(registered_cmd, shell=True, capture_output=True, text=True)
        
        if registered_result.returncode == 0 and registered_result.stdout:
            # Extract and display task names more cleanly
            task_types = set()
            for line in registered_result.stdout.split('\n'):
                if line.startswith('  '):
                    task_name = line.strip()
                    if task_name:
                        task_types.add(task_name)
            
            if task_types:
                print(f"{GREEN}Registered task types:{ENDC}")
                for task in sorted(task_types):
                    print(f"  - {task}")
            else:
                print(f"{YELLOW}No registered task types found{ENDC}")
        else:
            print(f"{RED}Error getting registered tasks: {registered_result.stderr}{ENDC}")
            
        return workers_found
        
    except Exception as e:
        print(f"{RED}Error checking Celery worker status: {str(e)}{ENDC}")
        return None

def get_redis_queue_status():
    """Check Redis queues and get queue information"""
    print_header("REDIS QUEUE STATUS")
    
    try:
        # Connect to Redis
        redis_client = redis.Redis.from_url('redis://localhost:6379/0', socket_connect_timeout=2)
        redis_info = redis_client.info()
        
        # Check Redis info
        print_subheader("REDIS SERVER INFO")
        print(f"Redis version: {redis_info.get('redis_version', 'unknown')}")
        print(f"Connected clients: {redis_info.get('connected_clients', 'unknown')}")
        print(f"Memory used: {redis_info.get('used_memory_human', 'unknown')}")
        print(f"Total connections received: {redis_info.get('total_connections_received', 'unknown')}")
        print(f"Total commands processed: {redis_info.get('total_commands_processed', 'unknown')}")
        
        # Get queue sizes
        print_subheader("QUEUE LENGTHS")
        queues = {
            'model_creation': redis_client.llen('model_creation'),
            'celery': redis_client.llen('celery'),
            # Add more queue names if needed
        }
        
        queue_table = []
        for queue_name, length in queues.items():
            status = f"{GREEN}Empty{ENDC}" if length == 0 else f"{YELLOW}{length} tasks{ENDC}"
            queue_table.append([queue_name, length, status])
        
        print(tabulate(
            queue_table,
            headers=["Queue Name", "Tasks Waiting", "Status"],
            tablefmt="pretty"
        ))
        
        # Get delayed tasks (if any)
        print_subheader("SCHEDULED TASKS")
        scheduled_count = redis_client.zcard('_kombu.binding.schedule')
        if scheduled_count > 0:
            print(f"{YELLOW}Found {scheduled_count} scheduled (delayed) tasks{ENDC}")
        else:
            print(f"{GREEN}No scheduled tasks found{ENDC}")
            
        return queues
        
    except redis.RedisError as e:
        print(f"{RED}Redis connection error: {str(e)}{ENDC}")
        return None
    except Exception as e:
        print(f"{RED}Error checking Redis queues: {str(e)}{ENDC}")
        return None

def analyze_user_tasks():
    """Analyze tasks by user from the task tracker log"""
    print_header("USER TASK ANALYSIS")
    
    try:
        # Path to the task tracker log
        log_file = '/data/SWATGenXApp/codes/web_application/logs/model_tasks.log'
        
        if not os.path.exists(log_file):
            print(f"{RED}Task log file not found: {log_file}{ENDC}")
            return None
            
        # Parse the log file to extract task information
        tasks = {}
        users = defaultdict(list)
        
        with open(log_file, 'r') as f:
            for line in f:
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
                    continue
        
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
            status_color = GREEN if status == 'SUCCESS' else YELLOW if status == 'STARTED' else RED if status == 'FAILURE' else BLUE
            status_table.append([f"{status_color}{status}{ENDC}", count, f"{count/len(tasks)*100:.1f}%"])
        
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
            print(f"{YELLOW}No user task data found{ENDC}")
            return None
        
        user_table = []
        for username, task_ids in sorted(users.items(), key=lambda x: len(x[1]), reverse=True):
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
                username, 
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
        
        # Show currently active tasks in detail
        if active_tasks:
            print_subheader("CURRENTLY ACTIVE TASKS")
            active_table = []
            for task in sorted(active_tasks, key=lambda x: x.get('created_at', ''), reverse=True):
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
                active_table.append([
                    username,
                    site_no,
                    f"{status} ({progress}%)",
                    created_str,
                    task_id[:8] + '...'  # Truncate task ID for display
                ])
            
            print(tabulate(
                active_table[:20],  # Limit to 20 tasks for readability
                headers=["User", "Site No", "Status", "Created", "Task ID"],
                tablefmt="pretty"
            ))
            
            if len(active_table) > 20:
                print(f"{YELLOW}(Showing 20 of {len(active_table)} active tasks){ENDC}")
        
        return users
            
    except Exception as e:
        print(f"{RED}Error analyzing user tasks: {str(e)}{ENDC}")
        import traceback
        print(traceback.format_exc())
        return None

def check_system_resources():
    """Check system resources (CPU, memory, disk)"""
    print_header("SYSTEM RESOURCES")
    
    try:
        # Check CPU load
        print_subheader("CPU LOAD")
        with open('/proc/loadavg', 'r') as f:
            load = f.read().strip().split()
            load_1min, load_5min, load_15min = load[0:3]
            
            # Get CPU count for context
            cpu_count = os.cpu_count() or 1
            
            # Format with color based on load
            def format_load(load_str, cpu_count):
                load_val = float(load_str)
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
        
        # Check memory usage
        print_subheader("MEMORY USAGE")
        mem_cmd = "free -h"
        mem_result = subprocess.run(mem_cmd, shell=True, capture_output=True, text=True)
        if mem_result.returncode == 0:
            print(mem_result.stdout)
        else:
            print(f"{RED}Error getting memory info: {mem_result.stderr}{ENDC}")
        
        # Check disk usage
        print_subheader("DISK USAGE")
        disk_cmd = "df -h /data"
        disk_result = subprocess.run(disk_cmd, shell=True, capture_output=True, text=True)
        if disk_result.returncode == 0:
            print(disk_result.stdout)
        else:
            print(f"{RED}Error getting disk info: {disk_result.stderr}{ENDC}")
            
    except Exception as e:
        print(f"{RED}Error checking system resources: {str(e)}{ENDC}")

def main():
    """Main function to run the monitoring script"""
    parser = argparse.ArgumentParser(description='Monitor Celery, Redis, and task status')
    parser.add_argument('--workers', action='store_true', help='Show worker status only')
    parser.add_argument('--queues', action='store_true', help='Show queue status only')
    parser.add_argument('--tasks', action='store_true', help='Show user tasks only')
    parser.add_argument('--resources', action='store_true', help='Show system resources only')
    parser.add_argument('--output', '-o', type=str, help='Write report to file')
    args = parser.parse_args()
    
    # Store original stdout for later restoration
    original_stdout = sys.stdout
    if args.output:
        try:
            sys.stdout = open(args.output, 'w')
            print(f"SWATGenX Celery Monitoring Report - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        except Exception as e:
            sys.stdout = original_stdout
            print(f"{RED}Error opening output file: {str(e)}{ENDC}")
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
                waiting_tasks = sum(queue_info.values())
                print(f"Tasks waiting in queues: {waiting_tasks}")
                
            if user_tasks:
                print(f"Total users with models: {len(user_tasks)}")
                
                # Get list of users with active tasks
                active_statuses = ['PENDING', 'STARTED', 'RECEIVED', 'RETRY']
                users_with_active = 0
                for username, task_ids in user_tasks.items():
                    # Check if user has any active tasks
                    has_active = False
                    for task_id in task_ids:
                        import redis
                        try:
                            r = redis.Redis.from_url('redis://localhost:6379/0')
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
                
                print(f"Users with active models: {users_with_active}")
            
    except Exception as e:
        print(f"{RED}Unexpected error: {str(e)}{ENDC}")
        import traceback
        print(traceback.format_exc())
    finally:
        # Restore original stdout if we redirected it
        if args.output:
            sys.stdout.close()
            sys.stdout = original_stdout
            print(f"Report written to {args.output}")

if __name__ == "__main__":
    main()
