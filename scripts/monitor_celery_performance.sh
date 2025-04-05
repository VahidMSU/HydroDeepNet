#!/bin/bash
# Celery performance monitoring script
# This script collects metrics on Celery workers and tasks

# Constants
LOG_DIR="/data/SWATGenXApp/codes/web_application/logs"
METRICS_DIR="${LOG_DIR}/metrics"
REPORT_FILE="${METRICS_DIR}/celery_performance_report.txt"
HISTORY_FILE="${METRICS_DIR}/celery_history.csv"
VENV_PATH="/data/SWATGenXApp/codes/.venv"
APP_DIR="/data/SWATGenXApp/codes/web_application"

# Ensure directories exist
mkdir -p "${METRICS_DIR}"

# Log function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "${REPORT_FILE}"
}

# Start new report
log "=== Celery Performance Report ==="
log "Date: $(date)"
log "Hostname: $(hostname)"

# Activate virtual environment
source "${VENV_PATH}/bin/activate"
cd "${APP_DIR}"

# System resources
log "--- System Resources ---"
log "CPU Usage:"
top -bn1 | head -n 5 | tail -n 4 >> "${REPORT_FILE}"
log "Memory Usage:"
free -h >> "${REPORT_FILE}"
log "Disk Usage:"
df -h /data >> "${REPORT_FILE}"

# Celery workers
log "--- Celery Workers ---"
WORKER_COUNT=$(pgrep -f "celery worker" | wc -l)
log "Active Worker Processes: ${WORKER_COUNT}"

# Try to get worker stats from Celery API
if which celery > /dev/null; then
    log "Worker Status:"
    celery -A celery_worker inspect stats 2>/dev/null >> "${REPORT_FILE}" || log "Could not get worker stats"
    
    log "Active Workers:"
    celery -A celery_worker inspect active 2>/dev/null >> "${REPORT_FILE}" || log "Could not get active workers"
    
    log "Registered Tasks:"
    celery -A celery_worker inspect registered 2>/dev/null | grep -c ":" | sed 's/^/Total registered tasks: /' >> "${REPORT_FILE}" || log "Could not get registered tasks"
    
    log "Active Tasks:"
    ACTIVE_TASKS=$(celery -A celery_worker inspect active 2>/dev/null | grep -c "{" || echo "Unknown")
    log "Currently Active Tasks: ${ACTIVE_TASKS}"
fi

# Redis queue stats
log "--- Redis Queue Stats ---"
REDIS_CLI=$(which redis-cli)
if [ -n "${REDIS_CLI}" ]; then
    log "Queue Lengths:"
    echo "model_creation: $($REDIS_CLI -h localhost llen model_creation 2>/dev/null || echo 'Error')" >> "${REPORT_FILE}"
    echo "celery: $($REDIS_CLI -h localhost llen celery 2>/dev/null || echo 'Error')" >> "${REPORT_FILE}"
    
    log "Redis Info:"
    $REDIS_CLI -h localhost info | grep -E 'connected_clients|used_memory_human|rejected_connections' >> "${REPORT_FILE}"
fi

# Task stats (use python to parse task tracker data)
log "--- Task Statistics ---"
python3 - << EOF >> "${REPORT_FILE}"
import os
import json
import time
from datetime import datetime, timedelta

def analyze_task_log(log_file):
    if not os.path.exists(log_file):
        print(f"Task log file not found: {log_file}")
        return
        
    tasks = {}
    task_statuses = {}
    now = time.time()
    recent_tasks = []
    
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
                    
                # Store latest status for each task
                tasks[task_id] = task_data
                
                # Count task statuses
                status = task_data.get('status', 'UNKNOWN')
                task_statuses[status] = task_statuses.get(status, 0) + 1
                
                # Store recent tasks
                created_at = task_data.get('created_at')
                if created_at:
                    try:
                        created_time = datetime.fromisoformat(created_at)
                        if datetime.now() - created_time < timedelta(hours=24):
                            recent_tasks.append(task_data)
                    except:
                        pass
            except:
                continue
    
    # Output statistics
    print(f"Total tasks tracked: {len(tasks)}")
    print("\nTask Status Counts:")
    for status, count in sorted(task_statuses.items()):
        print(f"  {status}: {count}")
    
    # Tasks by site
    sites = {}
    for task in tasks.values():
        site = task.get('site_no')
        if site:
            sites[site] = sites.get(site, 0) + 1
            
    print(f"\nUnique sites: {len(sites)}")
    
    # Calculate task completion times
    completion_times = []
    for task in tasks.values():
        if task.get('status') == 'SUCCESS':
            created = task.get('created_at')
            updated = task.get('updated_at')
            if created and updated:
                try:
                    start_time = datetime.fromisoformat(created)
                    end_time = datetime.fromisoformat(updated)
                    completion_time = (end_time - start_time).total_seconds() / 60  # in minutes
                    completion_times.append(completion_time)
                except:
                    pass
    
    if completion_times:
        avg_completion = sum(completion_times) / len(completion_times)
        min_completion = min(completion_times)
        max_completion = max(completion_times)
        print(f"\nTask Completion Times (minutes):")
        print(f"  Average: {avg_completion:.2f}")
        print(f"  Minimum: {min_completion:.2f}")
        print(f"  Maximum: {max_completion:.2f}")
        
    # Recent task volume
    print(f"\nRecent Tasks (last 24h): {len(recent_tasks)}")
    
    # Tasks by hour of day
    if recent_tasks:
        hour_counts = {}
        for task in recent_tasks:
            created_at = task.get('created_at')
            if created_at:
                try:
                    hour = datetime.fromisoformat(created_at).hour
                    hour_counts[hour] = hour_counts.get(hour, 0) + 1
                except:
                    pass
        
        print("\nTask Distribution by Hour:")
        for hour in sorted(hour_counts.keys()):
            count = hour_counts[hour]
            bar = "#" * (count // 2 + 1)
            print(f"  {hour:02d}:00 - {count:3d} {bar}")

# Run the analysis
analyze_task_log("${LOG_DIR}/model_tasks.log")
EOF

# Collect metrics for CSV history (timestamp, worker_count, active_tasks, load_avg)
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
LOAD_AVG=$(cat /proc/loadavg | awk '{print $1}')

# Create header if file doesn't exist
if [ ! -f "${HISTORY_FILE}" ]; then
    echo "timestamp,worker_count,active_tasks,load_avg" > "${HISTORY_FILE}"
fi

# Append metrics
echo "${TIMESTAMP},${WORKER_COUNT},${ACTIVE_TASKS},${LOAD_AVG}" >> "${HISTORY_FILE}"

# Finish
log "Performance report completed at $(date)"
echo "Report saved to ${REPORT_FILE}"
