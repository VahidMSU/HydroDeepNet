# SWATGenX Celery Management Utilities

This document describes tools for managing and monitoring the Celery task queue system used by SWATGenX.

## 1. Monitoring Celery Status

The `monitor_celery_status.py` script provides real-time information about Celery workers, Redis queues, and task status.

### Basic Usage:

```bash
# Run full system check
python /data/SWATGenXApp/codes/scripts/monitor_celery_status.py

# Check only specific components
python /data/SWATGenXApp/codes/scripts/monitor_celery_status.py --workers
python /data/SWATGenXApp/codes/scripts/monitor_celery_status.py --queues
python /data/SWATGenXApp/codes/scripts/monitor_celery_status.py --tasks
python /data/SWATGenXApp/codes/scripts/monitor_celery_status.py --resources

# Save report to file
python /data/SWATGenXApp/codes/scripts/monitor_celery_status.py -o /path/to/report.txt
```

### Advanced Usage:

```bash
# Watch mode - refresh every 30 seconds
python /data/SWATGenXApp/codes/scripts/monitor_celery_status.py -w 30

# Enable monitoring alerts
python /data/SWATGenXApp/codes/scripts/monitor_celery_status.py --alert-threshold 100 --email admin@example.com

# Show only failing tasks
python /data/SWATGenXApp/codes/scripts/monitor_celery_status.py --failures

# Disable colors for easier parsing
python /data/SWATGenXApp/codes/scripts/monitor_celery_status.py --clean
```

## 2. Managing Celery Queues

The `manage_celery_queues.py` script allows inspecting and maintaining Celery queues in Redis.

### Queue Inspection:

```bash
# List all queues and their sizes
python /data/SWATGenXApp/codes/scripts/manage_celery_queues.py --list

# Inspect tasks in a specific queue
python /data/SWATGenXApp/codes/scripts/manage_celery_queues.py --inspect celery

# Show detailed task information
python /data/SWATGenXApp/codes/scripts/manage_celery_queues.py --inspect celery --full

# Limit the number of tasks shown
python /data/SWATGenXApp/codes/scripts/manage_celery_queues.py --inspect celery --limit 50
```

### Queue Maintenance:

```bash
# Backup a queue before making changes
python /data/SWATGenXApp/codes/scripts/manage_celery_queues.py --backup celery

# Remove tasks of a specific type from a queue
python /data/SWATGenXApp/codes/scripts/manage_celery_queues.py --queue celery --remove-tasks fetch_station_geometries

# Remove multiple task types (comma-separated)
python /data/SWATGenXApp/codes/scripts/manage_celery_queues.py --queue celery --remove-tasks fetch_station_geometries,process_noaa_data

# Fix tasks with invalid site numbers
python /data/SWATGenXApp/codes/scripts/manage_celery_queues.py --fix-site-numbers celery

# Check invalid site numbers without removing (dry run)
python /data/SWATGenXApp/codes/scripts/manage_celery_queues.py --fix-site-numbers celery --dry-run

# Purge all tasks from a queue (with confirmation)
python /data/SWATGenXApp/codes/scripts/manage_celery_queues.py --flush celery

# Purge queue without confirmation (use with caution)
python /data/SWATGenXApp/codes/scripts/manage_celery_queues.py --flush celery --yes
```

## 3. Managing Celery Workers

Use these systemd commands to control the Celery worker service:

```bash
# Start Celery workers
sudo systemctl start celery-worker.service

# Stop Celery workers
sudo systemctl stop celery-worker.service

# Restart Celery workers
sudo systemctl restart celery-worker.service

# Check worker status
sudo systemctl status celery-worker.service

# View Celery worker logs
sudo journalctl -u celery-worker.service

# Follow Celery logs in real-time
sudo journalctl -f -u celery-worker.service
```

For multi-worker deployment:

```bash
# Start multiple workers
sudo systemctl start celery-multi-worker.service

# Check all worker status
sudo systemctl status celery-multi-worker.service
```

## 4. Troubleshooting Common Issues

### No tasks being processed:

```bash
# Check if workers are running
python /data/SWATGenXApp/codes/scripts/monitor_celery_status.py --workers

# If no workers found, restart the worker service
sudo systemctl restart celery-worker.service

# Check Redis connection
redis-cli ping
```

### Queue backed up with tasks:

```bash
# Inspect the queue contents
python /data/SWATGenXApp/codes/scripts/manage_celery_queues.py --inspect celery --full

# For tasks stuck in development/testing, selective purge by type:
python /data/SWATGenXApp/codes/scripts/manage_celery_queues.py --queue celery --remove-tasks fetch_station_geometries
```

### Tasks with invalid site numbers:

```bash
# Identify tasks with malformed site numbers (like [2025-04-0)
python /data/SWATGenXApp/codes/scripts/manage_celery_queues.py --fix-site-numbers celery --dry-run

# Remove tasks with invalid site numbers
python /data/SWATGenXApp/codes/scripts/manage_celery_queues.py --fix-site-numbers celery

# Bypass confirmation prompt
python /data/SWATGenXApp/codes/scripts/manage_celery_queues.py --fix-site-numbers celery --yes
```

### Stuck tasks in Redis (not in queue):

Sometimes tasks appear as "active" but they're not in the queue and are stuck in Redis. Use the `fix_stuck_tasks.py` script:

```bash
# List all active tasks in Redis
python /data/SWATGenXApp/codes/scripts/fix_stuck_tasks.py --list

# List all tasks (including completed) in Redis 
python /data/SWATGenXApp/codes/scripts/fix_stuck_tasks.py --list-all

# Clear tasks with invalid site numbers (dry run first)
python /data/SWATGenXApp/codes/scripts/fix_stuck_tasks.py --clear-invalid-site --dry-run

# Actually clear tasks with invalid site numbers
python /data/SWATGenXApp/codes/scripts/fix_stuck_tasks.py --clear-invalid-site

# Clear stale tasks (older than 1 day and still active)
python /data/SWATGenXApp/codes/scripts/fix_stuck_tasks.py --clear-stale

# In emergency situations, clear all active tasks
python /data/SWATGenXApp/codes/scripts/fix_stuck_tasks.py --clear-all-active
```

After clearing stuck tasks, always restart the Celery workers:
```bash
sudo systemctl restart celery-worker.service
```

### Worker crashes or high memory usage:

```bash
# Check system resources
python /data/SWATGenXApp/codes/scripts/monitor_celery_status.py --resources

# Adjust worker concurrency settings in:
#   /data/SWATGenXApp/codes/scripts/celery-worker.service
#   /data/SWATGenXApp/codes/scripts/celery-multi-worker.service

# Then restart the service
sudo systemctl daemon-reload
sudo systemctl restart celery-worker.service
```

### Handling WRONGTYPE errors in Redis:

If you encounter `WRONGTYPE Operation against a key holding the wrong kind of value` errors in logs:

```bash
# Keep only batch-oriented commands:
sudo fix-redis-batch --dry-run
sudo fix-redis-batch --batch-size 100
sudo systemctl start redis-fix.service
sudo journalctl -f -u redis-fix.service
```

# For emergency recovery, use the all-in-one script
sudo /data/SWATGenXApp/codes/scripts/restart_corrupted_celery.sh
```

> **Note:** The wrapper scripts ensure proper Python environment is used with sudo.
> 
> **For large Redis databases:** Use the batch tools (`fix-redis-batch`) which are designed to handle 
> thousands of corrupted keys without getting stuck.

After cleaning up WRONGTYPE errors, always restart the Celery workers:
```bash
sudo systemctl restart celery-worker.service
```

## 5. Production Monitoring Setup

For continuous monitoring, set up a cron job to run the monitor script and send alerts:

```bash
# Add to crontab (run every 15 minutes)
*/15 * * * * /data/SWATGenXApp/codes/.venv/bin/python /data/SWATGenXApp/codes/scripts/monitor_celery_status.py --alert-threshold 1000 --email admin@example.com --clean -o /data/SWATGenXApp/codes/web_application/logs/celery_monitor_$(date +\%Y\%m\%d).log
```