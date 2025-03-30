#!/bin/bash
# restart_celery_workers.sh
# This script restarts the Celery workers with improved configuration for handling multiple tasks

set -e

LOG_FILE="/data/SWATGenXApp/codes/web_application/logs/celery_restart.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Restarting Celery workers..." | tee -a $LOG_FILE

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "[$(date '+%Y-%m-%d %H:%M:%S')] This script must be run as root" | tee -a $LOG_FILE
   exit 1
fi

# Stop any existing Celery services
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stopping existing Celery services..." | tee -a $LOG_FILE
systemctl stop celery-multi-worker.service 2>/dev/null || echo "[$(date '+%Y-%m-%d %H:%M:%S')] celery-multi-worker service was not running" | tee -a $LOG_FILE
systemctl stop celery-worker.service 2>/dev/null || echo "[$(date '+%Y-%m-%d %H:%M:%S')] celery-worker service was not running" | tee -a $LOG_FILE

# Wait for processes to fully stop
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for Celery processes to stop..." | tee -a $LOG_FILE
sleep 5

# Make sure all Celery processes are killed
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking for remaining Celery processes..." | tee -a $LOG_FILE
pkill -f "celery" || echo "[$(date '+%Y-%m-%d %H:%M:%S')] No Celery processes found running" | tee -a $LOG_FILE

# Wait for processes to fully terminate
sleep 2

# Copy updated service file
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Copying updated service file..." | tee -a $LOG_FILE
cp /data/SWATGenXApp/codes/scripts/celery-worker.service /etc/systemd/system/

# Reload systemd to recognize the updated service
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Reloading systemd daemon..." | tee -a $LOG_FILE
systemctl daemon-reload

# Start the service
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Celery worker service..." | tee -a $LOG_FILE
systemctl start celery-worker.service

# Check status
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking service status..." | tee -a $LOG_FILE
systemctl status celery-worker.service --no-pager

# Enable the service to start on boot
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Enabling service to start on boot..." | tee -a $LOG_FILE
systemctl enable celery-worker.service

# Display task tracking log path
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task tracking log available at: /data/SWATGenXApp/codes/web_application/logs/model_tasks.log" | tee -a $LOG_FILE

# Display how to monitor running tasks
echo "[$(date '+%Y-%m-%d %H:%M:%S')] To monitor task status, use the API endpoints:" | tee -a $LOG_FILE
echo "  - GET /api/user_tasks - List all your tasks" | tee -a $LOG_FILE
echo "  - GET /api/task_status/<task_id> - Get status of a specific task" | tee -a $LOG_FILE

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Celery worker restart completed successfully" | tee -a $LOG_FILE