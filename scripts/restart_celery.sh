#!/bin/bash
# Restart Celery services for SWATGenX application
# This script ensures proper configuration of the Celery workers

# Exit on error
set -e

# Define log file
LOG_FILE="/data/SWATGenXApp/codes/web_application/logs/celery_restart.log"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   log "This script must be run as root"
   exit 1
fi

log "Starting Celery services restart procedure"

# Ensure log directory exists
mkdir -p /data/SWATGenXApp/codes/web_application/logs
chmod 755 /data/SWATGenXApp/codes/web_application/logs

# Copy the service files to systemd directory
log "Copying service files to systemd directory..."
cp /data/SWATGenXApp/codes/scripts/celery-worker.service /etc/systemd/system/
cp /data/SWATGenXApp/codes/scripts/celery-multi-worker.service /etc/systemd/system/

# Reload systemd to recognize the new service files
log "Reloading systemd..."
systemctl daemon-reload

# Check Redis status first
log "Checking Redis status..."
if systemctl is-active --quiet redis-server.service; then
    log "Redis is running"
elif systemctl is-active --quiet redis.service; then
    log "Redis is running (as redis.service)"
else
    log "Redis is not running. Attempting to start..."
    systemctl start redis-server.service || systemctl start redis.service || log "Failed to start Redis, but continuing anyway"
fi

# Stop any existing Celery services
log "Stopping existing Celery services..."
systemctl stop celery-worker.service 2>/dev/null || log "celery-worker service was not running"
systemctl stop celery-multi-worker.service 2>/dev/null || log "celery-multi-worker service was not running"

# Wait for processes to fully stop
log "Waiting for Celery processes to stop..."
sleep 5

# Check for any remaining Celery processes and kill them
log "Checking for remaining Celery processes..."
if pgrep -f "celery" > /dev/null; then
    log "Found remaining Celery processes. Sending TERM signal..."
    pkill -TERM -f "celery" || log "Failed to terminate Celery processes"
    sleep 3
    
    # Check again and use KILL if necessary
    if pgrep -f "celery" > /dev/null; then
        log "Processes still running. Sending KILL signal..."
        pkill -KILL -f "celery" || log "Failed to kill Celery processes"
        sleep 2
    fi
else
    log "No Celery processes found running"
fi

# Clean up any temporary files in case of previous crashes
log "Cleaning up any temporary files..."
find /tmp -name "celery-*" -type f -mtime +1 -delete 2>/dev/null || true

# Create redis directory with proper permissions if needed
if [ ! -d "/var/lib/redis" ]; then
    log "Creating Redis data directory..."
    mkdir -p /var/lib/redis
    chown redis:redis /var/lib/redis
    chmod 770 /var/lib/redis
fi

# Start the services
log "Starting Celery multi-worker service..."
systemctl start celery-multi-worker.service
sleep 3

# Check status
if systemctl is-active --quiet celery-multi-worker.service; then
    log "celery-multi-worker service started successfully"
else
    log "WARNING: celery-multi-worker service failed to start properly. Checking logs..."
    tail -n 20 /data/SWATGenXApp/codes/web_application/logs/celery-multi-worker-error.log | tee -a "$LOG_FILE"
fi

# Enable the services to start on boot
log "Enabling services to start on boot..."
systemctl enable celery-multi-worker.service

# Check worker status
log "Checking service status..."
systemctl status celery-multi-worker.service --no-pager

# Check for active workers
log "Verifying Celery workers are running..."
sleep 2  # Give workers time to register

# Use Celery inspect to check active workers
if /data/SWATGenXApp/codes/.venv/bin/celery -A celery_app inspect ping 2>/dev/null; then
    log "Celery workers are responding to ping"
    
    # Show active queues
    log "Active queues:"
    /data/SWATGenXApp/codes/.venv/bin/celery -A celery_app inspect active_queues 2>/dev/null || log "Could not get queue information"
else
    log "WARNING: Celery workers are not responding to ping. Check the worker logs."
fi

log "Celery services restart procedure completed"

# Output instructions for monitoring
log "To monitor logs:"
log "  - Celery multi-worker logs: tail -f /data/SWATGenXApp/codes/web_application/logs/celery-multi-worker.log"
log "  - Check for task status: curl -X GET http://localhost:5050/api/user_tasks"

exit 0
