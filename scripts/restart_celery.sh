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

# Copy the service files to systemd directory
log "Copying service files to systemd directory..."
cp /data/SWATGenXApp/codes/scripts/celery-worker.service /etc/systemd/system/
cp /data/SWATGenXApp/codes/scripts/celery-multi-worker.service /etc/systemd/system/

# Reload systemd to recognize the new service files
log "Reloading systemd..."
systemctl daemon-reload

# Stop any existing Celery services
log "Stopping existing Celery services..."
systemctl stop celery-worker.service 2>/dev/null || log "celery-worker service was not running"
systemctl stop celery-multi-worker.service 2>/dev/null || log "celery-multi-worker service was not running"

# Wait for processes to fully stop
log "Waiting for Celery processes to stop..."
sleep 5

# Check for any remaining Celery processes and kill them
log "Checking for remaining Celery processes..."
pkill -f "celery" || log "No Celery processes found running"

# Wait for processes to fully terminate
sleep 2

# Start the services
log "Starting Celery services..."
systemctl start celery-multi-worker.service
log "celery-multi-worker service started"

# Enable the services to start on boot
log "Enabling services to start on boot..."
systemctl enable celery-multi-worker.service

# Check status
log "Checking service status..."
systemctl status celery-multi-worker.service --no-pager

log "Celery services restart procedure completed"

# Output instructions for monitoring
log "To monitor logs:"
log "  - Celery multi-worker logs: tail -f /data/SWATGenXApp/codes/web_application/logs/celery-multi-worker.log"

exit 0
