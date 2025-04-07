#!/bin/bash
# Celery and Redis Queue Recovery Script for SWATGenX
# This script fixes corrupted tasks and restarts Celery workers after a "KeyError: 'properties'" crash
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "SCRIPT_DIR: $CURRENT_DIR"
source "${CURRENT_DIR}/../global_path.sh"


CELERY_SERVICES_DIR="${SCRIPT_DIR}/celery-tools-services"
SERVICES_DIR="${CELERY_SERVICES_DIR}/services"
UTILS_DIR="${CELERY_SERVICES_DIR}/utils"
LOG_FILE="${LOG_DIR}/celery_recovery_$(date +%Y%m%d_%H%M%S).log"

# Ensure the log directory exists
mkdir -p "${LOG_DIR}"

# Log function
log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a "${LOG_FILE}"
}

# Check if we're running as root
if [[ $EUID -ne 0 ]]; then
   log "ERROR: This script must be run as root for service restarts"
   exit 1
fi

# Header
log "===== SWATGenX Celery Recovery Script ====="
log "Starting recovery operations..."

# Check if Redis is running
log "Checking Redis status..."
if systemctl is-active --quiet redis.service; then
    log "Redis is running"
else
    log "Redis is not running, attempting to start it..."
    systemctl start redis.service
    sleep 2
    
    if systemctl is-active --quiet redis.service; then
        log "Redis successfully started"
    else
        log "ERROR: Failed to start Redis, aborting recovery"
        exit 1
    fi
fi

# Skip repair and cleanup steps as requested
log "Skipping Redis WRONGTYPE fixes and queue cleanup steps"

# Stop Celery workers
log "Stopping Celery workers..."
systemctl stop celery-worker.service
sleep 2

# Check if any celery processes are still running and kill them
CELERY_PIDS=$(pgrep -f "celery worker")
if [ ! -z "$CELERY_PIDS" ]; then
    log "Celery processes still running, sending SIGTERM..."
    kill $CELERY_PIDS 2>/dev/null
    sleep 3
    
    # Check again and use SIGKILL if needed
    CELERY_PIDS=$(pgrep -f "celery worker")
    if [ ! -z "$CELERY_PIDS" ]; then
        log "Celery processes still running, sending SIGKILL..."
        kill -9 $CELERY_PIDS 2>/dev/null
        sleep 1
    fi
fi

# Start Celery workers
log "Starting Celery workers..."
systemctl start celery-worker.service
sleep 3

# Check if workers started successfully
if systemctl is-active --quiet celery-worker.service; then
    log "Celery workers started successfully"
else
    log "WARNING: Celery worker service failed to start, checking logs..."
    journalctl -u celery-worker.service -n 20 | tee -a "${LOG_FILE}"
fi

# Check queue status
log "Checking Celery queue status..."
${PYTHON_ENV} ${UTILS_DIR}/monitor_celery_status.py --queues --clean 2>&1 | tee -a "${LOG_FILE}"

log "Recovery process completed"
log "Check the full log at: ${LOG_FILE}"
