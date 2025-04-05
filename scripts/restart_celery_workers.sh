#!/bin/bash

# Log file
LOG_FILE="/data/SWATGenXApp/codes/web_application/logs/celery_restart.log"

# Application directory
APP_DIR="/data/SWATGenXApp/codes/web_application"

# Number of worker processes to start - use system CPU count with a margin
CPU_COUNT=$(nproc)
WORKER_COUNT=${MAX_WORKER_COUNT:-$((CPU_COUNT * 8 / 10))}  # Use 80% of CPU cores for workers by default
if [[ "$WORKER_COUNT" -lt 5 ]]; then
    WORKER_COUNT=5  # Ensure at least 5 workers
fi

# Calculate optimal worker concurrency based on system resources
TOTAL_MEMORY_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
TOTAL_MEMORY_GB=$((TOTAL_MEMORY_KB / 1024 / 1024))
MEMORY_PER_WORKER=$((TOTAL_MEMORY_GB / WORKER_COUNT / 2))  # Use half of available memory per worker
if [[ "$MEMORY_PER_WORKER" -lt 1 ]]; then
    MEMORY_PER_WORKER=1  # Ensure at least 1GB per worker
fi

# Concurrency per worker - targeting ~200 total processes
CONCURRENCY=${WORKER_CONCURRENCY:-$((200 / WORKER_COUNT + 1))}
if [[ "$CONCURRENCY" -gt 16 ]]; then
    CONCURRENCY=16  # Cap concurrency at 16 per worker for stability
fi

# Venv path
VENV_PATH="/data/SWATGenXApp/codes/.venv"

# Log function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

# Make sure log directory exists
mkdir -p "$(dirname $LOG_FILE)"

log "Starting Celery worker restart procedure"
log "System resources: $CPU_COUNT CPUs, ${TOTAL_MEMORY_GB}GB RAM"
log "Deploying $WORKER_COUNT workers with concurrency $CONCURRENCY (total processes: $((WORKER_COUNT * CONCURRENCY)))"
log "Memory allocated per worker: ${MEMORY_PER_WORKER}GB"

# Activate virtual environment
source $VENV_PATH/bin/activate
cd $APP_DIR

# Stop all existing celery processes
log "Stopping existing Celery workers..."
pkill -f "celery worker" || true
sleep 5

# Make sure all processes are stopped
if pgrep -f "celery worker" > /dev/null; then
    log "Some workers still running, forcing termination..."
    pkill -9 -f "celery worker" || true
    sleep 2
fi

# Check if Redis is running
log "Checking Redis status..."
if ! pgrep -x "redis-server" > /dev/null; then
    log "Redis is not running, starting it..."
    systemctl start redis || log "Failed to start Redis"
    sleep 2
fi

# Clear any existing pid files
rm -f /tmp/celery-*.pid 2>/dev/null

# Distribute workers across queues based on workload
MODEL_CREATION_WORKERS=$((WORKER_COUNT * 7 / 10))  # 70% for model creation
DEFAULT_WORKERS=$((WORKER_COUNT - MODEL_CREATION_WORKERS))  # Remaining for default queue

log "Distributing workers: $MODEL_CREATION_WORKERS for model_creation, $DEFAULT_WORKERS for default queue"

# Start model creation workers with high concurrency
log "Starting $MODEL_CREATION_WORKERS model creation workers..."
for i in $(seq 1 $MODEL_CREATION_WORKERS); do
    log "Starting model creation worker $i"
    celery -A celery_worker worker \
        --loglevel=info \
        --concurrency=$CONCURRENCY \
        --hostname=model${i}@%h \
        --queues=model_creation \
        --pool=prefork \
        --max-tasks-per-child=20 \
        --max-memory-per-child=$((MEMORY_PER_WORKER * 1024 * 1024)) \
        --logfile="/data/SWATGenXApp/codes/web_application/logs/celery-model-worker-${i}.log" \
        --detach
    
    # Small delay to prevent overwhelming the system during startup
    sleep 0.5
done

# Start default queue workers
log "Starting $DEFAULT_WORKERS default queue workers..."
for i in $(seq 1 $DEFAULT_WORKERS); do
    log "Starting default queue worker $i"
    celery -A celery_worker worker \
        --loglevel=info \
        --concurrency=$CONCURRENCY \
        --hostname=default${i}@%h \
        --queues=celery \
        --pool=prefork \
        --max-tasks-per-child=30 \
        --max-memory-per-child=$((MEMORY_PER_WORKER * 1024 * 1024)) \
        --logfile="/data/SWATGenXApp/codes/web_application/logs/celery-default-worker-${i}.log" \
        --detach
    
    # Small delay to prevent overwhelming the system during startup
    sleep 0.5
done

# Wait for workers to start
sleep 5

# Verify workers are running
WORKER_COUNT_RUNNING=$(pgrep -f "celery worker" | wc -l)
log "Celery workers running: $WORKER_COUNT_RUNNING"

# Expected process count (with a wider acceptable range)
EXPECTED_PROCESSES_MIN=$((WORKER_COUNT * CONCURRENCY / 2))
EXPECTED_PROCESSES_MAX=$((WORKER_COUNT * CONCURRENCY * 2))

if [ $WORKER_COUNT_RUNNING -lt $EXPECTED_PROCESSES_MIN ]; then
    log "WARNING: Only $WORKER_COUNT_RUNNING worker processes started, expected at least $EXPECTED_PROCESSES_MIN"
    log "Check worker logs for errors"
elif [ $WORKER_COUNT_RUNNING -gt $EXPECTED_PROCESSES_MAX ]; then
    log "WARNING: $WORKER_COUNT_RUNNING worker processes started, which is more than expected maximum $EXPECTED_PROCESSES_MAX"
    log "This may indicate orphaned processes or incorrect counting"
else
    log "Worker count looks good: $WORKER_COUNT_RUNNING processes running"
fi

# Set proper permissions
chmod 755 "/data/SWATGenXApp/codes/web_application/logs"
chmod 644 "/data/SWATGenXApp/codes/web_application/logs"/*.log

# Show active workers via celery inspect
if which celery > /dev/null; then
    log "Active workers according to Celery:"
    celery -A celery_worker inspect active 2>/dev/null || log "Could not get active workers list"
fi

log "Celery worker restart completed"