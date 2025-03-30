#!/bin/bash
# restart_services.sh
# Complete service restart script for SWATGenX application
# This will restart Redis, Celery workers, and the Flask application

set -e

# Define colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Define log file
LOG_FILE="/data/SWATGenXApp/codes/web_application/logs/service_restart.log"

# Logging function
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a "$LOG_FILE"
}

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   error "This script must be run as root"
   exit 1
fi

log "Starting service restart procedure"

# Detect Redis service name - could be redis.service or redis-server.service
REDIS_SERVICE="redis-server.service"
if ! systemctl list-unit-files | grep -q "$REDIS_SERVICE"; then
    if systemctl list-unit-files | grep -q "redis.service"; then
        REDIS_SERVICE="redis.service"
    else
        warning "Redis service not found as redis-server.service or redis.service"
        # Try to find any Redis service
        REDIS_SERVICE=$(systemctl list-unit-files | grep redis | head -n 1 | awk '{print $1}')
        if [[ -z "$REDIS_SERVICE" ]]; then
            error "Cannot find any Redis service"
            REDIS_SERVICE="redis-server.service" # Default fallback
        else
            log "Found Redis service: $REDIS_SERVICE"
        fi
    fi
fi

# Copy service files to systemd directory
log "Copying service files to systemd directory..."
cp /data/SWATGenXApp/codes/scripts/celery-worker.service /etc/systemd/system/
cp /data/SWATGenXApp/codes/scripts/flask-app.service /etc/systemd/system/
# Don't copy our Redis service file, as the system already has one

# Reload systemd to recognize the new service files
log "Reloading systemd daemon..."
systemctl daemon-reload

# Check Redis configuration
REDIS_CONF="/etc/redis/redis.conf"
if [ -f "$REDIS_CONF" ]; then
    log "Checking Redis configuration..."
    # Ensure Redis is listening on localhost
    if ! grep -q "^bind 127.0.0.1" "$REDIS_CONF"; then
        warning "Redis is not configured to bind to localhost. Adding bind directive."
        # Add or update bind directive
        if grep -q "^#.*bind" "$REDIS_CONF"; then
            sed -i 's/^#.*bind.*/bind 127.0.0.1/' "$REDIS_CONF"
        else
            echo "bind 127.0.0.1" >> "$REDIS_CONF"
        fi
    fi
    
    # Ensure Redis is not protected-mode (which can cause connection issues)
    if grep -q "^protected-mode yes" "$REDIS_CONF"; then
        warning "Redis is in protected mode. Disabling for local connections."
        sed -i 's/^protected-mode yes/protected-mode no/' "$REDIS_CONF"
    fi
    
    log "Redis configuration checked and updated if necessary."
else
    error "Redis configuration file not found at $REDIS_CONF"
    # Create a minimal Redis config if it doesn't exist
    log "Creating minimal Redis configuration..."
    mkdir -p /etc/redis
    cat > "$REDIS_CONF" << EOF
bind 127.0.0.1
protected-mode no
port 6379
dir /var/lib/redis
EOF
    log "Created minimal Redis configuration."
fi

# Stop existing services in reverse dependency order
log "Stopping Flask application service..."
systemctl stop flask-app.service 2>/dev/null || log "Flask app service was not running"

log "Stopping Celery worker service..."
systemctl stop celery-worker.service 2>/dev/null || log "Celery worker service was not running"

log "Stopping Redis service..."
systemctl stop $REDIS_SERVICE 2>/dev/null || log "Redis service was not running"

# Wait for processes to fully stop
log "Waiting for services to stop completely..."
sleep 5

# Check for any remaining Redis processes
log "Checking for remaining Redis processes..."
if pgrep -f "redis-server" > /dev/null; then
    warning "Redis processes still running. Attempting to kill..."
    pkill -f "redis-server" || log "Failed to kill Redis processes"
    sleep 2
fi

# Check for any remaining Celery processes
log "Checking for remaining Celery processes..."
if pgrep -f "celery" > /dev/null; then
    warning "Celery processes still running. Attempting to kill..."
    pkill -f "celery" || log "Failed to kill Celery processes"
    sleep 2
fi

# Check for any remaining gunicorn processes on port 5001
log "Checking for gunicorn processes on port 5001..."
if lsof -i :5001 > /dev/null 2>&1; then
    warning "Gunicorn process still using port 5001. Attempting to kill..."
    pkill -f "gunicorn" || log "Failed to kill gunicorn processes"
    sleep 2
fi

# Check for any remaining gunicorn processes on port 5050
log "Checking for gunicorn processes on port 5050..."
if lsof -i :5050 > /dev/null 2>&1; then
    warning "Gunicorn process still using port 5050. Attempting to kill..."
    pkill -f "gunicorn" || log "Failed to kill gunicorn processes"
    # Try using lsof to find and kill the process directly
    pid=$(lsof -t -i:5050 2>/dev/null)
    if [ ! -z "$pid" ]; then
        warning "Killing process directly using PID: $pid"
        kill -9 $pid
    fi
    sleep 2
fi

# Create required directories with correct permissions
log "Ensuring proper directories and permissions..."
mkdir -p /data/SWATGenXApp/codes/web_application/logs
mkdir -p /var/lib/redis
chown -R www-data:www-data /data/SWATGenXApp/codes/web_application/logs
# Only attempt to change Redis directory permissions if it exists and we have the right user
if [ -d "/var/lib/redis" ] && getent passwd redis >/dev/null; then
    chown -R redis:redis /var/lib/redis
    chmod 750 /var/lib/redis
fi

# Start services in dependency order
log "Starting Redis service..."
systemctl start $REDIS_SERVICE
sleep 3

# Verify Redis is running
if systemctl is-active --quiet $REDIS_SERVICE; then
    log "Redis service started successfully"
    
    # Test Redis connection directly
    if redis-cli ping > /dev/null; then
        log "Redis connection test successful"
    else
        error "Redis connection test failed. Please check Redis configuration."
    fi
else
    error "Failed to start Redis service"
    systemctl status $REDIS_SERVICE --no-pager
fi

# Start Celery worker
log "Starting Celery worker service..."
systemctl start celery-worker.service
sleep 3

# Verify Celery worker is running
if systemctl is-active --quiet celery-worker.service; then
    log "Celery worker service started successfully"
else
    error "Failed to start Celery worker service"
    systemctl status celery-worker.service --no-pager
fi

# Start Flask application
log "Starting Flask application service..."
systemctl start flask-app.service
sleep 3

# Verify Flask application is running
if systemctl is-active --quiet flask-app.service; then
    log "Flask application service started successfully"
else
    error "Failed to start Flask application service"
    systemctl status flask-app.service --no-pager
fi

# Enable services to start on boot
log "Enabling services to start on boot..."
# Use conditional enabling to avoid errors
if systemctl is-active --quiet $REDIS_SERVICE; then
    systemctl enable $REDIS_SERVICE 2>/dev/null || warning "Could not enable Redis service - it may be linked or controlled by another unit"
fi
systemctl enable celery-worker.service
systemctl enable flask-app.service

# Final status check
log "Checking all service statuses..."
echo -e "\n${GREEN}=== Redis Service Status ===${NC}"
systemctl status $REDIS_SERVICE --no-pager
echo -e "\n${GREEN}=== Celery Worker Service Status ===${NC}"
systemctl status celery-worker.service --no-pager
echo -e "\n${GREEN}=== Flask Application Service Status ===${NC}"
systemctl status flask-app.service --no-pager

# Output final status for log
if systemctl is-active --quiet $REDIS_SERVICE && \
   systemctl is-active --quiet celery-worker.service && \
   systemctl is-active --quiet flask-app.service; then
    log "All services restarted successfully!"
else
    error "One or more services failed to start. Please check the logs."
fi

# Print monitoring instructions
echo -e "\n${GREEN}=== Monitoring Instructions ===${NC}"
echo -e "To monitor Redis: ${YELLOW}redis-cli monitor${NC}"
echo -e "To check Celery worker log: ${YELLOW}tail -f /data/SWATGenXApp/codes/web_application/logs/celery-worker.log${NC}"
echo -e "To check Flask application log: ${YELLOW}tail -f /data/SWATGenXApp/codes/web_application/logs/flask-app.log${NC}"
echo -e "To view task status: ${YELLOW}curl -X GET http://localhost:5050/api/user_tasks${NC}"

log "Service restart procedure completed"