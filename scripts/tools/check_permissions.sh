#!/bin/bash

# Log file
LOG_FILE="/data/SWATGenXApp/codes/web_application/logs/permission_check.log"

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Function to log messages
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Clear log file
> "$LOG_FILE"

log "Starting permission check..."

# Check Apache/www-data permissions
APACHE_USER="www-data"
log "Checking as user: $APACHE_USER"

# Check important directories
DIRS_TO_CHECK=(
  "/data/SWATGenXApp/codes/web_application"
  "/data/SWATGenXApp/GenXAppData"
  "/data/SWATGenXApp/Users"
  "/data/SWATGenXApp/codes/SWATGenX"
  "/data/SWATGenXApp/codes/GeoReporter"
  "/data/SWATGenXApp/codes/web_application/logs"
  "/data/SWATGenXApp/codes/web_application/frontend/build"
)

for dir in "${DIRS_TO_CHECK[@]}"; do
  log "Checking directory: $dir"
  
  if [ ! -d "$dir" ]; then
    log "❌ ERROR: Directory does not exist: $dir"
    continue
  fi
  
  # Check if www-data can read the directory
  if sudo -u $APACHE_USER test -r "$dir"; then
    log "✅ $APACHE_USER can read: $dir"
  else
    log "❌ ERROR: $APACHE_USER cannot read: $dir"
    # Try to fix permissions
    log "Attempting to fix read permissions..."
    sudo chmod -R +r "$dir"
    sudo chown -R $APACHE_USER:$APACHE_USER "$dir"
  fi
  
  # Check if www-data can execute (traverse) the directory
  if sudo -u $APACHE_USER test -x "$dir"; then
    log "✅ $APACHE_USER can traverse: $dir"
  else
    log "❌ ERROR: $APACHE_USER cannot traverse: $dir"
    # Try to fix permissions
    log "Attempting to fix execute permissions..."
    sudo chmod -R +x "$dir"
  fi
  
  # For write permissions, only check certain directories
  if [[ "$dir" == *"/logs"* ]] || [[ "$dir" == *"/Users"* ]]; then
    if sudo -u $APACHE_USER test -w "$dir"; then
      log "✅ $APACHE_USER can write to: $dir"
    else
      log "❌ ERROR: $APACHE_USER cannot write to: $dir"
      # Try to fix permissions
      log "Attempting to fix write permissions..."
      sudo chmod -R +w "$dir"
      sudo chown -R $APACHE_USER:$APACHE_USER "$dir"
    fi
  fi
done

# Check Apache config
log "Checking Apache configuration..."
if sudo apachectl configtest; then
  log "✅ Apache configuration is valid"
else
  log "❌ ERROR: Apache configuration test failed"
fi

# Check if Apache is running
log "Checking if Apache is running..."
if command -v systemctl &>/dev/null; then
  if systemctl is-active --quiet apache2; then
    log "✅ Apache is running"
  else
    log "❌ ERROR: Apache is not running"
    # Try to start Apache
    log "Attempting to start Apache..."
    sudo systemctl start apache2
  fi
else
  # Fallback to service command if systemctl is not available
  log "systemctl not found, using service command instead"
  if service apache2 status >/dev/null 2>&1; then
    log "✅ Apache is running"
  else
    log "❌ ERROR: Apache is not running"
    log "Attempting to start Apache..."
    sudo service apache2 start
  fi
fi

# Check if Flask app is running
log "Checking if Flask app is running on port 5050..."
if command -v netstat &>/dev/null; then
  if netstat -tuln | grep -q ":5050 "; then
    log "✅ Something is running on port 5050"
  else
    log "❌ ERROR: Nothing is running on port 5050"
  fi
elif command -v lsof &>/dev/null; then
  if lsof -i :5050 >/dev/null 2>&1; then
    log "✅ Something is running on port 5050"
  else 
    log "❌ ERROR: Nothing is running on port 5050"
  fi
else
  log "⚠️ WARNING: Cannot check if Flask app is running (netstat and lsof not available)"
fi

# Check Redis is running
log "Checking if Redis is running..."
if systemctl is-active --quiet redis-server; then
  log "✅ Redis is running"
else
  log "❌ ERROR: Redis is not running"
  # Try to start Redis
  log "Attempting to start Redis..."
  sudo systemctl start redis-server
fi

# Check Celery is running
log "Checking if Celery is running..."
if systemctl is-active --quiet celery-worker; then
  log "✅ Celery worker is running"
else
  log "❌ ERROR: Celery worker is not running"
  # Try to start Celery
  log "Attempting to start Celery worker..."
  sudo systemctl start celery-worker
fi

log "Permission check completed. See full results in $LOG_FILE"
