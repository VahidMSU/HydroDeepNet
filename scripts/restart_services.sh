#!/bin/bash

# Constants
LOG_FILE="/data/SWATGenXApp/codes/web_application/logs/service_restart.log"
APP_DIR="/data/SWATGenXApp/codes"
SCRIPT_DIR="$APP_DIR/scripts"
WEB_DIR="$APP_DIR/web_application"
VENV_PATH="$APP_DIR/.venv"
FRONTEND_DIR="$WEB_DIR/frontend"

# Create log directory
mkdir -p "$(dirname "$LOG_FILE")"

# Logging function
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Check for required commands
for cmd in systemctl netstat curl apache2ctl npm; do
  if ! command -v $cmd &> /dev/null; then
    log "❌ Required command not found: $cmd"
    echo "Please install $cmd and try again"
    exit 1
  fi
done

# Function to check service status
check_service() {
  if systemctl is-active --quiet "$1"; then
    log "✅ $2 is running"
  else
    log "❌ $2 is not running"
  fi
}

# Function to restart service
restart_service() {
  log "Restarting $2..."
  if ! sudo systemctl restart "$1"; then
    log "⚠️ Failed to restart $2"
    return 1
  fi
  return 0
}

# Copy configuration files
log "Setting up daemon services..."
cd "$SCRIPT_DIR" || exit 1
sudo cp ./ciwre-bae.conf /etc/apache2/sites-available/
sudo cp ./000-default.conf /etc/apache2/sites-available/
sudo cp ./celery-worker.service /etc/systemd/system/

# Reload systemd
log "Reloading systemd configuration..."
sudo systemctl daemon-reload

# Enable Apache modules
log "Enabling required Apache modules..."
sudo a2enmod proxy proxy_http proxy_wstunnel headers rewrite ssl


# Verify Apache configuration
log "Checking Apache configuration..."
if sudo apache2ctl configtest; then
  log "✅ Apache configuration is valid"
else
  log "❌ Apache configuration test failed - attempting to fix permissions"
fi

# Handle ports 5050 and 3000
log "Checking for processes on ports 5050.."
if [ -f "$SCRIPT_DIR/kill_port_process.sh" ]; then
  bash "$SCRIPT_DIR/kill_port_process.sh" 5050 || sudo fuser -k 5050/tcp
  sleep 2
fi

# Restart Apache
restart_service "apache2" "Apache"
restart_service "flask-app" "Flask app"
restart_service "redis-server" "Redis"
restart_service "celery-worker" "Celery worker"


# Check services status
log "Checking service status..."
check_service "apache2" "Apache"
check_service "redis-server" "Redis"
check_service "celery-worker" "Celery worker"
check_service "flask-app" "Flask app"

# Check ports 5050 and 3000
for PORT in 5050 3000; do
  if netstat -tuln | grep ":$PORT " >/dev/null; then
    log "✅ Service is running on port $PORT"
  else
    log "❌ Nothing is running on port $PORT"
  fi
done