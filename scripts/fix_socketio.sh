#!/bin/bash

LOG_FILE="/data/SWATGenXApp/codes/web_application/logs/socketio_fix.log"
mkdir -p "$(dirname "$LOG_FILE")"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting Socket.IO configuration check and fix..."

# Check for required commands
for cmd in python3 pip sudo a2enmod systemctl; do
  if ! command -v $cmd &> /dev/null; then
    log "❌ Required command not found: $cmd"
    echo "Error: Required command not found: $cmd"
    exit 1
  fi
done

# Check if eventlet is installed
log "Checking for eventlet..."
if python3 -c "import eventlet" 2>/dev/null; then
  log "✅ eventlet is installed"
else
  log "❌ eventlet is not installed, installing..."
  pip install eventlet
  if [ $? -ne 0 ]; then
    log "Failed to install eventlet"
    exit 1
  fi
fi

# Check if python-socketio is installed
log "Checking for python-socketio..."
if python3 -c "import socketio" 2>/dev/null; then
  log "✅ python-socketio is installed"
else
  log "❌ python-socketio is not installed, installing..."
  pip install python-socketio
  if [ $? -ne 0 ]; then
    log "Failed to install python-socketio"
    exit 1
  fi
fi

# Enable required Apache modules
log "Enabling required Apache modules..."
sudo a2enmod proxy_wstunnel headers rewrite
sudo systemctl restart apache2
log "✅ Apache modules enabled"

# Check if anything is already bound to port 5050
log "Checking for processes bound to port 5050..."
if lsof -i :5050 > /dev/null; then
  PORT_PROCESSES=$(lsof -i :5050 | grep -v PID | awk '{print $2}')
  log "Found processes using port 5050: $PORT_PROCESSES"
  log "Attempting to kill processes..."
  
  for pid in $PORT_PROCESSES; do
    log "Killing process $pid..."
    sudo kill -9 $pid
  done
  
  sleep 2
  
  if lsof -i :5050 > /dev/null; then
    log "❌ Failed to free port 5050"
    exit 1
  else
    log "✅ Port 5050 freed successfully"
  fi
else
  log "✅ Port 5050 is available"
fi

# Restart Flask app with SocketIO support
log "Starting Flask application with SocketIO support..."
cd /data/SWATGenXApp/codes/web_application
nohup python3 run.py > /data/SWATGenXApp/codes/web_application/logs/flask_app.log 2>&1 &
APP_PID=$!
log "Flask app started with PID: $APP_PID"

# Wait for app to start
log "Waiting for app to start..."
sleep 5

# Test connection
log "Testing Socket.IO connection..."
if lsof -i :5050 > /dev/null; then
  log "✅ Something is listening on port 5050"
  
  # Run the simple test script
  if [[ -f /data/SWATGenXApp/codes/web_application/simple_socketio_test.py ]]; then
    log "Running Socket.IO test script..."
    python3 /data/SWATGenXApp/codes/web_application/simple_socketio_test.py
    if [[ $? -eq 0 ]]; then
      log "✅ Socket.IO test script passed"
    else
      log "❌ Socket.IO test script failed"
    fi
  else
    log "❌ Test script not found"
  fi
else
  log "❌ Nothing is listening on port 5050"
fi

log "Socket.IO configuration check and fix completed."
