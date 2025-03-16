#!/bin/bash

LOG_FILE="/data/SWATGenXApp/codes/web_application/logs/socketio_reset.log"
mkdir -p "$(dirname "$LOG_FILE")"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting Socket.IO environment reset..."

# Make sure required packages are installed
log "Installing required packages..."
pip install flask-socketio python-socketio eventlet requests

# Check if eventlet is working properly
log "Testing eventlet..."
python3 -c 'import eventlet; eventlet.listen(("localhost", 0))'
if [ $? -eq 0 ]; then
  log "✅ Eventlet test passed"
else
  log "❌ Eventlet test failed"
fi

# Kill any processes using port 5050
log "Killing processes on port 5050..."
lsof -ti:5050 | xargs -r kill -9
sleep 2

# Enable required Apache modules
log "Enabling required Apache modules..."
sudo a2enmod proxy proxy_http proxy_wstunnel headers rewrite
sudo systemctl restart apache2

# Start the test server
log "Starting test Socket.IO server..."
cd /data/SWATGenXApp/codes/web_application
nohup python3 test_socketio_server.py > /data/SWATGenXApp/codes/web_application/logs/test_socketio.log 2>&1 &
TEST_PID=$!
log "Test server started with PID: $TEST_PID"

# Wait for server to start
log "Waiting for server to start..."
sleep 5

# Check if server is running
if ps -p $TEST_PID > /dev/null; then
  log "✅ Test server is running"
else
  log "❌ Test server failed to start"
  exit 1
fi

# Run diagnostics
log "Running Socket.IO diagnostics..."
python3 /data/SWATGenXApp/codes/web_application/socketio_diagnostics.py

# Provide next steps
log "Socket.IO reset and testing completed."
log "If tests passed, you can stop the test server with: kill $TEST_PID"
log "If tests failed, check the logs for more information:"
log "- Test server log: /data/SWATGenXApp/codes/web_application/logs/test_socketio.log"
log "- Socket.IO diagnostics log: /data/SWATGenXApp/codes/web_application/logs/socketio_reset.log"
