#!/bin/bash

LOG_FILE="/data/SWATGenXApp/codes/web_application/logs/advanced_socketio_fix.log"
mkdir -p "$(dirname "$LOG_FILE")"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=== Starting Advanced Socket.IO Diagnostics and Fix ==="

# Check what's actually running on port 5050
log "Checking what's running on port 5050..."
if sudo lsof -i :5050; then
  PROCESS_INFO=$(sudo lsof -i :5050)
  log "Process info: $PROCESS_INFO"
else
  log "No process appears to be listening on port 5050 according to lsof"
fi

# Try netstat as an alternative
log "Checking netstat for port 5050..."
if netstat -tuln | grep :5050; then
  NETSTAT_INFO=$(netstat -tuln | grep :5050)
  log "Netstat info: $NETSTAT_INFO"
else
  log "No port 5050 listing found in netstat"
fi

# Check socket connectivity with curl
log "Testing socket with curl..."
curl -v http://localhost:5050/ > /dev/null 2>> "$LOG_FILE"

# Install necessary packages
log "Installing necessary packages..."
pip install -U eventlet flask-socketio python-socketio gunicorn requests

# Kill any existing process using the port
log "Attempting to free port 5050..."
sudo fuser -k 5050/tcp || log "No process found using port 5050 by fuser"
sleep 2

# Restart networking services
log "Restarting networking services..."
sudo systemctl restart apache2

# Create a very simple test server to verify Socket.IO connectivity
TEMP_SERVER_FILE=$(mktemp)
log "Creating temporary test server at $TEMP_SERVER_FILE"

cat > "$TEMP_SERVER_FILE" << 'EOL'
import eventlet
eventlet.monkey_patch()

from flask import Flask
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def index():
    return 'Socket.IO Simple Test Server'

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    socketio.emit('message', {'data': 'Connected'})

@socketio.on('message')
def handle_message(data):
    print(f'Received: {data}')
    socketio.emit('message', {'echo': data})

if __name__ == '__main__':
    print("Starting minimal Socket.IO server on port 5050")
    socketio.run(app, host='0.0.0.0', port=5050)
EOL

# Start the simple test server
log "Starting simple test server..."
SIMPLE_SERVER_LOG=$(mktemp)
log "Simple server log will be at $SIMPLE_SERVER_LOG"
python3 "$TEMP_SERVER_FILE" > "$SIMPLE_SERVER_LOG" 2>&1 &
SIMPLE_SERVER_PID=$!
log "Simple server started with PID $SIMPLE_SERVER_PID"

# Wait for the server to start
sleep 3

# Check if the server is running
if ps -p $SIMPLE_SERVER_PID > /dev/null; then
  log "✅ Simple test server is running"
else
  log "❌ Simple test server failed to start"
  cat "$SIMPLE_SERVER_LOG" >> "$LOG_FILE"
  log "Testing port binding directly with Python socket..."
  
  # Test if we can bind to the port directly with a Python socket
  SOCKET_TEST_FILE=$(mktemp)
  cat > "$SOCKET_TEST_FILE" << 'EOL'
import socket
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('0.0.0.0', 5050))
    print("Successfully bound to port 5050")
    sock.close()
except Exception as e:
    print(f"Error binding to port: {e}")
EOL

  python3 "$SOCKET_TEST_FILE" 2>&1 | tee -a "$LOG_FILE"
fi

# Try a direct HTTP connection to the test server
log "Testing HTTP connection to test server..."
if curl -m 5 -v http://localhost:5050/ 2>&1 | tee -a "$LOG_FILE"; then
  log "✅ HTTP connection successful"
else
  log "❌ HTTP connection failed"
fi

# Try a WebSocket connection with wscat (if available)
if command -v wscat &> /dev/null; then
  log "Testing WebSocket connection with wscat..."
  echo '{"type": "message", "data": "test"}' | wscat -c ws://localhost:5050/socket.io/?transport=websocket -t 5 2>&1 | tee -a "$LOG_FILE" || log "❌ WebSocket connection failed"
else
  log "wscat not available, skipping WebSocket command-line test"
fi

# Create a simple Python WebSocket client
WS_CLIENT_FILE=$(mktemp)
log "Creating WebSocket client at $WS_CLIENT_FILE"

cat > "$WS_CLIENT_FILE" << 'EOL'
import socketio
import time
import sys

print("Testing Socket.IO connection...")
sio = socketio.Client(logger=True)

@sio.event
def connect():
    print("Connected!")
    sio.emit('message', {'data': 'Test message'})

@sio.event
def connect_error(error):
    print(f"Connection error: {error}")

@sio.event
def message(data):
    print(f"Received message: {data}")

try:
    print("Attempting polling connection...")
    sio.connect('http://localhost:5050', transports=['polling'], wait_timeout=5)
    time.sleep(2)
    sio.disconnect()
    
    print("Attempting WebSocket connection...")
    sio = socketio.Client(logger=True)
    sio.connect('http://localhost:5050', transports=['websocket'], wait_timeout=5)
    time.sleep(2)
    sio.disconnect()
    
    print("All connection tests passed")
    sys.exit(0)
except Exception as e:
    print(f"Connection failed: {e}")
    sys.exit(1)
EOL

log "Running WebSocket client test..."
if python3 "$WS_CLIENT_FILE" 2>&1 | tee -a "$LOG_FILE"; then
  log "✅ WebSocket client test passed"
else
  log "❌ WebSocket client test failed"
fi

# Test Apache proxy configuration
log "Testing if Apache correctly proxies Socket.IO requests..."
if curl -v https://ciwre-bae.campusad.msu.edu/socket.io/ 2>&1 | tee -a "$LOG_FILE"; then
  log "✅ Apache proxy test passed"
else
  log "❌ Apache proxy test failed"
fi

# Clean up
log "Cleaning up..."
if ps -p $SIMPLE_SERVER_PID > /dev/null; then
  kill $SIMPLE_SERVER_PID
  log "Stopped simple test server"
fi
rm -f "$TEMP_SERVER_FILE" "$WS_CLIENT_FILE" "$SIMPLE_SERVER_LOG"

# Output help information
log "=== Socket.IO Diagnostic Results ==="
log "Check $LOG_FILE for detailed results"
log ""
log "Suggestions:"
log "1. If port 5050 is showing as in use but no process is found, try rebooting the server"
log "2. Try using a different port in run.py (e.g. 5051 or 8080)"
log "3. Check the Apache configuration to ensure WebSocket proxy directives are correct:"
log "   - Make sure mod_proxy_wstunnel is enabled"
log "   - Make sure the RewriteRule for WebSocket is correct"
log ""
log "To restart the application with the fixes applied, run:"
log "bash /data/SWATGenXApp/codes/scripts/restart_services.sh"
