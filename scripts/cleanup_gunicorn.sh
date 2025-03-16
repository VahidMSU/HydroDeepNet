#!/bin/bash

# Cleanup script for Gunicorn processes
echo "Checking for gunicorn processes..."
GUNICORN_PROCESSES=$(ps aux | grep gunicorn | grep -v grep)
PROCESS_COUNT=$(echo "$GUNICORN_PROCESSES" | wc -l)
echo "Found $PROCESS_COUNT gunicorn processes"

# Print process details for debugging
if [ $PROCESS_COUNT -gt 0 ]; then
    echo "Process details:"
    echo "$GUNICORN_PROCESSES"
    
    # Check for zombie processes
    ZOMBIE_COUNT=$(echo "$GUNICORN_PROCESSES" | grep -c "Z")
    if [ $ZOMBIE_COUNT -gt 0 ]; then
        echo "WARNING: $ZOMBIE_COUNT zombie processes found!"
    fi
fi

# Stop the service first (more graceful)
echo "Stopping flask-app service..."
sudo systemctl stop flask-app.service

# Wait for service to stop
echo "Waiting for service to stop..."
sleep 5

# Check if processes still remain after service stop
REMAINING=$(ps aux | grep gunicorn | grep -v grep | wc -l)
if [ $REMAINING -gt 0 ]; then
    echo "Killing remaining $REMAINING processes..."
    # Kill all gunicorn processes
    sudo pkill -f gunicorn
    sleep 3
    
    # Check again and force kill if needed
    STILL_REMAINING=$(ps aux | grep gunicorn | grep -v grep | wc -l)
    if [ $STILL_REMAINING -gt 0 ]; then
        echo "Force killing $STILL_REMAINING stubborn processes..."
        sudo pkill -9 -f gunicorn
    fi
fi

# Check for parent processes that might be causing zombies
echo "Checking for potential zombie-causing parent processes..."
POTENTIAL_PARENTS=$(ps aux | grep -E "(gunicorn|flask)" | grep -v grep | grep -v cleanup_gunicorn)
if [ -n "$POTENTIAL_PARENTS" ]; then
    echo "Found processes that might be causing zombies:"
    echo "$POTENTIAL_PARENTS"
    
    # Kill these processes too
    echo "Killing potential parent processes..."
    sudo pkill -f flask
    sudo pkill -f "run.py"
    sleep 2
fi

# Reload systemd configuration
echo "Reloading systemd configuration..."
sudo systemctl daemon-reload

# Start the service
echo "Starting flask-app service..."
sudo systemctl start flask-app.service

# Check status
echo "Checking service status..."
sudo systemctl status flask-app.service

# Show running gunicorn processes after restart
echo "Current gunicorn processes:"
ps aux | grep gunicorn | grep -v grep

echo "Cleanup completed"
