#!/bin/bash

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Restarting Celery worker..."

# Stop the Celery worker service
sudo systemctl stop celery-worker
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stopped Celery worker service"

# Ensure Redis is running
sudo systemctl status redis || sudo systemctl start redis
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Ensured Redis is running"

# Force kill any existing Celery processes (in case they're stuck)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Killing any existing Celery processes"
sudo pkill -f 'celery worker' || true

# Start the service
sudo systemctl start celery-worker
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Started Celery worker service"

# Check if it's running
sleep 2
if sudo systemctl is-active --quiet celery-worker; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Celery worker is running"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ Celery worker failed to start"
    echo "Check logs with: sudo journalctl -u celery-worker -n 50"
fi

# Test Redis connection using redis-cli
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Testing Redis connection..."
redis-cli ping
if [ $? -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Redis is responding to pings"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ Redis is not responding to pings"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done!"
