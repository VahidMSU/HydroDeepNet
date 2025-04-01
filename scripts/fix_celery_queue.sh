#!/bin/bash
# Script to fix Celery queue issues and restart services
# Run with sudo: sudo bash fix_celery_queue.sh

echo "===== Celery Queue Fix Script ====="
echo "Run Date: $(date)"
echo

# Set environment variables
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/data/SWATGenXApp/codes/.venv/bin"
export FLASK_ENV="production"
export PYTHONPATH="/data/SWATGenXApp/codes:/data/SWATGenXApp/codes/web_application"

# Check queue status first
echo "Checking for tasks in the model_creation queue..."
MODEL_QUEUE=$(redis-cli -h 127.0.0.1 -p 6379 llen model_creation 2>/dev/null)
if [ -z "$MODEL_QUEUE" ] || [ "$MODEL_QUEUE" == "(nil)" ]; then
    echo "No model_creation queue found. This may be normal if no tasks have been created yet."
else
    echo "Found $MODEL_QUEUE tasks in model_creation queue"
    
    # Option to save stuck tasks
    read -p "Would you like to save stuck tasks before clearing the queue? (y/n): " SAVE_TASKS
    if [[ "$SAVE_TASKS" == "y" ]]; then
        echo "Saving tasks to /tmp/stuck_celery_tasks.json..."
        redis-cli -h 127.0.0.1 -p 6379 lrange model_creation 0 -1 > /tmp/stuck_celery_tasks.json
        echo "Tasks saved to /tmp/stuck_celery_tasks.json"
    fi
    
    # Clear the queue
    read -p "Clear the model_creation queue? (y/n): " CLEAR_QUEUE
    if [[ "$CLEAR_QUEUE" == "y" ]]; then
        echo "Clearing model_creation queue..."
        redis-cli -h 127.0.0.1 -p 6379 del model_creation
        echo "Queue cleared"
    fi
fi

# Stop Celery worker
echo "Stopping Celery worker service..."
systemctl stop celery-worker
echo "Waiting for Celery processes to terminate..."
sleep 5

# Check for any remaining Celery processes
CELERY_PROCS=$(ps aux | grep "[c]elery worker" | wc -l)
if [ "$CELERY_PROCS" -gt 0 ]; then
    echo "Found $CELERY_PROCS Celery processes still running. Forcing termination..."
    pkill -f "celery worker"
    sleep 3
    
    # Check again
    CELERY_PROCS=$(ps aux | grep "[c]elery worker" | wc -l)
    if [ "$CELERY_PROCS" -gt 0 ]; then
        echo "WARNING: Celery processes still running. Using SIGKILL..."
        pkill -9 -f "celery worker"
        sleep 2
    fi
fi

# Restart Redis for a clean state
echo "Restarting Redis server..."
systemctl restart redis-server
sleep 3

# Verify Redis is running
if systemctl is-active --quiet redis-server; then
    echo "✅ Redis restarted successfully"
else
    echo "❌ Failed to restart Redis! Attempting to start it..."
    systemctl start redis-server
    sleep 2
    
    if ! systemctl is-active --quiet redis-server; then
        echo "❌ CRITICAL: Could not start Redis. Fix Redis before continuing."
        exit 1
    fi
fi

# Restart Celery worker
echo "Starting Celery worker service..."
systemctl start celery-worker
sleep 5

# Verify Celery worker is running
if systemctl is-active --quiet celery-worker; then
    echo "✅ Celery worker service started successfully"
    
    # Check for worker processes
    CELERY_PROCS=$(ps aux | grep "[c]elery worker" | wc -l)
    if [ "$CELERY_PROCS" -gt 0 ]; then
        echo "✅ Found $CELERY_PROCS Celery worker processes"
        ps aux | grep "[c]elery worker" | head -2
    else
        echo "⚠️ No Celery worker processes found despite service being active"
        echo "Check logs for errors:"
        tail -20 /data/SWATGenXApp/codes/web_application/logs/celery-worker-error.log
    fi
else
    echo "❌ Failed to start Celery worker service"
    echo "Checking error logs:"
    tail -20 /data/SWATGenXApp/codes/web_application/logs/celery-worker-error.log
    journalctl -u celery-worker -n 20 --no-pager
fi

# Check queue status again
echo "Checking if model_creation queue was re-created..."
QUEUE_EXISTS=$(redis-cli -h 127.0.0.1 -p 6379 exists model_creation 2>/dev/null)
if [ "$QUEUE_EXISTS" == "1" ]; then
    echo "✅ model_creation queue exists"
    QUEUE_LEN=$(redis-cli -h 127.0.0.1 -p 6379 llen model_creation 2>/dev/null)
    echo "Queue contains $QUEUE_LEN tasks"
else
    echo "⚠️ model_creation queue does not exist yet. This is normal if no new tasks have been created."
    echo "The queue will be created when the first task is submitted."
fi

echo
echo "===== Final Status ====="
echo "Redis: $(systemctl is-active redis-server)"
echo "Celery Worker: $(systemctl is-active celery-worker)"
echo
echo "Script complete. If issues persist, run the diagnostic script: sudo bash celery_diagnostics.sh"