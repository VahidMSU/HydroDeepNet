#!/bin/bash
# Celery and Redis Diagnostics Script
# Run with sudo: sudo bash celery_diagnostics.sh

echo "===== Celery & Redis Diagnostics ====="
echo "Run Date: $(date)"
echo

# Set environment variables
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/data/SWATGenXApp/codes/.venv/bin"
export FLASK_ENV="production"
export PYTHONPATH="/data/SWATGenXApp/codes:/data/SWATGenXApp/codes/web_application"

# Check if Redis is running
echo "===== Redis Status ====="
if systemctl is-active --quiet redis-server; then
    echo "✅ Redis is running"
    redis-cli ping
else
    echo "❌ Redis is NOT running"
    echo "Starting Redis..."
    systemctl start redis-server
    sleep 2
    if systemctl is-active --quiet redis-server; then
        echo "✅ Redis successfully started"
    else
        echo "❌ Failed to start Redis"
    fi
fi

# Check Redis memory usage
echo
echo "===== Redis Memory Usage ====="
redis-cli info memory | grep "used_memory_human\|used_memory_peak_human"

# List celery queues
echo
echo "===== Redis Queue Contents ====="
echo "Checking for Celery queues in Redis DB..."
QUEUES=$(redis-cli -h 127.0.0.1 -p 6379 keys "*kombu.binding.celery*" 2>/dev/null)
if [ -z "$QUEUES" ]; then
    echo "No Celery queue bindings found."
else
    echo "Found Celery queue bindings:"
    echo "$QUEUES"
fi

# Check for tasks in the model_creation queue
echo
echo "===== Tasks in model_creation Queue ====="
MODEL_QUEUE=$(redis-cli -h 127.0.0.1 -p 6379 llen model_creation 2>/dev/null)
if [ -z "$MODEL_QUEUE" ] || [ "$MODEL_QUEUE" == "(nil)" ] || [ "$MODEL_QUEUE" -eq 0 ]; then
    echo "No tasks found in model_creation queue"
else
    echo "Found $MODEL_QUEUE tasks in model_creation queue"
    echo "First 5 tasks:"
    redis-cli -h 127.0.0.1 -p 6379 lrange model_creation 0 4
fi

# Check Celery worker status
echo
echo "===== Celery Worker Status ====="
if systemctl is-active --quiet celery-worker; then
    echo "✅ Celery worker service is running"
    ps aux | grep "[c]elery worker" | wc -l | xargs -I{} echo "Number of celery worker processes: {}"
    ps aux | grep "[c]elery worker"
else
    echo "❌ Celery worker service is NOT running"
fi

# Check for any running Celery processes
echo
echo "===== All Celery Processes ====="
ps aux | grep celery | grep -v grep

# Check Celery worker logs for errors
echo
echo "===== Recent Celery Worker Errors ====="
tail -n 20 /data/SWATGenXApp/codes/web_application/logs/celery-worker-error.log

# Check Celery worker logs
echo
echo "===== Recent Celery Worker Logs ====="
tail -n 20 /data/SWATGenXApp/codes/web_application/logs/celery-worker.log

# Check Redis connections
echo
echo "===== Redis Connections ====="
redis-cli client list | wc -l | xargs -I{} echo "Total Redis connections: {}"

# Check if the app.swatgenx_tasks.create_model_task is properly registered
echo
echo "===== Checking Task Registration ====="
cd /data/SWATGenXApp/codes/web_application
python3 -c "
from celery_app import celery
print('Registered tasks:')
for task in celery.tasks:
    print(f'- {task}')
print('\nTask routes:')
print(celery.conf.task_routes)
" 2>/dev/null || echo "Failed to check task registration"

echo
echo "===== Suggestions ====="
echo "If tasks are stuck in the queue, try:"
echo "1. sudo systemctl restart redis-server"
echo "2. sudo /data/SWATGenXApp/codes/scripts/restart_celery.sh"
echo "3. Check that the task routing configuration matches between app.swatgenx_tasks and celery_app.py"
echo "4. Verify Redis is not running out of memory"
echo "5. Ensure workers are listening to the correct queue (model_creation)"

echo
echo "Diagnostics complete."