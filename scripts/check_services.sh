#!/bin/bash

echo "Checking if backend services are running..."

# Check if Flask app is running on port 5050
if curl -s http://127.0.0.1:5050/api/health -m 2 > /dev/null; then
    echo "✅ Backend API is running on port 5050"
else
    echo "❌ Backend API is not responding on port 5050"
    echo "Checking service status..."
    sudo systemctl status flask-app.service --no-pager | head -n 20
    
    echo "Attempting to restart flask-app service..."
    sudo systemctl restart flask-app.service
    sleep 2
    
    if curl -s http://127.0.0.1:5050/api/health -m 2 > /dev/null; then
        echo "✅ Backend API is now running after restart"
    else
        echo "❌ Backend API failed to start. Please check logs at /data/SWATGenXApp/codes/web_application/logs/flask-app.log"
        echo "The frontend may not work correctly without the backend API."
    fi
fi

# Check if Redis is running
if sudo systemctl is-active --quiet redis-server; then
    echo "✅ Redis server is running"
else
    echo "❌ Redis server is not running"
    echo "Attempting to restart redis-server..."
    sudo systemctl restart redis-server
fi

# Check if Celery worker is running
if sudo systemctl is-active --quiet celery-worker.service; then
    echo "✅ Celery worker is running"
else
    echo "❌ Celery worker is not running"
    echo "Attempting to restart celery-worker service..."
    sudo systemctl restart celery-worker.service
fi

# Check if Apache is running
if sudo systemctl is-active --quiet apache2; then
    echo "✅ Apache2 server is running"
else
    echo "❌ Apache2 server is not running"
    echo "Attempting to restart apache2..."
    sudo systemctl restart apache2
fi

echo "Service check completed."
