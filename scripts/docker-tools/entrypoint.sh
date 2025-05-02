#!/bin/bash
set -e

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Create required directories
mkdir -p /data/SWATGenXApp/codes/web_application/logs
chown -R www-data:www-data /data/SWATGenXApp/codes/web_application/logs

# Create a simpler Nginx configuration for Docker
cat > /etc/nginx/sites-available/default << 'EOF'
server {
    listen 80;
    server_name localhost;
    
    root /data/SWATGenXApp/codes/web_application/frontend/build;
    index index.html;
    
    # API proxy settings
    location /api/ {
        proxy_pass http://127.0.0.1:5050;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
        proxy_connect_timeout 120s;
    }
    
    # WebSocket proxy
    location /ws {
        proxy_pass http://127.0.0.1:5050;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 120s;
    }
    
    # Special routes
    location ~ ^/(login|signup|model-settings|model-confirmation|verify|download|download_directory|reports)(.*)$ {
        proxy_pass http://127.0.0.1:5050$request_uri;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 120s;
    }
    
    # Static files
    location /static/ {
        alias /data/SWATGenXApp/codes/web_application/frontend/build/static/;
        expires 30d;
        add_header Cache-Control "public, max-age=2592000";
    }
    
    # React app routing fallback
    location / {
        try_files $uri $uri/ /index.html;
    }
    
    # Error logs
    error_log /data/SWATGenXApp/codes/web_application/logs/nginx-error.log;
    access_log /data/SWATGenXApp/codes/web_application/logs/nginx-access.log;
}
EOF

# Create symlink to enable config
ln -sf /etc/nginx/sites-available/default /etc/nginx/sites-enabled/default

# Start Redis server
log "Starting Redis server..."
service redis-server start || log "Failed to start Redis server"

# Start Nginx server
log "Starting Nginx server..."
service nginx start || log "Failed to start Nginx server"

# Start Celery worker as www-data user
log "Starting Celery worker..."
gosu www-data bash -c "cd /data/SWATGenXApp/codes/web_application && \
    /data/SWATGenXApp/codes/.venv/bin/celery -A celery_worker worker \
    --loglevel=info -n model_worker@%h -Q model_creation \
    --concurrency=4 --prefetch-multiplier=1 --max-tasks-per-child=10 \
    -O fair --without-heartbeat --without-gossip \
    >> /data/SWATGenXApp/codes/web_application/logs/celery-worker.log 2>&1 &"

# Wait for Celery to start
sleep 2

# Start Flask application with gunicorn
log "Starting Flask application with gunicorn..."
cd /data/SWATGenXApp/codes/web_application
exec gosu www-data /data/SWATGenXApp/codes/.venv/bin/gunicorn \
    --workers 4 --bind 127.0.0.1:5050 run:app \
    --timeout 120 \
    --access-logfile /data/SWATGenXApp/codes/web_application/logs/gunicorn-access.log \
    --error-logfile /data/SWATGenXApp/codes/web_application/logs/gunicorn-error.log
