#!/bin/bash
# Script to fix final Docker deployment issues

# Define colors for better output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Final Docker Deployment Fixes ===${NC}"

# 1. Fix Redis configuration and service
echo -e "${GREEN}Setting up Redis...${NC}"

# Check if Redis is installed
if ! command -v redis-server &>/dev/null; then
    echo -e "${YELLOW}Installing Redis server...${NC}"
    apt-get update
    apt-get install -y redis-server
fi

# Create Redis configuration directory if it doesn't exist
mkdir -p /etc/redis

# Create a proper Redis configuration file
echo -e "${GREEN}Creating Redis configuration...${NC}"
cat >/etc/redis/redis.conf <<EOL
bind 0.0.0.0
protected-mode no
port 6379
dir /var/lib/redis
daemonize yes
supervised systemd
logfile /var/log/redis/redis-server.log
pidfile /var/run/redis/redis-server.pid
EOL

# Create log and run directories for Redis
mkdir -p /var/log/redis /var/run/redis /var/lib/redis
chown -R redis:redis /var/log/redis /var/run/redis /var/lib/redis
chmod 755 /var/log/redis /var/run/redis /var/lib/redis

# Stop any existing Redis processes
echo -e "${GREEN}Stopping existing Redis processes...${NC}"
systemctl stop redis-server 2>/dev/null || true
killall redis-server 2>/dev/null || true
sleep 2

# Start Redis server
echo -e "${GREEN}Starting Redis server...${NC}"
redis-server /etc/redis/redis.conf

# Wait for Redis to start
sleep 2

# Verify Redis is listening
echo -e "${GREEN}Verifying Redis is listening on port 6379...${NC}"
if netstat -tuln | grep -q ":6379"; then
    echo -e "${GREEN}Redis is now listening on port 6379.${NC}"
else
    echo -e "${RED}Redis is not listening on port 6379. Starting in foreground mode...${NC}"
    redis-server /etc/redis/redis.conf --daemonize no &
    sleep 2
fi

# Test Redis connection
echo -e "${GREEN}Testing Redis connection...${NC}"
if redis-cli ping | grep -q "PONG"; then
    echo -e "${GREEN}Redis connection test successful!${NC}"
else
    echo -e "${RED}Redis connection test failed. Trying alternative approaches...${NC}"
    redis-cli -h 127.0.0.1 ping
fi

# 2. Fix GDAL Python module installation
echo -e "${GREEN}Installing GDAL Python module...${NC}"

# Install system dependencies for GDAL
apt-get update
apt-get install -y python3-gdal libgdal-dev

# Get GDAL version
GDAL_VERSION=$(gdal-config --version 2>/dev/null || echo "3.4.1")
echo -e "${GREEN}Detected GDAL version: ${YELLOW}$GDAL_VERSION${NC}"

# Install GDAL Python module in the virtual environment
pip install gdal==$GDAL_VERSION

# Verify GDAL installation
echo -e "${GREEN}Verifying GDAL Python module installation...${NC}"
if python3 -c "from osgeo import gdal; print(f'GDAL Python module version: {gdal.__version__}')"; then
    echo -e "${GREEN}GDAL Python module installed successfully.${NC}"
else
    echo -e "${RED}GDAL Python module installation failed. Trying alternative approach...${NC}"
    pip install --no-binary :all: gdal==$GDAL_VERSION
fi

# 3. Fix Nginx frontend
echo -e "${GREEN}Setting up Nginx frontend...${NC}"

# Create frontend directories
mkdir -p /data/SWATGenXApp/codes/web_application/frontend/build/static

# Create a simple index.html file if it doesn't exist
if [ ! -f "/data/SWATGenXApp/codes/web_application/frontend/build/index.html" ]; then
    echo -e "${YELLOW}Creating a simple index.html file...${NC}"
    cat >/data/SWATGenXApp/codes/web_application/frontend/build/index.html <<'EOL'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SWATGenX Web Application</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f4f4f4;
            color: #333;
        }
        .container {
            max-width: 1100px;
            margin: 0 auto;
            padding: 20px;
            background: white;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        header {
            text-align: center;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
        }
        h1 {
            color: #2c3e50;
        }
        .content {
            margin-bottom: 20px;
        }
        .api-endpoints {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #2c3e50;
        }
        footer {
            text-align: center;
            margin-top: 20px;
            padding-top: 10px;
            border-top: 1px solid #eee;
            color: #777;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>SWATGenX Web Application</h1>
            <p>A comprehensive platform for SWAT+ modeling and analysis</p>
        </header>
        
        <div class="content">
            <h2>Welcome to SWATGenX</h2>
            <p>This is the frontend web interface for the SWATGenX application. If you're seeing this page, Nginx is correctly serving the frontend content.</p>
            
            <div class="api-endpoints">
                <h3>Available API Endpoints:</h3>
                <ul>
                    <li><strong>Status Check:</strong> <code>/api/diagnostic/status</code></li>
                    <li><strong>Model Settings:</strong> <code>/api/model-settings</code></li>
                </ul>
            </div>
        </div>
        
        <footer>
            <p>SWATGenX &copy; 2023 | Running in Docker</p>
        </footer>
    </div>
</body>
</html>
EOL

    # Create a simple CSS file to verify static file serving
    mkdir -p /data/SWATGenXApp/codes/web_application/frontend/build/static/css
    cat >/data/SWATGenXApp/codes/web_application/frontend/build/static/css/main.css <<'EOL'
body {
    background-color: #f4f4f4;
    font-family: Arial, sans-serif;
}
EOL
    echo -e "${GREEN}Created a simple index.html and CSS file.${NC}"
fi

# Create a proper Nginx config file
echo -e "${GREEN}Creating Nginx configuration...${NC}"
cat >/etc/nginx/sites-available/default <<'EOL'
server {
    listen 80 default_server;
    server_name _;

    # Serve the React build output
    root /data/SWATGenXApp/codes/web_application/frontend/build;
    index index.html;

    # API proxy for port 5050
    location /api/ {
        proxy_pass http://127.0.0.1:5050;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket proxy
    location /ws {
        proxy_pass http://127.0.0.1:5050;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Static files
    location /static/ {
        alias /data/SWATGenXApp/codes/web_application/frontend/build/static/;
        expires 30d;
        add_header Cache-Control "public, max-age=2592000";
    }

    # All other requests go to the React app
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Log files
    access_log /var/log/nginx/access.log;
    error_log /var/log/nginx/error.log;
}
EOL

# Ensure Nginx config is valid
echo -e "${GREEN}Testing Nginx configuration...${NC}"
nginx -t

# Reload Nginx
echo -e "${GREEN}Restarting Nginx...${NC}"
service nginx reload || service nginx restart

# Verify Nginx is serving content
echo -e "${GREEN}Verifying Nginx is serving content...${NC}"
if curl -s http://localhost/ | grep -q "SWATGenX"; then
    echo -e "${GREEN}Nginx is successfully serving the frontend.${NC}"
else
    echo -e "${RED}Nginx is not serving the frontend properly.${NC}"
    # Check Nginx logs
    echo -e "${YELLOW}Last 10 lines of Nginx error logs:${NC}"
    tail -n 10 /var/log/nginx/error.log
fi

echo -e "${BLUE}=== All fixes have been applied. Running final verification... ===${NC}"
bash /data/SWATGenXApp/codes/scripts/docker-tools/verify_docker_deployment.sh
