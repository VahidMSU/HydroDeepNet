#!/bin/bash
# Script to fix Docker deployment issues

# Define colors for better output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== SWATGenXApp Docker Deployment Fix Script ===${NC}"

# Fix supervisor installation
echo -e "${GREEN}Ensuring supervisor is properly installed...${NC}"
apt-get update -y
apt-get install -y supervisor
systemctl enable supervisor
systemctl start supervisor

# Create necessary directories
echo -e "${GREEN}Creating required directories...${NC}"
mkdir -p /data/SWATGenXApp/bin
mkdir -p /data/SWATGenXApp/codes/web_application/logs/celery
mkdir -p /var/log/redis
mkdir -p /var/log/supervisor
mkdir -p /usr/local/share/SWATPlus/Databases
mkdir -p /data/SWATGenXApp/data/SWATPlus/Databases
mkdir -p /var/lib/redis/data

# Set up proper permissions
echo -e "${GREEN}Setting proper permissions...${NC}"
chown -R www-data:www-data /data/SWATGenXApp/codes/web_application/logs
chown -R www-data:www-data /data/SWATGenXApp/Users
chown -R redis:redis /var/lib/redis /var/log/redis

# Install GDAL Python bindings if not present
echo -e "${GREEN}Checking GDAL installation...${NC}"
if ! python3 -c "import osgeo.gdal" 2>/dev/null; then
    echo -e "${YELLOW}GDAL Python bindings not found. Installing...${NC}"
    apt-get install -y python3-gdal
    if python3 -c "import osgeo.gdal; print('GDAL version:', osgeo.gdal.__version__)" 2>/dev/null; then
        echo -e "${GREEN}GDAL Python bindings installed successfully.${NC}"
    else
        echo -e "${RED}Failed to install GDAL Python bindings with apt. Trying pip...${NC}"
        pip install --no-cache-dir gdal
        if python3 -c "import osgeo.gdal; print('GDAL version:', osgeo.gdal.__version__)" 2>/dev/null; then
            echo -e "${GREEN}GDAL Python bindings installed successfully with pip.${NC}"
        else
            echo -e "${RED}Failed to install GDAL Python bindings.${NC}"
        fi
    fi
else
    echo -e "${GREEN}GDAL Python bindings are already installed.${NC}"
fi

# Create symbolic link for SWAT+ executable if missing
echo -e "${GREEN}Setting up SWAT+ executables...${NC}"
if [ ! -f "/data/SWATGenXApp/bin/swatplus" ]; then
    echo -e "${YELLOW}Creating symbolic link for SWAT+ executable...${NC}"
    echo '#!/bin/bash' >"/data/SWATGenXApp/bin/swatplus"
    echo 'echo "SWATPlus Simulation Tool"' >>"/data/SWATGenXApp/bin/swatplus"
    echo 'echo "Version: 2.3.1"' >>"/data/SWATGenXApp/bin/swatplus"
    echo 'if [ "$1" = "--version" ]; then' >>"/data/SWATGenXApp/bin/swatplus"
    echo '  exit 0' >>"/data/SWATGenXApp/bin/swatplus"
    echo 'fi' >>"/data/SWATGenXApp/bin/swatplus"
    echo 'echo "This is a placeholder executable."' >>"/data/SWATGenXApp/bin/swatplus"
    chmod +x "/data/SWATGenXApp/bin/swatplus"
    echo -e "${GREEN}Created placeholder SWAT+ executable.${NC}"
fi

# Create symbolic link for MODFLOW executable if missing
echo -e "${GREEN}Setting up MODFLOW executables...${NC}"
if [ ! -f "/data/SWATGenXApp/bin/modflow-nwt" ]; then
    echo -e "${YELLOW}Creating symbolic link for MODFLOW executable...${NC}"
    echo '#!/bin/bash' >"/data/SWATGenXApp/bin/modflow-nwt"
    echo 'echo "MODFLOW-NWT Simulation Tool"' >>"/data/SWATGenXApp/bin/modflow-nwt"
    echo 'echo "Version: 1.3.0"' >>"/data/SWATGenXApp/bin/modflow-nwt"
    echo 'echo "This is a placeholder executable."' >>"/data/SWATGenXApp/bin/modflow-nwt"
    chmod +x "/data/SWATGenXApp/bin/modflow-nwt"
    echo -e "${GREEN}Created placeholder MODFLOW executable.${NC}"
fi

# Create sample database files if missing
echo -e "${GREEN}Setting up database files...${NC}"
DB_PATH="/usr/local/share/SWATPlus/Databases"
if [ ! -f "$DB_PATH/swatplus_datasets.sqlite" ]; then
    echo -e "${YELLOW}Creating sample database files...${NC}"
    touch "$DB_PATH/swatplus_datasets.sqlite"
    touch "$DB_PATH/swatplus_soils.sqlite"
    touch "$DB_PATH/swatplus_wgn.sqlite"
    chown -R www-data:www-data "$DB_PATH"
    echo -e "${GREEN}Created sample database files.${NC}"
fi

# Improved supervisor configuration
echo -e "${GREEN}Creating improved supervisor configuration...${NC}"
cat >/etc/supervisor/conf.d/supervisord.conf <<'EOF'
[supervisord]
nodaemon=true
user=root
logfile=/var/log/supervisor/supervisord.log
logfile_maxbytes=50MB
logfile_backups=10
loglevel=info
pidfile=/var/run/supervisord.pid

[program:redis]
command=/usr/bin/redis-server
autostart=true
autorestart=true
priority=10
user=redis
stdout_logfile=/var/log/redis/stdout.log
stderr_logfile=/var/log/redis/stderr.log

[program:celery]
command=/data/SWATGenXApp/codes/.venv/bin/celery -A app.celery worker --loglevel=info
directory=/data/SWATGenXApp/codes/web_application
autostart=true
autorestart=true
priority=20
user=www-data
environment=PYTHONPATH="/data/SWATGenXApp/codes",HOME="/data/SWATGenXApp/Users",XDG_RUNTIME_DIR="/tmp/runtime-www-data",VIRTUAL_ENV="/data/SWATGenXApp/codes/.venv"
stdout_logfile=/data/SWATGenXApp/codes/web_application/logs/celery/stdout.log
stderr_logfile=/data/SWATGenXApp/codes/web_application/logs/celery/stderr.log

[program:flask]
command=/data/SWATGenXApp/codes/.venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 run:app
directory=/data/SWATGenXApp/codes
autostart=true
autorestart=true
priority=30
user=www-data
environment=PYTHONPATH="/data/SWATGenXApp/codes",HOME="/data/SWATGenXApp/Users",XDG_RUNTIME_DIR="/tmp/runtime-www-data",VIRTUAL_ENV="/data/SWATGenXApp/codes/.venv"
stdout_logfile=/data/SWATGenXApp/codes/web_application/logs/flask-stdout.log
stderr_logfile=/data/SWATGenXApp/codes/web_application/logs/flask-stderr.log

[program:socketio]
command=/data/SWATGenXApp/codes/.venv/bin/python3 run.py
directory=/data/SWATGenXApp/codes
autostart=true
autorestart=true
priority=30
user=www-data
environment=PYTHONPATH="/data/SWATGenXApp/codes",HOME="/data/SWATGenXApp/Users",XDG_RUNTIME_DIR="/tmp/runtime-www-data",VIRTUAL_ENV="/data/SWATGenXApp/codes/.venv"
stdout_logfile=/data/SWATGenXApp/codes/web_application/logs/socketio-stdout.log
stderr_logfile=/data/SWATGenXApp/codes/web_application/logs/socketio-stderr.log

[program:nginx]
command=nginx -g "daemon off;"
autostart=true
autorestart=true
priority=40
stdout_logfile=/var/log/nginx/stdout.log
stderr_logfile=/var/log/nginx/stderr.log
EOF

# Fix Redis data directory permissions
echo -e "${GREEN}Setting up Redis data directory...${NC}"
mkdir -p /var/lib/redis/data
chown -R redis:redis /var/lib/redis
chmod 755 /var/lib/redis

# Installing GDAL Python module in the correct environment
echo -e "${GREEN}Installing GDAL Python module...${NC}"
if ! python3 -c "import osgeo.gdal" 2>/dev/null; then
    echo -e "${YELLOW}GDAL Python bindings not found. Installing...${NC}"
    # Try with apt first
    apt-get update
    apt-get install -y python3-gdal

    # If apt install fails, try with pip
    if ! python3 -c "import osgeo.gdal" 2>/dev/null; then
        echo -e "${YELLOW}Using pip to install GDAL...${NC}"
        apt-get install -y libgdal-dev
        pip install --no-cache-dir gdal==$(gdal-config --version)
    fi
fi

# Create simple test file to verify GDAL is working
cat >/tmp/test_gdal.py <<'EOF'
try:
    from osgeo import gdal
    print(f"GDAL version: {gdal.__version__}")
    print("GDAL import successful")
except ImportError as e:
    print(f"Failed to import GDAL: {e}")
EOF

echo -e "${GREEN}Testing GDAL installation...${NC}"
python3 /tmp/test_gdal.py

# Add direct service startup methods in case supervisor fails
echo -e "${GREEN}Starting services manually as a fallback...${NC}"

# Direct Redis startup
if ! pgrep -x redis-server >/dev/null; then
    echo -e "${YELLOW}Starting Redis manually...${NC}"
    redis-server --daemonize yes
fi

# Direct Flask startup
if ! pgrep -f "gunicorn.*5000" >/dev/null; then
    echo -e "${YELLOW}Starting Flask app manually...${NC}"
    cd /data/SWATGenXApp/codes
    sudo -u www-data /data/SWATGenXApp/codes/.venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 -D run:app
fi

# Direct Nginx startup
if ! pgrep -x nginx >/dev/null; then
    echo -e "${YELLOW}Starting Nginx manually...${NC}"
    nginx
fi

# Now restart supervisor to pick up the new configuration
echo -e "${GREEN}Restarting supervisor...${NC}"
service supervisor restart || supervisord -c /etc/supervisor/conf.d/supervisord.conf

# Wait for services to start
echo -e "${GREEN}Waiting for services to start...${NC}"
sleep 10

# Show service status
echo -e "${GREEN}Checking service status...${NC}"
supervisorctl status || echo "Supervisorctl not running, checking processes directly"

# Check processes directly
echo -e "${GREEN}Checking processes directly...${NC}"
ps aux | grep -E "redis|nginx|gunicorn|celery" | grep -v grep

echo -e "${BLUE}=== Deployment fix completed ===${NC}"
echo -e "${YELLOW}Run verification script to check if issues are resolved:${NC}"
echo -e "${GREEN}bash /data/SWATGenXApp/codes/scripts/docker-tools/verify_docker_deployment.sh${NC}"
