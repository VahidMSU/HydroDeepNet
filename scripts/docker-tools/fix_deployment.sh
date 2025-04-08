#!/bin/bash
# Script to fix common Docker deployment issues for SWATGenXApp

# Define colors for better output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== SWATGenXApp Docker Deployment Fix Script ===${NC}"

# Check if running inside Docker
if [ ! -f "/.dockerenv" ]; then
    echo -e "${YELLOW}Note: This script should be run inside the Docker container.${NC}"
    echo -e "${YELLOW}Run it with: docker exec -it swatgenx_app bash /data/SWATGenXApp/codes/scripts/docker-tools/fix_deployment.sh${NC}"
    read -p "Continue anyway? (y/n): " choice
    if [ "$choice" != "y" ]; then
        exit 0
    fi
fi

# Create necessary directories
echo -e "${GREEN}Creating required directories...${NC}"
mkdir -p /data/SWATGenXApp/bin
mkdir -p /data/SWATGenXApp/codes/web_application/logs/celery
mkdir -p /var/log/redis
mkdir -p /usr/local/share/SWATPlus/Databases
mkdir -p /data/SWATGenXApp/data/SWATPlus/Databases
mkdir -p /var/lib/redis/data

# Set up proper permissions
echo -e "${GREEN}Setting proper permissions...${NC}"
chown -R www-data:www-data /data/SWATGenXApp/codes/web_application/logs
chown -R www-data:www-data /data/SWATGenXApp/Users
chown -R redis:redis /var/lib/redis /var/log/redis

# Check Redis service
echo -e "${GREEN}Checking Redis service...${NC}"
if ! pgrep -x redis-server >/dev/null; then
    echo -e "${YELLOW}Redis server is not running. Starting Redis...${NC}"
    redis-server --daemonize yes
    sleep 2
    if pgrep -x redis-server >/dev/null; then
        echo -e "${GREEN}Redis server started successfully.${NC}"
    else
        echo -e "${RED}Failed to start Redis server.${NC}"
    fi
else
    echo -e "${GREEN}Redis server is already running.${NC}"
fi

# Check and fix GDAL installation
echo -e "${GREEN}Checking GDAL installation...${NC}"
if ! python3 -c "import osgeo.gdal" 2>/dev/null; then
    echo -e "${YELLOW}GDAL Python bindings not found. Installing...${NC}"
    pip install --no-cache-dir gdal
    if python3 -c "import osgeo.gdal; print('GDAL version:', osgeo.gdal.__version__)" 2>/dev/null; then
        echo -e "${GREEN}GDAL Python bindings installed successfully.${NC}"
    else
        echo -e "${RED}Failed to install GDAL Python bindings.${NC}"
    fi
else
    echo -e "${GREEN}GDAL Python bindings are already installed.${NC}"
fi

# Restart supervisor services
echo -e "${GREEN}Restarting supervisor services...${NC}"
if command -v supervisorctl >/dev/null; then
    supervisorctl update
    supervisorctl restart all
    echo -e "${GREEN}Supervisor services restarted.${NC}"
else
    echo -e "${RED}Supervisor not found. Cannot restart services.${NC}"
fi

# Check if services are now running
echo -e "${GREEN}Checking service status...${NC}"
if supervisorctl status | grep -q "RUNNING"; then
    echo -e "${GREEN}Services are now running.${NC}"
else
    echo -e "${RED}Some services are still not running properly.${NC}"
    supervisorctl status
fi

echo -e "${BLUE}=== Fix script completed ===${NC}"
echo -e "${YELLOW}Run verification script to check if issues are resolved:${NC}"
echo -e "${GREEN}bash /data/SWATGenXApp/codes/scripts/docker-tools/verify_docker_deployment.sh${NC}"
