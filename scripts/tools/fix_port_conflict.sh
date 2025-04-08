#!/bin/bash
# Script to diagnose and fix port conflicts with the Flask application

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== SWATGenX Port Conflict Resolver ===${NC}"
echo -e "This script will identify and resolve conflicts on port 5050"

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    echo -e "${RED}This script must be run as root${NC}"
    echo -e "Please run with: sudo bash $(basename "$0")"
    exit 1
fi

echo -e "\n${YELLOW}Checking what's using port 5050...${NC}"
# Use lsof to find processes using port 5050
if command -v lsof >/dev/null; then
    PORT_PROCESS=$(lsof -i:5050 -t)
    if [ -z "$PORT_PROCESS" ]; then
        echo -e "${GREEN}No process is currently using port 5050${NC}"
    else
        echo -e "${RED}Found process(es) using port 5050:${NC}"
        for pid in $PORT_PROCESS; do
            process_info=$(ps -f -p $pid)
            echo -e "$process_info"
            echo -e "PID: $pid"
        done

        echo -e "\n${YELLOW}Would you like to kill the process(es)? (y/n)${NC}"
        read -p "> " response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            for pid in $PORT_PROCESS; do
                echo -e "Killing process $pid..."
                kill -9 $pid
                sleep 1
                if kill -0 $pid 2>/dev/null; then
                    echo -e "${RED}Failed to kill process $pid${NC}"
                else
                    echo -e "${GREEN}Successfully killed process $pid${NC}"
                fi
            done
        else
            echo -e "${YELLOW}Skipping process termination${NC}"
        fi
    fi
else
    echo -e "${RED}lsof command not found. Installing...${NC}"
    apt-get update && apt-get install -y lsof
    echo -e "${GREEN}lsof installed. Please run this script again.${NC}"
    exit 1
fi

echo -e "\n${YELLOW}Checking for zombie or orphaned gunicorn processes...${NC}"
GUNICORN_PROCESSES=$(ps aux | grep gunicorn | grep -v grep | awk '{print $2}')
if [ -z "$GUNICORN_PROCESSES" ]; then
    echo -e "${GREEN}No gunicorn processes found${NC}"
else
    echo -e "${RED}Found gunicorn processes:${NC}"
    ps aux | grep gunicorn | grep -v grep

    echo -e "\n${YELLOW}Would you like to kill these gunicorn processes? (y/n)${NC}"
    read -p "> " response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        for pid in $GUNICORN_PROCESSES; do
            echo -e "Killing gunicorn process $pid..."
            kill -9 $pid
            sleep 0.5
        done
        echo -e "${GREEN}Killed all gunicorn processes${NC}"
    else
        echo -e "${YELLOW}Skipping gunicorn process termination${NC}"
    fi
fi

echo -e "\n${YELLOW}Checking Flask application service status...${NC}"
systemctl status flask-app.service --no-pager

echo -e "\n${GREEN}Port conflict resolution complete!${NC}"
echo -e "Next steps:"
echo -e "1. Restart the services with: ${YELLOW}sudo bash /data/SWATGenXApp/codes/scripts/restart_services.sh${NC}"
echo -e "2. Check logs at: ${YELLOW}/data/SWATGenXApp/codes/web_application/logs/flask-app.log${NC}"
echo -e "3. Test the API with: ${YELLOW}/data/SWATGenXApp/codes/scripts/test/api_model_creation_test.sh${NC}"

exit 0
