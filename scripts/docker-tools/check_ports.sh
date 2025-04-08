#!/bin/bash
# Script to check port availability and what's using them

# Define colors for better output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Checking Port Availability ===${NC}"

# Function to check if a port is in use
check_port() {
    local port=$1
    echo -e "${YELLOW}Checking port $port:${NC}"

    # Try lsof first (most detailed)
    if command -v lsof >/dev/null 2>&1; then
        echo -e "${GREEN}Using lsof to check port $port${NC}"
        lsof_result=$(lsof -i :$port 2>/dev/null)
        if [ -z "$lsof_result" ]; then
            echo -e "${GREEN}Port $port is available${NC}"
        else
            echo -e "${RED}Port $port is in use by:${NC}"
            echo "$lsof_result"
        fi
    # Then try netstat
    elif command -v netstat >/dev/null 2>&1; then
        echo -e "${GREEN}Using netstat to check port $port${NC}"
        netstat_result=$(netstat -tulpn 2>/dev/null | grep :$port)
        if [ -z "$netstat_result" ]; then
            echo -e "${GREEN}Port $port is available${NC}"
        else
            echo -e "${RED}Port $port is in use by:${NC}"
            echo "$netstat_result"
        fi
    # Finally try ss
    elif command -v ss >/dev/null 2>&1; then
        echo -e "${GREEN}Using ss to check port $port${NC}"
        ss_result=$(ss -tulpn 2>/dev/null | grep :$port)
        if [ -z "$ss_result" ]; then
            echo -e "${GREEN}Port $port is available${NC}"
        else
            echo -e "${RED}Port $port is in use by:${NC}"
            echo "$ss_result"
        fi
    else
        echo -e "${RED}No tools available to check port usage (lsof, netstat, ss)${NC}"
    fi

    echo ""
}

# Check commonly used ports
check_port 5000
check_port 5050
check_port 80
check_port 8080

echo -e "${BLUE}=== Python Environment Information ===${NC}"
echo -e "${YELLOW}Current Python path:${NC}"
echo $PYTHONPATH
echo ""

echo -e "${YELLOW}Current working directory:${NC}"
pwd
echo ""

echo -e "${YELLOW}Location of Python files:${NC}"
find /data/SWATGenXApp/codes -type f -name "*.py" | head -n 10
echo ""

echo -e "${BLUE}=== Flask Application Structure ===${NC}"
echo -e "${YELLOW}Checking app module structure:${NC}"
if [ -d "/data/SWATGenXApp/codes/app" ]; then
    echo -e "${GREEN}App directory exists at /data/SWATGenXApp/codes/app${NC}"
    ls -la /data/SWATGenXApp/codes/app
elif [ -f "/data/SWATGenXApp/codes/app.py" ]; then
    echo -e "${GREEN}App file exists at /data/SWATGenXApp/codes/app.py${NC}"
    head -n 10 /data/SWATGenXApp/codes/app.py
else
    echo -e "${RED}Cannot find app module at expected locations${NC}"
    echo "Searching for potential app modules:"
    find /data/SWATGenXApp/codes -type f -name "app.py" -o -name "__init__.py" | head -n 10
fi

echo ""
echo -e "${BLUE}=== Recommendations ===${NC}"
echo -e "1. If ports are in use: ${YELLOW}kill the processes or use different ports${NC}"
echo -e "2. If app module not found: ${YELLOW}check your Flask application structure and PYTHONPATH${NC}"
echo -e "3. Run Flask manually to see detailed errors: ${YELLOW}cd /data/SWATGenXApp/codes && python3 -m flask run -h 0.0.0.0 -p 5000${NC}"
