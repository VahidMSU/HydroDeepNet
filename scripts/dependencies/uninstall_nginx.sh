#!/bin/bash
# uninstall_nginx.sh
# This script uninstalls Nginx web server and removes all associated configuration files.

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Uninstalling Nginx web server ===${NC}"

if command -v nginx &>/dev/null; then
    echo -e "${YELLOW}Stopping Nginx service...${NC}"
    systemctl stop nginx

    echo -e "${YELLOW}Disabling Nginx service...${NC}"
    systemctl disable nginx

    echo -e "${YELLOW}Uninstalling Nginx packages...${NC}"
    apt-get remove --purge -y nginx nginx-common nginx-core nginx-full nginx-extras

    echo -e "${YELLOW}Removing Nginx configuration files...${NC}"
    rm -rf /etc/nginx

    echo -e "${YELLOW}Removing Nginx log files...${NC}"
    rm -rf /var/log/nginx

    echo -e "${YELLOW}Removing Nginx cache directories...${NC}"
    rm -rf /var/cache/nginx

    echo -e "${YELLOW}Removing Nginx runtime directories...${NC}"
    rm -rf /var/lib/nginx /run/nginx

    echo -e "${YELLOW}Cleaning up any remaining packages...${NC}"
    apt-get autoremove -y
    apt-get autoclean -y

    echo -e "${GREEN}Nginx has been completely uninstalled.${NC}"
else
    echo -e "${YELLOW}Nginx is not installed.${NC}"
fi

# Check if Nginx was actually removed
if command -v nginx &>/dev/null; then
    echo -e "${RED}Warning: Nginx binary still found in path. Uninstallation may not be complete.${NC}"
    echo -e "${YELLOW}You may need to manually remove the remaining Nginx components.${NC}"
    exit 1
else
    echo -e "${GREEN}Nginx uninstallation verified successfully.${NC}"
fi

exit 0
