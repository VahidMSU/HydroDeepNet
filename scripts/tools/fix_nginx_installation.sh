#!/bin/bash
# fix_nginx_installation.sh
# This script diagnoses and fixes Nginx installation issues

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== SWATGenX Nginx Installation Fixer ===${NC}"
echo -e "This script will identify and fix Nginx installation issues"

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    echo -e "${RED}This script must be run as root${NC}"
    echo -e "Please run with: sudo bash $(basename "$0")"
    exit 1
fi

# Check if Nginx is installed
if ! dpkg -s nginx &>/dev/null; then
    echo -e "${YELLOW}Nginx is not installed. Installing now...${NC}"
    apt-get update
    apt-get install -y nginx

    if ! dpkg -s nginx &>/dev/null; then
        echo -e "${RED}Failed to install Nginx.${NC}"
        exit 1
    fi
    echo -e "${GREEN}Nginx installed successfully.${NC}"
else
    echo -e "${GREEN}Nginx is already installed.${NC}"
fi

# Check for nginx.conf
echo -e "\n${YELLOW}Checking for main Nginx configuration file...${NC}"
if [ ! -f "/etc/nginx/nginx.conf" ]; then
    echo -e "${RED}Missing nginx.conf - creating default configuration...${NC}"

    # Create directory structure
    mkdir -p /etc/nginx/sites-available /etc/nginx/sites-enabled /etc/nginx/conf.d

    # Create main config
    cat >/etc/nginx/nginx.conf <<'EOF'
user www-data;
worker_processes auto;
pid /run/nginx.pid;
include /etc/nginx/modules-enabled/*.conf;

events {
    worker_connections 768;
    # multi_accept on;
}

http {
    # Basic Settings
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    # server_tokens off;

    # server_names_hash_bucket_size 64;
    # server_name_in_redirect off;

    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # SSL Settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;

    # Logging Settings
    access_log /var/log/nginx/access.log;
    error_log /var/log/nginx/error.log;

    # Gzip Settings
    gzip on;

    # Virtual Host Configs
    include /etc/nginx/conf.d/*.conf;
    include /etc/nginx/sites-enabled/*;
}
EOF

    # Create mime.types file
    if [ ! -f "/etc/nginx/mime.types" ]; then
        echo -e "${YELLOW}Creating mime.types file...${NC}"
        cat >/etc/nginx/mime.types <<'EOF'
types {
    text/html                             html htm shtml;
    text/css                              css;
    text/xml                              xml;
    image/gif                             gif;
    image/jpeg                            jpeg jpg;
    application/javascript                js;
    application/atom+xml                  atom;
    application/rss+xml                   rss;

    text/plain                            txt;
    image/png                             png;
    image/webp                            webp;
    image/svg+xml                         svg svgz;
    image/tiff                            tif tiff;
    image/x-icon                          ico;
    image/x-ms-bmp                        bmp;

    application/json                      json;
    application/pdf                       pdf;
    application/zip                       zip;
}
EOF
    fi

    echo -e "${GREEN}Created nginx.conf and necessary configuration files.${NC}"
else
    echo -e "${GREEN}Found nginx.conf at /etc/nginx/nginx.conf${NC}"
fi

# Check directory structure
echo -e "\n${YELLOW}Checking Nginx directory structure...${NC}"
for dir in /etc/nginx/sites-available /etc/nginx/sites-enabled /etc/nginx/conf.d; do
    if [ ! -d "$dir" ]; then
        echo -e "${YELLOW}Creating missing directory: $dir${NC}"
        mkdir -p "$dir"
    fi
done

# Create required runtime directories
echo -e "\n${YELLOW}Checking Nginx runtime directories...${NC}"
mkdir -p /var/lib/nginx/body /var/lib/nginx/fastcgi /var/lib/nginx/proxy /var/lib/nginx/scgi /var/lib/nginx/uwsgi /var/cache/nginx /run/nginx /var/log/nginx
chown -R www-data:www-data /var/lib/nginx /var/cache/nginx /var/log/nginx /run/nginx
chmod -R 755 /var/lib/nginx /var/cache/nginx /var/log/nginx /run/nginx
echo -e "${GREEN}Nginx runtime directories configured.${NC}"

# Test nginx configuration
echo -e "\n${YELLOW}Testing Nginx configuration...${NC}"
if nginx -t; then
    echo -e "${GREEN}Nginx configuration test successful!${NC}"
else
    echo -e "${RED}Nginx configuration test failed. Please check the errors above.${NC}"
    exit 1
fi

# Try to start/restart nginx
echo -e "\n${YELLOW}Attempting to start/restart Nginx...${NC}"
systemctl restart nginx

if systemctl is-active --quiet nginx; then
    echo -e "${GREEN}Nginx started successfully!${NC}"
    systemctl enable nginx
    echo -e "${GREEN}Nginx enabled to start on boot.${NC}"
else
    echo -e "${RED}Failed to start Nginx. Check status with: systemctl status nginx${NC}"
    systemctl status nginx --no-pager
fi

echo -e "\n${GREEN}Nginx installation check and fix completed!${NC}"
echo -e "Next steps:"
echo -e "1. If Nginx is now running, try: ${YELLOW}sudo bash /data/SWATGenXApp/codes/scripts/restart_services.sh${NC}"
echo -e "2. Select Nginx as the web server option when prompted."

exit 0
