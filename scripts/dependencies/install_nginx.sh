#!/bin/bash
# install_nginx.sh
# This script installs Nginx web server if it is not already installed and configures required directories.

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Installing and configuring Nginx web server ===${NC}"

if dpkg -s nginx &>/dev/null; then
    echo -e "${YELLOW}Nginx is already installed.${NC}"
else
    echo -e "${GREEN}Installing Nginx...${NC}"
    apt-get update || true
    apt-get install -y nginx && echo -e "${GREEN}Nginx has been installed.${NC}" || {
        echo -e "${RED}Failed to install Nginx. Please check your internet connection and apt repositories.${NC}"
        exit 1
    }
fi

# Create required Nginx directories and set permissions
echo -e "${GREEN}Configuring Nginx directories and permissions...${NC}"
mkdir -p /var/lib/nginx/body /var/lib/nginx/fastcgi /var/lib/nginx/proxy /var/lib/nginx/scgi /var/lib/nginx/uwsgi /var/cache/nginx /run/nginx /var/log/nginx
chown -R www-data:www-data /var/lib/nginx /var/cache/nginx /var/log/nginx /run/nginx
chmod -R 755 /var/lib/nginx /var/cache/nginx /var/log/nginx /run/nginx

# Check if main nginx.conf exists, create it if missing
if [ ! -f "/etc/nginx/nginx.conf" ]; then
    echo -e "${YELLOW}Main Nginx configuration file is missing. Creating default configuration...${NC}"
    mkdir -p /etc/nginx /etc/nginx/sites-available /etc/nginx/sites-enabled /etc/nginx/conf.d

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

    # Create basic mime.types file if it doesn't exist
    if [ ! -f "/etc/nginx/mime.types" ]; then
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

    # Create a default site configuration if it doesn't exist
    if [ ! -f "/etc/nginx/sites-available/default" ]; then
        cat >/etc/nginx/sites-available/default <<'EOF'
server {
    listen 80 default_server;
    listen [::]:80 default_server;

    root /var/www/html;
    index index.html index.htm;

    server_name _;

    location / {
        try_files $uri $uri/ =404;
    }
}
EOF
        # Create symbolic link to enable the default site
        if [ ! -f "/etc/nginx/sites-enabled/default" ]; then
            ln -s /etc/nginx/sites-available/default /etc/nginx/sites-enabled/
        fi
    fi

    echo -e "${GREEN}Default Nginx configuration created.${NC}"
fi

# Test the Nginx configuration
echo -e "${GREEN}Testing Nginx configuration...${NC}"
nginx -t
if [ $? -ne 0 ]; then
    echo -e "${RED}Nginx configuration test failed. Please check the error messages above.${NC}"
    exit 1
fi

# Make sure Nginx is started and enabled
echo -e "${GREEN}Starting Nginx service...${NC}"
systemctl start nginx
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to start Nginx service. Please check the systemd logs.${NC}"
    systemctl status nginx
    exit 1
fi

echo -e "${GREEN}Enabling Nginx to start on boot...${NC}"
systemctl enable nginx
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Warning: Failed to enable Nginx on boot.${NC}"
fi

# Verify Nginx is running
if systemctl is-active --quiet nginx; then
    echo -e "${GREEN}Nginx is now running and configured.${NC}"
else
    echo -e "${RED}Nginx installation completed but service is not running.${NC}"
    echo -e "${YELLOW}Please check logs with: systemctl status nginx${NC}"
    exit 1
fi

echo -e "${GREEN}Nginx installation and configuration completed successfully.${NC}"
echo -e "${GREEN}Nginx directories configured.${NC}"
