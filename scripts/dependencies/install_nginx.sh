#!/bin/bash
# install_nginx.sh
# This script installs Nginx web server if it is not already installed and configures required directories.

if dpkg -s nginx &>/dev/null; then
    echo "Nginx is already installed."
else
    echo "Installing Nginx..."
    apt-get update || true
    apt-get install -y nginx && echo "Nginx has been installed."
fi

# Create required Nginx directories and set permissions
echo "Configuring Nginx directories and permissions..."
mkdir -p /var/lib/nginx/body /var/lib/nginx/fastcgi /var/lib/nginx/proxy /var/lib/nginx/scgi /var/lib/nginx/uwsgi /var/cache/nginx /run/nginx /var/log/nginx
chown -R www-data:www-data /var/lib/nginx /var/cache/nginx /var/log/nginx /run/nginx
chmod -R 755 /var/lib/nginx /var/cache/nginx /var/log/nginx /run/nginx
echo "Nginx directories configured."
