#!/bin/bash
# install_apache.sh
# This script installs Apache web server if it is not already installed,
# then starts the Apache service and enables it on boot.

set -e  # Exit on any error

echo "Starting Apache installation process..."

# Use more reliable checks for Apache
if dpkg -s apache2 &>/dev/null && systemctl list-unit-files | grep -q apache2.service; then
    echo "Apache is already installed."
else
    echo "Installing Apache web server and dependencies..."
    apt-get update
    
    # Install Apache with dependencies
    apt-get install -y apache2 apache2-utils ssl-cert
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install Apache packages."
        exit 1
    fi
    
    # Enable required modules
    echo "Enabling required Apache modules..."
    a2enmod proxy proxy_http proxy_wstunnel rewrite headers ssl
    
    # Create default directories if they don't exist
    echo "Creating required directories..."
    mkdir -p /var/www/html
    chown -R www-data:www-data /var/www/html
fi

# Ensure Apache is configured to start on boot
echo "Enabling Apache service to start on boot..."
systemctl enable apache2

# Start/restart Apache
echo "Starting Apache service..."
systemctl restart apache2
if [ $? -ne 0 ]; then
    echo "WARNING: Failed to start Apache. Checking status..."
    systemctl status apache2 --no-pager
    exit 1
fi

# Verify Apache is running
if systemctl is-active --quiet apache2; then
    echo "SUCCESS: Apache installed and running successfully."
else
    echo "ERROR: Apache installation failed. Service is not running."
    exit 1
fi

# Print installation info
echo "Apache version: $(apache2 -v | head -n 1)"
echo "Configuration directory: /etc/apache2"
echo "Installation complete."
