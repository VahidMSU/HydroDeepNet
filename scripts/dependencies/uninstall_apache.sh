#!/bin/bash
# uninstall_apache.sh
# This script uninstalls Apache web server if it is installed.
# It stops and disables the Apache service, removes the package, and cleans up residual files.

if command -v apache2 &>/dev/null; then
    echo "Stopping Apache service..."
    systemctl stop apache2
    echo "Disabling Apache service..."
    systemctl disable apache2
    echo "Removing Apache..."
    apt-get remove -y apache2 && apt-get purge -y apache2
    echo "Cleaning up residual configuration files..."
    apt-get autoremove -y && apt-get autoclean -y
    echo "Apache has been uninstalled."
else
    echo "Apache is not installed."
fi
