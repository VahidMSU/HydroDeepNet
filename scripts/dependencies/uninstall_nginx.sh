#!/bin/bash
# uninstall_nginx.sh
# This script uninstalls Nginx web server if it is installed.

if command -v nginx &> /dev/null; then
    echo "Uninstalling Nginx..."
    apt-get remove -y nginx && apt-get purge -y nginx && echo "Nginx has been uninstalled."
else
    echo "Nginx is not installed."
fi
