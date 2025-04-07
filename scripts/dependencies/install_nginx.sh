#!/bin/bash
# install_nginx.sh
# This script installs Nginx web server if it is not already installed.

if dpkg -s nginx &>/dev/null; then
    echo "Nginx is already installed."
else
    echo "Installing Nginx..."
    apt-get update && apt-get install -y nginx && echo "Nginx has been installed."
fi
