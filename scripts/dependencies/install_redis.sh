#!/bin/bash
# install_redis.sh
# This script installs Redis server if it is not already installed.

if command -v redis-server &> /dev/null; then
    echo "Redis server is already installed."
else
    echo "Installing Redis server..."
    apt-get update && apt-get install -y redis-server && echo "Redis server has been installed."
fi
