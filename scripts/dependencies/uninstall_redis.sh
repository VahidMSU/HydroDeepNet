#!/bin/bash
# uninstall_redis.sh
# This script uninstalls Redis server if it is installed.

if command -v redis-server &> /dev/null; then
    echo "Uninstalling Redis server..."
    apt-get remove -y redis-server && apt-get purge -y redis-server && echo "Redis server has been uninstalled."
else
    echo "Redis server is not installed."
fi
