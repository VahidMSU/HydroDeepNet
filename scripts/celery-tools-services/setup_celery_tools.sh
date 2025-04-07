#!/bin/bash
# Setup script to make all Celery utility scripts executable
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "SCRIPT_DIR: $CURRENT_DIR"
source "${CURRENT_DIR}/../global_path.sh"


echo "Setting up Celery management tools..."

# Make all wrapper scripts executable
chmod +x ${SCRIPT_DIR}/fix_redis_wrongtype.py
chmod +x ${SCRIPT_DIR}/cleanup_corrupted_tasks.py
chmod +x ${SCRIPT_DIR}/cleanup_corrupted_tasks_wrapper.sh
chmod +x ${SCRIPT_DIR}/restart_corrupted_celery.sh
chmod +x ${SCRIPT_DIR}/monitor_celery_status.py

echo "Creating symbolic links in /usr/local/bin for easy access..."

# Create symbolic links in /usr/local/bin if running as root
if [[ $EUID -eq 0 ]]; then
    ln -sf ${SCRIPT_DIR}/cleanup_corrupted_tasks_wrapper.sh /usr/local/bin/cleanup-celery-tasks
    ln -sf ${SCRIPT_DIR}/restart_corrupted_celery.sh /usr/local/bin/restart-celery
    echo "Symbolic links created. You can now use these commands anywhere:"
    echo "  - cleanup-celery-tasks"
    echo "  - restart-celery"
else
    echo "Not running as root, skipping symbolic link creation."
    echo "Run this script with sudo to create symbolic links in /usr/local/bin."
fi

echo "Setup completed. Script executability set."
