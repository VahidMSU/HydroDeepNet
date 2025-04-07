#!/bin/bash
# Setup script to organize Celery services and utilities
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "SCRIPT_DIR: $CURRENT_DIR"
source "${CURRENT_DIR}/../global_path.sh"

CELERY_SERVICES_DIR="${SCRIPT_DIR}/celery-tools-services"
BIN_DIR="/usr/local/bin"

echo "Setting up Celery services directory structure..."

# Create the celery-tools-services directory if it doesn't exist
mkdir -p "${CELERY_SERVICES_DIR}"
mkdir -p "${CELERY_SERVICES_DIR}/utils"
mkdir -p "${CELERY_SERVICES_DIR}/services"
mkdir -p "${CELERY_SERVICES_DIR}/docs"

# Move existing scripts to the new directory structure
echo "Moving files to the new directory structure..."

# Utils
cp "${SCRIPT_DIR}/fix_redis_wrongtype.py" "${CELERY_SERVICES_DIR}/utils/"
cp "${SCRIPT_DIR}/fix_redis_wrongtype_batch.py" "${CELERY_SERVICES_DIR}/utils/"
cp "${SCRIPT_DIR}/cleanup_corrupted_tasks.py" "${CELERY_SERVICES_DIR}/utils/"
cp "${SCRIPT_DIR}/monitor_celery_status.py" "${CELERY_SERVICES_DIR}/utils/"
cp "${SCRIPT_DIR}/manage_celery_queues.py" "${CELERY_SERVICES_DIR}/utils/"
cp "${SCRIPT_DIR}/fix_stuck_tasks.py" "${CELERY_SERVICES_DIR}/utils/"
cp "${SCRIPT_DIR}/restart_stalled_tasks.py" "${CELERY_SERVICES_DIR}/utils/"
cp "${SCRIPT_DIR}/cleanup_redis_celery.py" "${CELERY_SERVICES_DIR}/utils/"

# Service wrappers
cp "${SCRIPT_DIR}/fix_redis_wrongtype_batch_wrapper.sh" "${CELERY_SERVICES_DIR}/services/"
cp "${SCRIPT_DIR}/cleanup_corrupted_tasks_wrapper.sh" "${CELERY_SERVICES_DIR}/services/"
cp "${SCRIPT_DIR}/restart_corrupted_celery.sh" "${CELERY_SERVICES_DIR}/services/"
cp "${SCRIPT_DIR}/setup_redis_fix.sh" "${CELERY_SERVICES_DIR}/services/"

# Documentation
cp "${SCRIPT_DIR}/celery_manager_readme.md" "${CELERY_SERVICES_DIR}/docs/"
cp "${SCRIPT_DIR}/cleanup_redis_celery_readme.md" "${CELERY_SERVICES_DIR}/docs/"
cp "${SCRIPT_DIR}/restart_stalled_tasks_readme.md" "${CELERY_SERVICES_DIR}/docs/"

# Systemd service files
cp "${SCRIPT_DIR}/redis-fix.service" "${CELERY_SERVICES_DIR}/services/"

# Make all scripts executable
echo "Setting executable permissions..."
find "${CELERY_SERVICES_DIR}" -name "*.py" -exec chmod +x {} \;
find "${CELERY_SERVICES_DIR}" -name "*.sh" -exec chmod +x {} \;

echo "Setup completed!"
echo "New directory structure created at: ${CELERY_SERVICES_DIR}"
