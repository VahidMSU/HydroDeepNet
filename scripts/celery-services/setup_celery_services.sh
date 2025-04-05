#!/bin/bash
# Setup script to organize Celery services and utilities

SCRIPTS_DIR="/data/SWATGenXApp/codes/scripts"
CELERY_SERVICES_DIR="${SCRIPTS_DIR}/celery-services"
BIN_DIR="/usr/local/bin"

echo "Setting up Celery services directory structure..."

# Create the celery-services directory if it doesn't exist
mkdir -p "${CELERY_SERVICES_DIR}"
mkdir -p "${CELERY_SERVICES_DIR}/utils"
mkdir -p "${CELERY_SERVICES_DIR}/services"
mkdir -p "${CELERY_SERVICES_DIR}/docs"

# Move existing scripts to the new directory structure
echo "Moving files to the new directory structure..."

# Utils
cp "${SCRIPTS_DIR}/fix_redis_wrongtype.py" "${CELERY_SERVICES_DIR}/utils/"
cp "${SCRIPTS_DIR}/fix_redis_wrongtype_batch.py" "${CELERY_SERVICES_DIR}/utils/"
cp "${SCRIPTS_DIR}/cleanup_corrupted_tasks.py" "${CELERY_SERVICES_DIR}/utils/"
cp "${SCRIPTS_DIR}/monitor_celery_status.py" "${CELERY_SERVICES_DIR}/utils/"
cp "${SCRIPTS_DIR}/manage_celery_queues.py" "${CELERY_SERVICES_DIR}/utils/"
cp "${SCRIPTS_DIR}/fix_stuck_tasks.py" "${CELERY_SERVICES_DIR}/utils/"
cp "${SCRIPTS_DIR}/restart_stalled_tasks.py" "${CELERY_SERVICES_DIR}/utils/"
cp "${SCRIPTS_DIR}/cleanup_redis_celery.py" "${CELERY_SERVICES_DIR}/utils/"

# Service wrappers
cp "${SCRIPTS_DIR}/fix_redis_wrongtype_batch_wrapper.sh" "${CELERY_SERVICES_DIR}/services/"
cp "${SCRIPTS_DIR}/cleanup_corrupted_tasks_wrapper.sh" "${CELERY_SERVICES_DIR}/services/"
cp "${SCRIPTS_DIR}/restart_corrupted_celery.sh" "${CELERY_SERVICES_DIR}/services/"
cp "${SCRIPTS_DIR}/setup_redis_fix.sh" "${CELERY_SERVICES_DIR}/services/"

# Documentation
cp "${SCRIPTS_DIR}/celery_manager_readme.md" "${CELERY_SERVICES_DIR}/docs/"
cp "${SCRIPTS_DIR}/cleanup_redis_celery_readme.md" "${CELERY_SERVICES_DIR}/docs/"
cp "${SCRIPTS_DIR}/restart_stalled_tasks_readme.md" "${CELERY_SERVICES_DIR}/docs/"

# Systemd service files
cp "${SCRIPTS_DIR}/redis-fix.service" "${CELERY_SERVICES_DIR}/services/"

# Make all scripts executable
echo "Setting executable permissions..."
find "${CELERY_SERVICES_DIR}" -name "*.py" -exec chmod +x {} \;
find "${CELERY_SERVICES_DIR}" -name "*.sh" -exec chmod +x {} \;

echo "Setup completed!"
echo "New directory structure created at: ${CELERY_SERVICES_DIR}"
