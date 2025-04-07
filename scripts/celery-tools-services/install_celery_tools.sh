#!/bin/bash
# Install script for SWATGenX Celery Management Tools
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "SCRIPT_DIR: $CURRENT_DIR"
source "${CURRENT_DIR}/../global_path.sh"
CELERY_SERVICES_DIR="${SCRIPT_DIR}/celery-tools-services"

echo "Installing SWATGenX Celery Management Tools..."

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    echo "This script must be run as root to install system components"
    echo "Please run with: sudo $0"
    exit 1
fi

# Make sure the directory structure is set up
if [ ! -d "${CELERY_SERVICES_DIR}" ]; then
    echo "Creating Celery services directory structure..."
    mkdir -p "${CELERY_SERVICES_DIR}/utils"
    mkdir -p "${CELERY_SERVICES_DIR}/services" 
    mkdir -p "${CELERY_SERVICES_DIR}/docs"
elif [ -f "${SCRIPT_DIR}/setup_celery_services.sh" ]; then
    echo "Setting up Celery services directory..."
    bash "${SCRIPT_DIR}/setup_celery_services.sh"
fi

# Make all scripts executable
echo "Setting executable permissions..."
find "${CELERY_SERVICES_DIR}" -name "*.py" -exec chmod +x {} \;
find "${CELERY_SERVICES_DIR}" -name "*.sh" -exec chmod +x {} \;

# Install the main command
echo "Installing the main command..."
if [ -f "${CELERY_SERVICES_DIR}/celery-tools" ]; then
    echo "Creating symlink to ${CELERY_SERVICES_DIR}/celery-tools in ${SYSTEM_BIN_DIR}/celery-tools"
    ln -sf "${CELERY_SERVICES_DIR}/celery-tools" "${SYSTEM_BIN_DIR}/celery-tools"
else
    echo "WARNING: celery-tools main script not found at ${CELERY_SERVICES_DIR}/celery-tools"
fi

# Install systemd services
echo "Installing systemd services..."
REDIS_SERVICE_PATH="${CELERY_SERVICES_DIR}/services/redis-fix.service"
if [ -f "${REDIS_SERVICE_PATH}" ]; then
    cp "${REDIS_SERVICE_PATH}" "${SYSTEMD_DIR}/"
    systemctl daemon-reload
else
    echo "WARNING: Redis fix service file not found at ${REDIS_SERVICE_PATH}"
fi

echo "Installation completed successfully!"
echo
echo "You can now use the following commands:"
echo "  celery-tools monitor           # Monitor Celery status"
echo "  celery-tools inspect           # Inspect queue contents"
echo "  sudo celery-tools fix-redis    # Fix Redis WRONGTYPE errors"
echo "  sudo celery-tools restart      # Restart Celery workers"
echo
echo "For more information:"
echo "  celery-tools help"
echo
echo "Documentation can be found at:"
echo "  ${CELERY_SERVICES_DIR}/docs/"
