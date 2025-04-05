#!/bin/bash
# Install script for SWATGenX Celery Management Tools

SCRIPTS_DIR="/data/SWATGenXApp/codes/scripts"
CELERY_SERVICES_DIR="${SCRIPTS_DIR}/celery-services"
SYSTEMD_DIR="/etc/systemd/system"
BIN_DIR="/usr/local/bin"

echo "Installing SWATGenX Celery Management Tools..."

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    echo "This script must be run as root to install system components"
    echo "Please run with: sudo $0"
    exit 1
fi

# Make sure the directory structure is set up
if [ ! -d "${CELERY_SERVICES_DIR}" ]; then
    echo "Setting up Celery services directory..."
    ${CELERY_SERVICES_DIR}/setup_celery_services.sh
fi

# Make all scripts executable
echo "Setting executable permissions..."
find "${CELERY_SERVICES_DIR}" -name "*.py" -exec chmod +x {} \;
find "${CELERY_SERVICES_DIR}" -name "*.sh" -exec chmod +x {} \;

# Install the main command
echo "Installing the main command..."
ln -sf "${CELERY_SERVICES_DIR}/celery-tools" "${BIN_DIR}/celery-tools"

# Install systemd services
echo "Installing systemd services..."
cp "${CELERY_SERVICES_DIR}/services/redis-fix.service" "${SYSTEMD_DIR}/"
systemctl daemon-reload

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
