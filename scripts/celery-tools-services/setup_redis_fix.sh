#!/bin/bash
# Setup script for Redis fix tools
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "SCRIPT_DIR: $CURRENT_DIR"
source "${CURRENT_DIR}/../global_path.sh"

SYSTEMD_DIR="/etc/systemd/system"

echo "Setting up Redis WRONGTYPE fix tools..."

# Make scripts executable
chmod +x ${SCRIPT_DIR}/fix_redis_wrongtype_batch.py
chmod +x ${SCRIPT_DIR}/fix_redis_wrongtype_batch_wrapper.sh

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    echo "This script must be run as root to install system services"
    echo "Please run with: sudo $0"
    exit 1
fi

# Install systemd service if it doesn't exist or if it's different
if [ ! -f "${SYSTEMD_DIR}/redis-fix.service" ] || ! cmp -s "${SCRIPT_DIR}/redis-fix.service" "${SYSTEMD_DIR}/redis-fix.service"; then
    echo "Installing Redis fix service..."
    cp ${SCRIPT_DIR}/redis-fix.service ${SYSTEMD_DIR}/
    systemctl daemon-reload
    echo "Service installed. You can run it with: sudo systemctl start redis-fix.service"
else
    echo "Redis fix service is already installed"
fi

# Create symbolic links in /usr/local/bin
echo "Creating symbolic link for easy access..."
ln -sf ${SCRIPT_DIR}/fix_redis_wrongtype_batch_wrapper.sh /usr/local/bin/fix-redis-batch

echo "Setup completed!"
echo ""
echo "You can now use the following commands:"
echo "  sudo fix-redis-batch              # Run the batch fix tool with default settings"
echo "  sudo fix-redis-batch --dry-run    # Test without making changes"
echo "  sudo systemctl start redis-fix    # Run the fix as a background service"
echo ""
echo "For help and more options:"
echo "  sudo fix-redis-batch --help"
