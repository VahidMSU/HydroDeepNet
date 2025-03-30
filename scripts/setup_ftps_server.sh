#!/bin/bash
# FTPS Server Setup and Start Script for SWATGenX
# This script sets up and manages the vsftpd FTPS server

set -e

# Configuration
SCRIPT_DIR="/data/SWATGenXApp/codes/scripts"
VSFTPD_CONF_FILE="/etc/vsftpd.conf"
VSFTPD_USER_LIST="/etc/vsftpd.allowed_users"
FTPS_ROOT="/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID"
LOG_FILE="/var/log/ftps_setup.log"

# Ensure the script is run as root
if [ "$(id -u)" -ne 0 ]; then
    echo "This script must be run as root" >&2
    exit 1
fi

# Create log file
mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"
chmod 640 "$LOG_FILE"

# Function to log operations
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Check if vsftpd is installed
if ! command -v vsftpd &> /dev/null; then
    log "vsftpd is not installed. Installing..."
    apt-get update
    apt-get install -y vsftpd
    log "vsftpd installed successfully"
else
    log "vsftpd is already installed"
fi

# Install dependencies needed for user management
if ! command -v setfacl &> /dev/null; then
    log "acl package is not installed. Installing..."
    apt-get update
    apt-get install -y acl
    log "acl package installed successfully"
fi

# Create the allowed users file if it doesn't exist
if [ ! -f "$VSFTPD_USER_LIST" ]; then
    touch "$VSFTPD_USER_LIST"
    chmod 600 "$VSFTPD_USER_LIST"
    log "Created vsftpd allowed users file"
fi

# Ensure the root directory exists and has proper permissions
if [ ! -d "$FTPS_ROOT" ]; then
    log "ERROR: FTPS root directory $FTPS_ROOT does not exist"
    exit 1
fi

# Copy the vsftpd configuration file
if [ -f "$SCRIPT_DIR/vsftpd.conf" ]; then
    log "Copying vsftpd configuration file to $VSFTPD_CONF_FILE"
    cp "$SCRIPT_DIR/vsftpd.conf" "$VSFTPD_CONF_FILE"
    chmod 644 "$VSFTPD_CONF_FILE"
else
    log "ERROR: vsftpd configuration file not found at $SCRIPT_DIR/vsftpd.conf"
    exit 1
fi

# Set proper permissions for the FTPS root directory
log "Setting permissions for FTPS root directory"
chown -R root:root "$FTPS_ROOT"
chmod -R 755 "$FTPS_ROOT"

# Copy the ftps_user_manager script
if [ -f "$SCRIPT_DIR/ftps_user_manager.sh" ]; then
    log "Setting permissions for FTPS user manager script"
    chmod 755 "$SCRIPT_DIR/ftps_user_manager.sh"
else
    log "ERROR: FTPS user manager script not found at $SCRIPT_DIR/ftps_user_manager.sh"
    exit 1
fi

# Setup sudoers entry for web app to manage FTPS users
SUDO_ENTRY="www-data ALL=(root) NOPASSWD: $SCRIPT_DIR/ftps_user_manager.sh"
SUDOERS_FILE="/etc/sudoers.d/vsftpd"

if [ ! -f "$SUDOERS_FILE" ] || ! grep -q "$SUDO_ENTRY" "$SUDOERS_FILE"; then
    log "Setting up sudoers entry for FTPS user management"
    echo "$SUDO_ENTRY" > "$SUDOERS_FILE"
    chmod 440 "$SUDOERS_FILE"
fi

# Restart vsftpd service
log "Restarting vsftpd service"
systemctl restart vsftpd
systemctl enable vsftpd

# Check if the service started successfully
if systemctl is-active --quiet vsftpd; then
    log "✅ vsftpd service is running"
else
    log "❌ Failed to start vsftpd service. Check logs with: systemctl status vsftpd"
    exit 1
fi

# Check if port 990 is listening
if ss -tln | grep -q ":990"; then
    log "✅ FTPS server is listening on port 990"
else
    log "❌ FTPS server is not listening on port 990. Check the configuration and logs."
    exit 1
fi

log "FTPS server setup is complete and service is running"
echo "========================================================="
echo "FTPS Server Information:"
echo "  - Server: ciwre-bae.campusad.msu.edu (35.9.219.73)"
echo "  - Port: 990"
echo "  - Protocol: FTPS (FTP over SSL/TLS)"
echo "  - Passive Port Range: 40000-50000"
echo "  - Root Directory: $FTPS_ROOT"
echo ""
echo "To manage FTPS users, use the ftps_user_manager.sh script:"
echo "  - Create user: sudo $SCRIPT_DIR/ftps_user_manager.sh create <username>"
echo "  - Delete user: sudo $SCRIPT_DIR/ftps_user_manager.sh delete <username>"
echo "  - List users: sudo $SCRIPT_DIR/ftps_user_manager.sh list"
echo "========================================================="

exit 0