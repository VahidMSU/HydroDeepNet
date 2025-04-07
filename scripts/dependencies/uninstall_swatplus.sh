#!/bin/bash
# SWAT+ Uninstallation Script
# This script removes SWAT+, SWAT+ Editor, and the QSWATPlus plugin

# Define colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   error "This script must be run as root to remove system files"
   exit 1
fi

# Process arguments
FORCE=false
KEEP_USER_DATA=false

for arg in "$@"; do
    case $arg in
        --force)
            FORCE=true
            shift
            ;;
        --keep-user-data)
            KEEP_USER_DATA=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --force           Skip confirmation prompts"
            echo "  --keep-user-data  Preserve user database files"
            echo "  --help            Show this help message"
            exit 0
            ;;
    esac
done

# Confirmation prompt
if [ "$FORCE" = false ]; then
    echo -e "${YELLOW}This script will remove SWAT+, SWAT+ Editor, and QSWATPlus components from your system.${NC}"
    echo -e "${YELLOW}It will delete the following directories:${NC}"
    echo "  - /usr/local/bin/swatplus*"
    echo "  - /usr/local/share/SWATPlus"
    echo "  - /usr/local/share/SWATPlusEditor"
    echo "  - /usr/share/qgis/python/plugins/QSWATPlusLinux3_64"
    
    if [ "$KEEP_USER_DATA" = false ]; then
        echo "  - User database files in ~/.local/share/SWATPlus"
    fi
    
    read -p "Are you sure you want to continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Uninstallation cancelled"
        exit 0
    fi
fi

log "Starting SWAT+ uninstallation..."

# 1. Remove SWAT+ executables
log "Removing SWAT+ executables..."
rm -f /usr/local/bin/swatplus* 2>/dev/null || warning "No SWAT+ executables found in /usr/local/bin"

# 2. Remove SWAT+ Editor
log "Removing SWAT+ Editor..."
if [ -d "/usr/local/share/SWATPlusEditor" ]; then
    rm -rf /usr/local/share/SWATPlusEditor
    log "SWAT+ Editor removed successfully"
else
    warning "SWAT+ Editor directory not found"
fi

# 3. Remove QSWATPlus plugin
log "Removing QSWATPlus plugin..."
if [ -d "/usr/share/qgis/python/plugins/QSWATPlusLinux3_64" ]; then
    rm -rf /usr/share/qgis/python/plugins/QSWATPlusLinux3_64
    log "QSWATPlus plugin removed successfully"
else
    warning "QSWATPlus plugin directory not found"
fi

# 4. Remove system-wide database files
log "Removing system-wide SWAT+ database files..."
if [ -d "/usr/local/share/SWATPlus" ]; then
    rm -rf /usr/local/share/SWATPlus
    log "System-wide SWAT+ database files removed successfully"
else
    warning "System-wide SWAT+ database directory not found"
fi

# 5. Optionally remove user database files
if [ "$KEEP_USER_DATA" = false ]; then
    log "Removing user SWAT+ database files..."
    # We need to handle this carefully as it involves user home directories
    
    # First, the current user
    if [ -d "$HOME/.local/share/SWATPlus" ]; then
        rm -rf "$HOME/.local/share/SWATPlus"
        log "Removed SWAT+ database files from current user's home directory"
    fi
    
    # Then, the www-data user
    if [ -d "/var/www/.local/share/SWATPlus" ]; then
        rm -rf "/var/www/.local/share/SWATPlus"
        log "Removed SWAT+ database files from www-data user's directory"
    fi
else
    log "Preserving user database files as requested"
fi

# 6. Clean up associated configuration files
log "Cleaning up configuration files..."

# Remove any .qgis files related to SWAT in /etc
find /etc -name "*swat*" -type f 2>/dev/null | while read file; do
    rm -f "$file"
    log "Removed configuration file: $file"
done

log "SWAT+ uninstallation completed successfully"
log "Note: If you installed QGIS specifically for SWAT+, you may want to uninstall it separately with:"
log "  apt-get remove --purge qgis qgis-plugin-grass"
