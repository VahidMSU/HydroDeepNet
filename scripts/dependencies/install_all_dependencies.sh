#!/bin/bash
# SWAT+ Complete Dependencies Installation Script
# This script installs all required dependencies for SWAT+

# Set colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Base directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/data/SWATGenXApp/codes"
BIN_DIR="${BASE_DIR}/bin"

# Ensure bin directory exists
mkdir -p "$BIN_DIR"

# Logging functions
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

section() {
    echo ""
    echo -e "${BLUE}======== $1 ========${NC}"
    echo ""
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   error "This script must be run as root to install system dependencies"
   exit 1
fi

# Function to execute a script and check its status
run_script() {
    local script="$1"
    local script_name=$(basename "$script")
    
    section "Running $script_name"
    
    if [ -f "$script" ]; then
        chmod +x "$script"
        if bash "$script"; then
            log "$script_name completed successfully"
            return 0
        else
            error "$script_name failed with exit code $?"
            return 1
        fi
    else
        error "Script not found: $script"
        return 1
    fi
}

# Main installation process
section "SWAT+ Dependencies Installation"
log "Starting installation of all SWAT+ dependencies"
log "Date: $(date)"
log "System: $(lsb_release -ds 2>/dev/null || cat /etc/*release 2>/dev/null | head -n1 || echo 'Unknown')"

# Step 1: Install system dependencies
section "Installing System Dependencies"
log "Updating package lists..."
apt-get update

log "Installing required packages..."
apt-get install -y build-essential cmake g++ gfortran wget unzip \
                   libsqlite3-dev python3 python3-pip python3-dev \
                   libcurl4-openssl-dev

# Step 2: Install GDAL
if ! command -v gdalinfo &> /dev/null; then
    run_script "${SCRIPT_DIR}/install_gdal.sh"
else
    log "GDAL is already installed: $(gdalinfo --version)"
fi

# Step 3: Install QGIS
if ! command -v qgis &> /dev/null; then
    run_script "${SCRIPT_DIR}/install_qgis.sh"
else
    log "QGIS is already installed: $(qgis --version 2>&1)"
fi

# Step 4: Install SWAT+
if [ ! -d "/usr/local/share/SWATPlus" ]; then
    run_script "${SCRIPT_DIR}/install_swatplus.sh"
else
    log "SWAT+ appears to be already installed"
fi

# Step 5: Install SWAT+ executable
if [ ! -x "${BIN_DIR}/swatplus" ]; then
    run_script "${SCRIPT_DIR}/install_swatplus_exe.sh"
else
    log "SWAT+ executable is already installed: ${BIN_DIR}/swatplus"
fi

# Step 6: Install MODFLOW
if [ ! -x "${BIN_DIR}/modflow-nwt" ]; then
    run_script "${SCRIPT_DIR}/install_modflow.sh"
else
    log "MODFLOW-NWT is already installed: ${BIN_DIR}/modflow-nwt"
fi

# Step 7: Check all dependencies
section "Verifying Installation"
log "Running dependency check..."
"${SCRIPT_DIR}/check_dependencies.sh"

# Done
section "Installation Complete"
log "All SWAT+ dependencies should now be installed"
log "If any components are still missing, please refer to the individual installation scripts"
echo ""
echo -e "${GREEN}To verify the installation again later, run:${NC}"
echo -e "  ${YELLOW}sudo bash /data/SWATGenXApp/codes/scripts/dependencies/check_dependencies.sh${NC}"
```

This comprehensive solution will:
1. Standardize the dependency check script with consistent formatting and organization
2. Improve the output clarity with organized sections
3. Provide accurate feedback on what dependencies are installed and what might be missing
4. Create an all-in-one installation script to ease the setup process
```
</file>
