#!/bin/bash
# SWAT+ Dependencies Validation Script
# This script checks if all required components are properly installed

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Paths
LOG_FILE="/data/SWATGenXApp/codes/web_application/logs/swat_dependencies_check.log"
BASE_DIR="/data/SWATGenXApp/codes/"
mkdir -p "$(dirname "$LOG_FILE")"

# Status tracking
ALL_CHECKS_PASSED=true

# Logging functions
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

section() {
    echo "" | tee -a "$LOG_FILE"
    echo -e "${BLUE}=== $1 ===${NC}" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[✓] $1${NC}" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[!] $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[✗] $1${NC}" | tee -a "$LOG_FILE"
    ALL_CHECKS_PASSED=false
}

# Header
section "SWAT+ Dependencies Validation"
log "Date: $(date)"
log "Hostname: $(hostname)"
log "User: $(whoami)"

# Check System Information
section "System Information"
log "OS: $(lsb_release -ds 2>/dev/null || cat /etc/*release 2>/dev/null | head -n1 || echo 'Unknown')"
log "Kernel: $(uname -r)"
log "Architecture: $(uname -m)"

# Check Python
section "Python Environment"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    success "Python installed: $PYTHON_VERSION"
    
    # Check key Python modules
    echo "Checking required Python modules:" | tee -a "$LOG_FILE"
    
    if python3 -c "from osgeo import gdal; print(f'  GDAL version: {gdal.__version__}')" 2>/dev/null; then
        success "GDAL Python bindings installed"
    else
        error "GDAL Python bindings missing"
    fi
    
    if python3 -c "import numpy; print(f'  NumPy version: {numpy.__version__}')" 2>/dev/null; then
        success "NumPy installed"
    else
        warning "NumPy not found (may be required for some operations)"
    fi
else
    error "Python 3 not found"
fi

# Check GDAL installation
section "GDAL Installation"
if command -v gdalinfo &> /dev/null; then
    GDAL_VERSION=$(gdalinfo --version | awk '{print $1}')
    success "GDAL installed: $GDAL_VERSION"
else
    error "GDAL not found"
fi

# Check QGIS installation
section "QGIS Installation"
if command -v qgis &> /dev/null; then
    QGIS_VERSION=$(qgis --version 2>&1)
    success "QGIS installed: $QGIS_VERSION"
    
    # Check Python bindings for QGIS
    if python3 -c "import qgis.core; print(f'  QGIS Python binding version: {qgis.core.Qgis.QGIS_VERSION}')" 2>/dev/null; then
        success "QGIS Python bindings installed"
    else
        warning "QGIS Python bindings not available - may need to set PYTHONPATH"
    fi
else
    error "QGIS not found"
fi

# Check SWAT+ executables
section "SWAT+ Executables"
SWAT_EXE="${BASE_DIR}bin/swatplus"
if [ -x "$SWAT_EXE" ]; then
    success "SWAT+ executable found: $SWAT_EXE"
    # Get version info without full execution
    VERSION_INFO=$($SWAT_EXE --version 2>&1 | head -n 3)
    log "Version: $VERSION_INFO"
else
    error "SWAT+ executable not found at: $SWAT_EXE"
fi

# Check MODFLOW
MODFLOW_EXE="${BASE_DIR}bin/modflow-nwt"
if [ -x "$MODFLOW_EXE" ]; then
    success "MODFLOW-NWT executable found: $MODFLOW_EXE"
else
    error "MODFLOW-NWT executable not found at: $MODFLOW_EXE"
fi

# Check SWAT+ Editor installation
section "SWAT+ Editor"
EDITOR_PATH="/usr/local/share/SWATPlusEditor/swatplus-editor"
if [ -d "$EDITOR_PATH" ]; then
    success "SWAT+ Editor found: $EDITOR_PATH"
    
    # Check for key files
    if [ -f "$EDITOR_PATH/src/api/swatplus_api.py" ]; then
        success "Main API script found"
    else
        warning "Main API script missing: src/api/swatplus_api.py"
    fi
    
    if [ -f "$EDITOR_PATH/src/api/swatplus_rest_api.py" ]; then
        success "REST API script found"
    else
        warning "REST API script missing: src/api/swatplus_rest_api.py"
    fi
else
    error "SWAT+ Editor not found at: $EDITOR_PATH"
fi

# Check QSWAT+ installation
section "QSWAT+ Plugin"
QSWAT_PATH="/usr/share/qgis/python/plugins/QSWATPlusLinux3_64"
if [ -d "$QSWAT_PATH" ]; then
    success "QSWAT+ plugin found: $QSWAT_PATH"
    
    # Check for key components
    if [ -f "$QSWAT_PATH/__init__.py" ]; then
        success "Plugin initialization file found"
    else
        warning "Plugin initialization file missing: __init__.py"
    fi
    
    if [ -d "$QSWAT_PATH/QSWATPlus" ]; then
        success "QSWATPlus directory found"
    else
        warning "QSWATPlus directory missing"
    fi
else
    error "QSWAT+ plugin not found at: $QSWAT_PATH"
fi

# Check database files
section "SWAT+ Database Files"
DB_PATH="/usr/local/share/SWATPlus/Databases"
USER_DB_PATH="${HOME}/.local/share/SWATPlus/Databases"

DB_FOUND=false
for path in "$DB_PATH" "$USER_DB_PATH"; do
    if [ -d "$path" ]; then
        success "SWAT+ database directory found: $path"
        DB_FOUND=true
        
        # Check required database files
        REQUIRED_DBS=("swatplus_datasets.sqlite" "swatplus_soils.sqlite" "swatplus_wgn.sqlite")
        for db in "${REQUIRED_DBS[@]}"; do
            if [ -f "$path/$db" ]; then
                SIZE=$(du -h "$path/$db" | cut -f1)
                success "Found $db ($SIZE)"
            else
                error "Missing required database: $db"
            fi
        done
        
        break
    fi
done

if [ "$DB_FOUND" = false ]; then
    error "SWAT+ database directory not found"
fi

# Final summary
section "Summary"
if [ "$ALL_CHECKS_PASSED" = true ]; then
    success "All critical dependencies are properly installed!"
else
    error "One or more required dependencies are missing or improperly installed"
    echo "" | tee -a "$LOG_FILE"
    echo -e "${YELLOW}To install missing components, you can run:${NC}" | tee -a "$LOG_FILE"
    echo -e "  ${GREEN}sudo bash /data/SWATGenXApp/codes/scripts/dependencies/install_swatplus.sh${NC}" | tee -a "$LOG_FILE"
    echo -e "  ${GREEN}sudo bash /data/SWATGenXApp/codes/scripts/dependencies/install_swatplus_exe.sh${NC}" | tee -a "$LOG_FILE"
    echo -e "  ${GREEN}sudo bash /data/SWATGenXApp/codes/scripts/dependencies/install_modflow.sh${NC}" | tee -a "$LOG_FILE"
    echo -e "  ${GREEN}sudo bash /data/SWATGenXApp/codes/scripts/dependencies/install_qgis.sh${NC}" | tee -a "$LOG_FILE"
    echo -e "  ${GREEN}sudo bash /data/SWATGenXApp/codes/scripts/dependencies/install_gdal.sh${NC}" | tee -a "$LOG_FILE"
fi

log "Check completed. Full log available at: $LOG_FILE"


