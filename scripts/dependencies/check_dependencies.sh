#!/bin/bash
# SWAT+ Dependencies Validation Script
# This script checks if all required components are properly installed

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get base directory from environment or default to relative path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${SWAT_BASE_DIR:-$(cd "$SCRIPT_DIR/../../" && pwd)}"
BIN_DIR="${BASE_DIR}/bin"

echo "Base directory: $BASE_DIR"
echo "Script directory: $SCRIPT_DIR"
echo "Bin directory: $BIN_DIR"

sleep 1000

# Paths
LOG_DIR="${BASE_DIR}/web_application/logs"
LOG_FILE="${LOG_DIR}/swat_dependencies_check.log"
mkdir -p "$LOG_DIR"

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
log "Base directory: $BASE_DIR"

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
    
    # Use a more reliable approach for Python module checking
    python_module_check() {
        MODULE=$1
        IMPORT=$2
        VERSION_ATTR=${3:-"__version__"}
        
        if python3 -c "import $IMPORT; print(f'  $MODULE version: {$IMPORT.$VERSION_ATTR}')" 2>/dev/null; then
            success "$MODULE installed"
            return 0
        else
            if [ "$4" = "required" ]; then
                error "$MODULE missing"
            else
                warning "$MODULE not found (may be required for some operations)"
            fi
            return 1
        fi
    }
    
    python_module_check "GDAL Python bindings" "osgeo.gdal" "__version__" "required"
    python_module_check "NumPy" "numpy" "__version__"
    python_module_check "Pandas" "pandas" "__version__"
    python_module_check "Matplotlib" "matplotlib" "__version__"
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
SWAT_EXE="${BIN_DIR}/swatplus"
if [ -x "$SWAT_EXE" ]; then
    success "SWAT+ executable found: $SWAT_EXE"
    # Get version info without full execution
    if "$SWAT_EXE" --version &>/dev/null; then
        VERSION_INFO=$("$SWAT_EXE" --version 2>&1 | head -n 3)
        log "Version: $VERSION_INFO"
    else
        log "Version: Unable to determine version"
    fi
else
    error "SWAT+ executable not found at: $SWAT_EXE"
fi

# Check MODFLOW
MODFLOW_EXE="${BIN_DIR}/modflow-nwt"
if [ -x "$MODFLOW_EXE" ]; then
    success "MODFLOW-NWT executable found: $MODFLOW_EXE"
else
    error "MODFLOW-NWT executable not found at: $MODFLOW_EXE"
fi

# Check for installation directories with flexible paths
check_dir() {
    DIR_NAME=$1
    DIR_PATH=$2
    ALT_PATH=$3
    
    if [ -d "$DIR_PATH" ]; then
        success "$DIR_NAME found: $DIR_PATH"
        return 0
    elif [ -n "$ALT_PATH" ] && [ -d "$ALT_PATH" ]; then
        success "$DIR_NAME found: $ALT_PATH"
        return 0
    else
        error "$DIR_NAME not found at: $DIR_PATH"
        if [ -n "$ALT_PATH" ]; then
            error "Also checked: $ALT_PATH"
        fi
        return 1
    fi
}

# Check directory content
check_file() {
    FILE_DESC=$1
    FILE_PATH=$2
    
    if [ -f "$FILE_PATH" ]; then
        SIZE=$(du -h "$FILE_PATH" 2>/dev/null | cut -f1)
        success "Found $FILE_DESC: $FILE_PATH ($SIZE)"
        return 0
    else
        warning "Missing $FILE_DESC: $FILE_PATH"
        return 1
    fi
}

# Check SWAT+ Editor installation
section "SWAT+ Editor"
EDITOR_PATH="/usr/local/share/SWATPlusEditor/swatplus-editor"
USER_EDITOR_PATH="${HOME}/.local/share/SWATPlusEditor/swatplus-editor"

check_dir "SWAT+ Editor" "$EDITOR_PATH" "$USER_EDITOR_PATH"
if [ $? -eq 0 ]; then
    # Use the directory that was found
    if [ -d "$EDITOR_PATH" ]; then
        FOUND_EDITOR="$EDITOR_PATH"
    else
        FOUND_EDITOR="$USER_EDITOR_PATH"
    fi
    
    check_file "Main API script" "$FOUND_EDITOR/src/api/swatplus_api.py"
    check_file "REST API script" "$FOUND_EDITOR/src/api/swatplus_rest_api.py"
fi

# Check QSWAT+ installation
section "QSWAT+ Plugin"
QSWAT_PATH="/usr/share/qgis/python/plugins/QSWATPlusLinux3_64"
USER_QSWAT_PATH="${HOME}/.local/share/qgis/python/plugins/QSWATPlusLinux3_64"

check_dir "QSWAT+ plugin" "$QSWAT_PATH" "$USER_QSWAT_PATH"
if [ $? -eq 0 ]; then
    # Use the directory that was found
    if [ -d "$QSWAT_PATH" ]; then
        FOUND_QSWAT="$QSWAT_PATH"
    else
        FOUND_QSWAT="$USER_QSWAT_PATH"
    fi
    
    check_file "Plugin initialization file" "$FOUND_QSWAT/__init__.py"
    check_dir "QSWATPlus directory" "$FOUND_QSWAT/QSWATPlus" ""
fi

# Check database files with flexible paths
section "SWAT+ Database Files"
DB_PATHS=(
    "/usr/local/share/SWATPlus/Databases"
    "${HOME}/.local/share/SWATPlus/Databases"
    "${BASE_DIR}/data/SWATPlus/Databases"
)

DB_FOUND=false
for path in "${DB_PATHS[@]}"; do
    if [ -d "$path" ]; then
        success "SWAT+ database directory found: $path"
        DB_FOUND=true
        
        # Check required database files
        REQUIRED_DBS=("swatplus_datasets.sqlite" "swatplus_soils.sqlite" "swatplus_wgn.sqlite")
        for db in "${REQUIRED_DBS[@]}"; do
            check_file "$db database" "$path/$db"
        done
        
        break
    fi
done

if [ "$DB_FOUND" = false ]; then
    error "SWAT+ database directory not found in standard locations"
    error "Checked paths: ${DB_PATHS[*]}"
fi

# Final summary
section "Summary"
if [ "$ALL_CHECKS_PASSED" = true ]; then
    success "All critical dependencies are properly installed!"
else
    error "One or more required dependencies are missing or improperly installed"
    echo "" | tee -a "$LOG_FILE"
    echo -e "${YELLOW}To install missing components, you can run:${NC}" | tee -a "$LOG_FILE"
    echo -e "  ${GREEN}bash $SCRIPT_DIR/install_all_dependencies.sh${NC}" | tee -a "$LOG_FILE"
    echo -e "Or run individual installation scripts:" | tee -a "$LOG_FILE"
    echo -e "  ${GREEN}bash $SCRIPT_DIR/install_swatplus.sh${NC}" | tee -a "$LOG_FILE"
    echo -e "  ${GREEN}bash $SCRIPT_DIR/install_swatplus_exe.sh${NC}" | tee -a "$LOG_FILE"
    echo -e "  ${GREEN}bash $SCRIPT_DIR/install_modflow.sh${NC}" | tee -a "$LOG_FILE"
    echo -e "  ${GREEN}bash $SCRIPT_DIR/install_qgis.sh${NC}" | tee -a "$LOG_FILE"
    echo -e "  ${GREEN}bash $SCRIPT_DIR/install_gdal.sh${NC}" | tee -a "$LOG_FILE"
fi

log "Check completed. Full log available at: $LOG_FILE"


