#!/bin/bash
## Install MODFLOW-NWT
set -e  # Exit immediately if a command exits with non-zero status

# Get base directory from environment or default to relative path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${SWAT_BASE_DIR:-$(cd "$SCRIPT_DIR/../../" && pwd)}"
BIN_DIR="${BASE_DIR}/bin"
BUILD_DIR="${SCRIPT_DIR}/modflow_build"
echo "Base directory: $BASE_DIR"
echo "Script directory: $SCRIPT_DIR"
echo "Build directory: $BUILD_DIR"
echo "Bin directory: $BIN_DIR"

mkdir -p "${BIN_DIR}"

# Check if the MODFLOW executable already exists
modflow_nwt="${BIN_DIR}/modflow-nwt"
if [ -f "$modflow_nwt" ]; then
    echo "MODFLOW-NWT is already installed at $modflow_nwt"
    exit 0
fi

# Check if MODFLOW-NWT is already built
# Check if executable exists
if [ -f "${BUILD_DIR}/MODFLOW-NWT/build/modflow-nwt" ]; then
    echo "MODFLOW-NWT is already built at ${BUILD_DIR}/MODFLOW-NWT/build/modflow-nwt"
    echo "Copying to bin directory..."
    cp "${BUILD_DIR}/MODFLOW-NWT/build/modflow-nwt" "${BIN_DIR}/"
    chmod +x "${BIN_DIR}/modflow-nwt"
    echo "Installation completed successfully!"
    exit 0
fi

# Check if the MODFLOW-NWT build directory exists
sleep 2

# Clean up any existing build directories
if [ -d "${BUILD_DIR}" ]; then
    echo "Removing existing MODFLOW-NWT build directory..."
    rm -rf "${BUILD_DIR}/MODFLOW-NWT/build"
fi

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "Downloading MODFLOW-NWT source code..."
# Use a more reliable download method - curl with retry
if [ -f "MODFLOW-NWT_1.3.0.zip" ]; then
    echo "MODFLOW-NWT source code already downloaded."
else
    if command -v curl &> /dev/null; then
        curl -L --retry 3 --retry-delay 3 -o MODFLOW-NWT_1.3.0.zip https://water.usgs.gov/water-resources/software/MODFLOW-NWT/MODFLOW-NWT_1.3.0.zip
    else
        wget --tries=3 --timeout=15 -O MODFLOW-NWT_1.3.0.zip https://water.usgs.gov/water-resources/software/MODFLOW-NWT/MODFLOW-NWT_1.3.0.zip
    fi

    # Verify the download was successful
    if [ ! -s MODFLOW-NWT_1.3.0.zip ]; then
        echo "Error: Failed to download MODFLOW-NWT source code."
        exit 1
    fi
fi

echo "Extracting MODFLOW-NWT source code..."
unzip -q MODFLOW-NWT_1.3.0.zip || {
    echo "Error: Failed to extract zip file. The download may be corrupted."
    echo "Please try running the script again."
    exit 1
}

# Create build subdirectory
mkdir -p "${BUILD_DIR}/MODFLOW-NWT/build"
cd "${BUILD_DIR}/MODFLOW-NWT/build"

# Set up Intel compiler environment if available
if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    source /opt/intel/oneapi/setvars.sh
fi

# Copy source files
echo "Copying source files to build directory..."
cp "${BUILD_DIR}/MODFLOW-NWT/src/"* ./

echo "Compiling MODFLOW-NWT..."

# First compile global module file - this is needed by many files
echo "Step 1: Compiling global module..."
for module in global.f global.f90; do
    if [ -f "$module" ]; then
        echo "  Compiling: $module"
        ifx -c -I. "$module"
    fi
done

# Then compile other foundational modules
echo "Step 2: Compiling other foundational modules..."
for module in mach_mod.f90 modules.f90 openspec.F90 parammodule.f; do
    if [ -f "$module" ]; then
        echo "  Compiling: $module"
        ifx -c -I. "$module"
    fi
done

# Then compile base components that other modules may need
echo "Step 3: Compiling base components..."
for base in utl7.f gsfmodflow.f; do
    if [ -f "$base" ]; then
        echo "  Compiling: $base"
        ifx -c -I. "$base"
    fi
done

# Then compile domain-specific modules
echo "Step 4: Compiling domain-specific modules..."
for module in NWT1_module.f NWT1_ilupc_mod.f90 gwfsfrmodule_NWT.f gwfuzfmodule_NWT.f gwflakmodule_NWT.f; do
    if [ -f "$module" ]; then
        echo "  Compiling: $module"
        ifx -c -I. "$module" || echo "Warning: Failed to compile $module, continuing..."
    fi
done

# Then compile solvers
echo "Step 5: Compiling solvers..."
for solver in NWT1_xmdlib.f NWT1_xmd.f NWT1_gmres.f90 NWT1_solver.f sip7_NWT.f pcg7_NWT.f; do
    if [ -f "$solver" ]; then
        echo "  Compiling: $solver"
        ifx -c -I. "$solver" || echo "Warning: Failed to compile $solver, continuing..."
    fi
done

# Then compile remaining base components
echo "Step 6: Compiling remaining base components..."
for base in parutl7.f gwf2bas7_NWT.f; do
    if [ -f "$base" ]; then
        echo "  Compiling: $base"
        ifx -c -I. "$base" || echo "Warning: Failed to compile $base, continuing..."
    fi
done

# Compile all remaining files
echo "Step 7: Compiling remaining files..."
for file in *.f*; do
    # Check if the object file already exists (compiled in previous steps or failed)
    obj_file="${file%.*}.o"
    if [ ! -f "$obj_file" ]; then
        echo "  Compiling: $file"
        ifx -c -I. "$file" || echo "Warning: Failed to compile $file, continuing..."
    fi
done

# Try again for any files that may have failed due to dependency issues
echo "Step 8: Retry compiling any failed files..."
for file in *.f*; do
    # Check if the object file still doesn't exist
    obj_file="${file%.*}.o"
    if [ ! -f "$obj_file" ]; then
        echo "  Retrying: $file"
        ifx -c -I. "$file" || echo "Warning: Failed to compile $file on retry, continuing..."
    fi
done

# Link all object files
echo "Linking object files..."
ifx -o modflow-nwt *.o || {
    echo "Error during linking. Trying an alternative approach..."
    # Try linking with specific object files first, then the rest
    ifx -o modflow-nwt mach_mod.o modules.o openspec.o NWT1_module.o *.o
}

# Move the executable to the bin directory
if [ -f "modflow-nwt" ]; then
    echo "Cleaning up..."
    # Ensure the bin directory exists
    mkdir -p "${BIN_DIR}"
    
    # Copy the executable to the bin directory
    cp modflow-nwt "${BIN_DIR}/"
    if [ -f "${BIN_DIR}/modflow-nwt" ]; then
        chmod +x "${BIN_DIR}/modflow-nwt"
        echo "Compilation completed successfully!"
        echo "Executable copied to ${BIN_DIR}/modflow-nwt"
    else
        echo "Error: Failed to copy the executable to ${BIN_DIR}/"
        exit 1
    fi
    
    exit 0
else
    echo "Compilation failed at linking stage. Object files preserved for debugging."
    echo "Build directory: ${BUILD_DIR}/MODFLOW-NWT/build"
    exit 1
fi