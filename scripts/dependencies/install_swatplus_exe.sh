#!/bin/bash

# Script to compile SWAT+ model
# This script uses CMake and the available presets to build SWAT+

# Colors for better output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

BIN_DIR="/data/SWATGenXApp/codes/bin"
if [ ! -d "$BIN_DIR" ]; then
    echo -e "${RED}Error: Bin directory not found at $BIN_DIR${NC}"
    exit 1
fi

## if already installed, exit
if [ -f "$BIN_DIR/swatplus" ]; then
    echo -e "${GREEN}SWAT+ is already installed at $BIN_DIR/swatplus${NC}"
    exit 0
fi

BASE_DIR="/data/SWATGenXApp/codes/scripts/dependencies"

if [ ! -d "$BASE_DIR" ]; then
    echo -e "${RED}Error: Base directory not found at $BASE_DIR${NC}"
    exit 1
fi


# Set the SWAT+ directory
SWATPLUS_DIR="$BASE_DIR/swatplus"


# Check if SWAT+ directory exists
if [ ! -d "$SWATPLUS_DIR" ]; then
    echo -e "${RED}Error: SWAT+ directory not found at $SWATPLUS_DIR${NC}"
    ### clone from https://github.com/VahidMSU/swatplus.git
    echo -e "${GREEN}Cloning SWAT+ repository...${NC}"
    cd $BASE_DIR
    git clone https://github.com/VahidMSU/swatplus.git
fi



# Detect system
SYSTEM=$(uname -s)
echo -e "${GREEN}Detected system: ${YELLOW}$SYSTEM${NC}"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for Intel oneAPI environment
intel_oneapi_detected=false
potential_oneapi_paths=(
    "/opt/intel/oneapi/setvars.sh"
    "/usr/local/intel/oneapi/setvars.sh"
    "$HOME/intel/oneapi/setvars.sh"
)

# Function to source Intel oneAPI environment
activate_oneapi() {
    echo -e "${GREEN}Activating Intel oneAPI environment...${NC}"
    
    # Check common paths for setvars.sh
    for oneapi_path in "${potential_oneapi_paths[@]}"; do
        if [ -f "$oneapi_path" ]; then
            echo -e "${GREEN}Found oneAPI at: ${YELLOW}$oneapi_path${NC}"
            # Source the oneAPI environment
            source "$oneapi_path"
            return 0
        fi
    done
    
    # Ask user for custom path if not found in common locations
    echo -e "${YELLOW}Intel oneAPI environment not found in common locations.${NC}"
    echo -e "${GREEN}Please enter the full path to setvars.sh (or leave empty to skip):${NC}"
    read custom_oneapi_path
    
    if [ -n "$custom_oneapi_path" ] && [ -f "$custom_oneapi_path" ]; then
        echo -e "${GREEN}Using oneAPI at: ${YELLOW}$custom_oneapi_path${NC}"
        source "$custom_oneapi_path"
        return 0
    elif [ -n "$custom_oneapi_path" ]; then
        echo -e "${RED}Error: File not found at $custom_oneapi_path${NC}"
    fi
    
    # If we get here, oneAPI wasn't activated
    echo -e "${YELLOW}Intel oneAPI environment not activated. Intel compilers may not work correctly.${NC}"
    return 1
}

# Check for required tools
echo -e "${GREEN}Checking required tools...${NC}"
if ! command_exists cmake; then
    echo -e "${RED}Error: CMake is not installed or not in PATH${NC}"
    echo -e "${YELLOW}Please install CMake from http://cmake.org${NC}"
    exit 1
fi

# Check for Intel compilers or environment
if command_exists ifort || command_exists ifx; then
    intel_oneapi_detected=true
else
    # Check if we can find the oneAPI environment script
    for oneapi_path in "${potential_oneapi_paths[@]}"; do
        if [ -f "$oneapi_path" ]; then
            intel_oneapi_detected=true
            break
        fi
    done
fi

# Ask if user wants to use Intel compilers
if [ "$intel_oneapi_detected" = true ]; then
    echo -e "${GREEN}Intel oneAPI environment detected. Do you want to use Intel compilers? (y/n)${NC}"
    read use_intel
    if [ "$use_intel" = "y" ] || [ "$use_intel" = "Y" ]; then
        activate_oneapi
    fi
fi

# Change to SWAT+ directory
cd "$SWATPLUS_DIR"

# Check available compilers and store them
available_compilers=()
recommended_preset=""

if command_exists gfortran; then
    echo -e "${GREEN}Found compiler: ${YELLOW}gfortran${NC}"
    available_compilers+=("gfortran")
    
    # Set recommended preset based on OS
    if [ "$SYSTEM" == "Linux" ]; then
        recommended_preset="gfortran_debug_linux"
    elif [ "$SYSTEM" == "Darwin" ]; then
        recommended_preset="gfortran_debug_macbook"
    elif [ "$SYSTEM" == "Windows_NT" ] || [[ "$SYSTEM" == MINGW* ]] || [[ "$SYSTEM" == MSYS* ]]; then
        recommended_preset="gfortran_debug_windows"
    fi
fi

if command_exists ifort; then
    echo -e "${GREEN}Found compiler: ${YELLOW}ifort${NC}"
    available_compilers+=("ifort")
    # If gfortran wasn't found or if Intel compilers are preferred
    recommended_preset="ifort_debug"
fi

if command_exists ifx; then
    echo -e "${GREEN}Found compiler: ${YELLOW}ifx${NC}"
    available_compilers+=("ifx")
    # If ifx is available, prefer it over ifort
    recommended_preset="ifx_debug"
fi

if [ ${#available_compilers[@]} -eq 0 ]; then
    echo -e "${RED}Error: No Fortran compiler found. Please install gfortran, ifort, or ifx.${NC}"
    exit 1
fi

# Detect available CMake presets
echo -e "${GREEN}Available CMake presets:${NC}"
cmake --list-presets

# Recommend a preset
if [ -n "$recommended_preset" ]; then
    echo -e "${GREEN}Recommended preset for your system: ${YELLOW}$recommended_preset${NC}"
fi

# Prompt for preset choice
echo -e "${GREEN}Please choose a preset to use (enter the name, or press Enter for recommended):${NC}"
read preset_choice

# Use recommended preset if none provided
if [ -z "$preset_choice" ] && [ -n "$recommended_preset" ]; then
    preset_choice="$recommended_preset"
    echo -e "${GREEN}Using recommended preset: ${YELLOW}$preset_choice${NC}"
fi

# Validate preset choice
if ! cmake --list-presets | grep -q "$preset_choice"; then
    echo -e "${RED}Error: Invalid preset name${NC}"
    exit 1
fi

# Check if the selected preset is compatible with available compilers
preset_compiler=""
if [[ "$preset_choice" == *"gfortran"* ]]; then
    preset_compiler="gfortran"
elif [[ "$preset_choice" == *"ifort"* ]]; then
    preset_compiler="ifort"
elif [[ "$preset_choice" == *"ifx"* ]]; then
    preset_compiler="ifx"
fi

# Verify the chosen preset's compiler is available
compiler_available=false
for comp in "${available_compilers[@]}"; do
    if [ "$comp" == "$preset_compiler" ]; then
        compiler_available=true
        break
    fi
done

if [ "$compiler_available" = false ] && [ -n "$preset_compiler" ]; then
    echo -e "${RED}Warning: The preset '$preset_choice' requires the $preset_compiler compiler, which is not available on your system.${NC}"
    echo -e "${YELLOW}Available compilers: ${available_compilers[*]}${NC}"
    echo -e "${YELLOW}You might need to activate the Intel oneAPI environment first.${NC}"
    echo -e "${GREEN}Do you want to continue anyway? (y/n)${NC}"
    read continue_choice
    if [ "$continue_choice" != "y" ] && [ "$continue_choice" != "Y" ]; then
        echo -e "${YELLOW}Exiting. Please choose a preset compatible with your available compilers.${NC}"
        exit 1
    fi
fi

# Configure with chosen preset
echo -e "${GREEN}Configuring SWAT+ with preset: ${YELLOW}$preset_choice${NC}"
if ! cmake --preset "$preset_choice" .; then
    echo -e "${RED}Error: Configuration failed${NC}"
    echo -e "${YELLOW}Check if the compiler specified in the preset is available on your system.${NC}"
    echo -e "${YELLOW}You might need to install the compiler or activate Intel oneAPI.${NC}"
    exit 1
fi

# Build
echo -e "${GREEN}Building SWAT+...${NC}"
build_dir=$(cmake --preset "$preset_choice" --fresh -N . | grep "Build files have been written to:" | sed 's/Build files have been written to: //')
if [ -z "$build_dir" ]; then
    build_dir="build/debug"  # Use default if couldn't detect
fi

cd "$build_dir"
if ! cmake --build .; then
    echo -e "${RED}Error: Build failed${NC}"
    exit 1
fi

echo -e "${GREEN}Build completed successfully!${NC}"
echo -e "${GREEN}Executable location: ${YELLOW}$build_dir/swatplus_exe${NC}"

# Optional: Install
echo -e "${GREEN}Do you want to install SWAT+? (y/n)${NC}"
read install_choice
if [ "$install_choice" = "y" ] || [ "$install_choice" = "Y" ]; then
    echo -e "${GREEN}Installing SWAT+...${NC}"
    if ! cmake --install .; then
        echo -e "${RED}Error: Installation failed${NC}"
        exit 1
    fi
    echo -e "${GREEN}Installation completed successfully!${NC}"
fi

echo -e "${GREEN}SWAT+ compilation process completed.${NC}"

# Locate and copy the compiled swatplus executable
final_exe=$(find . -maxdepth 1 -type f -executable -name "swatplus*" | head -n 1)

if [ -f "$final_exe" ]; then
    # Ensure the bin directory exists
    mkdir -p "$BIN_DIR"
    
    # Copy the executable to the bin directory
    cp "$final_exe" "$BIN_DIR/swatplus"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Successfully copied executable to ${YELLOW}$BIN_DIR/swatplus${NC}"
        # Make it executable
        chmod +x "$BIN_DIR/swatplus"
    else
        echo -e "${RED}Error: Failed to copy executable to $BIN_DIR/swatplus${NC}"
        exit 1
    fi
else
    echo -e "${RED}Error: Compiled executable not found in $build_dir${NC}"
    # Try harder to find the executable anywhere in the build directory
    final_exe=$(find "$build_dir" -type f -executable -name "swatplus*" | head -n 1)
    
    if [ -f "$final_exe" ]; then
        mkdir -p "$BIN_DIR"
        cp "$final_exe" "$BIN_DIR/swatplus" && \
        chmod +x "$BIN_DIR/swatplus" && \
        echo -e "${GREEN}Successfully copied executable to ${YELLOW}$BIN_DIR/swatplus${NC}" || \
        echo -e "${RED}Error: Failed to copy executable to $BIN_DIR/swatplus${NC}"
    else
        echo -e "${RED}Error: Could not find any swatplus executable in the build directory${NC}"
        exit 1
    fi
fi
