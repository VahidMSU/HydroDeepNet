#!/bin/bash

# Colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Add the current directory to PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Ensure logs directory exists
LOG_DIR="/data/SWATGenXApp/codes/AI_agent/logs"
mkdir -p "$LOG_DIR"

# Function to print colored text
print_color() {
    color=$1
    text=$2
    echo -e "${color}${text}${NC}"
}

# Function to check if Python modules are available
check_dependencies() {
    print_color $BLUE "Checking dependencies..."
    
    # Check for required Python packages
    if ! python3 -c "import agno" 2>/dev/null; then
        print_color $YELLOW "Warning: agno module not found. Installing required packages..."
        if [ -f "${SCRIPT_DIR}/requirements/requirements.txt" ]; then
            pip install -r "${SCRIPT_DIR}/requirements/requirements.txt"
        else
            print_color $RED "Error: requirements.txt not found"
            exit 1
        fi
    fi
    
    # Check for spaCy model
    if ! python3 -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null; then
        print_color $YELLOW "Installing spaCy model..."
        python3 -m spacy download en_core_web_sm
    fi
    
    print_color $GREEN "Dependencies verified."
}

# Function to run the application
run_app() {
    local base_dir=$1
    
    if [ -z "$base_dir" ]; then
        base_dir="/data/SWATGenXApp/Users/admin/Reports/"
    fi
    
    print_color $GREEN "Starting Enhanced Hydrology Report Analyzer"
    print_color $BLUE "Report directory: $base_dir"
    print_color $BLUE "Log directory: $LOG_DIR" 
    
    # Run the integration script
    python3 "${SCRIPT_DIR}/integration.py" --base-dir "$base_dir"
}

# Main function
main() {
    # Check command line arguments
    local base_dir=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dir)
                base_dir="$2"
                shift 2
                ;;
            --help)
                print_color $GREEN "Enhanced Hydrology Report Analyzer"
                echo "Usage: ./run.sh [options]"
                echo ""
                echo "Options:"
                echo "  --dir PATH    Specify reports directory"
                echo "  --help        Show this help message"
                exit 0
                ;;
            *)
                print_color $RED "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Check dependencies
    check_dependencies
    
    # Run application
    run_app "$base_dir"
}

# Call main function with all arguments
main "$@" 