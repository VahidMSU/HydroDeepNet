#!/bin/bash

# Colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TOOLS_DIR="${SCRIPT_DIR}/tools"

# Add the current directory to PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}:${TOOLS_DIR}:${PYTHONPATH}"

# Print diagnostic info
echo "Script directory: ${SCRIPT_DIR}"
echo "Tools directory: ${TOOLS_DIR}"
echo "PYTHONPATH: ${PYTHONPATH}"

# Ensure logs directory exists
LOG_DIR="/data/SWATGenXApp/codes/AI_agent/logs"
mkdir -p "$LOG_DIR"

# Parse command line arguments
INPUT_FILE=""
OUTPUT_FILE=""
BASE_DIR="/data/SWATGenXApp/Users/admin/Reports/"
MODEL_ID="gpt-4o"
INTERACTIVE=true

# Process arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_FILE="$2"
            INTERACTIVE=false
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            INTERACTIVE=false
            shift 2
            ;;
        --dir)
            BASE_DIR="$2"
            shift 2
            ;;
        --model)
            MODEL_ID="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./run_ai.sh [options]"
            echo ""
            echo "Options:"
            echo "  --input PATH    Path to input JSON file containing message"
            echo "  --output PATH   Path to output JSON file for response"
            echo "  --dir PATH      Specify reports directory"
            echo "  --model ID      Specify model to use (default: gpt-4o)"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Function to print colored text
print_color() {
    color=$1
    text=$2
    echo -e "${color}${text}${NC}"
}

# Function to check if Python modules are available
check_dependencies() {
    print_color $BLUE "Checking dependencies..."
    
    # Set OUTPUT_FILE for error reporting if it's defined
    local output_file=$1
    
    # Print Python version for diagnosis
    print_color $BLUE "Python version:"
    python3 --version
    
    # List available Python packages
    print_color $BLUE "Available Python packages:"
    pip list | grep -E "agno|spacy|ollama"
    
    # Check for required Python packages
    if ! python3 -c "import agno" 2>/dev/null; then
        print_color $YELLOW "Warning: agno module not found. Installing required packages..."
        if [ -f "${SCRIPT_DIR}/requirements/requirements.txt" ]; then
            pip install -r "${SCRIPT_DIR}/requirements/requirements.txt"
        else
            print_color $RED "Error: requirements.txt not found"
            if [ -n "$output_file" ]; then
                echo '{"response": "Error: requirements.txt not found for installing dependencies"}' > "$output_file"
            fi
            exit 1
        fi
    fi
    
    # Check for spaCy model
    if ! python3 -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null; then
        print_color $YELLOW "Installing spaCy model..."
        python3 -m spacy download en_core_web_sm
    fi
    
    # Check for interactive_agent dependencies
    python3 -c "import sys; sys.path.append('${TOOLS_DIR}'); import interactive_agent; print('Successfully imported interactive_agent');" || {
        print_color $RED "Error importing interactive_agent module"
        if [ -n "$output_file" ]; then
            echo '{"response": "Error importing interactive_agent module. Check Python paths."}' > "$output_file"
        fi
        exit 1
    }
    
    print_color $GREEN "Dependencies verified."
}

# Function to run the application in interactive mode
run_interactive() {
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

# Function to run in API mode (non-interactive)
run_api_mode() {
    local input_file=$1
    local output_file=$2
    local base_dir=$3
    local model_id=$4
    
    # Check if input and output files are provided
    if [ -z "$input_file" ] || [ -z "$output_file" ]; then
        print_color $RED "Error: Input and output files must be specified in API mode"
        echo '{"response": "Error: Input and output files must be specified"}' > "$output_file"
        exit 1
    fi
    
    # Check if input file exists
    if [ ! -f "$input_file" ]; then
        print_color $RED "Error: Input file does not exist: $input_file"
        echo '{"response": "Error: Input file does not exist"}' > "$output_file"
        exit 1
    fi
    
    print_color $BLUE "Running in API mode"
    print_color $BLUE "Input file: $input_file"
    print_color $BLUE "Output file: $output_file"
    print_color $BLUE "Model: $model_id"
    
    # Read input data from the input file
    input_json=$(cat "$input_file")
    print_color $BLUE "Input data: $input_json"
    
    # Extract the message, and potentially other fields like session_id
    query=$(python3 -c "import json; print(json.load(open('$input_file')).get('message', ''))")
    session_id=$(python3 -c "import json; print(json.load(open('$input_file')).get('session_id', 'None'))")
    model=$(python3 -c "import json; print(json.load(open('$input_file')).get('model', '$model_id'))")
    
    if [ -z "$query" ]; then
        print_color $RED "Error: No message found in input file"
        echo '{"response": "Error: No message found in input file"}' > "$output_file"
        exit 1
    fi
    
    print_color $BLUE "Processing query: $query"
    print_color $BLUE "Using model: $model"
    print_color $BLUE "Session ID: $session_id"
    
    # Write an empty response to the output file first
    echo '{"response": "Processing..."}' > "$output_file"
    
    # Process the query using EnhancedReportAnalyzer
    python_script="
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
sys.path.insert(0, '${TOOLS_DIR}')

print('Python path:', sys.path)

try:
    from integration import EnhancedReportAnalyzer
    print('Successfully imported EnhancedReportAnalyzer')
    analyzer = EnhancedReportAnalyzer(base_dir='$base_dir', model_id='$model')
    print('Initialized analyzer')
    response = analyzer.process_query('$query', session_id='$session_id')
    print('Response generated')
    print(response)
except ImportError as e:
    print('Import error:', str(e))
    print('Could not import required modules')
    print('Python path:', sys.path)
    exit(1)
except Exception as e:
    print('Error processing query:', str(e))
    import traceback
    traceback.print_exc()
    exit(2)
"
    
    # Run the Python script and capture the output
    echo "$python_script" > /tmp/run_ai_script.py
    response=$(python3 /tmp/run_ai_script.py 2>&1)
    python_exit_code=$?
    
    # Check if Python script ran successfully
    if [ $python_exit_code -ne 0 ]; then
        print_color $RED "Error running Python script (exit code $python_exit_code):"
        print_color $RED "$response"
        echo "{\"response\": \"Error processing query. Technical details: $response\"}" > "$output_file"
        return 1
    fi
    
    # Extract the actual response from the output
    # This extracts the last line of output, which should be the response
    actual_response=$(echo "$response" | grep -v "^Python path:" | grep -v "^Successfully imported" | grep -v "^Initialized" | grep -v "^Response generated" | tail -1)
    
    # Write the response to the output file
    echo "{\"response\": \"${actual_response//\"/\\\"}\", \"model\": \"$model\", \"session_id\": \"$session_id\"}" > "$output_file"
    
    print_color $GREEN "Query processed and response written to $output_file"
    return 0
}

# Main function
main() {
    # Check dependencies (pass output file for error reporting)
    check_dependencies "$OUTPUT_FILE"
    
    # Run in appropriate mode
    if [ "$INTERACTIVE" = true ]; then
        run_interactive "$BASE_DIR"
    else
        run_api_mode "$INPUT_FILE" "$OUTPUT_FILE" "$BASE_DIR" "$MODEL_ID"
    fi
}

# Call main function
main "$@" 