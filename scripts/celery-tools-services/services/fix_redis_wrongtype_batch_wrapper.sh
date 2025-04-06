#!/bin/bash
# Wrapper script for the batch-oriented Redis WRONGTYPE fix tool

PYTHON_ENV="/data/SWATGenXApp/codes/.venv/bin/python"
SCRIPT_PATH="/data/SWATGenXApp/codes/scripts/celery-tools-services/services/fix_redis_wrongtype_batch.py"
LOG_DIR="/data/SWATGenXApp/codes/web_application/logs"
LOG_FILE="${LOG_DIR}/redis_batch_fix_$(date +%Y%m%d_%H%M%S).log"

# Ensure the log directory exists
mkdir -p "${LOG_DIR}"

# Print usage information
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --pattern, -p       Redis key pattern to scan (default: celery-task-meta-*)"
    echo "  --dry-run, -d       Show what would be done without making changes"
    echo "  --batch-size, -b    Number of keys to process in each batch (default: 50)"
    echo "  --timeout           Timeout in seconds for the entire operation (default: 3600)" 
    echo "  --yes, -y           Skip confirmation prompts"
    echo "  --no-backup         Skip creating backups (faster but less safe)"
    echo "  --no-color          Disable colored output"
    echo "  --help, -h          Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --batch-size 100 --yes    # Process in batches of 100 keys with auto-confirmation"
}

# Check if help is requested
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    usage
    exit 0
fi

# Log the command execution
echo "Running Redis WRONGTYPE batch fix tool at $(date)" | tee -a "${LOG_FILE}"
echo "Command: $0 $@" | tee -a "${LOG_FILE}"

# Check if we're running as root
if [[ $EUID -ne 0 ]]; then
   echo "This script should be run with sudo" | tee -a "${LOG_FILE}"
   exit 1
fi

# Execute the Python script with the virtual environment Python
echo "Starting Redis batch fix operation with Python at ${PYTHON_ENV}" | tee -a "${LOG_FILE}"
echo "------------------------" | tee -a "${LOG_FILE}"

# Try to detect if we're in a terminal that supports interactive prompts
if [ -t 0 ]; then
    # Run in interactive mode
    ${PYTHON_ENV} ${SCRIPT_PATH} "$@" 2>&1 | tee -a "${LOG_FILE}"
else
    # Run in non-interactive mode (add --yes if not already present)
    if [[ "$*" != *"--yes"* && "$*" != *"-y"* ]]; then
        echo "Running in non-interactive mode, adding --yes automatically" | tee -a "${LOG_FILE}"
        ${PYTHON_ENV} ${SCRIPT_PATH} --yes "$@" 2>&1 | tee -a "${LOG_FILE}"
    else
        ${PYTHON_ENV} ${SCRIPT_PATH} "$@" 2>&1 | tee -a "${LOG_FILE}"
    fi
fi

EXIT_CODE=${PIPESTATUS[0]}

echo "------------------------" | tee -a "${LOG_FILE}"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Redis batch fix operation completed successfully" | tee -a "${LOG_FILE}"
else
    echo "Redis batch fix operation failed with exit code ${EXIT_CODE}" | tee -a "${LOG_FILE}"
fi

echo "Log file: ${LOG_FILE}"
exit $EXIT_CODE
