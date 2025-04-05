#!/bin/bash
# Wrapper script to run the Celery task cleanup tool with correct Python environment

PYTHON_ENV="/data/SWATGenXApp/codes/.venv/bin/python"
SCRIPT_PATH="/data/SWATGenXApp/codes/scripts/celery-services/services/cleanup_corrupted_tasks.py"
LOG_DIR="/data/SWATGenXApp/codes/web_application/logs"
LOG_FILE="${LOG_DIR}/celery_cleanup_$(date +%Y%m%d_%H%M%S).log"

# Ensure the log directory exists
mkdir -p "${LOG_DIR}"

# Print usage information
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --dry-run, -d    Identify corrupted messages without removing them"
    echo "  --repair, -r     Try to repair corrupted messages instead of just removing them"
    echo "  --queue, -q      Specific queue(s) to check (default: all Celery queues)"
    echo "  --yes, -y        Skip confirmation prompts"
    echo "  --clean          Disable colored output"
    echo "  --fix-wrongtype  Scan and fix WRONGTYPE Redis keys only"
    echo "  --pattern        Redis key pattern to scan for WRONGTYPE issues"
    echo "  --help, -h       Show this help message"
}

# Check if help is requested
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    usage
    exit 0
fi

# Log the command execution
echo "Running Celery task cleanup tool at $(date)" | tee -a "${LOG_FILE}"
echo "Command: $0 $@" | tee -a "${LOG_FILE}"

# Check if we're running as root
if [[ $EUID -ne 0 ]]; then
   echo "This script should be run with sudo" | tee -a "${LOG_FILE}"
   exit 1
fi

# Execute the Python script with the virtual environment Python
echo "Starting Celery task cleanup with Python at ${PYTHON_ENV}" | tee -a "${LOG_FILE}"
echo "------------------------" | tee -a "${LOG_FILE}"

${PYTHON_ENV} ${SCRIPT_PATH} "$@" 2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

echo "------------------------" | tee -a "${LOG_FILE}"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Celery task cleanup completed successfully" | tee -a "${LOG_FILE}"
else
    echo "Celery task cleanup failed with exit code ${EXIT_CODE}" | tee -a "${LOG_FILE}"
fi

echo "Log file: ${LOG_FILE}"
exit $EXIT_CODE
