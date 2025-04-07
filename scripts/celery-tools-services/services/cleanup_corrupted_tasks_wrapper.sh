#!/bin/bash
# Wrapper script for the cleanup_corrupted_tasks.py tool
# This script provides a simpler interface for the cleanup tool

# Define paths
SCRIPT_DIR="${BASE_DIR}/scripts/celery-tools-services"
PYTHON_SCRIPT="${SCRIPT_DIR}/utils/cleanup_corrupted_tasks.py"
VENV_PYTHON="${BASE_DIR}/.venv/bin/python"
LOG_DIR="${BASE_DIR}/web_application/logs"
LOG_FILE="${LOG_DIR}/celery_cleanup_$(date +%Y%m%d_%H%M%S).log"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Output header
echo "Running Celery task cleanup tool at $(date)"
echo "Command: $0 $@"

# Check if Python virtual environment exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Python virtual environment not found at $VENV_PYTHON"
    echo "Please make sure the virtual environment is properly set up."
    exit 1
fi

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    echo "Please make sure the script is properly installed."
    exit 1
fi

# Add timeout configuration
TIMEOUT_SECONDS=60

# Run the Python script with all arguments and a timeout
echo "Starting Celery task cleanup with Python at $VENV_PYTHON (timeout: ${TIMEOUT_SECONDS}s)"
echo "------------------------"

# Use timeout command to prevent hanging
timeout $TIMEOUT_SECONDS $VENV_PYTHON "$PYTHON_SCRIPT" "$@" 2>&1 | tee "$LOG_FILE"
RESULT=${PIPESTATUS[0]}
echo "------------------------"

# Check result
if [ $RESULT -eq 0 ]; then
    echo "Celery task cleanup completed successfully"
elif [ $RESULT -eq 124 ]; then
    echo "Celery task cleanup timed out after ${TIMEOUT_SECONDS} seconds"
    echo "This may indicate the script is stuck or processing too much data."
    echo "Consider running with --limit option to process fewer items at once."
    exit 124
else
    echo "Celery task cleanup failed with exit code $RESULT"
fi

echo "Log file: $LOG_FILE"

exit $RESULT
