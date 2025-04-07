#!/bin/bash
# uninstall_celery.sh
# This script uninstalls Celery if it is installed and removes related
# systemd service files for celery-worker and celery-beat.
# Do not run this script with sudo.

CURRENT_DIR=$(dirname "$(readlink -f "$0")")
source "${CURRENT_DIR}/../global_path.sh"

# Use PYTHONPATH to run pip from the virtual environment
if "${PYTHONPATH}/pip" show celery &>/dev/null; then
    echo "Uninstalling Celery..."
    "${PYTHONPATH}/pip" uninstall -y celery && echo "Celery has been uninstalled."
else
    echo "Celery is not installed."
fi


