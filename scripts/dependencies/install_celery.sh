#!/bin/bash
# install_celery.sh
# This script installs Celery if it is not already installed.

CURRENT_DIR=$(dirname "$(readlink -f "$0")")
source "${CURRENT_DIR}/../global_path.sh"

if [ "$EUID" -eq 0 ]; then
    echo "Please do not run this script with sudo."
    exit 1
fi

if "${PYTHONPATH}/pip" show celery &>/dev/null; then
    echo "Celery is already installed."
else
    echo "Installing Celery..."
    pip install celery && echo "Celery has been installed."
fi