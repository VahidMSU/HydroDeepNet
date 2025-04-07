#!/bin/bash

# Get absolute path of the current script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Go two levels up to get HOME_DIR (i.e., /data/SWATGenXApp)
HOME_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_NAME="$(basename $(dirname "$SCRIPT_DIR"))"  # e.g., 'codes'

BASE_DIR="${HOME_DIR}/${REPO_NAME}"
WEBAPP_DIR="${BASE_DIR}/web_application"
DEPENDENCIES_DIR="${SCRIPT_DIR}/dependencies"
USER_DIR="${HOME_DIR}/Users"
DATA_DIR="${HOME_DIR}/GenXAppData"
LOG_DIR="${WEBAPP_DIR}/logs"
CONFIG_DIR="${SCRIPT_DIR}/config"
SYSTEMD_DIR="/etc/systemd/system"
APACHE_DIR="/etc/apache2/sites-available"
BIN_DIR="${BASE_DIR}/bin"
PYTHONPATH="${BASE_DIR}/.venv/bin"
SWATGENXAPP_DIR="${BASE_DIR}/SWATGenXApp"

echo "$SCRIPT_DIR"
echo "GLOBAL PATHS"
echo "===================="
echo "HOME_DIR: $HOME_DIR"
echo "REPO_NAME: $REPO_NAME"
echo "BASE_DIR: $BASE_DIR"
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "WEBAPP_DIR: $WEBAPP_DIR"
echo "DEPENDENCIES_DIR: $DEPENDENCIES_DIR"
echo "USER_DIR: $USER_DIR"
echo "DATA_DIR: $DATA_DIR"
echo "LOG_DIR: $LOG_DIR"
echo "CONFIG_DIR: $CONFIG_DIR"
echo "SYSTEMD_DIR: $SYSTEMD_DIR"
echo "APACHE_DIR: $APACHE_DIR"
echo "BIN_DIR: $BIN_DIR"
echo "PYTHONPATH: $PYTHONPATH"
echo "SWATGENXAPP_DIR: $SWATGENXAPP_DIR"
echo "===================="
