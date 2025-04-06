#!/bin/bash

set -e  # Exit on any error

BASE_DIR="/data/SWATGenXApp"
CODES_DIR="${BASE_DIR}/codes"
APPDATA_DIR="${BASE_DIR}/GenXAppData"
USERS_DIR="${BASE_DIR}/Users"
GROUP_NAME="www-data"

# 1. Set ownership and permissions for the base directories
echo "Setting ownership and permissions for base directories..."
sudo chown -R www-data:${GROUP_NAME} "${BASE_DIR}"
sudo chmod -R 2775 "${BASE_DIR}"

# Apply ACL to ensure future files inherit permissions
sudo setfacl -R -m u:www-data:rwx -m g:${GROUP_NAME}:rwx "${BASE_DIR}"
sudo setfacl -R -d -m u:www-data:rwx -m g:${GROUP_NAME}:rwx "${BASE_DIR}"

# 2. Set permissions for specific subdirectories
echo "Setting permissions for ${CODES_DIR}..."
sudo chown -R www-data:${GROUP_NAME} "${CODES_DIR}"
sudo chmod -R 2775 "${CODES_DIR}"
sudo setfacl -R -m u:www-data:rwx -m g:${GROUP_NAME}:rwx "${CODES_DIR}"
sudo setfacl -R -d -m u:www-data:rwx -m g:${GROUP_NAME}:rwx "${CODES_DIR}"

echo "Setting permissions for ${APPDATA_DIR}..."
sudo chown -R www-data:${GROUP_NAME} "${APPDATA_DIR}"
sudo chmod -R 2775 "${APPDATA_DIR}"
sudo setfacl -R -m u:www-data:rwx -m g:${GROUP_NAME}:rwx "${APPDATA_DIR}"
sudo setfacl -R -d -m u:www-data:rwx -m g:${GROUP_NAME}:rwx "${APPDATA_DIR}"

# 3. Set up permissions for user directories
echo "Setting permissions for user directories..."
for userdir in "${USERS_DIR}"/*; do
    if [ -d "$userdir" ]; then
        username=$(basename "$userdir")
        echo "Configuring permissions for $userdir (User: $username)..."
        sudo chown -R "${username}:${GROUP_NAME}" "$userdir"
        sudo chmod -R 2775 "$userdir"
        sudo setfacl -R -m u:"${username}":rwx -m u:www-data:rwx -m g:${GROUP_NAME}:rwx "$userdir"
        sudo setfacl -R -d -m u:"${username}":rwx -m u:www-data:rwx -m g:${GROUP_NAME}:rwx "$userdir"
    fi
done

echo "Permissions and ownership setup completed successfully!"
