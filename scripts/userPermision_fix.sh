#!/bin/bash

echo "🔍 Fixing SFTP User Directory Permissions & ACLs..."

# 1️⃣ Set correct base directory permissions
echo "📌 Updating SFTP user directory permissions..."
sudo chmod -R 770 /data/SWATGenXApp/Users

# 2️⃣ Apply ACLs for only necessary users (Restrict access)
echo "📌 Applying Correct ACL Rules..."
for user in $(ls /data/SWATGenXApp/Users); do
    USER_HOME="/data/SWATGenXApp/Users/$user"

    # Grant full access to www-data and the user, REMOVE other users
    sudo setfacl -R -m u:www-data:rwx "$USER_HOME"
    sudo setfacl -R -m u:$user:rwx "$USER_HOME"
    sudo setfacl -R -m g:www-data:rwx "$USER_HOME"

    # Remove explicit ACLs for other users (Fixes Malformed ACL Error)
    sudo setfacl -R -x u:rafieiva "$USER_HOME" 2>/dev/null
    sudo setfacl -R -x u:alishaha "$USER_HOME" 2>/dev/null

    # Remove 'other' access completely
    sudo setfacl -R -m o::--- "$USER_HOME"
    sudo setfacl -R -d -m o::--- "$USER_HOME"

    # Apply default ACL rules to persist new files/folders
    sudo setfacl -R -d -m u:www-data:rwx "$USER_HOME"
    sudo setfacl -R -d -m u:$user:rwx "$USER_HOME"
    sudo setfacl -R -d -m g:www-data:rwx "$USER_HOME"
done

echo "✅ SFTP User Permissions and ACLs Fixed!"
