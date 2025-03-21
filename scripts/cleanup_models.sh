#!/bin/bash
# filepath: /data/SWATGenXApp/codes/scripts/cleanup_models.sh

# Base path for user directories
PATH_BASE="/data/SWATGenXApp/Users"
# List of usernames
USERNAMES=("guest" "admin" "vahidr32" "rafieiva")

# Function to handle directory removal with permission fixes
remove_dir() {
    local dir=$1
    echo "Removing $dir"
    
    # First try simple removal
    if rm -rf "$dir" 2>/dev/null; then
        echo "Successfully removed $dir"
        return 0
    fi
    
    # If simple removal failed, try fixing permissions first
    echo "Permission issues with $dir, trying to fix permissions"
    
    # Try to make everything writable
    find "$dir" -type d -exec chmod u+w {} \; 2>/dev/null
    find "$dir" -type f -exec chmod u+w {} \; 2>/dev/null
    
    # Try removal again
    if rm -rf "$dir" 2>/dev/null; then
        echo "Successfully removed $dir after permission fix"
        return 0
    else
        echo "Failed to remove $dir, you may need sudo"
        return 1
    fi
}

# Process each username
for username in "${USERNAMES[@]}"; do
    swat_dir="$PATH_BASE/$username/SWATplus_by_VPUID"
    
    # Check if directory exists
    if [ -d "$swat_dir" ]; then
        remove_dir "$swat_dir"
    else
        echo "Path $swat_dir does not exist"
    fi
done

# Check for any remaining directories and suggest sudo command
echo ""
echo "If any directories couldn't be removed, try using sudo:"
for username in "${USERNAMES[@]}"; do
    swat_dir="$PATH_BASE/$username/SWATplus_by_VPUID"
    if [ -d "$swat_dir" ]; then
        echo "sudo rm -rf $swat_dir"
    fi
done