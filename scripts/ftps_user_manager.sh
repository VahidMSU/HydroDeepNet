#!/bin/bash
# FTPS user management script for SWATGenX

set -e

# Configuration
FTPS_ROOT="/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID"
USERLIST_FILE="/etc/vsftpd.allowed_users"
PASSWORD_LENGTH=12

# Log file for operations
LOG_FILE="/var/log/ftps_user_management.log"

# Ensure the script is run as root
if [ "$(id -u)" -ne 0 ]; then
    echo "This script must be run as root" >&2
    exit 1
fi

# Create directories and files if they don't exist
mkdir -p "$(dirname $LOG_FILE)"
touch $LOG_FILE
chmod 640 $LOG_FILE

if [ ! -f "$USERLIST_FILE" ]; then
    touch "$USERLIST_FILE"
    chmod 600 "$USERLIST_FILE"
fi

# Function to log operations
log_operation() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> $LOG_FILE
}

# Function to generate a secure random password
generate_password() {
    # Generate a secure random password with a mix of characters
    password=$(tr -dc 'A-Za-z0-9_!@#$%^&*()' < /dev/urandom | head -c $PASSWORD_LENGTH)
    echo $password
}

# Function to create a new FTPS user
create_user() {
    local username=$1
    
    # Check if username is valid
    if [[ ! $username =~ ^[a-zA-Z][a-zA-Z0-9_-]*$ ]]; then
        echo "Invalid username format. Username must start with a letter and contain only letters, numbers, underscores, and hyphens."
        log_operation "Failed to create user $username: Invalid username format"
        return 1
    fi
    
    # Check if user already exists
    if id "$username" &>/dev/null; then
        echo "User $username already exists"
        log_operation "Failed to create user $username: User already exists"
        return 1
    fi
    
    # Generate a secure password
    password=$(generate_password)
    
    # Create the user with no login shell and a home directory in the FTP root
    useradd -m -d "$FTPS_ROOT" -s /usr/sbin/nologin "$username"
    
    # Set the password
    echo "$username:$password" | chpasswd
    
    # Add user to the allowed users list
    if ! grep -q "^$username$" "$USERLIST_FILE"; then
        echo "$username" >> "$USERLIST_FILE"
    fi
    
    # Restrict user permissions to read-only
    chown -R root:root "$FTPS_ROOT"
    chmod -R 755 "$FTPS_ROOT"
    
    # Set user's permissions to read-only access
    setfacl -m u:$username:r-x "$FTPS_ROOT"
    
    # Apply ACLs recursively
    find "$FTPS_ROOT" -type d -exec setfacl -m u:$username:r-x {} \;
    find "$FTPS_ROOT" -type f -exec setfacl -m u:$username:r-- {} \;
    
    log_operation "Created FTPS user $username"
    
    # Output the credentials
    echo "FTPS User created successfully:"
    echo "Username: $username"
    echo "Password: $password"
    echo "Server: ciwre-bae.campusad.msu.edu (35.9.219.73)"
    echo "Port: 990"
    echo "Protocol: FTPS (FTP over SSL/TLS)"
    echo "Directory: /SWATplus_by_VPUID"
}

# Function to delete an FTPS user
delete_user() {
    local username=$1
    
    # Check if user exists
    if ! id "$username" &>/dev/null; then
        echo "User $username does not exist"
        log_operation "Failed to delete user $username: User does not exist"
        return 1
    fi
    
    # Remove user from the allowed users list
    sed -i "/^$username$/d" "$USERLIST_FILE"
    
    # Delete the user
    userdel -r "$username" 2>/dev/null || true
    
    log_operation "Deleted FTPS user $username"
    echo "FTPS User $username deleted successfully"
}

# Function to list all FTPS users
list_users() {
    if [ -f "$USERLIST_FILE" ]; then
        echo "FTPS Users:"
        cat "$USERLIST_FILE"
    else
        echo "No FTPS users found"
    fi
}

# Main script logic
if [ $# -lt 1 ]; then
    echo "Usage: $0 [create|delete|list] [username]"
    echo ""
    echo "Commands:"
    echo "  create <username>  - Create a new FTPS user"
    echo "  delete <username>  - Delete an existing FTPS user"
    echo "  list              - List all FTPS users"
    exit 1
fi

command=$1
username=$2

case $command in
    create)
        if [ -z "$username" ]; then
            echo "Error: Username is required for create command"
            exit 1
        fi
        create_user "$username"
        ;;
    delete)
        if [ -z "$username" ]; then
            echo "Error: Username is required for delete command"
            exit 1
        fi
        delete_user "$username"
        ;;
    list)
        list_users
        ;;
    *)
        echo "Unknown command: $command"
        echo "Usage: $0 [create|delete|list] [username]"
        exit 1
        ;;
esac

exit 0