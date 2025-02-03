#!/bin/bash
echo "🔍 Running SFTP Security Check..."

# Define the SFTP base directory
SFTP_BASE_DIR="/data/SWATGenXApp/Users"

# List all users
echo "📌 Checking SFTP user directories..."
for user in $(ls "$SFTP_BASE_DIR"); do
    USER_HOME="$SFTP_BASE_DIR/$user"
    UPLOAD_DIR="$USER_HOME/uploads"

    echo "🔹 User: $user"
    
    # 1️⃣ Check ownership of user home directory
    OWNER=$(stat -c "%U:%G" "$USER_HOME")
    if [[ "$OWNER" == "www-data:www-data" ]]; then
        echo "   ✅ Ownership is correct (www-data)"
    else
        echo "   ❌ WARNING: Incorrect ownership ($OWNER)"
    fi

    # 2️⃣ Check permissions
    PERMS=$(stat -c "%A" "$USER_HOME")
    if [[ "$PERMS" == "drwxrwsr-x" ]]; then
        echo "   ✅ Directory permissions are correct"
    else
        echo "   ❌ WARNING: Permissions incorrect ($PERMS)"
    fi

    # 3️⃣ Ensure users cannot access each other’s files
    OTHER_USERS_ACCESS=$(find "$SFTP_BASE_DIR" -type d ! -group www-data -perm /o+rwx)
    if [[ -z "$OTHER_USERS_ACCESS" ]]; then
        echo "   ✅ No unauthorized access detected"
    else
        echo "   ❌ WARNING: Other users may have access to files!"
    fi

    # 4️⃣ Check that the uploads folder is owned by the user
    UPLOAD_OWNER=$(stat -c "%U:%G" "$UPLOAD_DIR")
    if [[ "$UPLOAD_OWNER" == "$user:www-data" ]]; then
        echo "   ✅ Uploads folder ownership is correct"
    else
        echo "   ❌ WARNING: Incorrect ownership for uploads ($UPLOAD_OWNER)"
    fi

    # 5️⃣ Check ACL settings
    ACLS=$(getfacl -p "$USER_HOME" 2>/dev/null)
    if echo "$ACLS" | grep -q "user:www-data:rwx"; then
        echo "   ✅ ACLs correctly set for Apache"
    else
        echo "   ❌ WARNING: Missing ACL permissions for Apache"
    fi

    echo "----------------------------------"
done

# 6️⃣ Check SSH Configuration for SFTP restrictions
echo "📌 Checking SSH SFTP configuration..."
if grep -q "Match Group sftp_users" /etc/ssh/sshd_config; then
    echo "   ✅ SSH SFTP restrictions are enabled"
else
    echo "   ❌ WARNING: SFTP restrictions are missing in /etc/ssh/sshd_config"
fi

if grep -q "ChrootDirectory $SFTP_BASE_DIR" /etc/ssh/sshd_config; then
    echo "   ✅ ChrootDirectory is correctly configured"
else
    echo "   ❌ WARNING: ChrootDirectory is missing or incorrect"
fi

# 7️⃣ Check firewall rules
echo "📌 Checking firewall settings..."
if sudo ufw status | grep -q "OpenSSH"; then
    echo "   ✅ SSH/SFTP access is properly configured"
else
    echo "   ❌ WARNING: Firewall rules might be incorrect"
fi

echo "🔍 SFTP Security Check Completed!"
