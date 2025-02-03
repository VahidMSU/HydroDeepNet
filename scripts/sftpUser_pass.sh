#!/bin/bash
echo "🔍 Running Final SFTP Security Check..."

# Check Password Authentication
echo "📌 Checking SSH password authentication..."
ssh_password_auth=$(sudo grep "^PasswordAuthentication" /etc/ssh/sshd_config | awk '{print $2}')
if [ "$ssh_password_auth" == "yes" ]; then
    echo "✅ Password authentication is enabled."
else
    echo "❌ WARNING: Password authentication is disabled!"
fi

# Check if any SFTP user can log in without a password
echo "📌 Checking SFTP users for empty passwords..."
empty_password_users=$(sudo awk -F: '($2 == "" ) {print $1}' /etc/shadow)
if [ -z "$empty_password_users" ]; then
    echo "✅ All SFTP accounts have passwords."
else
    echo "❌ WARNING: The following users have empty passwords:"
    echo "$empty_password_users"
fi

# Check directory permissions
echo "📌 Checking SFTP user directory permissions..."
for dir in /data/SWATGenXApp/Users/*; do
    user=$(basename "$dir")
    perm=$(ls -ld "$dir")
    echo "🔹 User: $user"
    echo "   $perm"
    if [[ "$perm" =~ "drwxrwx---" ]]; then
        echo "   ✅ Directory permissions are correct."
    else
        echo "   ❌ WARNING: Incorrect permissions!"
    fi
done

# Check ACL rules
echo "📌 Checking ACL rules..."
for dir in /data/SWATGenXApp/Users/*; do
    getfacl "$dir"
done

echo "🔍 Final SFTP Security Check Completed!"
