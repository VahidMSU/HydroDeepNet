import subprocess

def create_sftp_user(username):
    """Creates an SFTP user with restricted access while ensuring Apache access."""
    try:
        user_home = f"/data/SWATGenXApp/Users/{username}"
        upload_dir = f"{user_home}/uploads"

        # 1. Create the user (no shell access, belongs to 'sftp_users' group)
        subprocess.run(["sudo", "useradd", "-m", "-d", user_home, "-s", "/usr/sbin/nologin", "-g", "sftp_users", username], check=True)
        
        # 2. Create necessary directories
        subprocess.run(["sudo", "mkdir", "-p", upload_dir], check=True)

        # 3. Set permissions to allow Apache full control
        subprocess.run(["sudo", "chown", "www-data:www-data", user_home], check=True)  # Apache owns home dir
        subprocess.run(["sudo", "chmod", "2775", user_home], check=True)  # SetGID ensures group ownership persists
        subprocess.run(["sudo", "chown", f"{username}:www-data", upload_dir], check=True)  # User owns uploads, but Apache has access
        subprocess.run(["sudo", "chmod", "2770", upload_dir], check=True)  # No access for others

        # 4. Set ACLs to persist permissions
        subprocess.run(["sudo", "setfacl", "-R", "-m", "u:www-data:rwx", user_home], check=True)
        subprocess.run(["sudo", "setfacl", "-R", "-m", "g:www-data:rwx", user_home], check=True)
        subprocess.run(["sudo", "setfacl", "-R", "-d", "-m", "u:www-data:rwx", user_home], check=True)
        subprocess.run(["sudo", "setfacl", "-R", "-d", "-m", "g:www-data:rwx", user_home], check=True)

        return {"message": f"SFTP user {username} created successfully with Apache access."}

    except subprocess.CalledProcessError as e:
        return {"error": f"Failed to create SFTP user {username}: {e}"}


def delete_sftp_user(username):
    """Deletes an SFTP user and cleans up their directories."""
    try:
        user_home = f"/data/SWATGenXApp/Users/{username}"

        # Remove user and home directory
        subprocess.run(["sudo", "userdel", "-r", username], check=True)
        subprocess.run(["sudo", "rm", "-rf", user_home], check=True)

        return {"message": f"SFTP user {username} deleted successfully."}
    
    except subprocess.CalledProcessError as e:
        return {"error": f"Failed to delete SFTP user {username}: {e}"}
