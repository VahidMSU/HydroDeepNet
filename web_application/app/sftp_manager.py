## /data/SWATGenXApp/codes/web_application/app/sftp_manager.py
import subprocess
import subprocess
import subprocess

def create_sftp_user(username):
    """Creates an SFTP user with restricted access while ensuring Apache access."""
    try:
        user_home = f"/data/SWATGenXApp/Users/{username}"
        upload_dir = f"{user_home}/uploads"

        SUDO = "/usr/bin/sudo"
        USERADD = "/usr/sbin/useradd"
        MKDIR = "/bin/mkdir"
        CHOWN = "/bin/chown"
        CHMOD = "/bin/chmod"
        SETFACL = "/usr/bin/setfacl"
        USERDEL = "/usr/sbin/userdel"
        RM = "/bin/rm"

        ## if user already exists, delete it by sudo userdel -r {username}
        previously_existed = False
        try:
            subprocess.run([SUDO, "-n", USERDEL, "-r", username], check=True)
            previously_existed = True
        except subprocess.CalledProcessError as e:
            pass



        subprocess.run([SUDO, "-n", USERADD, "-m", "-d", user_home, "-s", "/usr/sbin/nologin", "-g", "sftp_users", username], check=True)
        subprocess.run([SUDO, "-n", MKDIR, "-p", upload_dir], check=True)
        subprocess.run([SUDO, "-n", CHOWN, "www-data:www-data", user_home], check=True)
        subprocess.run([SUDO, "-n", CHMOD, "2775", user_home], check=True)
        subprocess.run([SUDO, "-n", CHOWN, f"{username}:www-data", upload_dir], check=True)
        subprocess.run([SUDO, "-n", CHMOD, "2770", upload_dir], check=True)

        subprocess.run([SUDO, "-n", SETFACL, "-R", "-m", "u:www-data:rwx", user_home], check=True)
        subprocess.run([SUDO, "-n", SETFACL, "-R", "-m", "g:www-data:rwx", user_home], check=True)
        subprocess.run([SUDO, "-n", SETFACL, "-R", "-d", "-m", "u:www-data:rwx", user_home], check=True)
        subprocess.run([SUDO, "-n", SETFACL, "-R", "-d", "-m", "g:www-data:rwx", user_home], check=True)

        return {"status": "success", "message": f"SFTP user {username} created successfully with Apache access. Previously exists? {previously_existed}"}

    except subprocess.CalledProcessError as e:
        return {"status": "error", "error": f"Failed to create SFTP user {username}: {e}"}

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
