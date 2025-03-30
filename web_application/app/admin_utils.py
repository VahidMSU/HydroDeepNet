import sys
import os
import subprocess
from app.models import User
from app.extensions import db
from app.sftp_manager import delete_sftp_user

def remove_user(username, delete_files=True):
    """
    Completely remove a user from the system, including database entry, 
    SFTP account, and optionally all files.
    
    Args:
        username (str): The username to remove
        delete_files (bool): Whether to delete user files (default: True)
        
    Returns:
        dict: Result of the operation with status and message
    """
    results = {
        "status": "success",
        "message": f"User {username} removed successfully",
        "details": {}
    }
    
    # Step 1: Remove database entry
    try:
        user = User.query.filter_by(username=username).first()
        if user:
            db.session.delete(user)
            db.session.commit()
            results["details"]["database"] = "User removed from database"
        else:
            results["details"]["database"] = "User not found in database"
    except Exception as e:
        results["status"] = "error"
        results["details"]["database"] = f"Error removing user from database: {str(e)}"
    
    # Step 2: Remove SFTP account
    try:
        sftp_result = delete_sftp_user(username)
        if sftp_result.get("error"):
            results["details"]["sftp"] = f"Error removing SFTP account: {sftp_result.get('error')}"
        else:
            results["details"]["sftp"] = "SFTP account removed successfully"
    except Exception as e:
        results["details"]["sftp"] = f"Error removing SFTP account: {str(e)}"
    
    # Step 3: Remove user files if requested
    if delete_files:
        try:
            user_dir = f"/data/SWATGenXApp/Users/{username}"
            if os.path.exists(user_dir):
                # Use subprocess for permission handling
                subprocess.run(["sudo", "rm", "-rf", user_dir], check=True)
                results["details"]["files"] = "User files removed successfully"
            else:
                results["details"]["files"] = "User directory not found"
        except Exception as e:
            results["status"] = "error"
            results["details"]["files"] = f"Error removing user files: {str(e)}"
    
    # If any steps failed, update overall status
    if "error" in results["status"]:
        results["message"] = f"Error removing user {username}. See details for more information."
    
    return results

def run_system_maintenance():
    """
    Perform system maintenance tasks like cleaning temporary files and
    checking system health.
    
    Returns:
        dict: Results of maintenance operations
    """
    results = {
        "status": "success",
        "tasks": {}
    }
    
    # Clean temporary files
    try:
        temp_dir = "/data/SWATGenXApp/temp"
        if os.path.exists(temp_dir):
            # Remove files older than 7 days
            cmd = f"find {temp_dir} -type f -mtime +7 -delete"
            subprocess.run(cmd, shell=True, check=True)
            results["tasks"]["temp_cleanup"] = "Cleaned temporary files older than 7 days"
        else:
            results["tasks"]["temp_cleanup"] = "Temporary directory not found"
    except Exception as e:
        results["status"] = "partial"
        results["tasks"]["temp_cleanup"] = f"Error cleaning temporary files: {str(e)}"
    
    # Check disk space
    try:
        df_output = subprocess.check_output(["df", "-h", "/data"]).decode("utf-8")
        results["tasks"]["disk_space"] = df_output.strip()
    except Exception as e:
        results["status"] = "partial"
        results["tasks"]["disk_space"] = f"Error checking disk space: {str(e)}"
    
    # Check Redis connection
    try:
        from app.comm_utils import check_redis_health
        redis_health = check_redis_health()
        results["tasks"]["redis_health"] = redis_health
    except Exception as e:
        results["status"] = "partial"
        results["tasks"]["redis_health"] = f"Error checking Redis health: {str(e)}"
    
    return results

if __name__ == "__main__":
    # Simple command-line interface for admin utilities
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python admin_utils.py remove_user <username> [--keep-files]")
        print("  python admin_utils.py maintenance")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "remove_user" and len(sys.argv) >= 3:
        username = sys.argv[2]
        keep_files = "--keep-files" in sys.argv
        print(f"Removing user {username} (keep files: {keep_files})")
        result = remove_user(username, delete_files=not keep_files)
        print(result)
    
    elif command == "maintenance":
        print("Running system maintenance...")
        result = run_system_maintenance()
        print(result)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)