"""
FTPS Manager Module for SWATGenX App
Handles interaction with the FTPS server through the system's ftps_user_manager.sh script
"""
import subprocess
import json
import os
from flask import current_app
import secrets
import string
import logging

# Set up logging
logger = logging.getLogger(__name__)

def _run_ftps_command(command, username=None):
    """
    Run the FTPS user management script with the given command and username.
    
    Args:
        command (str): The command to run (create, delete, list)
        username (str, optional): The username for the command
        
    Returns:
        dict: A dictionary containing the result of the operation
    """
    script_path = current_app.config['FTPS_SCRIPT_PATH']
    
    # Check if the script exists and is executable
    if not os.path.isfile(script_path):
        logger.error(f"FTPS manager script not found at {script_path}")
        return {"success": False, "message": "FTPS manager script not found"}
    
    # Make sure the script is executable
    try:
        os.chmod(script_path, 0o755)  # rwxr-xr-x
    except Exception as e:
        logger.error(f"Failed to set executable permissions on FTPS script: {e}")
        # Continue anyway, in case it's already executable
    
    # Build the command
    cmd = ["sudo", script_path, command]
    if username:
        cmd.append(username)
    
    try:
        # Execute the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30  # Set a reasonable timeout
        )
        
        # Check for errors
        if result.returncode != 0:
            logger.error(f"FTPS command failed: {result.stderr}")
            return {
                "success": False,
                "message": f"Error executing FTPS command: {result.stderr}",
                "details": result.stdout
            }
        
        # Process the output
        output = result.stdout.strip()
        
        # For create command, parse credentials from output
        if command == "create":
            # Extract credentials from output
            username_line = next((line for line in output.split('\n') if line.startswith("Username:")), None)
            password_line = next((line for line in output.split('\n') if line.startswith("Password:")), None)
            
            if username_line and password_line:
                username = username_line.split(":", 1)[1].strip()
                password = password_line.split(":", 1)[1].strip()
                
                return {
                    "success": True,
                    "message": "FTPS user created successfully",
                    "username": username,
                    "password": password,
                    "server": "ciwre-bae.campusad.msu.edu",
                    "port": 990,
                    "protocol": "FTPS (FTP over SSL/TLS)",
                    "details": output
                }
        
        # Default response for other commands
        return {
            "success": True,
            "message": output,
            "details": output
        }
        
    except subprocess.TimeoutExpired:
        logger.error("FTPS command timed out")
        return {"success": False, "message": "FTPS command timed out"}
    except Exception as e:
        logger.error(f"Error executing FTPS command: {e}")
        return {"success": False, "message": f"Error: {str(e)}"}

def create_ftps_user(username):
    """
    Create a new FTPS user with access to the SWATGenX data directory.
    
    Args:
        username (str): The username to create
        
    Returns:
        dict: A dictionary containing the result of the operation and credentials
    """
    logger.info(f"Creating FTPS user: {username}")
    return _run_ftps_command("create", username)

def delete_ftps_user(username):
    """
    Delete an existing FTPS user.
    
    Args:
        username (str): The username to delete
        
    Returns:
        dict: A dictionary containing the result of the operation
    """
    logger.info(f"Deleting FTPS user: {username}")
    return _run_ftps_command("delete", username)

def list_ftps_users():
    """
    List all FTPS users.
    
    Returns:
        dict: A dictionary containing the list of users
    """
    logger.info("Listing FTPS users")
    result = _run_ftps_command("list")
    
    # Process the output to extract usernames
    if result["success"]:
        output = result["message"]
        users = []
        
        # Skip the header line "FTPS Users:" and extract usernames
        for line in output.split('\n'):
            if line and not line.startswith("FTPS Users:"):
                users.append(line.strip())
        
        result["users"] = users
    
    return result