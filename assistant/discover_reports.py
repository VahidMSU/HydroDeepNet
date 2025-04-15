import os
import json
from pathlib import Path

def add_file_to_group(group_data, file_path, file_name, logger=None):
    """Helper function to add a file to a group's file structure"""
    logger.info(f"Adding file to group... {file_path}")
    if os.path.isfile(file_path):
        file_ext = Path(file_name).suffix.lower()
        
        # Categorize files by extension
        if file_ext not in group_data["files"]:
            group_data["files"][file_ext] = {}
        
        # Add file to its category using filename as key
        # If file with same name exists, create a unique key by adding the directory
        if file_name in group_data["files"][file_ext]:
            # Create a relative path from the file location to make it unique
            rel_dir = os.path.basename(os.path.dirname(file_path))
            unique_name = f"{rel_dir}_{file_name}"
            group_data["files"][file_ext][unique_name] = {
                "path": file_path
            }
        else:
            group_data["files"][file_ext][file_name] = {
                "path": file_path
            }


def discover_reports(base_dir, silent=False, recursive=True, logger=None):
    """
    Discovers all reports in the base directory and organizes them in a structured dictionary.
    
    Args:
        base_dir: Base directory containing report folders
        silent: If True, suppress output messages
        recursive: If True, search subdirectories recursively
        
    Returns:
        Dictionary containing report structures organized by report name and group
    """
    logger.info("Discovering reports...")
    # Dictionary to store report structures
    reports_dict = {}
    
    # Get all report folders (timestamps)
    report_names = [d for d in os.listdir(base_dir) 
                   if os.path.isdir(os.path.join(base_dir, d))]
    
    for report_name in report_names:
        report_path = os.path.join(base_dir, report_name)
        
        # Initialize report entry
        reports_dict[report_name] = {
            "path": report_path,
            "config": None,
            "groups": {}
        }
        
        # Check for config file
        config_path = os.path.join(report_path, "config.json")
        if os.path.exists(config_path):
            reports_dict[report_name]["config"] = config_path
        
        # Get report groups (subdirectories)
        report_groups = [d for d in os.listdir(report_path) 
                        if os.path.isdir(os.path.join(report_path, d))]
        
        # Process each report group
        for group_name in report_groups:
            group_path = os.path.join(report_path, group_name)
            
            # Initialize group entry
            reports_dict[report_name]["groups"][group_name] = {
                "path": group_path,
                "files": {}
            }
            
            # Process files recursively or non-recursively
            if recursive:
                # Walk through all subdirectories
                for root, dirs, files in os.walk(group_path):
                    for file_name in files:
                        file_path = os.path.join(root, file_name)
                        # Add the file to the group
                        add_file_to_group(reports_dict[report_name]["groups"][group_name], file_path, file_name, logger)
            else:
                # Just get files in the group directory (non-recursive)
                group_files = os.listdir(group_path)
                for file_name in group_files:
                    file_path = os.path.join(group_path, file_name)
                    if os.path.isfile(file_path):
                        add_file_to_group(reports_dict[report_name]["groups"][group_name], file_path, file_name, logger)
    
    # Save the report structure to a JSON file
    output_path = os.path.join(base_dir, "report_structure.json")
    with open(output_path, 'w') as f:
        json.dump(reports_dict, f, indent=2)
    
    if not silent:
        logger.info(f"Report structure saved to {output_path}")
    return reports_dict
