import os
import json
from pathlib import Path

def discover_reports(base_dir="/data/SWATGenXApp/Users/admin/Reports/"):
    """
    Discovers all reports in the base directory and organizes them in a structured dictionary.
    
    Args:
        base_dir: Base directory containing report folders
        
    Returns:
        Dictionary containing report structures organized by report name and group
    """
    print(f"Scanning for reports in: {base_dir}")
    
    # Dictionary to store report structures
    reports_dict = {}
    
    # Get all report folders (timestamps)
    report_names = [d for d in os.listdir(base_dir) 
                   if os.path.isdir(os.path.join(base_dir, d))]
    
    for report_name in report_names:
        report_path = os.path.join(base_dir, report_name)
        print(f"Processing report: {report_name}")
        
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
            print(f"  Found config: {config_path}")
        
        # Get report groups (subdirectories)
        report_groups = [d for d in os.listdir(report_path) 
                        if os.path.isdir(os.path.join(report_path, d))]
        
        # Process each report group
        for group_name in report_groups:
            group_path = os.path.join(report_path, group_name)
            print(f"  Processing group: {group_name}")
            
            # Initialize group entry
            reports_dict[report_name]["groups"][group_name] = {
                "path": group_path,
                "files": {}
            }
            
            # Get all files in the group
            group_files = os.listdir(group_path)
            
            # Process files by type
            for file_name in group_files:
                file_path = os.path.join(group_path, file_name)
                if os.path.isfile(file_path):
                    file_ext = Path(file_name).suffix.lower()
                    
                    # Categorize files by extension
                    if file_ext not in reports_dict[report_name]["groups"][group_name]["files"]:
                        reports_dict[report_name]["groups"][group_name]["files"][file_ext] = []
                    
                    # Add file to its category
                    reports_dict[report_name]["groups"][group_name]["files"][file_ext].append({
                        "name": file_name,
                        "path": file_path
                    })
                    
                    print(f"    Added file: {file_name}")
    
    # Save the report structure to a JSON file
    output_path = os.path.join(base_dir, "report_structure.json")
    with open(output_path, 'w') as f:
        json.dump(reports_dict, f, indent=2)
    
    print(f"Report structure saved to {output_path}")
    return reports_dict

if __name__ == "__main__":
    reports = discover_reports()
    
    # Print summary
    print("\nReport Summary:")
    for report_name, report_data in reports.items():
        print(f"Report: {report_name}")
        print(f"  Config: {'Yes' if report_data['config'] else 'No'}")
        
        for group_name, group_data in report_data["groups"].items():
            file_count = sum(len(files) for files in group_data["files"].values())
            print(f"  Group: {group_name} ({file_count} files)")
            
            for ext, files in group_data["files"].items():
                print(f"    {ext}: {len(files)} files")
    
    print("\nUse reports_dict to access all report paths organized by report and group")  



