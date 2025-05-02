import os
import json
from pathlib import Path
try:
    from .Logger import LoggerSetup
except ImportError:
    from Logger import LoggerSetup
def discover_reports(base_dir):
    """
    Discovers all files in the base directory and organizes them in a structured dictionary.
    
    Args:
        base_dir: Base directory containing report folders

    Returns:
        Dictionary containing report structures organized by report name and group
    """
    # Dictionary to store report structures
    reports_dict = {}
    logger = LoggerSetup(rewrite=False, verbose=True)

    for report_name in os.listdir(base_dir):
        report_path = os.path.join(base_dir, report_name)
        if os.path.isdir(report_path):
            reports_dict[report_name] = {
                "path": report_path,
                "files": {}
            }
            ### now collect all files in the report and group them by file type
            for file_name in os.listdir(report_path):
                file_path = os.path.join(report_path, file_name)
                if os.path.isfile(file_path):
                    reports_dict[report_name]["files"][file_name] = {
                        "path": file_path
                    }
    
    ## save the reports_dict to a file
    with open(Path(base_dir) / "report_structure.json", "w") as f:
        json.dump(reports_dict, f)

    # Remove debug print statement
    logger.info(f"Discovered {len(reports_dict)} report groups")
    
    return reports_dict

if __name__ == "__main__":
    base_dir = "/data/SWATGenXApp/Users/admin/Reports/20250412_172208"
    reports_dict = discover_reports(base_dir)
    print(f"report groups: {reports_dict}")
