import sys
import os
import sqlite3

# Add a function to initialize the database
def initialize_database(project_db):
    try:
        conn = sqlite3.connect(project_db)
        cursor = conn.cursor()
        # Example initialization query (replace with actual initialization logic if needed)
        cursor.execute("PRAGMA foreign_keys = ON;")
        conn.commit()
        conn.close()
        print("Database initialized successfully")
    except sqlite3.Error as e:
        print(f"Error initializing database: {e}")
        raise

def run_swatplus_editor(SWATGenXPaths, vpuid: str, level: str, name: str, model_name: str) -> None:
    """Run the SWAT+ Editor for the specified model."""
    sys.path.append(SWATGenXPaths.SWATPlusEditor_path)
    from actions.run_all import RunAll
    from rest.setup import automatic_updates
    base_model = os.path.join(SWATGenXPaths.swatgenx_outlet_path, vpuid, level, name)
    assert os.path.exists(base_model), f"Model does not exist in {base_model}"
    print(f"Base model: {base_model}")
    ###
    project_db = os.path.join(base_model, model_name, f"{model_name}.sqlite")
    input_files_dir = os.path.join(base_model, model_name, "Scenarios", "Default", "TxtInOut")
    
    print(f"Project DB: {project_db}")
    
    # Check if the database file exists
    if not os.path.exists(project_db):
        print(f"Error: Database file does not exist at {project_db}")
        return
    
    # Check if the database file is accessible
    try:
        # Try to open the database to verify it's accessible
        conn = sqlite3.connect(project_db)
        conn.close()
        print("Successfully connected to the database")
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        return

    # Initialize the database
    try:
        initialize_database(project_db)
    except Exception as e:
        print(f"Error during database initialization: {e}")
        return

    # Use automatic_updates with proper error handling
    try:
        automatic_updates(project_db=project_db)
        print("Successfully ran automatic updates")
    except Exception as e:
        print(f"Error during automatic updates: {e}")
        return

    # Run SWAT+ with error handling
    try:

        RunAll(
            project_db=project_db,
            editor_version="3.0.8",
            swat_exe=SWATGenXPaths.swat_exe,
            weather_dir=os.path.join(base_model, "PRISM"),
            weather_save_dir=input_files_dir,
            weather_import_format="plus",
            wgn_import_method="database",
            wgn_db=SWATGenXPaths.wgn_db,
            wgn_table="wgn_cfsr_world",
            wgn_csv_sta_file=None,
            wgn_csv_mon_file=None,
            year_start=SWATGenXPaths.exe_start_year,
            day_start=1,
            year_end=SWATGenXPaths.exe_end_year,
            day_end=1,
            input_files_dir=input_files_dir,
            swat_version="SWAT+"
        )
        print("SWAT+ model simulation completed")

        ### there might a problem with the print.prt duplicate prints. we need to trim the print.prt file and run the program one more tjme
        print_path = os.path.join(input_files_dir, "print.prt")
        if os.path.exists(print_path):
            with open(print_path, 'r') as file:
                lines = file.readlines()
            # Remove duplicate lines
            unique_lines = list(dict.fromkeys(lines))
            ## continue if there is no duplicate lines
            if len(unique_lines) == len(lines):
                print("No duplicate lines found in print.prt")
                return
            print(f"Duplicate lines found in {print_path}. Writing unique lines to the file.")
            
            with open(print_path, 'w') as file:
                file.writelines(unique_lines)
            ### now run the executable again
            import subprocess
            subprocess.run([SWATGenXPaths.swat_exe, project_db], cwd=input_files_dir)
    except Exception as e:
        print(f"Error running SWAT+ model: {e}")
        return


if __name__ == '__main__':
    from runQSWATPlus import runQSWATPlus
    from SWATGenXConfigPars import SWATGenXPaths
    SWATGenXPaths = SWATGenXPaths(username = "admin" )
    SWATGenXPaths.exe_start_year = 2000
    SWATGenXPaths.exe_end_year = 2001
    print(f" self.swatgenx_outlet_path: {SWATGenXPaths.swatgenx_outlet_path}")
    runQSWATPlus("0408", "huc12", "04141000", "SWAT_MODEL",SWATGenXPaths)
    run_swatplus_editor(SWATGenXPaths, "0408", "huc12", "04141000", "SWAT_MODEL")
