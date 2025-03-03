import sys
import os

def run_swatplus_editor(SWATGenXPaths, vpuid: str, level: str, name: str, model_name: str) -> None:
    """Run the SWAT+ Editor for the specified model."""
    sys.path.append(SWATGenXPaths.SWATPlusEditor_path)
    from actions.run_all import RunAll
    from rest.setup import check_config
    base_model = os.path.join(SWATGenXPaths.swatgenx_outlet_path, vpuid, level, name)
    assert os.path.exists(base_model), f"Model does not exist in {base_model}"
    print(f"Base model: {base_model}")
    ###
    project_db = os.path.join(base_model, model_name, f"{model_name}.sqlite")
    input_files_dir = os.path.join(base_model, model_name, "Scenarios", "Default", "TxtInOut")
    
    print(f"Project DB: {project_db}")

    check_config(project_db=project_db)

    RunAll(
        project_db=project_db,
        editor_version="2.3.3",
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


if __name__ == '__main__':

    from SWATGenXConfigPars import SWATGenXPaths
    SWATGenXPaths = SWATGenXPaths(username = "vahidr32" )
    SWATGenXPaths.exe_start_year = 2000
    SWATGenXPaths.exe_end_year = 2021
    print(f" self.swatgenx_outlet_path: {SWATGenXPaths.swatgenx_outlet_path}")
    run_swatplus_editor(SWATGenXPaths, "0712", "huc12", "05536265", "SWAT_MODEL_Web_Application")
