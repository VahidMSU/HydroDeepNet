import sys
sys.path.append('C:/SWAT/SWATPlus/SWATPlusEditor/swatplus.editor/src/api')
from actions.run_all import RunAll
from rest.setup import check_config
import os



def run_swatplusEditor(VPUID, LEVEL, NAME, MODEL_NAME):
	base_model = f"/data/MyDataBase/SWATGenXAppData/SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/"
	project_db_file = os.path.join(base_model,f"{MODEL_NAME}/{MODEL_NAME}.sqlite")
	wgn_db = "C:/SWAT/SWATPlus/Databases/swatplus_wgn.sqlite"
	swat_exe_file = "swatplus"
	weather_dir = os.path.join(base_model,"PRISM")
	input_files_dir = os.path.join(base_model,f"{MODEL_NAME}/Scenarios/Default/TxtInOut")
	weather_save_dir = os.path.join(base_model,f"{MODEL_NAME}/Scenarios/Default/TxtInOut")
	weather_import_format = "plus"
	wgn_import_method = "database"
	wgn_table = "wgn_cfsr_world"
	wgn_csv_sta_file = None
	wgn_csv_mon_file = None
	year_start = 2000
	day_start = 1
	year_end = 2001
	day_end = 1
	editor_version = "2.3.3"

	check_config(project_db_file)
	RunAll(project_db_file, editor_version, swat_exe_file,
			weather_dir, weather_save_dir, weather_import_format,
			wgn_import_method, wgn_db, wgn_table, wgn_csv_sta_file, wgn_csv_mon_file,
			year_start, day_start, year_end, day_end,
			input_files_dir, swat_version="SWAT+")




if __name__ == '__main__':
    VPUID = "0202"
    LEVEL = "huc12"
    NAME = "01330000"
    MODEL_NAME = "SWAT_MODEL"
    run_swatplusEditor(VPUID, LEVEL, NAME, MODEL_NAME)