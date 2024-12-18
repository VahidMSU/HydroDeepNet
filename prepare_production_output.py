import os
import shutil
import tarfile

# Define streamflow station name
NAME = "04176500"

# Define destination and source paths
destination_path = f"/data/Generated_models/full_model/{NAME}"
os.makedirs(destination_path, exist_ok=True)

source_path = f'/data2/MyDataBase/SWATGenXAppData/SWAT_input/huc12/{NAME}'
modflow_model = os.path.join(source_path, "MODFLOW_250m")
swat_model = os.path.join(source_path, "SWAT_MODEL")

source_path_2 = f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}"
swat_gwflow_model = os.path.join(source_path_2, "SWAT_gwflow_MODEL")
cal_parameters = os.path.join(source_path_2, "cal_parms_SWAT_gwflow_MODEL.cal")
best_parameters = os.path.join(source_path_2, "best_solution_SWAT_gwflow_MODEL.txt")
streamflow_data = os.path.join(source_path_2, "streamflow_data")
modis_et = os.path.join(source_path_2, "MODIS_ET/MODIS_ET.csv")
verification_videos = os.path.join(source_path_2, "figures_SWAT_gwflow_MODEL/verifications_videos")
verification_plots = os.path.join(source_path_2, "figures_SWAT_gwflow_MODEL/watershed_static_plots")
GlobalBestImprovement_png = os.path.join(source_path_2, "figures_SWAT_gwflow_MODEL/GlobalBestImprovement.png")
SF = os.path.join(source_path_2,"figures_SWAT_gwflow_MODEL/SF")
GW = os.path.join(source_path_2,"figures_SWAT_gwflow_MODEL/GW")
SWL = os.path.join(source_path_2,"figures_SWAT_gwflow_MODEL/SWL")
ET = os.path.join(source_path_2,"figures_SWAT_gwflow_MODEL/ET")
YLD = os.path.join(source_path_2,"figures_SWAT_gwflow_MODEL/YLD")

# Helper function to copy files or directories
def copy_if_not_exists(src, dest):
    if not os.path.exists(dest):
        if os.path.isdir(src):
            shutil.copytree(src, dest)
        else:
            shutil.copy(src, dest)
os.makedirs(os.path.join(destination_path, "figs"), exist_ok=True)
# Ensure required files and directories exist
assert os.path.exists(modflow_model), f"MODFLOW model not found at {modflow_model}"
assert os.path.exists(swat_model), f"SWAT model not found at {swat_model}"
assert os.path.exists(swat_gwflow_model), f"SWAT_gwflow_MODEL not found at {swat_gwflow_model}"
assert os.path.exists(cal_parameters), f"Calibration parameters not found at {cal_parameters}"
assert os.path.exists(best_parameters), f"Best parameters not found at {best_parameters}"
assert os.path.exists(streamflow_data), f"Streamflow data not found at {streamflow_data}"
assert os.path.exists(modis_et), f"MODIS ET data not found at {modis_et}"
assert os.path.exists(verification_videos), f"Verification videos not found at {verification_videos}"
assert os.path.exists(verification_plots), f"Verification plots not found at {verification_plots}"
assert os.path.exists(SF), f"Verification plots not found at {SF}"
assert os.path.exists(GW), f"Verification plots not found at {GW}"
assert os.path.exists(SWL), f"Verification plots not found at {SWL}"
assert os.path.exists(ET), f"Verification plots not found at {ET}"
assert os.path.exists(YLD), f"Verification plots not found at {YLD}"
assert os.path.exists(GlobalBestImprovement_png), f"Verification plots not found at {GlobalBestImprovement_png}"
## 
# Copy necessary files and directories to the destination
copy_if_not_exists(modflow_model, os.path.join(destination_path, "MODFLOW_250m"))
copy_if_not_exists(swat_model, os.path.join(destination_path, "SWAT_MODEL"))
copy_if_not_exists(swat_gwflow_model, os.path.join(destination_path, "SWAT_gwflow_MODEL"))
copy_if_not_exists(cal_parameters, os.path.join(destination_path, "cal_parms_SWAT_gwflow_MODEL.cal"))
copy_if_not_exists(best_parameters, os.path.join(destination_path, "best_solution_SWAT_gwflow_MODEL.txt"))
copy_if_not_exists(streamflow_data, os.path.join(destination_path, "streamflow_data"))
copy_if_not_exists(modis_et, os.path.join(destination_path, "MODIS_ET.csv"))
copy_if_not_exists(verification_videos, os.path.join(destination_path, "figs/verifications_videos"))
copy_if_not_exists(verification_plots, os.path.join(destination_path, "figs/watershed_static_plots"))
copy_if_not_exists(GlobalBestImprovement_png, os.path.join(destination_path, "figs/GlobalBestImprovement.png"))



copy_if_not_exists(SF, os.path.join(destination_path, "figs/SF"))
copy_if_not_exists(GW, os.path.join(destination_path, "figs/GW"))
copy_if_not_exists(SWL, os.path.join(destination_path, "figs/SWL"))
copy_if_not_exists(ET, os.path.join(destination_path, "figs/ET"))
copy_if_not_exists(YLD, os.path.join(destination_path, "figs/YLD"))




### remove verficiation_stage_1-5
# Remove additional scenarios
def remove_additional_scenarios(directory):
    for i in range(1, 6):
        scenario_path = os.path.join(directory, f"verification_stage_{i}")
        if os.path.exists(scenario_path):
            shutil.rmtree(scenario_path)

remove_additional_scenarios(os.path.join(destination_path, "SWAT_gwflow_MODEL/Scenarios"))


### trim the figures and only keep the last 100 files based on the creation time
def trim_figures(directory):
    files = os.listdir(directory)
    files.sort(key=lambda x: os.path.getctime(os.path.join(directory, x)))
    for file in files[:-100]:
        os.remove(os.path.join(directory, file))

trim_figures(os.path.join(destination_path, "figs/SF/calibration/daily"))
trim_figures(os.path.join(destination_path, "figs/SF/calibration/monthly"))
trim_figures(os.path.join(destination_path, "figs/GW"))
trim_figures(os.path.join(destination_path, "figs/SWL"))
trim_figures(os.path.join(destination_path, "figs/ET"))
trim_figures(os.path.join(destination_path, "figs/YLD"))



# Create a README file
readme_path = os.path.join(destination_path, "readme.md")
with open(readme_path, "w") as readme:
    readme.write(f"# Model Documentation for Streamflow Station: {NAME}\n")
    readme.write("This directory contains models and supporting data for the watershed.\n\n")
    readme.write("## Directory Contents\n")
    readme.write("1. `MODFLOW_250m/`: MODFLOW model (5 layers, 250m resolution).\n")
    readme.write("   - 1-1: `MODFLOW_250m/rasters_input`: Contains the input rasters for the MODFLOW model.\n")
    readme.write("2. `SWAT_MODEL/`: Raw SWAT+ model built upon NHDPlus HR (1:24k resolution), NLCD (250m resolution), gSSURGO (250m resolution), and DEM (30m resolution).\n")
    readme.write("3. `SWAT_gwflow_MODEL/`: SWAT model coupled with GWFlow and calibrated using PSO.\n")
    readme.write("   - The objective function is the sum of NSE performance for streamflow, ET, GW, SWL, and Yield.\n")
    readme.write("   - The weight of NSE for streamflow, SWL, and GW is 1, and for ET and Yield is 0.1.\n")
    readme.write("   - The GWFlow is a simplified version of the MODFLOW model and has 1 layer divided into 4 homogeneous zones (compared to 5 layers with heterogeneous properties in MODFLOW).\n")
    readme.write("4. `cal_parms_SWAT_gwflow_MODEL.cal`: Calibration parameters and ranges.\n")
    readme.write("5. `streamflow_data/`: Observed streamflow data (named by SWAT+ Channel gis_id and USGS station ID, e.g., `1_04176500` where gis_id is 1 and USGS station ID is 04176500).\n")
    readme.write("6. `MODIS_ET.csv`: MODIS ET monthly data for 20 years, aggregated for each landscape unit in SWAT+ models.\n")
    readme.write("7. `verifications_videos/`: Animations of verification results.\n")
    readme.write("8. `watershed_static_plots/`: Static plots of watershed properties.\n\n")
    readme.write("### Calibration Details\n")
    readme.write("- Model already executed for 20 years of daily outputs with the best parameters: `SWAT_gwflow_MODEL/Scenarios/verification_stage_0`.\n")
    readme.write("- ##### Important note: SWAT+ H5 database containing monthly `hru_wb` (water balance) data in 2D format for snowmelt, precipitation, generated surface runoff, percolation, ET, and other static variables of the model: `verification_stage_0/SWATplus_output.h5`.\n")
    readme.write("- Additional scenarios: `SWAT_gwflow_MODEL/Scenarios/verification_stage_1-5` (top-performing parameters).\n")
    readme.write("\n### Streamflow Data\n")
    readme.write("- Located in `streamflow_data/`.\n")
    readme.write("- Naming format: `channel_id_station_id` (e.g., `1_04176500`).\n")
    readme.write("\n### Notes:\n")
    readme.write("- The figures contain the last 100 executed model output performances during calibration and validation.\n")
    readme.write("- Models are calibrated in bulk; fine-tuning by users to achieve desirable performance might still be needed.\n")
    readme.write("- `GlobalBestImprovement.png` shows the performance of the model during the calibration period.\n")

import os
import subprocess

# Define the tarball path
tarball_path = f"/data/Generated_models/full_model/{NAME}.tar.gz"

# Remove the tarball if it already exists
if os.path.exists(tarball_path):
    ### copy to /data2/MyDataBase/out/generated_models
    shutil.copy(tarball_path, f"/data2/MyDataBase/out/generated_models/{NAME}.tar.gz")
else:

    # Create the tarball using tar with pigz for parallel compression
    try:
        subprocess.run(
            [
                "tar",
                "--use-compress-program=pigz",
                "-cf",
                tarball_path,
                "-C",
                "/data/Generated_models/full_model",
                NAME,
            ],
            check=True,
        )
        print(f"Tarball created successfully: {tarball_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during compression: {e}")
