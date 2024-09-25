import os
import shutil

def remove_git_directories(base_directory):
    for root, dirs, files in os.walk(base_directory):
        if '.git' in dirs:
            git_dir_path = os.path.join(root, '.git')
            print(f"Removing {git_dir_path}")
            shutil.rmtree(git_dir_path)

def remove_image_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                file_path = os.path.join(root, file)
                print(f"Removing image file {file_path}")
                os.remove(file_path)


def remove_ml_models(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pth") or file.endswith(".h5"):
                file_path = os.path.join(root, file)
                print(f"Removing model file {file_path}")
                os.remove(file_path)

def remove_large_files(directory, size_limit_mb):
    size_limit_bytes = size_limit_mb * 1024 * 1024
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.getsize(file_path) > size_limit_bytes:
                print(f"Removing large file {file_path}")
                os.remove(file_path)

if __name__ == "__main__":
    directories_to_clean = [
        "./SWATplus_development/aarthi_work/SWATplus_development/40601020807/",
        "./GeoCNN/models/",
        "./HydroGeo/SWATplus_performance/input_shape/",
        "./PostProcessing/model_bounds/",
        "./HydroGeo/SWATplus_performance/results/model_bounds_avg/",
        "./HydroGeo/SWATplus_performance/results/model_bounds_daily/",
        "./HydroGeo/SWATplus_performance/results/model_bounds_huc8/",
        "./HydroGeo/SWATplus_performance/results/model_bounds_monthly/",
        "./GeoCNN/figs/grids/",
        "./GeoCNN/figs_scatters/",
        "./PFAS_GNN/Hetero_data/",
        "./gw_machine_learning/SWAT_recharge_hru/results/",
        "./gw_machine_learning/input_videos/",
        "./PFAS_GNN/figs/",
    ]
    size_limit_mb = 0
    for directory in directories_to_clean:
        remove_large_files(directory, size_limit_mb)

    main_directory = "./"
    remove_git_directories(main_directory)

    remove_image_files(main_directory)

    remove_ml_models(main_directory)