import os
import shutil
import filecmp
from datetime import datetime

def sync_directories(path1, path2):
    # Walk through all subdirectories and files in path1
    print(f"Syncing directories {path1} and {path2}")
    for root, dirs, files in os.walk(path1):
        #print(f"Checking directory {root}")
        for file in files:
            print(f"Checking file {file}")
            # Determine the relative path of the current file
            rel_dir = os.path.relpath(root, path1)
            rel_file = os.path.join(rel_dir, file)
            file1 = os.path.join(path1, rel_file)
            file2 = os.path.join(path2, rel_file)

            # Check if the file exists in path2
            if not os.path.exists(file2):
                print(f"File {file2} does not exist, copying from {file1}")
                os.makedirs(os.path.dirname(file2), exist_ok=True)
               # shutil.copy2(file1, file2)

            else:
                # Compare files by content and modification time
                if filecmp.cmp(file1, file2):
                    #print(f"File {file1} and {file2} are identical, skipping.")
                    pass
                else:
                    # Check modification times
                    file1_mtime = os.path.getmtime(file1)
                    file2_mtime = os.path.getmtime(file2)

                    if file1_mtime > file2_mtime:
                        print(f"File {file1} is newer than {file2}, copying.")
               #         shutil.copy2(file1, file2)
                    else:
                        print(f"File {file2} is up-to-date, skipping.")

        # Handle subdirectories by ensuring they exist in path2
        for dir in dirs:
            dir1 = os.path.join(root, dir)
            rel_dir = os.path.relpath(dir1, path1)
            dir2 = os.path.join(path2, rel_dir)
            if not os.path.exists(dir2):
                print(f"Directory {dir2} does not exist, creating.")
                os.makedirs(dir2, exist_ok=True)

# Example usage
path1 = "/data/MyDataBase/SWATGenXAppData/codes/HydroGeoDataSet/"
path2 = "/home/rafieiva/MyDataBase/codes/HydroGeoDataBase"

# Call the sync function
sync_directories(path1, path2)
