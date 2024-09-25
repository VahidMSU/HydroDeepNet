<<<<<<< HEAD
import h5py
import os
import glob

def merge_h5_files(output_path, input_dir):
    input_files = glob.glob(os.path.join(input_dir, "*.h5"))
    with h5py.File(output_path, 'w') as out_h5:
        for file_path in input_files:
            with h5py.File(file_path, 'r') as in_h5:
                for key in in_h5.keys():
                    in_h5.copy(key, out_h5)
            os.remove(file_path)  # Remove the individual file after copying

if __name__ == "__main__":
    linux = False

    if linux:
        output_path = "/data/climate_change/LOCA2_merged.h5"
        input_dir = "/data/climate_change/h5_files/"
    else:
        output_path = "E:/MyDataBase/climate_change/LOCA2_merged.h5"
        input_dir = "E:/MyDataBase/climate_change/h5_files/"

    merge_h5_files(output_path, input_dir)
=======
import h5py
import os
import glob

def merge_h5_files(output_path, input_dir):
    input_files = glob.glob(os.path.join(input_dir, "*.h5"))
    with h5py.File(output_path, 'w') as out_h5:
        for file_path in input_files:
            with h5py.File(file_path, 'r') as in_h5:
                for key in in_h5.keys():
                    in_h5.copy(key, out_h5)
            os.remove(file_path)  # Remove the individual file after copying

if __name__ == "__main__":
    linux = False

    if linux:
        output_path = "/data/MyDataBase/climate_change/LOCA2_merged.h5"
        input_dir = "/data/MyDataBase/climate_change/h5_files/"
    else:
        output_path = "E:/MyDataBase/climate_change/LOCA2_merged.h5"
        input_dir = "E:/MyDataBase/climate_change/h5_files/"

    merge_h5_files(output_path, input_dir)
>>>>>>> 4151fd4 (initiate linux version)
