<<<<<<< HEAD
import os

def find_nc_files(root_directory):
    nc_files = []
    for dirpath, _, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename.endswith('.nc'):
                full_path = os.path.join(dirpath, filename)
                nc_files.append(full_path)
    nc_files.sort(key=lambda x: os.path.getmtime(x), reverse=False)  # Sort files based on modified time
    return nc_files

def main():
    root_directory = r'E:/MyDataBase/climate_change/cirrus.ucsd.edu/~pierce/LOCA2'  # Replace with your root directory
    nc_files = find_nc_files(root_directory)
    for file in nc_files:
        print(file)
    print("Total number of files:", len(nc_files))

if __name__ == "__main__":
    main()
=======
import os

def find_nc_files(root_directory):
    nc_files = []
    for dirpath, _, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename.endswith('.nc'):
                full_path = os.path.join(dirpath, filename)
                nc_files.append(full_path)
    nc_files.sort(key=lambda x: os.path.getmtime(x), reverse=False)  # Sort files based on modified time
    return nc_files

def main():
    root_directory = r'/data/MyDataBase/CIWRE-BAE/climate_change/cirrus.ucsd.edu/~pierce/LOCA2'  # Replace with your root directory
    nc_files = find_nc_files(root_directory)
    for file in nc_files:
        print(file)
    print("Total number of files:", len(nc_files))

if __name__ == "__main__":
    main()
>>>>>>> 4151fd4 (initiate linux version)
