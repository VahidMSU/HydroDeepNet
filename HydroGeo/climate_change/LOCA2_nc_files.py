import os

### USE this code to find all the nc files in the downloaded directory


def find_nc_files(root_directory):
    
    """
    Find all the .nc files in the root directory and its subdirectories.
    Args:
        root_directory (str): The root directory to search for .nc files.
    Returns:
        list: A list of .nc files.
    """

    nc_files = []
    for dirpath, _, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename.endswith('.nc'):
                full_path = os.path.join(dirpath, filename)
                nc_files.append(full_path)
    nc_files.sort(key=lambda x: os.path.getmtime(x), reverse=False)  # Sort files based on modified time
    return nc_files

def main(root_directory):
    ## define the root directory
    nc_files = find_nc_files(root_directory)
    for file in nc_files:
        print(file)
    print("Total number of files:", len(nc_files))

if __name__ == "__main__":

    """
    This script finds all the .nc files in the root directory and its subdirectories.
    
    """


    root_directory = r'/data/LOCA2'  # Replace with your root directory
    main(root_directory)

