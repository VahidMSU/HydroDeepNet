import os
import requests
import zipfile
import shutil
def download_prism(year, variable):

    
    ver = "M2"


    path = f"https://ftp.prism.oregonstate.edu/monthly/{variable}/{year}/PRISM_{variable}_stable_4km{ver}_{year}_all_bil.zip"


    os.makedirs(f"/data/PRISM/CONUS_monthly/{variable}/", exist_ok=True)

    os.makedirs(f"/data/PRISM/CONUS_monthly/{variable}/{year}", exist_ok=True)

    output_path = f"/data/PRISM/CONUS_monthly/{variable}/{year}/{os.path.basename(path)}"


    ### remove the directory if it exists

    ### check if the file address is valid
    response = requests.head(path)
    if response.status_code != 200:
        print(f"Error: {path} is not a valid address.")
        return

    
    os.makedirs(f"/data/PRISM/CONUS_monthly/{variable}/", exist_ok=True)

    response = requests.get(path)
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        
        try:
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(f"/data/PRISM/CONUS_monthly/{variable}/{year}")
            os.remove(output_path)
        except zipfile.BadZipFile:
            print(f"Error: {output_path} is not a valid zip file.")
            os.remove(output_path)
    else:
        print(f"Error: Failed to download {path}")

if __name__ == "__main__":
    variables = [ "tmax", "tmin", "ppt"]
    years = range(1950, 2024)
    for variable in variables:
        ### remove the directory if it exists
        #shutil.rmtree(f"/data/PRISM/CONUS_monthly/{variable}", ignore_errors=True)

        for year in years:
            download_prism(year, variable)
            print(f"Downloaded {year} {variable}")