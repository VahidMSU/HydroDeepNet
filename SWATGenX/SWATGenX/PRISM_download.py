import os
import zipfile
import calendar
from multiprocessing import Process
import itertools

import urllib.request

def download_and_unzip_data(TYPE, YEAR, MONTH, DAY, zip_dir, extract_dir):
    zip_path = os.path.join(zip_dir, f"PRISM_{TYPE}_stable_4kmD2_{YEAR}{MONTH:02}{DAY:02}_bil.zip")
    url = f"https://ftp.prism.oregonstate.edu/daily/{TYPE}/{YEAR}/PRISM_{TYPE}_stable_4kmD2_{YEAR}{MONTH:02}{DAY:02}_bil.zip"
    
    if not os.path.exists(zip_path):
        print(f"Downloading {zip_path}")
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)
        urllib.request.urlretrieve(url, zip_path)
    else:
        print(f"{zip_path} exists")
    
    extract_path = os.path.join(extract_dir, f"PRISM_{TYPE}_stable_4kmD2_{YEAR}{MONTH:02}{DAY:02}_bil")
    
    if not os.path.exists(extract_path):
        print(f"Extracting {zip_path} to {extract_path}")
        zipfile.ZipFile(zip_path, 'r').extractall(extract_path)
    else:
        print(f"{extract_path} exists")


try:
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
except ImportError:
    from SWATGenXConfigPars import SWATGenXPaths

if __name__ == "__main__":
    TYPES = ['ppt', 'tmax', 'tmin']
    YEARS = range(1990, 2023)
    MONTHS = range(1, 13)

    processes = []
    
    for TYPE, YEAR, MONTH in itertools.product(TYPES, YEARS, MONTHS):
        _, last_day = calendar.monthrange(YEAR, MONTH)
        for DAY in range(1, last_day + 1):
            process = Process(target=download_and_unzip_data, args=(TYPE, YEAR, MONTH, DAY, SWATGenXPaths.PRISM_zipped_path, SWATGenXPaths.PRISM_unzipped_path))
            processes.append(process)
            process.start()

    for process in processes:
        process.join()
