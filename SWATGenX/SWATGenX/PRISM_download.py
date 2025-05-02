import os
import zipfile
import calendar
from multiprocessing import Process
import itertools
import urllib.request
import shutil
import time
import hashlib
try:
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
except ImportError:
    from SWATGenXConfigPars import SWATGenXPaths


def calculate_file_hash(file_path):
    """Calculate SHA-256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def is_valid_zip(file_path):
    """Check if a file is a valid zip file"""
    if not os.path.exists(file_path) or os.path.getsize(file_path) < 50000:
        return False
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            # Check if the zip file contains the expected files
            expected_files = [f for f in zip_ref.namelist() if f.endswith('.bil') or f.endswith('.hdr')]
            if not expected_files:
                return False
            # Test the zip file integrity
            zip_ref.testzip()
            return True
    except (zipfile.BadZipFile, zipfile.LargeZipFile):
        return False


def download_file(url, zip_path, max_retries=3):
    """Download a file with retry logic"""
    for attempt in range(max_retries):
        try:
            print(f"Download attempt {attempt + 1} for {zip_path}")
            # Create a temporary file for download
            temp_path = f"{zip_path}.temp"
            urllib.request.urlretrieve(url, temp_path)

            # Verify the downloaded file
            if is_valid_zip(temp_path):
                # If valid, move the temp file to the final location
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                os.rename(temp_path, zip_path)
                return True
            else:
                print(f"Downloaded file is corrupted or invalid")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except Exception as e:
            print(f"Download failed: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # Exponential backoff
    return False


def download_and_unzip_data(TYPE, YEAR, MONTH, DAY, zip_dir, extract_dir):
    zip_path = os.path.join(zip_dir, f"PRISM_{TYPE}_stable_4kmD2_{YEAR}{MONTH:02}{DAY:02}_bil.zip")
    url = f"https://ftp.prism.oregonstate.edu/daily/{TYPE}/{YEAR}/PRISM_{TYPE}_stable_4kmD2_{YEAR}{MONTH:02}{DAY:02}_bil.zip"

    # Ensure directories exist
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)

    extract_path = os.path.join(extract_dir, f"PRISM_{TYPE}_stable_4kmD2_{YEAR}{MONTH:02}{DAY:02}_bil")
    bil_file = os.path.join(extract_path, f"PRISM_{TYPE}_stable_4kmD2_{YEAR}{MONTH:02}{DAY:02}_bil.bil")
    hdr_file = os.path.join(extract_path, f"PRISM_{TYPE}_stable_4kmD2_{YEAR}{MONTH:02}{DAY:02}_bil.hdr")

    # Check if we already have valid files
    if os.path.exists(bil_file) and os.path.exists(hdr_file) and os.path.getsize(bil_file) > 0:
        print(f"Valid files already exist for {YEAR}-{MONTH:02}-{DAY:02}")
        return

    # If we have empty files, remove them
    if os.path.exists(bil_file) and os.path.getsize(bil_file) == 0:
        print(f"Removing empty file {bil_file}")
        os.remove(bil_file)
    if os.path.exists(hdr_file) and os.path.getsize(hdr_file) == 0:
        print(f"Removing empty file {hdr_file}")
        os.remove(hdr_file)

    max_retries = 3
    for attempt in range(max_retries):
        # Check if we need to download
        if not os.path.exists(zip_path) or not is_valid_zip(zip_path):
            print(f"Download attempt {attempt + 1} for {zip_path}")
            if not download_file(url, zip_path):
                print(f"Failed to download {zip_path} on attempt {attempt + 1}")
                continue
        else:
            print(f"{zip_path} exists and is valid")

        # Extract the files
        print(f"Extracting {zip_path} to {extract_path}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

            # Verify extraction
            if not os.path.exists(bil_file) or not os.path.exists(hdr_file):
                print(f"Extraction failed - missing required files")
                shutil.rmtree(extract_path)
                continue

            if os.path.getsize(bil_file) == 0 or os.path.getsize(hdr_file) == 0:
                print(f"Extracted files are empty")
                shutil.rmtree(extract_path)
                continue

            # If we get here, we have valid files
            print(f"Successfully downloaded and extracted files for {YEAR}-{MONTH:02}-{DAY:02}")
            return

        except (zipfile.BadZipFile, zipfile.LargeZipFile) as e:
            print(f"Error extracting {zip_path}: {e}")
            if os.path.exists(extract_path):
                shutil.rmtree(extract_path)
            continue

    print(f"Failed to download and extract files for {YEAR}-{MONTH:02}-{DAY:02} after {max_retries} attempts")


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

            ## maximum 5 processes at a time
            if len(processes) >= 200:
                for process in processes:
                    process.join()
                processes = []

    for process in processes:
        process.join()
