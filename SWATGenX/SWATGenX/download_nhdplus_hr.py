import os
import requests
from pathlib import Path
import logging
import xml.etree.ElementTree as ET
from urllib.parse import urljoin
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

output_path = "/data/SWATGenXApp/GenXAppData/NHDPlusHR/CONUS/"
# Modified base URL to use the S3 bucket listing format
base_url = "https://prd-tnm.s3.amazonaws.com/?prefix=StagedProducts/Hydrography/NHDPlusHR/VPU/Current/GDB/"



def get_file_list():
    """
    Get the list of available files from the S3 bucket.
    Returns a dictionary mapping VPUIDs to their corresponding file URLs.
    """
    try:
        # Get the XML listing of the S3 bucket
        response = requests.get(base_url)
        logging.info(f"Response status code: {response.status_code}")
        logging.info(f"Response content type: {response.headers.get('content-type', 'unknown')}")

        response.raise_for_status()

        # Parse the XML response
        root = ET.fromstring(response.content)

        # Define the namespace
        ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}

        # Extract file information
        files = {}
        for contents in root.findall('.//s3:Contents', ns):
            key = contents.find('s3:Key', ns)
            if key is not None:
                key_text = key.text
                if key_text.endswith('.zip'):
                    # Extract VPUID from filename
                    if 'NHDPLUS_H_' in key_text and '_HU4_GDB.zip' in key_text:
                        vpuid = key_text.split('NHDPLUS_H_')[1].split('_HU4_GDB.zip')[0]
                        files[vpuid] = f"https://prd-tnm.s3.amazonaws.com/{key_text}"
                        logging.info(f"Found file for VPUID {vpuid}: {key_text}")

        if not files:
            logging.warning("No NHDPlus HR files found in the response")
        else:
            logging.info(f"Found {len(files)} NHDPlus HR files")

        return files
    except Exception as e:
        logging.error(f"Error getting file list: {str(e)}")
        return {}

def download_nhdplus_hr(vpuid, file_url, output_dir):
    """
    Download NHDPlus HR data for a specific VPUID.

    Args:
        vpuid (str): The VPUID to download
        file_url (str): The URL of the file to download
        output_dir (str): Directory to save the downloaded files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set the output file path
    output_file = os.path.join(output_dir, f"NHDPLUS_H_{vpuid}_HU4_GDB.zip")

    try:
        logging.info(f"Downloading NHDPlus HR data for VPUID {vpuid}")
        response = requests.get(file_url, stream=True)
        response.raise_for_status()

        # Get the total file size
        total_size = int(response.headers.get('content-length', 0))

        # Download the file with progress tracking
        with open(output_file, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        # Calculate and log progress
                        progress = (downloaded / total_size) * 100
                        logging.info(f"Download progress: {progress:.1f}%")

        logging.info(f"Successfully downloaded NHDPlus HR data for VPUID {vpuid}")
        return True

    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading NHDPlus HR data for VPUID {vpuid}: {str(e)}")
        return False

def main(VPUIDs):
    """
    Main function to download NHDPlus HR data for all specified VPUIDs.
    """
    # Get the list of available files
    available_files = get_file_list()

    if not available_files:
        logging.error("Failed to get the list of available files")
        return

    # Log available VPUIDs
    logging.info(f"Available VPUIDs: {list(available_files.keys())}")

    # Download files for requested VPUIDs
    for vpuid in VPUIDs:
        if vpuid in available_files:
            success = download_nhdplus_hr(vpuid, available_files[vpuid], output_path)
            if not success:
                logging.error(f"Failed to download data for VPUID {vpuid}")
        else:
            logging.error(f"VPUID {vpuid} not found in available files")

if __name__ == "__main__":
    # Get all available VPUIDs and download them
    available_files = get_file_list()
    VPUIDs = list(available_files.keys())
    print(VPUIDs)
    import time
    time.sleep(1000)
    if available_files:
        logging.info(f"Starting download of {len(available_files)} NHDPlus HR files")
        main(list(available_files.keys()))
    else:
        logging.error("No files available to download")
