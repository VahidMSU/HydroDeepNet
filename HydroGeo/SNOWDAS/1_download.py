###################################  downlaading NOAA SNODAS daily modeling results ################################################
import os

import requests
from datetime import datetime, timedelta

def download_snodas_data(start_date, end_date):
    base_url = "https://noaadata.apps.nsidc.org/NOAA/G02158/masked"

    current_date = start_date
    while current_date <= end_date:
        year = current_date.strftime('%Y')
        month = current_date.strftime('%m')
        month_name = current_date.strftime('%B')
        day = current_date.strftime('%Y%m%d')
        filename = fr"D:\MyDataBase\snow\snow\SNODAS_{day}.tar"
        if os.path.exists(filename):
            print(f"File {filename} already exists, skipping")
            current_date += timedelta(days=1)
            continue
        # Construct the URL for the day's data file
        file_url = f"{base_url}/{year}/{month}_{month_name[:3]}/SNODAS_{day}.tar"

        try:
            response = requests.get(file_url)
            response.raise_for_status()

            # Write the file to disk


            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"Downloaded {filename}")
        except requests.exceptions.HTTPError as err:
            print(f"HTTP Error for {file_url}: {err}")
        except Exception as e:
            print(f"Failed to download data for {current_date.strftime('%Y-%m-%d')}: {e}")

        # Increment the date
        current_date += timedelta(days=1)

# Define the date range
start_date = datetime(2004, 1, 1)
end_date = datetime(2024, 1, 1)

# Call the function
download_snodas_data(start_date, end_date)
