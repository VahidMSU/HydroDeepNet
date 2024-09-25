import os
import time
import numpy as np
import geopandas as gpd
import requests

import concurrent.futures

API_KEY = 'gYCmv3KTPwPGgnCSuBuG1cVXSwHdUjbLwbWp9dpK'
EMAIL = "rafieiva@msu.edu"
NAME = "Vahid"
YEARS = np.arange(1999, 2020)

def retrieve_nsrdb_data(file_name, url, params):
    # Make the request
    response = requests.get(url, params=params)
    # Check if the request was successful
    if response.status_code == 200:
        with open(file_name, 'w') as file:
            file.write(response.text)
        print(f"Data retrieved successfully for {file_name}.")
    else:
        print(f"Failed to retrieve data: {response.status_code}, {response.text}")

def process_point(point):
    wtk = point.geometry
    row = point['row']
    col = point['col']

    file_name = f'/data/MyDataBase/SWATGenXAppData/SWATplus_by_VPUID/0407/huc12/40700040303/NSRDB/{year}/r{row}_c{col}.csv'

    if os.path.exists(file_name):
        print(f'{file_name} exists, continue...')
        return
    time.sleep(0.5)
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    params = {
        'api_key': API_KEY,
        'wkt': wtk,  # Longitude and Latitude
        'names': year,  # Year
        'leap_year': 'false',
        'interval': '60',
        'utc': 'false',
        'full_name': NAME,
        'email': EMAIL,
        'affiliation': 'MSU',
        'mailing_list': 'false',
        'reason_for_use': 'research',
        'attributes': 'ghi,dhi,dni,wind_speed,air_temperature,solar_zenith_angle,relative_humidity,wind_direction'
    }

    retrieve_nsrdb_data(file_name, url, params)

if __name__ == "__main__":
    PRISM_dir = "/data/MyDataBase/SWATGenXAppData/SWATplus_by_VPUID/0407/huc12/40700040303/PRISM/"
    points = gpd.read_file(os.path.join(PRISM_dir, "PRISM_grid.shp")).to_crs("EPSG:4326")
    url = "https://developer.nrel.gov/api/solar/nsrdb_psm3_download.csv"

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        for year in YEARS:
            for i in range(len(points)):
                print(f'Processing {i+1} out of {len(points)}')
                point = points.iloc[i]
                future = executor.submit(process_point, point)
                futures.append(future)

        # Wait for all futures to complete
        concurrent.futures.wait(futures)
