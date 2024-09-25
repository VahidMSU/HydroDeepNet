<<<<<<< HEAD
import requests
from bs4 import BeautifulSoup
import os
from concurrent.futures import ThreadPoolExecutor

def download_file(session, base_url, current_path, href):
    file_url = base_url + current_path + href
    file_response = session.get(file_url)
    if file_response.status_code == 200:

        download_folder = os.path.join('/data/climate_change/cirrus.ucsd.edu/~pierce/LOCA2/CONUS_regions_split', current_path)
        os.makedirs(download_folder, exist_ok=True)
        file_path = os.path.join(download_folder, href)
        if os.path.exists(file_path):
            print(f'{file_path} exists, continue...')
            return
        with open(file_path, 'wb') as file:
            print(f'Downloading: {file_path}')
            file.write(file_response.content)
        print(f'Downloaded: {file_path}')
    else:
        print(f'Failed to download {href}')

def explore_and_download(base_url, session, current_path='', executor=None):
    full_url = base_url + current_path
    response = session.get(full_url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        futures = []
        for link in soup.find_all('a'):
            href = link.get('href')
            if href.endswith('/'):
                # It's a directory, navigate into it
                explore_and_download(base_url, session, current_path + href, executor)
            elif href.endswith('.nc') and 'e_n_cent' in href:
                # It's a .nc file with 'e_n_cent' in the URL, download it in parallel
                futures.append(executor.submit(download_file, session, base_url, current_path, href))

        # Wait for all downloads in this directory to complete
        for future in futures:
            future.result()

# Base URL
base_url = 'https://cirrus.ucsd.edu/~pierce/LOCA2/CONUS_regions_split/'

# Start session and ThreadPoolExecutor
with requests.Session() as session, ThreadPoolExecutor(max_workers=100) as executor:
    explore_and_download(base_url, session, executor=executor)

print("Download process completed.")
=======
import requests
from bs4 import BeautifulSoup
import os
from concurrent.futures import ThreadPoolExecutor

def download_file(session, base_url, current_path, href):
    file_url = base_url + current_path + href
    file_response = session.get(file_url)
    if file_response.status_code == 200:

        download_folder = os.path.join('/data/MyDataBase/climate_change/cirrus.ucsd.edu/~pierce/LOCA2/CONUS_regions_split', current_path)
        os.makedirs(download_folder, exist_ok=True)
        file_path = os.path.join(download_folder, href)
        if os.path.exists(file_path):
            print(f'{file_path} exists, continue...')
            return
        with open(file_path, 'wb') as file:
            print(f'Downloading: {file_path}')
            file.write(file_response.content)
        print(f'Downloaded: {file_path}')
    else:
        print(f'Failed to download {href}')

def explore_and_download(base_url, session, current_path='', executor=None):
    full_url = base_url + current_path
    response = session.get(full_url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        futures = []
        for link in soup.find_all('a'):
            href = link.get('href')
            if href.endswith('/'):
                # It's a directory, navigate into it
                explore_and_download(base_url, session, current_path + href, executor)
            elif href.endswith('.nc') and 'e_n_cent' in href:
                # It's a .nc file with 'e_n_cent' in the URL, download it in parallel
                futures.append(executor.submit(download_file, session, base_url, current_path, href))

        # Wait for all downloads in this directory to complete
        for future in futures:
            future.result()

# Base URL
base_url = 'https://cirrus.ucsd.edu/~pierce/LOCA2/CONUS_regions_split/'

# Start session and ThreadPoolExecutor
with requests.Session() as session, ThreadPoolExecutor(max_workers=100) as executor:
    explore_and_download(base_url, session, executor=executor)

print("Download process completed.")
>>>>>>> 4151fd4 (initiate linux version)
