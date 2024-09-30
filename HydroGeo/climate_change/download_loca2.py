import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

class ClimateDataDownloader:
    def __init__(self, base_url, model_file_path, download_dir, max_workers=4):
        self.base_url = base_url
        self.model_file_path = model_file_path
        self.download_dir = download_dir
        self.max_workers = max_workers
        self.data = self._load_model_data()

    def _load_model_data(self):
        data = []
        with open(self.model_file_path, 'r') as file:
            for line in file:
                if parts := line.split():
                    model = parts[1]
                    scen = parts[2]
                    ensembles = parts[3:]
                    data.extend([model, scen, ens] for ens in ensembles)
                    if "99" in parts[0]:
                        break
        return pd.DataFrame(data, columns=['model', 'scenario', 'ensemble'])

    def list_nc_files(self, model, scenario, ensemble, variable):
        url = f"{self.base_url}/{model}/e_n_cent/0p0625deg/{ensemble}/{scenario}/{variable}/"
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            return [
                link.get('href')
                for link in soup.find_all('a')
                if link.get('href').endswith('.nc')
            ]
        else:
            print(f"Failed to access {url}")
            return []

    def download_files(self, variables=None):
        if variables is None:
            variables = ['pr', 'tasmax', 'tasmin']
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for i in range(len(self.data)):
                model = self.data.loc[i, 'model']
                scenario = self.data.loc[i, 'scenario']
                ensembles = self.data.loc[i, 'ensemble'].split(',')
                for ensemble in ensembles:
                    futures.extend(
                        executor.submit(
                            self._process_variable,
                            model,
                            scenario,
                            ensemble,
                            variable,
                        )
                        for variable in variables
                    )
            for future in as_completed(futures):
                future.result()

    def _process_variable(self, model, scenario, ensemble, variable):
        nc_files = self.list_nc_files(model, scenario, ensemble, variable)
        if not nc_files:
            return

        download_dir = f"{self.download_dir}/{model}/e_n_cent/0p0625deg/{ensemble}/{scenario}/{variable}"
        os.makedirs(download_dir, exist_ok=True)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._download_file, os.path.join(download_dir, file), model, scenario, ensemble, variable, file) for file in nc_files]
            for future in as_completed(futures):
                future.result()

    def _download_file(self, file_path, model, scenario, ensemble, variable, file):
        if os.path.exists(file_path):
            return
        file_url = f"{self.base_url}/{model}/e_n_cent/0p0625deg/{ensemble}/{scenario}/{variable}/{file}"
        file_response = requests.get(file_url)
        if file_response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(file_response.content)
            print(f'Downloaded: {file_path}')
        else:
            print(f'Failed to download {file_url}')

if __name__ == "__main__":

    """
    
    This script downloads the LOCA2 data from the list of all models and scenarios.

    THe region of donwload is e_n_cent and the resolution is 0p0625deg.
    
    
    """


    base_url = 'https://cirrus.ucsd.edu/~pierce/LOCA2/CONUS_regions_split'
    model_file_path = "/data/MyDataBase/climate_change/list_of_all_models.txt"
    download_dir = "/data/LOCA2/CONUS_regions_split"
    downloader = ClimateDataDownloader(base_url, model_file_path, download_dir, max_workers=100)
    downloader.download_files()
