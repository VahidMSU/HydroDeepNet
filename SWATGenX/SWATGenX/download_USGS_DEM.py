import os

try:
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
except ImportError:
    from SWATGenXConfigPars import SWATGenXPaths

def download_USGS_DEM():
    DEM_13_arc_second_list = SWATGenXPaths.DEM_13_arc_second_list


    with open(DEM_13_arc_second_list, 'r') as file:
        lines = file.readlines()
        urls = [line.strip() for line in lines]

    Downloaded_CONUS_DEM_path = SWATGenXPaths.Downloaded_CONUS_DEM_path
    if not os.path.exists(Downloaded_CONUS_DEM_path):
        os.makedirs(Downloaded_CONUS_DEM_path)

    downloaded_files = os.listdir(Downloaded_CONUS_DEM_path)
    downloaded_files = [file.split(".")[0] for file in downloaded_files]
    if len(downloaded_files) == len(urls):
        print("All files are already downloaded")
        return
    else:
        #raise ValueError("Some files are missing")

        os.makedirs(Downloaded_CONUS_DEM_path, exist_ok=True)

        ## donwload all the files in parallel
        import requests
        from concurrent.futures import ThreadPoolExecutor

        def download_file(url):
            filename = url.split('/')[-1]
            output_file = os.path.join(Downloaded_CONUS_DEM_path, filename)
            if not os.path.exists(output_file):
                print(f"Downloading {url} to {output_file}")
                r = requests.get(url)
                with open(output_file, 'wb') as f:
                    f.write(r.content)
            else:
                print(f"File already exists: {output_file}")

        with ThreadPoolExecutor(10) as executor:
            executor.map(download_file, urls)

        ## Reproject the downloaded files


if __name__ == "__main__":
    download_USGS_DEM()