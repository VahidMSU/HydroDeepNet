import os
import requests

epochs = {"1995-2004": "https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/data-bundles/Annual_NLCD_LndCov_1995-2004_CU_C1V0.zip",
          "2005-2014": "https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/data-bundles/Annual_NLCD_LndCov_2005-2014_CU_C1V0.zip",
          "2015-2023": "https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/data-bundles/Annual_NLCD_LndCov_2015-2023_CU_C1V0.zip"}



## Download the files

for epoch, url in epochs.items():
    filename = url.split("/")[-1]
    dest = os.path.join("/data/SWATGenXApp/GenXAppData/NLCD", filename)
    print(f"Downloading {filename}...")
    r = requests.get(url, stream=True, allow_redirects=True)
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)