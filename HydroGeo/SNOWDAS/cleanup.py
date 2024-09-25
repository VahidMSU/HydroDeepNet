import os

file_path = "/data/MyDataBase/SWATGenXAppData/snow/snow/michigan/2004/1/16/Modeled_melt_rate_250m_EPSG26990.tif"

if os.path.exists(file_path):
    os.remove(file_path)
    print(f"File {file_path} has been deleted.")
else:
    print("The file does not exist.")