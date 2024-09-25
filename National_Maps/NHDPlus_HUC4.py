import geopandas as gpd
import os
from multiprocessing import Pool, cpu_count
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

output_save = "/data/MyDataBase/SWATGenXAppData/NHDPlusData/"
NHDPlusHR_path = r"D:\MyDataBase\NHDPlusData\NHDPlus_H_National_Release_1_GDB\NHDPlus_H_National_Release_1_GDB.gdb"

# Define the number of chunks and the chunk size
num_chunks = 50
chunk_size = 10000

VPUID_shape = gpd.read_file(NHDPlusHR_path, layer='NHDPlusBoundaryUnit')
print("shapefile read")
# Set the CRS to a proper one that covers the entire US and islands in the Pacific
VPUID_shape = VPUID_shape.to_crs("EPSG:4326")
print("shapefile crs set ")
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
VPUID_shape.plot(ax=ax)
plt.show()
