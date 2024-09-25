import os
import pandas as pd
import geopandas as gpd



df = gpd.read_file("/data/MyDataBase/CIWRE-BAE/NHDPlusData/NHDPlus_H_National_Release_1_GDB/NHDPlus_H_National_Release_1_GDB.gdb", layer ="WBDHU8")
## print layers in the gdb
print(df)