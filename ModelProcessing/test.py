import pandas as pd
import os 
import shutil	
names = os.listdir('/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/')
names.remove('log.txt')

## remvoe recharge folders

for name in names:
	path = f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{name}/recharg_output_SWAT_gwflow_MODEL"
	shutil.rmtree(path, ignore_errors=True)