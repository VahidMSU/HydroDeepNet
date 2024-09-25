import pandas as pd 
path = "/data/MyDataBase/SWAT_ML/ORGANIZED_40500010102_MPI-ESM1-2-HR_ssp245_r1i1p1f1.pkl"

data = pd.read_pickle(path)
print(data.columns)
print(data.shape)   