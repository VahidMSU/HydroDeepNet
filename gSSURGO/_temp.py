import os
import pandas as pd

df =  pd.read_csv("/data/MyDataBase/SWATGenXAppData/codes/gSSURGO/ssurgo_data_original.csv")

## check where the difference between muid and seqn is more than 0

print(df[df['muid'] - df['seqn'] > 0])