


import os 

import pandas as pd

df = pd.read_csv('/data/SWATGenXApp/codes/Michigan/baseflow/08095300.prn', delim_whitespace=True, header=None)

print(df.head())