import pandas as pd 
import numpy as np

path = "/home/rafieiva/MyDataBase/codebase/ModelProcessing/example_inputs/Yield_4states_bu_acr.csv"
df = pd.read_csv(path)
df.iloc[:, 3:] = np.multiply(0.06277 , df.iloc[:, 3:]).round(2)
df = df[df.State=="MICHIGAN"].reset_index(drop=True).drop(columns=["Unnamed: 0"])
df.to_csv("/home/rafieiva/MyDataBase/codebase/ModelProcessing/example_inputs/Yield_Michigan_t_ha.csv", index=False)
