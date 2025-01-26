import os
import pandas as pd
from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
import matplotlib.pyplot as plt

gauge_id = pd.read_csv(SWATGenXPaths.camel_hydro_path, sep=";", dtype={"gauge_id": str})['gauge_id'].values

station_output = "/data/camel/model_output_maurer/model_output/flow_timeseries/maurer/04/04057800_72_model_output.txt"
df = pd.read_csv(station_output, delim_whitespace= True)

simulated = df['MOD_RUN'].values
observed = df['OBS_RUN'].values 
fig = plt.figure(figsize=(10, 5))   
NSE = 1 - sum((simulated - observed)**2)/sum((observed - observed.mean())**2)
plt.plot(simulated, label='Simulated', linewidth=0.5)
plt.plot(observed, label='Observed', linewidth=0.5)
plt.title(f" NSE: {NSE:.2f}")
plt.grid(linewidth=0.5, linestyle='--')
plt.legend()
plt.tight_layout()
plt.savefig("output.png", dpi=300)
plt.close()