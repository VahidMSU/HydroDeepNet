### the purpose of this code is to compare the performance of the different models for the different watersheds

import os
import pandas as pd
import numpy as np
import seaborn as sns

BASE_PATH = "/data2/MyDataBase/SWATGenXAppData/"
RESOLUTION = 250
NAMES = os.listdir(fr'/data2/MyDataBase/SWATGenXAppData/SWAT_input/huc12/')


modelnames = [f'MODFLOW_{RESOLUTION}m', f'MODFLOW_ML_{RESOLUTION}m']

all_metrics = []

for NAME in NAMES:
	for MODEL_NAME in modelnames:
		path_to_metrics = os.path.join(BASE_PATH, f"SWAT_input/huc12/{NAME}/{MODEL_NAME}/metrics.csv")

		if os.path.exists(path_to_metrics):
			metrics = pd.read_csv(path_to_metrics)
			print(f"Model: {MODEL_NAME}, Watershed: {NAME}")
			print(metrics.describe())
			print()
			all_metrics.append(metrics)
		else:
			print(f"Metrics file not found for {MODEL_NAME}, {NAME}")
			print()

metrics_df = pd.concat(all_metrics)
os.makedirs(os.path.join(BASE_PATH, "Documentations/MODFLOW"), exist_ok=True)
metrics_df.to_csv(os.path.join(BASE_PATH, "Documentations/MODFLOW/EBK_vs_Michigan_metrics.csv"), index=False)
metrics_df.to_csv("EBK_vs_Michigan_metrics.xlsx", index=False)
## plot and compare the performance of the two models for different metrics
import matplotlib.pyplot as plt
EBK_MODELS_Performance = metrics_df[metrics_df['MODEL_NAME'] == f'MODFLOW_{RESOLUTION}m']
ML_MODELS_Performance = metrics_df[metrics_df['MODEL_NAME'] == f'MODFLOW_ML_{RESOLUTION}m']

EBK_MODELS_Performance = EBK_MODELS_Performance[EBK_MODELS_Performance.NAME.isin(ML_MODELS_Performance.NAME)]
ML_MODELS_Performance = ML_MODELS_Performance[ML_MODELS_Performance.NAME.isin(EBK_MODELS_Performance.NAME)]

### make box plot for each model and show the range of NSE, MSE and PBias among all the watersheds
fig, ax = plt.subplots(1, 3, figsize=(20, 5))
sns.boxplot(x='NSE', y='MODEL_NAME', data=EBK_MODELS_Performance, ax=ax[0])
sns.boxplot(x='MSE', y='MODEL_NAME', data=EBK_MODELS_Performance, ax=ax[1])
sns.boxplot(x='PBIAS', y='MODEL_NAME', data=EBK_MODELS_Performance, ax=ax[2])

plt.savefig(os.path.join(BASE_PATH, "Documentations/MODFLOW/EBK_vs_Michigan_metrics.jpeg"))
plt.close()	


### histrogram for ML and EBK models individually
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
## calculate RMSE
EBK_MODELS_Performance['RMSE'] = np.sqrt(EBK_MODELS_Performance['MSE'])
EBK_MODELS_Performance = EBK_MODELS_Performance[EBK_MODELS_Performance['MODEL_NAME'] == f'MODFLOW_250m']
NEW_NAMES = os.listdir(fr'/data2/MyDataBase/SWATplus_by_VPUID/0000/huc12/')
NEW_NAMES.remove('log.txt')	
NEW_NAMES = [int(name) for name in NEW_NAMES]
EBK_MODELS_Performance = EBK_MODELS_Performance[EBK_MODELS_Performance.NAME.isin(NEW_NAMES)]

sns.histplot(EBK_MODELS_Performance['NSE'], ax=ax[0])
sns.histplot(EBK_MODELS_Performance['RMSE'], ax=ax[1])
sns.histplot(EBK_MODELS_Performance['PBIAS'], ax=ax[2])

## titles for the plots
ax[0].set_title("NSE Histogram for EBK Models")
ax[1].set_title("RMSE Histogram for EBK Models")
ax[2].set_title("PBIAS Histogram for EBK Models")

## grid
ax[0].grid(True, linestyle='--')
ax[1].grid(True, linestyle='--')
ax[2].grid(True, linestyle='--')

## tight layout
plt.tight_layout()

plt.savefig("EBK_metrics.jpeg", dpi=300)
