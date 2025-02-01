import pandas as pd 
import os 
import numpy as np
import matplotlib.pyplot as plt
import os 
import shutil

from ModelProcessing.SWATGenXConfigPars import SWATGenXPaths
### i wrote this code to check the streamflow data and make sure the no-values are filled correctly
### we first remove all the streamflow data and then copy the new data from the source

def remove_streamflow_data():
    VPUID = "0000"
    target_path = f"{SWATGenXPaths.swatgenx_outlet_path}/{VPUID}/huc12/"
    NAMES = os.listdir(target_path)
    NAMES.remove("log.txt")
    for NAME in NAMES:

        source_path = f"{SWATGenXPaths.swatgenx_outlet_path}/{VPUID}/huc12/{NAME}/streamflow_data"
        target_path = f"{SWATGenXPaths.swatgenx_outlet_path}/{VPUID}/huc12/{NAME}/streamflow_data"
        ## remove target and copy source
        if os.path.exists(source_path):
            shutil.rmtree(target_path, ignore_errors=True)
            os.makedirs(target_path)
            files = os.listdir(source_path)
            for file in files:
                shutil.copy2(os.path.join(source_path, file), target_path)
        else:
            print(f"#####################{NAME} SWAT_input does not have streamflow_data")
            ### remove if from target if exists
            if os.path.exists(target_path):
                shutil.rmtree(target_path)
                print(f"#####################{NAME} streamflow_data removed from target")
            continue

def correct_streamflows():
    import sys
    sys.path.append("/data/SWATGenXApp/codes/SWATGenX")
    from ModelProcessing.SWATGenXConfigPars import SWATGenXPaths
    from SWATGenX.utils import find_VPUID
    VPUIDs = find_VPUID()
    VPUID = "0407"
    base_path = f"{SWATGenXPaths.swatgenx_outlet_path}/{VPUID}/huc12/"
    NAMES = os.listdir(base_path)
    NAMES.remove("log.txt")
    for NAME in NAMES:

        path = f"{base_path}/{NAME}/streamflow_data"
        names = os.listdir(path)
        files = [os.path.join(path, name) for name in names if name.endswith(".csv") and "filled" not in name]

        start_time = 2000
        end_time = 2022
        date_range = pd.date_range(f"{start_time}-01-01", f"{end_time}-12-31", freq='D')

        for file in files:
            try:
                df = pd.read_csv(file, parse_dates=['date'])
            except Exception as e:
                ## remove the file
                os.remove(file)
                continue
            #print(df.head())
            ## Ensure 'date' column in date_range DataFrame is of datetime type
            date_range_df = pd.DataFrame(date_range, columns=['date'])
            ## merge on date
            df = df.merge(date_range_df, on='date', how='right')
            ## fill NaN with -999
            df = df.fillna(-1)
            ## rewrite the file
            df['streamflow'] = np.where(df['streamflow'] == -1, df['streamflow'].shift(1), df['streamflow']) # this will fill the first row with the value of the second row, if there are several consecutive NaNs, they will be filled with the value of the first non-NaN row
            df.to_csv(file.replace(".csv", "_daily.csv"), index=False)
            ## resample to monthly
            df = df.set_index('date')

            df['yr'] = df.index.year
            df['mon'] = df.index.month
            df = df.reset_index()
            ## fillna with -1
            df['streamflow'] = np.where(df['streamflow'] == -1, np.nan, df['streamflow'])
            df = df.dropna()
            ## groupby yr, mon and fillna with mean
            df = df.groupby(['yr', 'mon']).agg({'streamflow': 'sum'}).reset_index()

            df[['yr','mon','streamflow']].to_csv(file.replace(".csv", "_monthly.csv"), index=True)
            ## plot 
            plt.figure(figsize=(10, 5))
            plt.grid(linestyle='--', linewidth=0.5)
            plt.plot(df.index, df['streamflow'])
            plt.title(f"{os.path.basename(file)}")
            plt.xlabel("Date")
            plt.ylabel("cumulative streamflow (cfs)")
            plt.savefig(file.replace(".csv", "_monthly.png"), dpi=300)
            plt.close()
            print(f"Processed {file}")
            ## remove the original file
            os.remove(file)

if __name__ == "__main__":
    #remove_streamflow_data()  ### if you want to remove the streamflow data
    correct_streamflows()
    print("Done!")