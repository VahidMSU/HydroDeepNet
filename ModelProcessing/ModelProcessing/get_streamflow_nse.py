import os 
import pandas as pd


def get_streamflow_nse(channel_sd_day_path, streamflow_path):
    gis_id = os.path.basename(streamflow_path).split("_")[0]
    # sourcery skip: extract-method
    simulated_streamflow = []
    dates = []
    with open(channel_sd_day_path, "r") as f:
        content = f.readlines()
        columns = content[1].split()
        flo_out_column_num = columns.index("flo_out")
        gis_id_column_num = columns.index("gis_id")
        mon_column_num = columns.index("mon")
        day_column_num = columns.index("day")
        year_column_num = columns.index("yr")

        for line in content[2:]:
            values = line.split()
            if values[gis_id_column_num] == gis_id:
                simulated_streamflow.append(float(values[flo_out_column_num]))
                dates.append(f"{values[year_column_num]}-{values[mon_column_num]}-{values[day_column_num]}")
    ## make the date 
    dates = pd.to_datetime(dates)    
    simulated = pd.DataFrame({"date": dates, "flo_out": simulated_streamflow})
    cms_to_cfs = 35.3147
    simulated['flo_out'] = simulated['flo_out'] * cms_to_cfs
    observed = pd.read_csv(streamflow_path, parse_dates=['date'], dtype={'streamflow': float})

    df = observed.merge(simulated, on='date', how='inner')

    ### NSE
    daily_nse = 1 - (df['flo_out'] - df['streamflow']).pow(2).sum() / (df['streamflow'] - df['streamflow'].mean()).pow(2).sum()
    print(f"NSE: {daily_nse}")

    ### resample to monthly and calculate nse
    df = df.set_index('date')
    df_monthly = df.resample('ME').mean()
    nse_monthly = 1 - (df_monthly['flo_out'] - df_monthly['streamflow']).pow(2).sum() / (df_monthly['streamflow'] - df_monthly['streamflow'].mean()).pow(2).sum()
    print(f"NSE monthly: {nse_monthly}")

    
    return daily_nse, nse_monthly

if __name__ == "__main__":
    
    streamflow_path = "/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0407/huc12/04127997/streamflow_data/9_04127997_daily.csv"
    channel_sd_day_path = "/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0407/huc12/04127997/SWAT_MODEL/Scenarios/verification_stage_0/channel_sd_day.txt"

    nse = get_streamflow_nse(channel_sd_day_path,streamflow_path)
