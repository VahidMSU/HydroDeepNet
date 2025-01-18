import numpy as np
import os
import geopandas as gpd
import pandas as pd
import glob
import contextily as ctx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def get_all_VPUIDs():
    path = "/data/SWATGenXApp/GenXAppData/NHDPlusData/NHDPlus_VPU_National/"
    files = glob.glob(f"{path}*.zip")
    return [os.path.basename(file).split('_')[2] for file in files]

def fetch_station_information(VPUIDs, Region):
    """_summary_

    Args:
        VPUIDs (str): Vector Processing Unit IDs
        Region (str): Region number

    Returns:
        dataframe, dataframe, list 
    """
    all_streamflow_stations = pd.DataFrame()
    all_meta_data = pd.DataFrame()
    all_list_of_huc12s = []
    for VPUID in VPUIDs:
        if VPUID[:2] != Region:
            continue
        meta_data_path = os.path.join(streamflow_base_path, f"{VPUID}/meta_{VPUID}.csv")
        if os.path.exists(meta_data_path):
            meta_data = pd.read_csv(meta_data_path)
            meta_data = meta_data[meta_data.GAP_percent < 5]
            all_list_of_huc12s.extend(iter(meta_data.list_of_huc12s))
            streamflow_stations = gpd.read_file(os.path.join(streamflow_base_path, f"{VPUID}/streamflow_stations_{VPUID}.shp"))
            streamflow_stations = streamflow_stations[streamflow_stations.site_no.isin(meta_data.site_no)]
            if all_streamflow_stations.empty:
                all_streamflow_stations = streamflow_stations
                all_meta_data = meta_data
            else:
                all_streamflow_stations = pd.concat([all_streamflow_stations, streamflow_stations])
                all_meta_data = pd.concat([all_meta_data, meta_data])
        else:
            print(f"Meta data path does not exist for {VPUID}")
    data = set()
    for list_of_huc12s in all_list_of_huc12s:
        list_of_huc12s = list_of_huc12s[1:-1].split(", ")
        for huc12 in list_of_huc12s:
            data.add(huc12.strip("'"))
    all_list_of_huc12s = list(data)
    
    ### save all_meta_data and all_streamflow_stations
    all_meta_data.to_csv(f"/data/SWATGenXApp/GenXAppData/Documentations/Region{Region}/all_meta_data.csv", index=False)
    all_streamflow_stations.to_file(f"/data/SWATGenXApp/GenXAppData/Documentations/Region{Region}/all_streamflow_stations.shp")
    
    return all_streamflow_stations, all_meta_data, all_list_of_huc12s

VPUIDs = get_all_VPUIDs()
Region = "02"
streamflow_base_path = "/data/SWATGenXApp/GenXAppData/USGS/streamflow_stations/VPUID"
national_nhdplus_path = "/data/SWATGenXApp/GenXAppData/NHDPlusData/NHDPlus_H_National_Release_1_GDB/NHDPlus_H_National_Release_1_GDB.gdb"
national_doc_region_path = "/data/SWATGenXApp/GenXAppData/Documentations/Region02/"

all_streamflow_stations, all_meta_data, all_list_of_huc12s = fetch_station_information(VPUIDs, Region)
national_huc12s_filtered_path = os.path.join(national_doc_region_path, "huc12s.pkl")

if not os.path.exists(national_huc12s_filtered_path):
    huc12s = gpd.read_file(national_nhdplus_path, layer="WBDHU12")
    huc12s = huc12s[huc12s.huc12.isin(all_list_of_huc12s)]
    os.makedirs(national_doc_region_path, exist_ok=True)
    huc12s.to_pickle(national_huc12s_filtered_path)
    unique_huc12s = huc12s.copy()
    print(f"HUC12s are saved in {national_huc12s_filtered_path}")
else:
    print(f"Reading huc12 from existing data in {national_huc12s_filtered_path}")
    unique_huc12s = pd.read_pickle(national_huc12s_filtered_path)
msd = all_streamflow_stations.merge(all_meta_data, on="site_no")
msd.reset_index(inplace=True)

msd['color'] = 'black'
msd.loc[msd.drainage_area_sqkm < 100, 'color'] = 'blue'
msd.loc[(msd.drainage_area_sqkm >= 100) & (msd.drainage_area_sqkm < 500), 'color'] = 'green'
msd.loc[(msd.drainage_area_sqkm >= 500) & (msd.drainage_area_sqkm < 1000), 'color'] = 'yellow'
msd.loc[(msd.drainage_area_sqkm >= 1000) & (msd.drainage_area_sqkm < 1500), 'color'] = 'red'
msd.loc[msd.drainage_area_sqkm >= 1500, 'color'] = 'black'

print('Size of metadata:', len(all_meta_data))
print("Number of streamflow stations in each category of drainage area:")

for category in ['<100 km2', '100-499 km2', '500-999 km2', '1000-1499 km2', '>=1500 km2']:
    if category == '<100 km2':
        print(f'category: {category}, number of streamflow stations: {len(msd[msd.drainage_area_sqkm < 100])}')
        below_100 = len(msd[msd.drainage_area_sqkm < 100])
    elif category == '100-499 km2':
        print(f'category: {category}, number of streamflow stations: {len(msd[(msd.drainage_area_sqkm >= 100) & (msd.drainage_area_sqkm < 500)])}')
        between_100_500 = len(msd[(msd.drainage_area_sqkm >= 100) & (msd.drainage_area_sqkm < 500)])
    elif category == '500-999 km2':
        print(f'category: {category}, number of streamflow stations: {len(msd[(msd.drainage_area_sqkm >= 500) & (msd.drainage_area_sqkm < 1000)])}')
        between_500_1000 = len(msd[(msd.drainage_area_sqkm >= 500) & (msd.drainage_area_sqkm < 1000)])
    elif category == '1000-1499 km2':
        print(f'category: {category}, number of streamflow stations: {len(msd[(msd.drainage_area_sqkm >= 1000) & (msd.drainage_area_sqkm < 1500)])}')
        between_1000_1500 = len(msd[(msd.drainage_area_sqkm >= 1000) & (msd.drainage_area_sqkm < 1500)])
    elif category == '>=1500 km2':
        print(f'category: {category}, number of streamflow stations: {len(msd[msd.drainage_area_sqkm >= 1500])}')
        over_1500 = len(msd[msd.drainage_area_sqkm >= 1500])

fig, ax = plt.subplots(figsize=(10, 10))
unique_huc12s.plot(ax=ax, color='gray', label='HUC12s')
msd.plot(ax=ax, color=msd.color, label='Streamflow Stations', markersize=10)
ctx.add_basemap(ax, crs=unique_huc12s.crs.to_string(), source='https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}', zoom=5)
plt.title('Streamflow Stations in Region 02')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
legend_handles = [mpatches.Patch(color=color, label=label) for color, label in zip(['blue', 'green', 'yellow', 'red', 'black'], [f'<100 km2, {below_100} stations', f'100-499 km2, {between_100_500} stations', f'500-999 km2, {between_500_1000} stations', f'1000-1499 km2, {between_1000_1500} stations', f'>=1500 km2, {over_1500} stations'])]
plt.legend(handles=legend_handles, title='Drainage Area (km2)', loc='upper left')
plt.savefig("/data/SWATGenXApp/GenXAppData/Documentations/Region02/streamflow_stations.jpeg", dpi=300)
plt.show()
