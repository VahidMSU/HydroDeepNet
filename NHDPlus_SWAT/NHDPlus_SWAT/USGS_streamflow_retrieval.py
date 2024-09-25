import requests
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import os
import pandas as pd
import utm
from hydrofunctions import NWIS
import seaborn as sns
## set the base directory


## use all states


def fetching_streamflow_stations(base_directory,VPUID) -> pd.DataFrame:
    """ Get the streamflow stations for a specific VPUID
    return a dataframe"""

    try:
        df= gpd.read_file(
        os.path.join(
            base_directory, f"streamflow_stations/VPUID/{VPUID}/streamflow_stations_{VPUID}.shp"
        )
    )
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

    return df
def find_upstrean_huc12s(huc12, WBDHU12):
    """
    Find all upstream huc12s for a given huc12.
    Returns:
    - A list of huc12 IDs that are upstream of the provided huc12.
    """
    huc12_df = WBDHU12[WBDHU12.tohuc == huc12].huc12.values
    huc12s_list = list(huc12_df)
    for u in huc12_df:
        huc12s_list.extend(find_upstrean_huc12s(u, WBDHU12))
    return huc12s_list


def get_nhdplus_huc12(VPUID) -> gpd.GeoDataFrame:
    """ Get the NHDPlus catchments for a specific VPUID
    return a dataframe"""
    base_directory = f"/data/MyDataBase/SWATGenXAppData/NHDPlusData/SWATPlus_NHDPlus/{VPUID}/unzipped_NHDPlusVPU/"

    ## raise error if the directory does not exist
    if not os.path.exists(base_directory):
        raise FileNotFoundError(f"Directory {base_directory} does not exist\n You first need to run the NHDPlus_extract_by_VPUID function to extract the data")


    gdb_name = os.listdir(base_directory)
    gdb_name = next(file for file in gdb_name if file.endswith('.gdb'))
    return gpd.read_file(
        os.path.join(
            base_directory,
            gdb_name,
        ),
        layer="WBDHU12",
    ).to_crs("EPSG:4326").rename(columns={"HUC12": "huc12", "ToHUC": "tohuc"})

def process_streamflow_station(huc12, stations_nhplus, WBDHU12, VPUID, base_directory, start_date='2000-01-01', end_date='2022-12-31'):
    first_huc = huc12
    list_of_huc12s = find_upstrean_huc12s(huc12, WBDHU12)
    list_of_huc12s.append(huc12)
    crs_proj = utm.from_latlon(stations_nhplus[stations_nhplus.huc12 == huc12].geometry.y.values[0], stations_nhplus[stations_nhplus.huc12 == huc12].geometry.x.values[0])
    EPSG = f"EPSG:{32700 - round((45 + stations_nhplus[stations_nhplus.huc12 == huc12].geometry.y.values[0])/90,0)*100 + round((183 + stations_nhplus[stations_nhplus.huc12 == huc12].geometry.x.values[0])/6,0)}"
    drainage_area = WBDHU12[WBDHU12.huc12.isin(list_of_huc12s)].to_crs(EPSG).geometry.area.sum()
    drainage_area_sqkm = drainage_area/1000000
    print("drainage area sqkm:", drainage_area/1000000)
    ## now plot the station and its drainage geometry in one plot
    fig, ax = plt.subplots()
    WBDHU12.plot(ax=ax, color='lightgrey', edgecolor='black')

    WBDHU12[WBDHU12.huc12.isin(list_of_huc12s)].plot(ax=ax, color='blue', edgecolor='black')
    site_no = f'{str(stations_nhplus[stations_nhplus.huc12 == huc12].site_no.values[0])}'

    stations_nhplus[stations_nhplus.huc12==huc12].plot(ax=ax, color='red')

    save_path = os.path.join(base_directory, f"streamflow_stations/VPUID/{VPUID}/streamflow_{site_no}.csv")

    print(site_no)

    # Create an instance of the NWIS class with the desired parameters

    try:
        station = NWIS(site=site_no, start_date = start_date, end_date= end_date)
    except Exception as e:
        print(f"Error: {e}")
        return site_no, first_huc, drainage_area_sqkm, list_of_huc12s, 0

    try:
        streamflow_data = station.df('discharge')
    except Exception as e:
        print(f"Error: {e}")
        return site_no, first_huc, drainage_area_sqkm, list_of_huc12s, 0

    if len(streamflow_data) == 0:
        print(f"No streamflow data for {site_no}")
        return site_no, first_huc, drainage_area_sqkm, list_of_huc12s, 0

    streamflow_data.reset_index().to_csv(save_path, header = ['date','streamflow'], index = False)
    ## remove hrs from the date column example of now: 2015-01-01 00:00:00+00:00,230.0
    streamflow_data.loc[:, 'date'] = streamflow_data.index
    streamflow_data.loc[:, 'date'] = streamflow_data['date'].dt.date
    streamflow_data.to_csv(save_path, index = False,header = ['streamflow','date'])
    ## read the streamflow data
    streamflow_data = pd.read_csv(save_path)
    number_of_streamflow_data = len(streamflow_data)
    plt.title(f"Streamflow station {site_no}, drainage area: {drainage_area_sqkm:.2f} sqkm")
    jpeg_save_path = os.path.join(base_directory, f"streamflow_stations/VPUID/{VPUID}/streamflow_{site_no}.jpeg")

    plt.savefig(jpeg_save_path,dpi=300)

    return site_no, first_huc, drainage_area_sqkm, list_of_huc12s, number_of_streamflow_data


def adding_huc12_to_stations(stations, VPUID):

    WBDHU12 = get_nhdplus_huc12(VPUID)
    stations_nhplus = stations[['VPUID','site_no','geometry']].sjoin(WBDHU12[['huc12','tohuc','geometry']], how="inner", predicate='intersects')

    return stations_nhplus, WBDHU12




def get_streamflow_by_VPUID(VPUID, start_date, end_date) -> pd.DataFrame:
    """ Generate the streamflow stations data for a specific VPUID


    #### return the meta data containing streamflow information  ####

    #### generate meta data or read if it is already exists ########

    ## calculate the total number of days
    # get the the huc12s of the streamflow flow stations
    # use a recursive function to get the huc12s of the streamflow stations
    # also include the huc12 of the station as the first huc12 in the list
    # get the drainage area of the huc12s
    # get the streamflow data for the station
    # save the streamflow data to a csv file
    # save the metadata to a csv file and return it to the user

    # Initialize a list to store row data"""""

    base_directory = "/data/MyDataBase/SWATGenXAppData/USGS/"
    meta_data_directory = os.path.join(base_directory, f"streamflow_stations/VPUID/{VPUID}/meta_{VPUID}.csv")
    if not os.path.exists(meta_data_directory):
        data = []
        total_expected_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
        stations = fetching_streamflow_stations(base_directory, VPUID)
        stations_nhplus, WBDHU12 = adding_huc12_to_stations(stations, VPUID)
        huc12s = stations_nhplus.huc12.values



        for huc12 in huc12s:
            try:
                site_no, first_huc, drainage_area_sqkm, list_of_huc12s, number_of_streamflow_data = process_streamflow_station(huc12, stations_nhplus, WBDHU12, VPUID, base_directory,start_date , end_date)
                #print(f"site_no: {site_no}, first_huc: {first_huc}, drainage_area_sqkm: {drainage_area_sqkm}, list_of_huc12s: {list_of_huc12s}, number_of_streamflow_data: {number_of_streamflow_data}")

                # Append a tuple (or list) of the data to the data list
                data.append((site_no, first_huc, drainage_area_sqkm, list_of_huc12s, number_of_streamflow_data, total_expected_days))

            except Exception as e:
                print(f"Error: {e}")
                continue

        # Once all data is collected, create the DataFrame
        meta_data_df = pd.DataFrame(data, columns=['site_no', 'first_huc', 'drainage_area_sqkm', 'list_of_huc12s', 'number_of_streamflow_data','total_expected_days'])
        meta_data_df = meta_data_df [['site_no', 'first_huc', 'drainage_area_sqkm', 'number_of_streamflow_data','total_expected_days', 'list_of_huc12s']]
        # Save to CSV
        # calculate the % of gaps in the data
        meta_data_df['GAP_percent'] = (1 - meta_data_df['number_of_streamflow_data']/meta_data_df['total_expected_days'])*100

        meta_data_df = meta_data_df.drop_duplicates('site_no')
        meta_data_df.to_csv(meta_data_directory, index=False)
    else:
        print(f"Data for {VPUID} already exists")
        meta_data_df = pd.read_csv(meta_data_directory)

    return meta_data_df



def get_all_VPUIDs(base_directory):

    path = os.path.join(base_directory, "NHDPlus_VPU_National")
    files = [f for f in os.listdir(path) if f.endswith('.zip')]
    return [file.split('_')[2] for file in files]


def plot_streamflow_data(meta_data_df, VPUID, start_date, end_date):

    print(meta_data_df)

    # example of how to read the data

    site_no = meta_data_df['site_no'].values[0]

    print(' #################### site no',site_no)

    for site_no, GAP_percent in zip(meta_data_df['site_no'], meta_data_df['GAP_percent']):

        if GAP_percent < 5:

            station = pd.read_csv(f"/data/MyDataBase/SWATGenXAppData/USGS/streamflow_stations/VPUID/{VPUID}/streamflow_{site_no}.csv")

            sns.set_style("whitegrid")

            # Get the discharge data for the station
            try:
                print(f"Plotting streamflow data for {site_no}")
                df = station.copy()
                df ['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)

                # Calculate the annual max, min, and median averages
                max_average_annual = df.resample('YE').mean().max()
                min_average_annual = df.resample('YE').mean().min()
                median_average_annual = df.resample('YE').mean().median()

                # Create a plot of the data
                fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
                ax.plot(df.index, df.iloc[:,0])
                ax.legend([site_no], loc='upper left')
                ax.set_xlabel('Date', fontsize=14)
                ax.set_ylabel('Streamflow (cfs)', fontsize=14)
                ax.set_title('Streamflow Data', fontsize=16)

                # Annotate the plot with the max, min, and median averages
                ax.annotate(f"Av Annual Max: {max_average_annual.values[0]:.2f}", xy=(0.8,1), xycoords='axes fraction', fontsize=12)
                ax.annotate(f"Av Annual Min: {min_average_annual.values[0]:.2f}", xy=(0.8,0.95), xycoords='axes fraction', fontsize=12)
                ax.annotate(f"Av Annual Median: {median_average_annual.values[0]:.2f}", xy=(0.8,0.9), xycoords='axes fraction', fontsize=12)

                # Save the plot
                plt.savefig(f"/data/MyDataBase/SWATGenXAppData/USGS/streamflow_stations/VPUID/{VPUID}/streamflow_record_{site_no}.jpeg", dpi=300)
                plt.close()
            except Exception as e:
                print(f"Error: {e}")
                continue

def USGS_streamflow_retrieval_by_VPUID(VPUID, start_date = '2000-01-01' , end_date = '2023-06-31'):

    print(f"Retrieving streamflow data foe VPUID:{VPUID}...")

    meta_data_path = fr"/data/MyDataBase/SWATGenXAppData/USGS/streamflow_stations/VPUID/{VPUID}/meta_{VPUID}.csv"

    if not os.path.exists(meta_data_path):
        meta_data_df = get_streamflow_by_VPUID(VPUID, start_date, end_date)
    else:
        print(f"Data for {VPUID} already exists")
        meta_data_df = pd.read_csv(meta_data_path, dtype={'site_no': str})

    plot_streamflow_data(meta_data_df, VPUID, start_date, end_date)
