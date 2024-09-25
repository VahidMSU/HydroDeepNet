import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import seaborn as sns
import geopandas as gpd
import fiona
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import glob
import matplotlib.lines as mlines

def get_the_number_of_simulated_stations(path):
    files = os.listdir(path)
    print('files',files, path)
    return len(np.unique([x.split('_')[2] for x in files if len(x.split('_')) > 2]))


def get_best_performance(MODEL_NAME):

    best_performance = []
    VPUIDS = os.listdir('/data/MyDataBase/SWATplus_by_VPUID/')
    VPUIDS = ['0000']
    for VPUID in VPUIDS:
        base_dir = fr'/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12'
        NAMES = os.listdir(base_dir)
        NAMES.remove('log.txt')
        for NAME in NAMES:
            best_performance_dir = os.path.join(base_dir, NAME,  f'best_solution_{MODEL_NAME}.txt')

            if not os.path.exists(best_performance_dir):

                print(MODEL_NAME, NAME, 'best performance file does not exist')
                continue

            found_best_line = False

            with open(best_performance_dir, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if 'best' in line or "Best" in line:
                        found_best_line = True
                        best_performance.append([NAME, -1*float(line.split(':')[1])])

            if not found_best_line:
                print(f'Best performance not found for {NAME}')

    return pd.DataFrame(best_performance, columns=['NAME', 'best_performance'])



def checking_the_number_true_stations(BASE_PATH, LEVEL, NAME, END_YEAR= 2007, START_YEAR = 2002, nyskip= 2):

    """checking the number of stations based on a defined period and percentage of missing observations (less than 10% is only accepted)
    """


# Correctly concatenate the directory path with the wildcard pattern for .csv files
    stations_path = glob.glob(fr"/data/MyDataBase/CIWRE-BAE/SWAT_input/{LEVEL}/{NAME}/streanflow_data/*.csv")



    number_of_stations = 0

    for station_path in stations_path:
        obs = pd.read_csv(station_path, index_col='Unnamed: 0', parse_dates=['date'])
        date_range = pd.date_range(start=f'{START_YEAR+nyskip}-01-01', end=f'{END_YEAR}-12-31', freq='D')
        df_complete = pd.DataFrame(date_range, columns=['date'])
        missing_dates = df_complete[~df_complete['date'].isin(obs['date'])]
        gap_length = len(missing_dates)
        total_length = len(df_complete)
        gap_percent = gap_length/total_length
        if gap_percent<0.10:
            number_of_stations = number_of_stations+1
    #print('number of stations',number_of_stations)
    return number_of_stations



def update_best_performance_level(gdf_huc12, BASE_PATH, LEVEL):

    for NAME in gdf_huc12.NAME:
        number_of_stations = checking_the_number_true_stations(BASE_PATH, LEVEL, NAME)
        if number_of_stations > 0:
            gdf_huc12.loc[gdf_huc12['NAME'] == NAME, 'best_performance'] = (gdf_huc12.loc[gdf_huc12['NAME'] == NAME].best_performance.values) / (number_of_stations)
    return gdf_huc12


def plot_all_models_bound(output_path, model_id, model_name):
    NAMES = os.listdir('/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/')
    NAMES.remove('log.txt')
    all_stations = streamflow_stations('/data/MyDataBase/SWATplus_by_VPUID/0000/')
    lakes = pd.read_pickle('model_bounds/lakes_model_bounds.pkl').to_crs('EPSG:4326')
    gdf_huc12 = model_id[model_id.LEVEL=='huc12'].assign(AREA=model_id.geometry.area).sort_values('AREA', ascending=True).to_crs('EPSG:4326')
    gdf_huc8 = pd.read_pickle('/home/rafieiva/MyDataBase/codes/PostProcessing/model_bounds/huc8_model_bounds.pkl').to_crs('EPSG:4326')

    print(gdf_huc12.head())
    print(gdf_huc12.columns)

    gdf_huc12 = gdf_huc12[gdf_huc12.NAME.isin(NAMES)]

    # Create a professional plot
    fig, ax = plt.subplots(figsize=(12, 8))

    gdf_huc8.plot(ax=ax, color='gray', edgecolor='black', alpha=0.7)
    lakes.plot(ax=ax, color='skyblue', edgecolor='black', alpha=0.7)
    gdf_huc12.plot(ax=ax, facecolor='none', edgecolor='blue', alpha=1, linewidth=1.5)
    all_stations.plot(ax=ax, color='black', markersize=13, alpha=1, label='Streamflow Stations', zorder=5, marker='o')

    # Create legend handles
    red_patch = mpatches.Patch(facecolor='none', label='SWAT+ models', edgecolor='blue', linewidth=1.5)
    gray_patch = mpatches.Patch(color='gray', label='HUC8 watersheds')
    station_patch = mlines.Line2D([], [], color='black', label='Streamflow Stations', marker='o', linestyle='None')

    # Add legend to the plot
    ax.legend(handles=[red_patch, gray_patch, station_patch], loc='upper right')
    ax.set_title(f'#{len(gdf_huc12)} SWAT+models & #{len(np.unique(all_stations.site_no))} Streamflow Stations')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    os.makedirs(os.path.join(output_path, model_name), exist_ok=True)

    # Set plot limits
    ax.set_xlim(-86.8, -82.4)
    ax.set_ylim(41.6, 46)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, model_name, 'models_boundary_huc12_huc8.jpeg'), dpi=300)
    plt.close()

# Example usage:
# plot_all_models_bound('output_path', model_id_dataframe, 'model_name')


def check_10_year_daily_data(stations_path):

    ### check if we have more than 10 years of daily data between 2007 to 2020
    all_site_nos = []
    for station_path in stations_path:
        print('station path', station_path)
        try:
            obs = pd.read_csv(station_path)
        except Exception as e:
            print(f'Error: {e}')
            continue
        ## if empty continue
        if len(obs) == 0:
            continue
        import datetime
        obs['date'] = pd.to_datetime(obs['date'])
        obs = obs[(obs.date.dt.year >= 2007) & (obs.date.dt.year <= 2020)]
        if len(obs) > 10*365:
            all_site_nos.append(os.path.basename(station_path).split('_')[1].split('.')[0])

    return all_site_nos


def streamflow_stations(BASE_PATH):
    ### get the location of streamflow stations Z:\MyDataBase\SWATplus_by_VPUID\0000\huc12\04167000\streamflow_data
    LEVEL = "huc12"
    NAMES = os.listdir(f'{BASE_PATH}/{LEVEL}')
    all_stations = []
    NAMES.remove('log.txt')
    for NAME in NAMES:
        stations = f'{BASE_PATH}/{LEVEL}/{NAME}/streamflow_data/stations.shp'
        stations = gpd.read_file(stations).to_crs('EPSG:4326')
        import glob
        stations_path = glob.glob(f'{BASE_PATH}/{LEVEL}/{NAME}/streamflow_data/*.csv')
        all_site_nos = check_10_year_daily_data(stations_path)
        stations = stations[stations.site_no.isin(all_site_nos)]
        all_stations.append(stations)
    all_stations = pd.concat(all_stations)
    print('all stations', all_stations.head())
    ### save all the stations
    all_stations.drop(columns=['geometry']).to_csv('model_bounds/all_stations.csv')
    return all_stations
def creating_distribution_plots(output_path, df_models, model_name):
    """
    Creates distribution plots for different land use types based on the provided DataFrame.

    Args:
        df_models (pandas.DataFrame): A DataFrame containing land use data.

    Returns:
        None

    """

    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")

    # Prepare the data for plotting
    land_use_columns = ['percent_urban_landuse', 'percent_agriculture_landuse', 'percent_wetland_landuse', 'percent_forest_landuse']

    # Create histograms for each type of land use
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    ## set a title for all the plots
    fig.suptitle(f'Land Use Distribution for {len(df_models)} Models', fontsize=16)
    axes = axes.flatten()

    for ax, column in zip(axes, land_use_columns):
        # pass if n

        if df_models[column].isna().sum() == len(df_models):
            continue

        sns.histplot(df_models[column], kde=True, ax=ax,bins=10)
        ax.set_xlabel(f'{column.split("_")[1].capitalize()} Land Use (% of total area)')
        ax.set_ylabel('Models count')
        ax.set_xlim(0, 100)  # Set x-axis limit to be between 0 and 100
        # set title and number of total models in the title
    os.makedirs(os.path.join(output_path, model_name), exist_ok=True)
    plt.savefig(
        os.path.join(output_path, model_name,'landuse_models_distribution.jpeg'),
        dpi=300,
    )


    print('Maximum number of HRUs' , df_models.HRU_Count.max()   , 'Maximum Area (km\u00b2)', 0.01*df_models.Total_Area.max()  )
    print('average number of HRUs' , df_models.HRU_Count.mean()  , 'average Area (km\u00b2)', 0.01*df_models.Total_Area.mean()  )
    print('median number of HRUs'  , df_models.HRU_Count.median(), 'median  Area (km\u00b2)', 0.01*df_models.Total_Area.median())
    print('minimum number of HRUs' , df_models.HRU_Count.min()   , 'minimum Area (km\u00b2)', 0.01*df_models.Total_Area.min()  )
    # Histograms for HRU Counts and Total Area
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    sns.histplot(df_models['HRU_Count'], kde=True)
    plt.xlabel('Models total number of HRUs')
    plt.ylabel('Models count')
    plt.subplot(1, 2, 2)
    sns.histplot(df_models['Total_Area']*0.01, kde=True)  # Converted to sqkm
    plt.xlabel('Models total drainage area (km\u00b2)')
    plt.ylabel('Models count')
    #plt.title(f' #{len(df_models)} Models')
    plt.tight_layout()
    os.makedirs(os.path.join(output_path,model_name), exist_ok=True)
    plt.savefig(os.path.join(output_path, model_name,r'SWATplus_hrus_area.jpeg'), dpi=600)


def fetching_lake_char(NAMES):
    """
    Fetches lake characteristics for each NAME in NAMES.

    Args:
        NAMES (list): A list of names.

    Returns:
        pandas.DataFrame: A DataFrame containing lake characteristics including NAME, total lake area in square kilometers,
        average lake area in square kilometers, and number of lakes.

    """

    lakes = []
    for NAME in NAMES:
        lakes_path = f"/data/MyDataBase/CIWRE-BAE/SWAT_input/huc12/{NAME}/SWAT_MODEL/Watershed/Shapes/SWAT_plus_lakes.shp"

        if not os.path.exists(lakes_path):
            continue

        lake_name = gpd.read_file(lakes_path)
        lake_name['total_Lake_area_sqkm'] = lake_name.geometry.area.sum()/1e6
        lake_name['average_lake_area_sqkm'] = lake_name.geometry.area.mean()/1e6
        lake_name['total_Lake_area_sqkm'] = lake_name['total_Lake_area_sqkm'].astype(float).round(2)
        lake_name['average_lake_area_sqkm'] = lake_name['average_lake_area_sqkm'].astype(float).round(2)
        lake_name['NAME'] = NAME
        lake_name['number_of_lakes'] = len(lake_name.LakeId.unique())
        lake_name['number_of_lakes'] = lake_name['number_of_lakes'].astype(int)
        lakes.append(lake_name[['NAME','total_Lake_area_sqkm','average_lake_area_sqkm','number_of_lakes']].drop_duplicates(subset=['NAME']))

    return pd.concat(lakes).reset_index(drop=True)



def process_shapefile(NAME):
    """
    Processes a shapefile for a given NAME and returns relevant information.

    Args:
        NAME (str): The name of the shapefile.

    Returns:
        dict: A dictionary containing the processed information including NAME, HRU_Count, Total_Area,
        percent_urban_landuse, percent_agriculture_landuse, percent_wetland_landuse, percent_forest_landuse,
        rivers_length, and n_rivers.

    """


    base_dir = '/data/MyDataBase/CIWRE-BAE/SWAT_input/huc12/'

    hru_path = os.path.join(base_dir, f'{NAME}/SWAT_MODEL/Watershed/Shapes/hrus2.shp')
    riv_path = os.path.join(base_dir, f'{NAME}/SWAT_MODEL/Watershed/Shapes/rivs1.shp')

    if not os.path.exists(riv_path):
        return None
    if not os.path.exists(hru_path):
        return None

    with fiona.open(riv_path) as src:
        records = [record['properties'] for record in src]


    riv = pd.DataFrame.from_records(records)
    n_riv = len(riv.Channel.unique())
    len_riv = riv.Len2.sum()

    with fiona.open(hru_path) as src:
        records = [record['properties'] for record in src]
    HRUS = pd.DataFrame.from_records(records)

    hru_count = len(HRUS)
    total_area = HRUS['Area'].sum()

    percent_urban_landuse       = 100*HRUS[HRUS.Landuse.isin(['URMD', 'URLD', 'URHD', 'UIDU'])].Area.sum()/total_area
    percent_agriculture_landuse = 100*HRUS[HRUS.Landuse.isin(['AGRR', 'HAY', 'RNGB', 'RNGE', 'SWRN'])].Area.sum()/total_area
    percent_wetland_landuse     = 100*HRUS[HRUS.Landuse.isin(['WETF', 'WETL', 'WATR'])].Area.sum()/total_area
    percent_forest_landuse      = 100*HRUS[HRUS.Landuse.isin(['FRSD', 'FRSE', 'FRST'])].Area.sum()/total_area

    return {

            'NAME':                        NAME,
            'HRU_Count':                   hru_count,
            'Total_Area':                  total_area,
            'percent_urban_landuse':       percent_urban_landuse,
            'percent_agriculture_landuse': percent_agriculture_landuse,
            'percent_wetland_landuse':     percent_wetland_landuse,
            'percent_forest_landuse':      percent_forest_landuse,
            'rivers_length' : len_riv,
            'n_rivers' : n_riv
            }


def update_best_performance(best_performance_df, BASE_PATH, LEVEL, MODEL_NAME):

    temp_df = get_best_performance(MODEL_NAME)

    for NAME in temp_df.NAME.unique():
        number_of_stations = checking_the_number_true_stations(BASE_PATH, LEVEL, NAME)

        if number_of_stations > 0:
            # Calculate adjusted best performance
            adjusted_performance = temp_df.loc[temp_df['NAME'] == NAME, 'best_performance'].values / (2 * number_of_stations)

            # Update the original DataFrame
            best_performance_df.loc[best_performance_df['NAME'] == NAME, f'best_{MODEL_NAME}'] = adjusted_performance
            best_performance_df.loc[best_performance_df['NAME'] == NAME, 'number_of_stations'] = number_of_stations
        else:
            print(f'Error: No stations found for {NAME}')

    return best_performance_df

def reading_HRUS(NAMES):
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_shapefile, NAMES))
    return [r for r in results if r is not None]

class SWATAnalysisWorkflow:
    def __init__(self, base_dir, output_path, model_name, dic, level):
        self.base_dir = base_dir
        self.output_path = output_path
        self.model_name = model_name
        self.dic = dic
        self.level = level
        self.names = [name for name in os.listdir(base_dir) if name != ['log.txt']]

    def run(self):
        results = reading_HRUS(self.names)
        df_models = pd.DataFrame(results)
        df_models.set_index('NAME', inplace=True)
        print('df_models are generated', df_models.head())
        ### get all stations

        creating_distribution_plots(self.output_path, df_models, self.model_name)

        lakes = fetching_lake_char(self.names)
        print('lakes are fetched', lakes.head())
        best_performance_df = get_best_performance(self.model_name)
        df_models = df_models.join(lakes.set_index('NAME'), on='NAME')
        df_models = df_models.join(best_performance_df.set_index('NAME'), on='NAME')
        print('best performance is fetched', best_performance_df.head())

        df_models.to_csv(f'{self.output_path}/{self.model_name}/df_models.csv')
        os.makedirs(f'model_bounds/{self.model_name}', exist_ok=True)
        model_bounds = pd.read_pickle('model_bounds/model_bounds_huc12.pkl')
        print('model bounds are fetched', model_bounds.head())

        plot_all_models_bound(self.output_path, model_bounds, self.model_name)
        print('model bounds are plotted')



if __name__ == '__main__':
    """
    This program will generate
    1- A distribution plot for different land use types based on the provided DataFrame.
    3- A map showing the boundaries of all models.
    """
    base_dir = '/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/'
    output_path = "model_characteristics"
    os.makedirs(output_path, exist_ok=True)
    model_name = 'SWAT_gwflow_MODEL'
    dic ='/data/MyDataBase/CIWRE-BAE/SWAT_input/huc12/'
    level = 'huc12'

    workflow = SWATAnalysisWorkflow(base_dir, output_path, model_name, dic, level)
    workflow.run()
