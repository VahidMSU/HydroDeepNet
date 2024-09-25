
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from libs.plot_funs import plot_distribution, plot_site_samples, plot_rivers    
import time
from libs.hetero_data_creation import create_hetdata
from libs.utils import get_features_string

def create_sum_PFAS(pfas_gw, logger, column='WSSN'):
    logger.info(f"COLUMNS\tMAX\tMIN\tMEDIAN\tNAN\tZEROS\tUniqueWSSN")

    for col in pfas_gw.columns:
        if col.endswith('Result'):
            ### maximum and minimum of the column
            logger.info(f"{col}\t{pfas_gw[col].max():.2f}\t{pfas_gw[col].min():.2f}\t{pfas_gw[col].median():.2f}\t{pfas_gw[col].isnull().sum():.2f}\t{len(pfas_gw[pfas_gw[col] == 0]):.2f}\t{pfas_gw[pfas_gw[col] > 0][column].nunique()}")

    pfas_gw['sum_PFAS'] = pfas_gw[[col for col in pfas_gw.columns if col.endswith('Result')]].sum(axis=1)
    logger.info(f"sum_PFAS\t{pfas_gw['sum_PFAS'].max():.2f}\t{pfas_gw['sum_PFAS'].min():.2f}\t{pfas_gw['sum_PFAS'].median():.2f}\t{pfas_gw['sum_PFAS'].isnull().sum():.2f}\t{len(pfas_gw[pfas_gw['sum_PFAS'] == 0]):.2f}\t{pfas_gw[pfas_gw.sum_PFAS > 0][column].nunique()}")
    logger.info(
        "stage:create_sum_PFAS ==##== Maximum Threshold: 2000, Minimum Threshold: 0"
    )
    pfas_gw = pfas_gw[pfas_gw['sum_PFAS'] < 2000]
    pfas_gw = pfas_gw[pfas_gw['sum_PFAS'] >= 0]
    assert pfas_gw['sum_PFAS'].isnull().sum() == 0, logger.error("There are nan values in sum_PFAS")
    logger.info(f"stage:create_sum_PFAS ==##== Number of pfas_gw: {len(pfas_gw)}")
    assert len(pfas_gw[column].unique()) == len(pfas_gw), logger.error(f"There are duplicate {column} values in pfas_gw")
    logger.info(f"stage:create_sum_PFAS ==##== Range of sum_PFAS: {pfas_gw['sum_PFAS'].min()} - {pfas_gw['sum_PFAS'].max()}")

    return pfas_gw


def average_over_time(pfas_gw, logger, column='WSSN'):
    logger.info(f"stage:average_over_time ==##== Total number of samples: {len(pfas_gw)} with {pfas_gw[column].nunique()} unique {column} ID")
    pfas_gw = pfas_gw[[column] + [col for col in pfas_gw.columns if col.endswith('Result')]]
    pfas_gw = pfas_gw.fillna(0, downcast='infer')
    pfas_gw = pfas_gw.groupby(column).mean().reset_index()

    logger.info(f"stage:average_over_time ==##== Number of samples after averaging over time {len(pfas_gw)} with {pfas_gw[column].nunique()} unique {column} ID")

    return pfas_gw


def load_sampled_gw(data_dir, gw_features, logger):

    pfas_gw = gpd.GeoDataFrame(
    pd.read_pickle(os.path.join(data_dir, "Huron_PFAS_GW_Features.pkl")),
    geometry='geometry',
    crs='EPSG:4326').to_crs("EPSG:26990").reset_index(drop=True)

    logger.info(f"stage:load_sampled_gw ==##== number of Nulls in WSSN: {pfas_gw['WSSN'].isnull().sum()}")
    logger.info(f"stage:load_sampled_gw ==##== sampled gw columns: {list(pfas_gw.columns)}")
    logger.info(f"stage:load_sampled_gw ==##== Number of sampled gw: {len(pfas_gw)}")
    logger.info(f"stage:load_sampled_gw ==##== Range of unique WSSN ID: {pfas_gw['WSSN'].nunique()}")
    logger.info(f"stage:load_sampled_gw ==##== Range of unique geometry: {pfas_gw['geometry'].nunique()}")
    logger.info(f"stage:load_sampled_gw ==##== Number of unique SiteCode: {pfas_gw['SiteCode'].nunique()}")



    pfas_gw_no_geometry = average_over_time(pfas_gw, logger)

    try:
        pfas_gw = pfas_gw.drop_duplicates(subset=['WSSN'])[['WSSN','geometry']+gw_features].merge(
            pfas_gw_no_geometry, on='WSSN', how='inner'
        )
    except Exception as e:
        logger.error(e)

    assert all(
        col in pfas_gw.columns for col in gw_features
    ), f"Allowed gw_features are {list(pfas_gw.columns)}"


    pfas_gw = create_sum_PFAS(pfas_gw, logger, "WSSN")
    return pfas_gw

def load_all_wssn(data_dir, gw_features, logger):

    all_wssn = pd.read_pickle(os.path.join(data_dir, "WSSN_Huron_with_features.pkl")).to_crs("EPSG:26990").reset_index(drop=True)[['WSSN', 'geometry']+gw_features]
    all_wssn['WSSN'] = all_wssn['WSSN'].astype(int).astype(str)
    logger.info(f"stage:load_all_wssn ==##== Number of all_wssn: {len(all_wssn)}")
    logger.info(f"stage:load_all_wssn ==##== Range of all_wssn with unique WSSN ID: {all_wssn['WSSN'].nunique()}")
    ## drop duplicates
    all_wssn['WSSN'] = pd.factorize(all_wssn['WSSN'])[0]
    all_wssn = all_wssn.drop_duplicates(subset='WSSN')
    return all_wssn

def load_grids(data_dir, gw_features, logger):

    all_wssn = pd.read_pickle(os.path.join(data_dir, "Huron_River_Grid_250m_with_features.pkl")).to_crs("EPSG:26990").reset_index(drop=True)[['geometry']+gw_features]
    all_wssn = all_wssn.reset_index(drop=True)[['geometry']+gw_features]
    all_wssn['WSSN'] = pd.factorize(all_wssn.index)[0]
    all_wssn['WSSN'] = all_wssn['WSSN'].astype(int).astype(str)
    logger.info(f"stage:load_all_wssn ==##== Number of all_wssn: {len(all_wssn)}")
    logger.info(f"stage:load_all_wssn ==##== Range of all_wssn with unique WSSN ID: {all_wssn['WSSN'].nunique()}")
    ## drop duplicates
    all_wssn['WSSN'] = pd.factorize(all_wssn['WSSN'])[0]
    all_wssn = all_wssn.drop_duplicates(subset='WSSN')


    return all_wssn



def load_pfas_gw(data_dir, gw_features, logger):


    pfas_gw = load_sampled_gw(data_dir, gw_features, logger)
    pfas_gw['sampled'] = 1

    wssn_gw = load_all_wssn(data_dir,gw_features, logger)
    #wssn_gw = load_grids(data_dir,gw_features, logger)
    wssn_gw['sampled'] = 0

    logger.info(f"stage:load_pfas_gw ==##== Number of sampled gw: {len(pfas_gw[pfas_gw['sampled'] == 1])}")
    logger.info(f"stage:load_pfas_gw ==##== Number of unsampled gw: {len(wssn_gw[wssn_gw['sampled'] == 0])}")

    ### concat them
    pfas_gw = pd.concat([pfas_gw, wssn_gw]).reset_index(drop=True)
    ## fillna with 0
    pfas_gw = pfas_gw.fillna(0, downcast='infer')
    #print(pfas_gw.columns)
    #time.sleep(100)
    #pfas_gw.drop(columns=['geometry']).drop_duplicates(subset='WSSN').to_csv(f"temp/pfas_gw{time.time()}.csv", index=False)


    pfas_gw['WSSN'] = pfas_gw['WSSN'].astype(str)
    pfas_gw['gw_node_index'] = pd.factorize(pfas_gw['WSSN'])[0]

    assert pfas_gw['sum_PFAS'].isnull().sum() == 0, "There are nan values in sum_PFAS"

    logger.info(f"stage:load_pfas_gw ==##== Number of pfas_gw: {len(pfas_gw)}")
    logger.info(f"stage:load_pfas_gw ==##== Range of sum_PFAS: {pfas_gw['sum_PFAS'].min()} - {pfas_gw['sum_PFAS'].max()}")

    return pfas_gw

def add_confirmed_sites(data_dir, logger):
    logger.info("stage:add_confirmed_sites ==##== Loading confirmed sites")
    confirmed_pfas_sites = gpd.GeoDataFrame(
        pd.read_pickle(os.path.join(data_dir, "Huron_confirmed_PFAS_SITE_Features.pkl")),
        geometry='geometry',
        crs='EPSG:4326'
    ).to_crs("EPSG:26990").reset_index(drop=True)

    confirmed_pfas_sites['Industry'] = confirmed_pfas_sites['Type'].astype(str)
    confirmed_pfas_sites['inv_status'] = 1

    logger.info(f"stage:add_confirmed_sites ==##== Number of confirmed sites: {len(confirmed_pfas_sites)}")

    return confirmed_pfas_sites

def add_biosolid_sites(data_dir, logger):
    logger.info("stage:add_biosolid_sites ==##== Loading biosolid sites")
    biosolid_sites = gpd.GeoDataFrame(
        pd.read_pickle(os.path.join(data_dir, "Huron_Biosolid_sites_with_features.pkl")),
        geometry='geometry',
        crs='EPSG:4326'
    ).to_crs("EPSG:26990").reset_index(drop=True)

    biosolid_sites['Industry'] = "Biosolid"
    biosolid_sites['inv_status'] = 0
    biosolid_sites['Latitude'] = biosolid_sites.to_crs("EPSG:4326").geometry.y
    biosolid_sites['Longitude'] = biosolid_sites.to_crs("EPSG:4326").geometry.x
    biosolid_sites['State'] = "MI"
    biosolid_sites['index'] = biosolid_sites.index +10000
    logger.info(f"stage:add_biosolid_sites ==##== Number of biosolid sites: {len(biosolid_sites)}")

    return biosolid_sites

def add_unconfirmed_sites(data_dir, logger):
    logger.info("stage:add_unconfirmed_sites ==##== Loading unconfirmed sites")
    unconfirmed_pfas_sites = gpd.GeoDataFrame(
        pd.read_pickle(os.path.join(data_dir, "Huron_PFAS_SITE_Features.pkl")),
        geometry='geometry',
        crs='EPSG:4326'
    ).to_crs("EPSG:26990").reset_index(drop=True)

    unconfirmed_pfas_sites['inv_status'] = 0

    logger.info(f"stage:add_unconfirmed_sites ==##== Number of unconfirmed sites: {len(unconfirmed_pfas_sites)}")

    return unconfirmed_pfas_sites


def load_pfas_sites(data_dir, logger, load_biosolid=False):
    logger.info("stage:load_pfas_sites ==##== Loading PFAS sites")

    unconfirmed_pfas_sites = add_unconfirmed_sites(data_dir, logger)
    confirmed_pfas_sites = add_confirmed_sites(data_dir, logger)
    if load_biosolid:
        biosolid_pfas_sites = add_biosolid_sites(data_dir, logger)
        # Only keep shared columns
        common_columns = list(set(unconfirmed_pfas_sites.columns) & set(confirmed_pfas_sites.columns) & set(biosolid_pfas_sites.columns))
        biosolid_pfas_sites = biosolid_pfas_sites[common_columns]
        combined_pfas_sites = pd.concat([unconfirmed_pfas_sites, confirmed_pfas_sites, biosolid_pfas_sites]).reset_index(drop=True)
    else:
        common_columns = list(set(unconfirmed_pfas_sites.columns) & set(confirmed_pfas_sites.columns))
        unconfirmed_pfas_sites = unconfirmed_pfas_sites[common_columns]
        confirmed_pfas_sites = confirmed_pfas_sites[common_columns]
        combined_pfas_sites = pd.concat([unconfirmed_pfas_sites, confirmed_pfas_sites]).reset_index(drop=True)

    logger.info(f"common columns in all PFAS sites: {common_columns}")
    # Combine all sites
    #time.sleep(100)
    logger.info("stage:load_pfas_sites ==##== Combining all sites")

    ## factorize the well Industry
    combined_pfas_sites['Industry'] = pd.factorize(combined_pfas_sites['Industry'])[0]
    # Set site_node_index after combining all sites
    combined_pfas_sites['site_node_index'] = pd.factorize(combined_pfas_sites.index)[0]
    ## assert no nan in all columns
    #combined_pfas_sites.to_csv(f"temp/combined_pfas_sites{time.time()}.csv", index=False)

    assert combined_pfas_sites.isnull().sum().sum() == 0, "There are nan values in combined_pfas_sites"

    logger.info("stage:load_pfas_sites ==##== PFAS sites are all loaded")

    return combined_pfas_sites

def classify_pfas_gw(pfas_gw, logger):
    ### assert no negative in pfas_gw sum_PFAS
    assert pfas_gw['sum_PFAS'].min() >= 0, logger.error("There are negative values in sum_PFAS")
    assert pfas_gw['sampled'].nunique() == 2, logger.error("There are more than 2 unique values in sampled")

    pfas_gw['sum_PFAS_class'] = pd.cut(pfas_gw['sum_PFAS'], bins=[-1, 1, 10, 1000], labels=[0, 1, 2])  ## the range will be 0-1, 1-10, 10-1000
    # if there are wells that are not sampled, assign them to class 3
    pfas_gw['sum_PFAS_class'] = np.where(pfas_gw['sampled'] == 0, 3, pfas_gw['sum_PFAS_class'])
    logger.info(f"stage:classify_pfas_gw ==##== Number of pfas_gw classes 0 (representing 0 PFAS): {len(pfas_gw[pfas_gw['sum_PFAS_class'] == 0])}")
    logger.info(f"stage:classify_pfas_gw ==##== Number of pfas_gw classes 1 (representing 0-10 PFAS): {len(pfas_gw[pfas_gw['sum_PFAS_class'] == 1])}")
    logger.info(f"stage:classify_pfas_gw ==##== Number of pfas_gw classes 2 (representing 10-1000 PFAS): {len(pfas_gw[pfas_gw['sum_PFAS_class'] == 2])}")
    logger.info(f"stage:classify_pfas_gw ==##== Number of pfas_gw classes 3 (representing not sampled wells): {len(pfas_gw[pfas_gw['sum_PFAS_class'] == 3])}")

    #pfas_gw.to_csv(f"temp/classified_pfas_gw{time.time()}.csv")


    return pfas_gw




def classify_pfas_sw(pfas_sw, logger):
    ### assert no negative in pfas_sw sum_PFAS
    assert pfas_sw['sum_PFAS'].min() >= 0, logger.error("There are negative values in sum_PFAS")
    assert pfas_sw['sampled'].nunique() == 1, logger.error("There are more than 2 unique values in sampled")
    #print(pfas_sw['sum_PFAS'].min(), pfas_sw['sum_PFAS'].max(), pfas_sw['sum_PFAS'].isnull().sum())
    
    pfas_sw['sum_PFAS_class'] = pd.cut(pfas_sw['sum_PFAS'], bins=[-1, 200,500, 2000], labels=[0, 1, 2])
    # if there are wells that are not sampled, assign them to class 3
    pfas_sw['sum_PFAS_class'] = np.where(pfas_sw['sampled'] == 0, 3, pfas_sw['sum_PFAS_class'])
    logger.info(f"stage:classify_pfas_sw ==##== Number of pfas_sw classes 0 (representing 0 PFAS): {len(pfas_sw[pfas_sw['sum_PFAS_class'] == 0])}")
    logger.info(f"stage:classify_pfas_sw ==##== Number of pfas_sw classes 1 (representing 0-10 PFAS): {len(pfas_sw[pfas_sw['sum_PFAS_class'] == 1])}")
    logger.info(f"stage:classify_pfas_sw ==##== Number of pfas_sw classes 2 (representing 10-1000 PFAS): {len(pfas_sw[pfas_sw['sum_PFAS_class'] == 2])}")
    logger.info(f"stage:classify_pfas_sw ==##== Number of pfas_sw classes 3 (representing not sampled wells): {len(pfas_sw[pfas_sw['sum_PFAS_class'] == 3])}")


    assert 0 in pfas_sw['sum_PFAS_class'].unique(),   logger.error("There is no class 0 in sum_PFAS_class")
    assert 1 in pfas_sw['sum_PFAS_class'].unique(),   logger.error("There is no class 1 in sum_PFAS_class")
    assert 2 in pfas_sw['sum_PFAS_class'].unique(),   logger.error("There is no class 2 in sum_PFAS_class")

    #pfas_sw.to_csv(f"temp/classified_pfas_sw{time.time()}.csv")


    return pfas_sw



def split_sampled_gw_pfas_data(pfas_data, logger, name):

    logger.info(f"############## Splitting {name} samples ##############")
    logger.info(f"stage:split_sampled_pfas_{name} ==##== Number of sampled pfas_{name}: {len(pfas_data[pfas_data['sampled'] == 1])}")
    sampled_pfas = pfas_data[pfas_data['sampled'] == 1]
    logger.info("splitting method: 70% train, 20% validation, 10% test")
    train, temp = train_test_split(sampled_pfas, test_size=0.5, random_state=42, stratify=sampled_pfas['sum_PFAS_class'])
    val, test = train_test_split(temp, test_size=1/3, random_state=42, stratify=temp['sum_PFAS_class'])
    unsampled_pfas = pfas_data[pfas_data['sampled'] == 0]
    logger.info(f"stage:split_sampled_pfas_{name} ==##== Number of train_{name} range: {train['sum_PFAS'].min()} - {train['sum_PFAS'].max()} with {len(train[train['sum_PFAS'] == 0])} zeros")
    logger.info(f"stage:split_sampled_pfas_{name} ==##== Number of val_{name} range: {val['sum_PFAS'].min()} - {val['sum_PFAS'].max()} with {len(val[val['sum_PFAS'] == 0])} zeros")
    logger.info(f"stage:split_sampled_pfas_{name} ==##== Number of test_{name} range: {test['sum_PFAS'].min()} - {test['sum_PFAS'].max()} with {len(test[test['sum_PFAS'] == 0])} zeros")


    return train, val, test, unsampled_pfas

def split_sampled_sw_pfas_data(pfas_data, logger, name):
    logger.info(f"############## Splitting {name} samples ##############")
    logger.info(f"stage:split_sampled_pfas_{name} ==##== Number of sampled pfas_{name}: {len(pfas_data[pfas_data['sampled'] == 1])}")
    logger.info("splitting method: 70% train, 20% validation, 10% test")
    train, temp = train_test_split(pfas_data, test_size=0.3, random_state=42, stratify=pfas_data['sum_PFAS_class'])
    val, test = train_test_split(temp, test_size=1/3, random_state=42, stratify=temp['sum_PFAS_class'])
    logger.info(f"stage:split_sampled_pfas_{name} ==##== Number of train_{name} range: {train['sum_PFAS'].min()} - {train['sum_PFAS'].max()} with {len(train[train['sum_PFAS'] == 0])} zeros")
    logger.info(f"stage:split_sampled_pfas_{name} ==##== Number of val_{name} range: {val['sum_PFAS'].min()} - {val['sum_PFAS'].max()} with {len(val[val['sum_PFAS'] == 0])} zeros")
    logger.info(f"stage:split_sampled_pfas_{name} ==##== Number of test_{name} range: {test['sum_PFAS'].min()} - {test['sum_PFAS'].max()} with {len(test[test['sum_PFAS'] == 0])} zeros")
    ## assert no index value exceed the length of pfas_data
    assert max(train.index) < len(pfas_data), f"Train index {max(train.index)} out of bounds"
    assert max(val.index) < len(pfas_data), f"Val index {max(val.index)} out of bounds"
    assert max(test.index) < len(pfas_data), f"Test index {max(test.index)} out of bounds"
    return train, val, test

def identify_zero_points(pfas_gw, logger):
    excluded_zeros = pfas_gw[pfas_gw['sum_PFAS'] == 0]
    assert excluded_zeros['sum_PFAS'].sum() == 0, "There are non-zero values in excluded_zeros"
    return excluded_zeros



def load_rivers(logger):
    import geopandas as gpd
    import pandas as pd
    import matplotlib.pyplot as plt
    path = "/data/MyDataBase/CIWRE-BAE/SWAT_input/huc8/4100013/SWAT_MODEL/Watershed/Shapes/rivs1.shp"
    # Load the shapefile containing river data
    rivs = gpd.read_file(path)
    ## replace -1 in ChannelR with a unique value
    rivs['ChannelR'] = rivs['ChannelR'].replace(-1, -999)

    logger.info(f"stage:create_river_nodes_edges ==##== Number of rows: {rivs.shape[0]}")
    # Assign node indices based on the unique 'Channel' values
    rivs['riv_node_index'] = pd.factorize(rivs['Channel'])[0]
    rivs['sampled'] = 0
    logger.info(f"stage:create_river_nodes_edges ==##== Number of unique Channel values: {rivs['Channel'].nunique()}")
    logger.info(f"stage:create_river_nodes_edges ==##== Number of unique riv_node_index values: {rivs['riv_node_index'].nunique()}")

    plot_rivers(rivs, logger)

    ## reduce geometry to only centroid
    rivs['geometry'] = rivs['geometry'].centroid
    assert "ChannelR" in rivs.columns, "ChannelR is missing in rivs"
    assert "Channel" in rivs.columns, "Channel is missing in rivs"
    return rivs

def load_pfas_sw(data_dir,gw_features, logger):
    pfas_sw = gpd.GeoDataFrame(
        pd.read_pickle(os.path.join(data_dir, "Huron_PFAS_SW_Features.pkl")),
        geometry='geometry',
        crs='EPSG:4326'
    ).to_crs("EPSG:26990").reset_index(drop=True)

    bounds = pd.read_pickle(os.path.join(data_dir, "Huron_River_basin_bound.pkl")).to_crs("EPSG:26990")

    ## clip the pfas_sw to the bounds
    pfas_sw = gpd.clip(pfas_sw, bounds)

    logger.info(f"stage:load_pfas_sw ==##== Number of pfas_sw: {len(pfas_sw)}")

    pfas_sw_no_geometry = average_over_time(pfas_sw, logger, column='SiteCode')




    pfas_sw = pfas_sw.drop_duplicates(subset=['SiteCode'])[['SiteCode','geometry']+gw_features].merge(
        pfas_sw_no_geometry, on='SiteCode', how='inner'
    )


    assert all(
        col in pfas_sw.columns for col in gw_features
    ), f"Allowed gw_features are {list(pfas_sw.columns)}"
    ## drop not SiteCode duplicates
    assert len(pfas_sw) == len(pfas_sw.drop_duplicates(subset='SiteCode')), "There are duplicates in SiteCode"
    pfas_sw['sw_node_index'] = pd.factorize(pfas_sw.index)[0]
    ## assert insdex does not exceed the length of pfas_sw
    assert max(pfas_sw['sw_node_index']) < len(pfas_sw), f"Train index {max(pfas_sw['sw_node_index'])} out of bounds"

    pfas_sw['sampled'] = 1
    pfas_sw = create_sum_PFAS(pfas_sw, logger, "SiteCode")
    pfas_sw['SiteCode'] = pd.factorize(pfas_sw['SiteCode'])[0]
    pfas_sw = pfas_sw.reset_index(drop=True)
    return pfas_sw
def load_dataset(args, device, logger):

    data_dir = args["data_dir"]
    pfas_gw_columns = args["pfas_gw_columns"]
    pfas_sites_columns = args["pfas_sites_columns"]
    gw_features = args["gw_features"]
    distance_threshold = args["distance_threshold"]
    pfas_sw_station_columns = args["pfas_sw_station_columns"]
    gw_gw_distance_threshold = args["gw_gw_distance_threshold"]

    pfas_sites = load_pfas_sites(data_dir, logger)

    pfas_gw = load_pfas_gw(data_dir, gw_features, logger)
    pfas_gw = classify_pfas_gw(pfas_gw, logger)

    pfas_sw = load_pfas_sw(data_dir,gw_features, logger)
    rivs = load_rivers(logger)
    
    pfas_sw = classify_pfas_sw(pfas_sw, logger)
    pfas_sw = pfas_sw.reset_index(drop=True)
    
    if args.get("plot", False):
        plot_distribution(pfas_gw, logger, name="gw")
        plot_distribution(pfas_sw, logger, name="sw")
        

    train_gw, val_gw, test_gw, unsampled_gw = split_sampled_gw_pfas_data(pfas_gw, logger, name="gw")
    train_sw, val_sw, test_sw = split_sampled_sw_pfas_data(pfas_sw, logger, name="sw")

    if args.get("plot", False):
        plot_site_samples(train_gw, val_gw, test_gw, pfas_sites, logger, name="gw")
        plot_site_samples(train_sw, val_sw, test_sw, pfas_sites, logger, name="sw")

    var_names = get_features_string(gw_features)
    REWRITE = True
    if REWRITE:
        data = create_hetdata(pfas_gw, pfas_sw, unsampled_gw, pfas_sites,rivs, device, pfas_gw_columns, pfas_sites_columns, pfas_sw_station_columns, gw_features, distance_threshold, logger, gw_gw_distance_threshold)
        torch.save(data, f"Hetero_data/{var_names}_{distance_threshold}_{gw_gw_distance_threshold}.pth")
    elif not os.path.exists(f"Hetero_data/{var_names}_{distance_threshold}_{gw_gw_distance_threshold}.pth"):
        data = create_hetdata(pfas_gw, pfas_sw, unsampled_gw, pfas_sites,rivs, device, pfas_gw_columns, pfas_sites_columns, pfas_sw_station_columns, gw_features, distance_threshold, logger, gw_gw_distance_threshold)
        torch.save(data, f"Hetero_data/{var_names}_{distance_threshold}_{gw_gw_distance_threshold}.pth")
    else:
        data = torch.load(f"Hetero_data/{var_names}_{distance_threshold}_{gw_gw_distance_threshold}.pth")
    data = data.to(device)  

    data = split_data(data, unsampled_gw, train_gw, val_gw, test_gw, "gw_wells", logger).to(device)
    data = split_data(data, None, train_sw, val_sw, test_sw, "sw_stations", logger).to(device)

    ## assert geometry is in pfas_gw
    assert 'geometry' in pfas_gw.columns, "geometry column is missing in pfas_gw"

    return data, pfas_gw, pfas_sw

def split_data(data,unsampled_gw,  train, val, test, node_name, logger):

    train_indices = train.index.values
    val_indices = val.index.values
    test_indices = test.index.values


    print(f"train_indices: {np.sort(train_indices)}")
    assert max(train_indices) < data[node_name].num_nodes, f"Train index {max(train_indices)} out of bounds"

    train_mask = torch.zeros(data[node_name].num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data[node_name].num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data[node_name].num_nodes, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    if unsampled_gw is not None:
        unsampled_indices = unsampled_gw.index.values
        unsampled_mask = torch.zeros(data[node_name].num_nodes, dtype=torch.bool)
        unsampled_mask[unsampled_indices] = True
        data[node_name].unsampled_mask = unsampled_mask
        logger.info(f"stage:split_data ==##== Number of unsampled samples: {unsampled_mask.sum()} (NONE ZERO: {unsampled_gw['sum_PFAS'].gt(0).sum()})")

    data[node_name].train_mask = train_mask
    data[node_name].val_mask = val_mask
    data[node_name].test_mask = test_mask

    logger.info(f"stage:split_data ==##== Number of training samples: {train_mask.sum()} (NONE ZERO: {train['sum_PFAS'].gt(0).sum()})")
    logger.info(f"stage:split_data ==##== Number of validation samples: {val_mask.sum()} (NONE ZERO: {val['sum_PFAS'].gt(0).sum()})")
    logger.info(f"stage:split_data ==##== Number of test samples: {test_mask.sum()} (NONE ZERO: {test['sum_PFAS'].gt(0).sum()})")

    return data
