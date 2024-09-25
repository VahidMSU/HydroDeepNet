import pandas as pd
import os
import time 
import geopandas as gpd
import logging



def setup_logger(log_file_path):
    # Clear the log file at the start
    with open(log_file_path, 'w') as f:
        f.write("")

    # Create a custom logger
    logger = logging.getLogger(log_file_path)

    # Set the logging level
    logger.setLevel(logging.INFO)

    # Create handlers
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatters and add them to handlers
    formatter = logging.Formatter('%(message)s')  # Only log the message itself
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def clean_Graph_dir(graph_path, logger):
    if not os.path.exists(graph_path):
        os.makedirs(graph_path)
        return
    files = os.listdir(graph_path)
    for file in files:
        if file == "log.txt":
            continue  # Skip deleting the log file
        logger.info(f"Removing {file}")
        os.remove(f"{graph_path}/{file}")


def read_swat_riv_shapefile(model_path, logger):
    swat_streams = gpd.read_file(model_path)
    swat_streams = swat_streams[['WSNO', 'LINKNO', 'DSLINKNO', 'Drop', 'NHDPlusID']]
    logger.info(f'swat_streams columns: {swat_streams.columns}')

    return swat_streams

def read_cell_riv(TxtInOut, logger):
    path = f"{TxtInOut}/gwflow.rivcells"
    df = pd.read_csv(path, sep='\s+', skiprows=2)
    df = df[['CELL_ID', 'CHANNEL_ID', 'ZONE','CHANNEL_length_m']]
    logger.info(f"gwflow.rivcells columns: {df.columns}")
    logger.info(f"number of unique CELL_ID: {len(df.CELL_ID.unique())}")
    logger.info(f"number of unique CHANNEL_ID: {len(df.CHANNEL_ID.unique())}")
    logger.info(f"number of nan in df: {df.isna().sum().sum()}")
    return df 


def read_cell_hru(TxtInOut, logger):
    path = f"{TxtInOut}/gwflow.cellhru"
    df = pd.read_csv(path, sep='\s+', skiprows=3)
    df = df[['CELL_ID', 'HRU_ID', 'overlap_area_m2', 'CELL_AREA']]
    logger.info(f"number of unique CELL_ID: {len(df.CELL_ID.unique())}")
    logger.info(f"number of unique HRU_ID: {len(df.HRU_ID.unique())}")  
    logger.info(f"number of nan in df: {df.isna().sum().sum()}")    
    logger.info(f"number of unique CELL_ID: {len(df.CELL_ID.unique())}")
    logger.info(f"cell hru columns: {df.columns}")
    return df


def read_hru_con(TxtInOut, logger):
    path = f"{TxtInOut}/hru.con"
    df = pd.read_csv(path, sep='\s+',skiprows=1).drop(columns=['cst', 'ovfl', 'rule' ,'out_tot','gis_id','name']).rename(columns={'hru': 'HRU_ID','elev':'hru_elev','lat':'hru_lat','lon':'hru_lon','area':'hru_area'})
    df = df[['HRU_ID','wst', 'hru_elev', 'hru_lat', 'hru_lon', 'hru_area']]
    logger.info(f"number of unique HRU_ID: {len(df.HRU_ID.unique())}")
    logger.info(f"number of nan in df: {df.isna().sum().sum()}")
    logger.info(f"hru.con columns: {df.columns}")
    return df



def combine_all_data_frame(SWAT_SHAPES,TxtInOut, logger):
    gwflow_rivcell = read_cell_riv(TxtInOut, logger)  # ['CELL_ID', 'HRU_ID']
    gwflow_cellhru = read_cell_hru(TxtInOut, logger) # ['CELL_ID', 'CHANNEL_ID']
    hru_con = read_hru_con(TxtInOut, logger)        # ['HRU_ID','wst']
    swat_riv = read_swat_riv_shapefile(f"{SWAT_SHAPES}/watershed/Shapes/SWAT_plus_streams.shp",logger) # ['WSNO', 'LINKNO', 'DSLINKNO', 'Drop', 'NHDPlusID']
    cell_hru_river = gwflow_cellhru.merge(gwflow_rivcell, on='CELL_ID', how='left')
    cell_hru_river = cell_hru_river.merge(hru_con, on='HRU_ID', how='left')
    ## now merge with swat riv
    cell_hru_river = cell_hru_river.merge(swat_riv, left_on='CHANNEL_ID', right_on='WSNO', how='left')
    logger.info(f"cell_hru_river columns: {cell_hru_river.columns}")
    return cell_hru_river


def print_information(df, logger):
    ## print unique for each column. Make sure we have the same number of unique values as expected for cell, hrus, wst, rivers
    logger.info(f"number of unique CHANNEL_ID: {len(df.cell_id.unique())}")
    logger.info(f"number of unique CHANNEL_ID: {len(df.channel_id.unique())}")
    logger.info(f"number of unique HRU_ID: {len(df.hru_id.unique())}")
    logger.info(f"number of unique wst: {len(df.wst.unique())}")

def generate_relation_table(name, logger_path):
    BASE_PATH = "/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/"
    graph_path = f"{BASE_PATH}/{name}/Graphs"
    
    
    logger = setup_logger(logger_path)
    #clean_Graph_dir(graph_path, logger)
    logger.info(f"Processing {name}")
    TxtInOut = f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{name}/SWAT_gwflow_MODEL/Scenarios/Default/TxtInOut/"
    SWAT_SHAPES = f"/data/MyDataBase/CIWRE-BAE/SWAT_input/huc12/{name}/SWAT_MODEL"

    df = combine_all_data_frame(SWAT_SHAPES,TxtInOut, logger)
    ### convert all column names to lower case
    df.columns = map(str.lower, df.columns)
    df.to_csv(f'{graph_path}/cell_hru_riv_wst.csv', index=False)
    #df.to_csv('cell_hru_riv_wst.csv', index=False)
    print_information(df, logger)


if __name__ == "__main__":
    BASE_PATH = "/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/"

    names = os.listdir(BASE_PATH)
    names.remove("log.txt")
    swat_output_base_path = "/data/MyDataBase/SWATplus_by_VPUID/0000/huc12"
    for name in names:
        cell_hru_riv_wst = os.path.join(swat_output_base_path,name,"Graphs/cell_hru_riv_wst.csv")
        logger_path = os.path.join(swat_output_base_path,name,"Graphs/log.txt")

        if "04115000" in name:
            
            generate_relation_table(name, logger_path)
