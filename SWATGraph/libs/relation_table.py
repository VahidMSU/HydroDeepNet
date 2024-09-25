import pandas
import numpy as np
import os
import time 
import geopandas as gpd

def clean_Graph_dir(graph_path):
    if not os.path.exists(graph_path):
        os.makedirs(graph_path)
        return
    files = os.listdir(graph_path)
    for file in files:
        print(f"Removing {file}")
        os.remove(f"{graph_path}/{file}")

def read_swat_riv_shapefile(model_path):
    swat_streams = gpd.read_file(model_path)
    print(f'swat_streams columns: {swat_streams.columns}')

    return swat_streams[['WSNO', 'LINKNO', 'DSLINKNO', 'Drop', 'NHDPlusID']]

def read_cell_riv(TxtInOut):
    path = f"{TxtInOut}/gwflow.rivcells"
    df = pandas.read_csv(path, sep='\s+', skiprows=2)
    print(f"gwflow.rivcells columns: {df.columns}")
    print(f"number of unique CELL_ID: {len(df.CELL_ID.unique())}")
    print(f"number of unique CHANNEL_ID: {len(df.CHANNEL_ID.unique())}")
    print(f"number of nan in df: {df.isna().sum().sum()}")
    return df[['CELL_ID', 'CHANNEL_ID', 'ZONE','CHANNEL_length_m']]


def read_cell_hru(TxtInOut):
    path = f"{TxtInOut}/gwflow.cellhru"
    df = pandas.read_csv(path, sep='\s+', skiprows=3)
    print(f"gwflow.cellhru columns: {df.columns}")
    print(f"number of unique CELL_ID: {len(df.CELL_ID.unique())}")
    print(f"number of unique HRU_ID: {len(df.HRU_ID.unique())}")
    print(f"number of nan in df: {df.isna().sum().sum()}")
    return df[['CELL_ID', 'HRU_ID', 'overlap_area_m2', 'CELL_AREA']]


def read_hru_con(TxtInOut):
    path = f"{TxtInOut}/hru.con"
    df = pandas.read_csv(path, sep='\s+',skiprows=1).drop(columns=['cst', 'ovfl', 'rule' ,'out_tot','gis_id','name']).rename(columns={'hru': 'HRU_ID','elev':'hru_elev','lat':'hru_lat','lon':'hru_lon','area':'hru_area'})
    print(f"hru.con columns: {df.columns}")
    print(f"number of unique id: {len(df.id.unique())}")
    print(f"number of nan in df: {df.isna().sum().sum()}")
    return df[['HRU_ID','wst', 'hru_elev', 'hru_lat', 'hru_lon', 'hru_area']]



def combine_all_data_frame(SWAT_SHAPES,TxtInOut):
    gwflow_rivcell = read_cell_riv(TxtInOut)  # ['CELL_ID', 'HRU_ID']
    gwflow_cellhru = read_cell_hru(TxtInOut)  # ['CELL_ID', 'CHANNEL_ID']
    hru_con = read_hru_con(TxtInOut)          # ['HRU_ID','wst']
    swat_riv = read_swat_riv_shapefile(f"{SWAT_SHAPES}/watershed/Shapes/SWAT_plus_streams.shp") # ['WSNO', 'LINKNO', 'DSLINKNO', 'Drop', 'NHDPlusID']
    cell_hru_river = gwflow_cellhru.merge(gwflow_rivcell, on='CELL_ID', how='left')
    cell_hru_river = cell_hru_river.merge(hru_con, on='HRU_ID', how='left')
    ## now merge with swat riv
    cell_hru_river = cell_hru_river.merge(swat_riv, left_on='CHANNEL_ID', right_on='WSNO', how='left')
    return cell_hru_river


def print_information(df):
    ## print unique for each column. Make sure we have the same number of unique values as expected for cell, hrus, wst, rivers
    print("###########################################################")
    print(f"number of unique CELL_ID: {len(df.cell_id.unique())}")
    print(f"number of unique CHANNEL_ID: {len(df.channel_id.unique())}")
    print(f"number of unique HRU_ID: {len(df.hru_id.unique())}")
    print(f"number of unique wst: {len(df.wst.unique())}")

def generate_relation_table(name):
    BASE_PATH = "/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/"
    graph_path = f"{BASE_PATH}/{name}/Graphs"
    print(f"Processing {name}")
    clean_Graph_dir(graph_path)
    TxtInOut = f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{name}/SWAT_gwflow_MODEL/Scenarios/Default/TxtInOut/"
    SWAT_SHAPES = f"/data/MyDataBase/CIWRE-BAE/SWAT_input/huc12/{name}/SWAT_MODEL"

    df = combine_all_data_frame(SWAT_SHAPES,TxtInOut)
    ### convert all column names to lower case
    df.columns = map(str.lower, df.columns)
    df.to_csv(f'{graph_path}/cell_hru_riv_wst.csv', index=False)
    df.to_csv('cell_hru_riv_wst.csv', index=False)
    print_information(df)

if __name__ == "__main__":
    BASE_PATH = "/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/"

    names = os.listdir(BASE_PATH)
    names.remove("log.txt")
    for name in names:
        swat_output_base_path = "/data/MyDataBase/SWATplus_by_VPUID/0000/huc12"
        cell_hru_riv_wst = os.path.join(swat_output_base_path,name,"/Graphs/cell_hru_riv_wst.csv")
        if os.path.exists(cell_hru_riv_wst):
            print(f"Already generated data for {name}")
            continue
        try:
            generate_relation_table(name)
        except Exception as e:
            print(f"Error in {name} {e}")