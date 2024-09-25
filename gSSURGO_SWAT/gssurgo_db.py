import arcpy
import geopandas as gpd
import pandas as pd
import sqlite3
import os
import logging
from libs.utilts import setup_logger, load_data
import numpy as np 

def create_ssurgo_sqlite_db(ssurgo_data, ssurgo_layer_data):
    logger = setup_logger()
    # Connect to SQLite database and create tables
    ssurgo_layer_schema = """
    CREATE TABLE ssurgo_layer (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        soil_id INTEGER NOT NULL,
        layer_num INTEGER NOT NULL,
        dp REAL NOT NULL,
        bd REAL NOT NULL,
        awc REAL NOT NULL,
        soil_k REAL NOT NULL,
        carbon REAL NOT NULL,
        clay REAL NOT NULL,
        silt REAL NOT NULL,
        sand REAL NOT NULL,
        rock REAL NOT NULL,
        alb REAL NOT NULL,
        usle_k REAL NOT NULL,
        ec REAL NOT NULL,
        caco3 REAL,
        ph REAL
    );
    """

    # Create 'ssurgo' table schema
    ssurgo_schema = """
    CREATE TABLE ssurgo (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name VARCHAR(255) NOT NULL,
        muid VARCHAR(255),
        seqn INTEGER,
        s5id VARCHAR(255),
        cmppct INTEGER,
        hyd_grp VARCHAR(255) NOT NULL,
        dp_tot REAL NOT NULL,
        anion_excl REAL NOT NULL,
        perc_crk REAL NOT NULL,
        texture VARCHAR(255) NOT NULL
    );
    """
    if os.path.exists('database/swatplus_soils.sqlite'):
        os.remove('database/swatplus_soils.sqlite')
    conn = sqlite3.connect('database/swatplus_soils.sqlite')
    cursor = conn.cursor()

    # Create tables
    cursor.execute("DROP TABLE IF EXISTS ssurgo")
    cursor.execute("DROP TABLE IF EXISTS ssurgo_layer")
    cursor.execute(ssurgo_schema)
    cursor.execute(ssurgo_layer_schema)

    # Insert data into tables
    ## only 100 rows will be printed
    ssurgo_data.iloc[:100].to_csv('database/ssurgo_data.csv')
    ssurgo_data.to_sql('ssurgo', conn, if_exists='append', index=False)
    ssurgo_layer_data.iloc[:100].to_csv('database/ssurgo_layer_data.csv')
    ssurgo_layer_data.to_sql('ssurgo_layer', conn, if_exists='append', index=False)

    # Commit and close the connection
    conn.commit()
    conn.close()

    logger.info("Data has been successfully inserted into the database.")


def concatenate_textures(textures):
    return ' '.join(textures)

def aggregation(ssurgo_data):
    logger = setup_logger()
    logging.info(f"Number of NaN values in the ssurgo_data: {ssurgo_data.isna().sum()}")
    ## ssurgo_data_original has id,name,muid,seqn,s5id,cmppct,hyd_grp,dp_tot,anion_excl,perc_crk,texture
    ssurgo_data['id'] = ssurgo_data.index
    ssurgo_data['name'] = ssurgo_data['compname']
    ssurgo_data['muid'] = ssurgo_data['mukey']
    ssurgo_data['seqn'] = ssurgo_data['mukey']
    ssurgo_data['s5id'] = ssurgo_data['mukey']
    ssurgo_data['cmppct'] = ssurgo_data['comppct_r']
    ssurgo_data['hyd_grp'] = ssurgo_data['hydgrp']
    ssurgo_data['dp_tot'] = ssurgo_data['hzthk_r']   ### we need to use thinkness to calculate the total depth (dp_tot)
    ssurgo_data['anion_excl'] = 0.5
    ssurgo_data['perc_crk'] = 0.5
    ssurgo_data['texture'] = ssurgo_data['texture']
    
    logger.info(f"hyd_grp: {ssurgo_data['hyd_grp']}")
    logger.info(f'number of NaN values in the hyd_grp column: {ssurgo_data["hyd_grp"].isna().sum()}')
    ssurgo_data.fillna(value={'hyd_grp':'Unknown'}, inplace=True)
    logger.info(f'number of NaN values in the hyd_grp column after fillna: {ssurgo_data["hyd_grp"].isna().sum()}')
    ssurgo_data.fillna(value={'texture':'Unknown'}, inplace=True)
    logger.info(f'number of NaN values in the dp_tot column after fillna: {ssurgo_data["dp_tot"].isna().sum()}')
    ssurgo_data.fillna(value={'dp_tot':0.0}, inplace=True)
    assert ssurgo_data['hyd_grp'].isna().sum() == 0, "There is a NaN value in the hyd_grp column"
    assert ssurgo_data['texture'].isna().sum() == 0, "There is a NaN value in the texture column"

    ssurgo_data = ssurgo_data[['id','name', 'muid', 'seqn', 's5id', 'cmppct', 'hyd_grp', 'dp_tot', 'anion_excl', 'perc_crk', 'texture']]


    logger.info(f"Checking if 377988 is in muid: {'377988' in ssurgo_data['muid'].astype(str).values}")
    return (
        ssurgo_data.groupby('muid')
        .agg(
            {
                'name': 'first',
                'seqn': 'first',
                's5id': 'first',
                'cmppct': 'first',
                'hyd_grp': 'first',
                'dp_tot': 'sum',
                'anion_excl': 'first',
                'perc_crk': 'first',
                'texture': concatenate_textures,
            }
        )
        .reset_index()
    )



def merge_data(chorizon, component, chtexturegrp, cointerp):
    ssurgo_data = pd.merge(component, chorizon, on='cokey', how='left').merge(chtexturegrp, on='chkey', how='left').merge(cointerp, on='cokey', how='left')
    logger.info(f"unique values in seqnum: {ssurgo_data['seqnum'].unique()}")
    #ssurgo_data.to_csv('database/merged_ssurgo_data.csv')

    return ssurgo_data



def create_ssurgo_layer_data(chorizon, component):
    # Transform data to match the 'ssurgo_layer' schema
    ssurgo_layer_data = pd.merge(chorizon, component, on='cokey', how='left')
    ## sort by depth to the top of the horizon, from the top to the bottom
    ssurgo_layer_data = ssurgo_layer_data.sort_values(['cokey', 'hzdept_r'], ascending=[True, True])
    ssurgo_layer_data = ssurgo_layer_data.assign(
        soil_id = ssurgo_layer_data['cokey'],
        layer_num = ssurgo_layer_data.groupby('cokey').cumcount() + 1
    )

    # Add muid to the layer data by using the mukey from the component table
    ssurgo_layer_data['muid'] = ssurgo_layer_data['mukey']

    ssurgo_layer_data = ssurgo_layer_data.rename(columns={
        'hzdept_r': 'dp',
        'dbthirdbar_r': 'bd',
        'awc_r': 'awc',
        'ksat_r': 'soil_k',
        'om_r': 'carbon',
        'claytotal_r': 'clay',
        'silttotal_r': 'silt',
        'sandtotal_r': 'sand',
        'fraggt10_r': 'rock',
        'albedodry_r': 'alb',
        'kwfact': 'usle_k',
        'ec_r': 'ec',
        'caco3_r': 'caco3',
        'ph1to1h2o_r': 'ph'
    })

    fill_values = {
        'usle_k': 0.1, 'alb': 0.23, 'rock': 0.0, 'ec': 0.0,
        'caco3': 0.0, 'ph': 6.5, 'carbon': 1.0, 'awc': 0.1,
        'bd': 1.3, 'clay': 0.1, 'silt': 0.1, 'sand': 0.1,
        'soil_k': 0.1, 'dp': 0.1
    }
    ssurgo_layer_data.fillna(value=fill_values, inplace=True)
    ssurgo_layer_data = ssurgo_layer_data[['muid','soil_id', 'layer_num', 'dp', 'bd', 'awc', 'soil_k', 'carbon', 'clay', 'silt', 'sand', 'rock', 'alb', 'usle_k', 'ec', 'caco3', 'ph']]

    return ssurgo_layer_data


def check_with_reference_rasters(soil_layer_data, ssurgo_data):
    path = '/data/MyDataBase/SWATGenXAppData/all_rasters/gSURRGO_swat_250m.tif'
    ## open the raster and get the unique values
    assert os.path.exists(path), "The raster file does not exist"   
    import rasterio
    import matplotlib.pyplot as plt 
    with rasterio.open(path) as src:
        ## get the array
        array = src.read(1)
        unique_values = src.read(1).flatten()
        unique_values = unique_values[unique_values != src.nodata]
        unique_values = unique_values[unique_values != 0]
        unique_values = unique_values[~pd.isnull(unique_values)]
        unique_values = pd.Series(unique_values).unique()




    print(f"unique values in the raster: {np.sort(unique_values)}") 
    print(f"unique values in the soil layer data: {soil_layer_data['muid'].unique()}")
    print(f"unique values in the ssurgo data: {soil_layer_data[soil_layer_data['muid'].isin(ssurgo_data['muid'])]['muid'].unique()}")
    print(f"unique values in the ssurgo data: {ssurgo_data['muid'].unique()}")
    print(f"unique values in the ssurgo data: {ssurgo_data[ssurgo_data['muid'].isin(soil_layer_data['muid'])]['muid'].unique()}")
    ### also length of those that we printed above should be equal to the length of the unique values in the raster
    print(f"length of unique values in the raster: {len(unique_values)}")
    print(f"length of unique values in the soil layer data: {len(soil_layer_data['muid'].unique())}")   
    print(f"length of unique values in the soil_layer_data: {len(soil_layer_data[soil_layer_data['muid'].isin(unique_values)]['muid'].unique())}")
    print(f"length of unique values in the ssurgo data: {len(ssurgo_data[ssurgo_data['muid'].isin(unique_values)]['muid'].unique())}")


    # Ensure the replacement values array has the correct shape
    unique_muid = soil_layer_data['muid'].unique()
    soil_k_values = soil_layer_data['soil_k'].values

    if len(unique_muid) != len(soil_k_values):
        ### drop not matching values
        unique_muid = unique_muid[unique_muid.isin(unique_values)]
        soil_k_values = soil_k_values[soil_k_values.isin(unique_values)]
    # Replace values in the raster with values in the soil layer data for soil_k
    arrayk = np.where(np.isin(array, unique_muid), soil_k_values, array)

    # Plot and save the resulting array
    plt.imshow(arrayk)
    plt.savefig('soil_k_raster.png')



def create_gssurgo_database():
    os.makedirs('database', exist_ok=True)
    chorizon, component, chtexturegrp, cointerp, mapunit = load_data()
    ssurgo_data = merge_data(chorizon, component, chtexturegrp, cointerp)
    ssurgo_data = aggregation(ssurgo_data)
    ssurgo_layer_data = create_ssurgo_layer_data(chorizon, component)
    
    
    print(ssurgo_data)

    print(ssurgo_layer_data)
    
    ## make sure muid in ssurgo data and soil layer data are the same
    ## assert all unique values in muid of the soil layer data are in the raster
    ssurgo_layer_data['muid'] = ssurgo_layer_data['muid'].astype(int)
    ssurgo_data['muid'] = ssurgo_data['muid'].astype(int)
    
    ssurgo_layer_data = ssurgo_layer_data[ssurgo_layer_data['muid'].isin(ssurgo_data['muid'])]
    ssurgo_data = ssurgo_data[ssurgo_data['muid'].isin(ssurgo_layer_data['muid'])]
    ssurgo_layer_data.to_pickle('database/ssurgo_layer_data.pkl')
    ssurgo_data.to_pickle('database/ssurgo_data.pkl')

    check_with_reference_rasters(ssurgo_layer_data, ssurgo_data)

    ### assert all muid in soil layers persent in the soil data
    assert all(ssurgo_layer_data['muid'].isin(ssurgo_data['muid'])), "There is a missing muid in the ssurgo data"
    ssurgo_layer_data.drop_duplicates(subset=['muid'], inplace=True)

    create_ssurgo_sqlite_db(ssurgo_data, ssurgo_layer_data)




def reset_logger():
    with open('gssurgo_db.log', 'w'):
        pass

if __name__ == '__main__':
    reset_logger()
    logger = setup_logger()
    create_gssurgo_database()
