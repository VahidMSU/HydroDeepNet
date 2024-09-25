
import logging
import os
import geopandas as gpd
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import sqlite3
def setup_logger():
    """Set up a logger for the script."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    # Create a file handler
    handler = logging.FileHandler('gssurgo_db.log')
    handler.setLevel(logging.INFO)
    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # Add the handlers to the logger
    logger.addHandler(handler)
    return logger

def read_gdb_chunk(gdb_path, layer_name, columns):
    """Read the entire layer from the GDB file."""
    return gpd.read_file(
        gdb_path,
        layer=layer_name,
        include_fields=columns,
        ignore_geometry=True
    )

def load_gdb(gdb_path, layer_name, columns):
    logger = setup_logger()
    logger.info(f"Reading the {layer_name} data from the gdb file")

    if os.path.exists(f"database/{layer_name}.pkl"):
        logger.info(f"Skipping {layer_name} as it already exists")
        data = pd.read_pickle(f"database/{layer_name}.pkl")
        logger.info(f"Loaded {layer_name} columns:\n{list(data.columns)}")
        return data
    else:
        data = read_gdb_chunk(gdb_path, layer_name, columns)
        if data.empty:
            logger.error(f"No data found in the {layer_name} layer.")
            return pd.DataFrame()  # Return empty dataframe to avoid concatenation error

        # Save to pickle
        data.to_pickle(f"database/{layer_name}.pkl")

    logger.info(f"Data has been successfully read from the {layer_name} layer")
    return data




def load_data():
    logger = setup_logger()
    gdb_path = '/data/MyDataBase/SWATGenXAppData/Soil/gSSURGO_MI/gSSURGO_MI.gdb'
    assert os.path.exists(gdb_path), "The GDB file does not exist"  
    logger.info("Loading data from GDB and creating cache")
    chorizon_columns = ['chkey', 'cokey', 'hzdepb_r', 'hzdept_r', 'dbthirdbar_r', 'awc_r', 'ksat_r','hzthk_r',
                        'om_r', 'claytotal_r', 'silttotal_r', 'sandtotal_r', 'fraggt10_r',
                        'kwfact', 'ec_r', 'caco3_r', 'ph1to1h2o_r']
    component_columns = ['cokey', 'compname', 'comppct_r', 'hydgrp', 'mukey','albedodry_r']
    chtexturegrp_columns = ['chkey', 'texture']
    cointerp = ['cokey', 'seqnum']
    MUPOLYGON_columns = ['mukey', 'muname', 'areasymbol']
    MUPOLYGON = load_gdb(gdb_path, 'MUPOLYGON', MUPOLYGON_columns)
    MUPOLYGON.to_csv('database/MUPOLYGON.csv')
    component = load_gdb(gdb_path, 'component', component_columns)
    chtexturegrp = load_gdb(gdb_path, 'chtexturegrp', chtexturegrp_columns)
    chorizon = load_gdb(gdb_path, 'chorizon', chorizon_columns)
    cointerp = load_gdb(gdb_path, 'cointerp', cointerp)

    logger.info("Data has been successfully loaded.")

    return chorizon, component, chtexturegrp, cointerp, MUPOLYGON
