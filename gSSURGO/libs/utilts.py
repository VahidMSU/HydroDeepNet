
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

def read_gdb_chunk(gdb_path, layer_name, columns, chunk_slice, start):
    """Read a chunk of data from the GDB file."""
    print(f"Reading chunk {start} to {start + chunk_slice}")
    return gpd.read_file(
        gdb_path,
        layer=layer_name,
        driver='FileGDB',
        include_fields=columns,
        ignore_geometry=True,
        rows=slice(start, start + chunk_slice),
    )

def load_gdb(gdb_path, layer_name, columns, num_processes=40):
    logger = setup_logger()
    logger.info(f"Reading the {layer_name} data from the gdb file")

    if os.path.exists(f"database/{layer_name}.pkl"):
        logger.info(f"Skipping {layer_name} as it already exists")
        data = pd.read_pickle(f"database/{layer_name}.pkl")
        logger.info(f"Loaded {layer_name} columns:\n{list(data.columns)}")
        return data
    else:
        chunk_slice = 500000  # Number of rows per chunk
        start = 0
        chunks = []

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            while True:
                futures.append(executor.submit(read_gdb_chunk, gdb_path, layer_name, columns, chunk_slice, start))
                start += chunk_slice

                future = futures[-1]
                data_chunk = future.result()
                if data_chunk.empty:
                    logger.info("No more data to read. Exiting.")
                    break
                chunks.append(data_chunk)

        # Combine all chunks into a single DataFrame
        data = pd.concat(chunks, ignore_index=True)

        # Write column names to a text file
        with open(f"{layer_name}_columns.txt", 'w') as f:
            for column in data.columns:
                f.write(f"{column}\n")

        # Save the combined DataFrame to a pickle file
        data.to_pickle(f"database/{layer_name}.pkl")

    logger.info(f"Data has been successfully read from the {layer_name} layer")
    return data



def load_data():
    logger = setup_logger()
    gdb_path = '/data/MyDataBase/SWATGenXAppData/Soil/gSSURGO_MI/gSSURGO_MI.gdb'
    logger.info("Loading data from GDB and creating cache")
    chorizon_columns = ['chkey', 'cokey', 'hzdepb_r', 'hzdept_r', 'dbthirdbar_r', 'awc_r', 'ksat_r','hzthk_r',
                        'om_r', 'claytotal_r', 'silttotal_r', 'sandtotal_r', 'fraggt10_r',
                        'kwfact', 'ec_r', 'caco3_r', 'ph1to1h2o_r']
    component_columns = ['cokey', 'compname', 'comppct_r', 'hydgrp', 'mukey','albedodry_r']
    chtexturegrp_columns = ['chkey', 'texture']
    cointerp = ['cokey', 'seqnum']
    MUPOLYGON_columns = ['MUKEY', 'MUSYM', 'AREASYMBOL']
    MUPOLYGON = load_gdb(gdb_path, 'MUPOLYGON', MUPOLYGON_columns)
    ### decapitalize columns
    MUPOLYGON.columns = map(str.lower, MUPOLYGON.columns)
    MUPOLYGON.to_csv('database/MUPOLYGON.csv')
    component = load_gdb(gdb_path, 'component', component_columns)
    chtexturegrp = load_gdb(gdb_path, 'chtexturegrp', chtexturegrp_columns)
    chorizon = load_gdb(gdb_path, 'chorizon', chorizon_columns)
    cointerp = load_gdb(gdb_path, 'cointerp', cointerp)

    logger.info("Data has been successfully loaded.")

    return chorizon, component, chtexturegrp, cointerp, MUPOLYGON
