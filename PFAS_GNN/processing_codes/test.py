import time
import os
import pandas as pd
import geopandas as gpd
import numpy as np
from libs.utils import setup_logging
from libs.load_data import load_pfas_gw

# Load the data
data_dir = '/data/MyDataBase/HuronRiverPFAS'
def load_all_wssn(data_dir, logger):
    
    all_wssn = pd.read_pickle(os.path.join(data_dir, "WSSN_Huron_with_features.pkl"))
    logger.info(f"stage:load_all_wssn ==##== Number of all_wssn: {len(all_wssn)}")
    
    return all_wssn
logger = setup_logging()
all_wssn = load_all_wssn(data_dir, logger)