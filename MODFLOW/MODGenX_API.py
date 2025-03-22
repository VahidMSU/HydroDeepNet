import os
from MODGenX.MODGenXCore import MODGenXCore
from multiprocessing import Process, Queue
from functools import partial
import shutil
import warnings
import geopandas as gpd
import rasterio
import sys
sys.path.append('/data/SWATGenXApp/codes/SWATGenX/')
from SWATGenX.utils import find_VPUID
"""
/***************************************************************************
    SWATGenX
                            -------------------
        begin                : 2023-05-15
        copyright            : (C) 2024 by Vahid Rafiei
        email                : rafieiva@msu.edu
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
#warnings.filterwarnings("ignore", category=DeprecationWarning)

if __name__ == '__main__':

	print('start')
	BASE_PATH = "/data/SWATGenXApp/GenXAppData/"
	LEVEL = 'huc12'
	NAME = '04112500'
	VPUID = find_VPUID(NAME)
	username = "vahidr32"
	RESOLUTION = 250
	modelnames = ['MODFLOW_30m']

	print('start')
	models = []
	test = True
	parallel = False

	MODEL_NAME = 'MODFLOW_250m'
	SWAT_MODEL_NAME = 'SWAT_MODEL_Web_Application'
	ML = False
	modflow_model = MODGenXCore(username, NAME, VPUID, BASE_PATH, LEVEL, RESOLUTION, MODEL_NAME, ML, SWAT_MODEL_NAME)
	modflow_model.create_modflow_model()
