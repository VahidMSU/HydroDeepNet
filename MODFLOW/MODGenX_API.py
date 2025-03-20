import os
from MODGenX.MODGenXCore import MODGenXCore
from multiprocessing import Process, Queue
from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
from functools import partial
import shutil
import warnings
import geopandas as gpd
import rasterio
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
	LEVEL = 'huc12'
	VPUID = "0712"
	RESOLUTION = 250
	NAME = '05536265'  ### testing, keep it or condition it
	MODFLOW_MODEL_NAME = 'MODFLOW_250m'
	SWAT_MODEL_NAME = 'SWAT_MODEL'

	config = SWATGenXPaths(
			MODFLOW_MODEL_NAME=MODFLOW_MODEL_NAME,
			SWAT_MODEL_NAME=SWAT_MODEL_NAME,
			LEVEL=LEVEL,
			VPUID=VPUID,
			NAME=NAME,
			RESOLUTION=RESOLUTION,
			username='vahidr32',
	)

	modflow_model = MODGenXCore(config)
	modflow_model.create_modflow_model()
