import os
from utils import *
from multiprocessing import Process, Queue
from functools import partial

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

if __name__ == '__main__':
    LEVEL = "huc12"
    RESOLUTION = 250
    BASE_PATH = "/data/MyDataBase/SWATGenXAppData/"
    NAMES = os.listdir(f"/data/MyDataBase/SWATGenXAppData/SWAT_input/{LEVEL}")
    NAMES.remove('log.txt')
    start_year=2002
    end_year =2005
    SWAT_MODEL_NAME = "SWAT_MODEL_30m"
    MODEL_NAME= 'SWAT_gwflow_MODEL_30m'
    #NAME = NAMES[0]
    MODFLOW_MODEL_NAME = "MODFLOW_250m"

    test = False
    parallel = True

    if test:
        NAME = "04166000"
        for i, NAME in enumerate(NAMES):
            if len(NAME) >10:
                continue
            model_base_path = os.path.join(BASE_PATH, f'SWAT_input/{LEVEL}/{NAME}/{SWAT_MODEL_NAME}')
            if not os.path.exists(model_base_path):
                print(f"{i} Model {model_base_path} not exists")
                continue
            if os.path.exists(f"/data/MyDataBase/SWATGenXAppData/SWAT_input/huc12/{NAME}/{MODEL_NAME}/Scenarios/Default/TxtInOut/gwflow.input") and os.path.exists(f"/data/MyDataBase/SWATGenXAppData/SWAT_input/huc12/{NAME}/streamflow_data/"):
                print(f"{i} Model {NAME} already exists")
                continue
            else:
                print(f"{i} need to create model {NAME}")
            try:
                completion_status, GRIDS_TABLE = creating_gwflow(BASE_PATH, LEVEL, NAME, RESOLUTION,start_year, end_year, MODEL_NAME, MODFLOW_MODEL_NAME, SWAT_MODEL_NAME)
                print(completion_status)
            except Exception as e:
                print(f"Error in {NAME} : {e}")

    else:

        processes = []
        queue = Queue()
        max_processes = 40
        n = 0

        for NAME in NAMES:
            if len(NAME) >10:
                continue
            if not os.path.exists(f"/data/MyDataBase/SWATGenXAppData/SWAT_input/huc12/{NAME}/{SWAT_MODEL_NAME}"):
                print(f"{n} - Model {NAME} {SWAT_MODEL_NAME} not exists")
                n += 1
                continue
            if os.path.exists(f"/data/MyDataBase/SWATGenXAppData/SWAT_input/huc12/{NAME}/{MODEL_NAME}/Scenarios/Default/TxtInOut/gwflow.input"):
                print(f"{n} - Model {NAME} already exists")
                n += 1
                continue

            model_base_path = os.path.join(BASE_PATH, f'SWAT_input/{LEVEL}/{NAME}/{MODFLOW_MODEL_NAME}/metrics.csv')
            if os.path.exists(model_base_path):
                wrapped_model = partial(creating_gwflow, BASE_PATH, LEVEL, NAME, RESOLUTION,start_year, end_year, MODEL_NAME, MODFLOW_MODEL_NAME, SWAT_MODEL_NAME)
                queue.put(wrapped_model)
        while not queue.empty():
            if len(processes) < max_processes:
                p = Process(target=queue.get())
                p.start()
                processes.append(p)
            else:
                for p in processes:
                    p.join()
                processes = []
        for p in processes:
            p.join()
