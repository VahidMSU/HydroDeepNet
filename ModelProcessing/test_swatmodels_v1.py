import pandas as pd
import numpy as np
import os
import glob
import shutil
import zipfile
import requests

def test_swatmodels_v1(VPUIDs=None):
    if VPUIDs is None:
        VPUIDs = os.listdir("/data/SWATGenXApp/Users/swatmodels_v1/")
    else:
        VPUIDs = [VPUID]
    for VPUID in VPUIDs:
        models = os.listdir(f"/data/SWATGenXApp/Users/swatmodels_v1/{VPUID}/huc12")
        for model in models:
            success = False
            simulation_path = f"/data/SWATGenXApp/Users/swatmodels_v1/{VPUID}/huc12/{model}/SWAT_MODEL/Scenarios/Default/TxtInOut/simulation.out"
            with open(simulation_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                if "Execution successfully completed" in line:
                    success = True
                    break
            if not success:
                print(f"Execution failed for {VPUID} {model}")

if __name__ == "__main__":
    test_swatmodels_v1()