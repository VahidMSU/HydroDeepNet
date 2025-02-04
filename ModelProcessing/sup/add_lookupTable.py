import pandas as pd
import os
import h5py
import numpy as np
def add_lookup_table(NAME, ver, lookup_csv):
    """
    Add the lookup table to the SWAT+ output HDF5 file.
    """
    SWATplus_output = f"/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12/{NAME}/SWAT_gwflow_MODEL/Scenarios/verification_stage_{ver}/SWATplus_output.h5"
    print(f"Processing {SWATplus_output}")
    with h5py.File(SWATplus_output, "r+") as f:
        # Check if Landuse group exists
        if "Landuse" not in f:
            print(f"Group 'Landuse' does not exist for {NAME}")
            raise ValueError(f"Group 'Landuse' does not exist in {SWATplus_output}")

        # Save lookup table as a CSV string in the HDF5 file
        if "lookup_table" in f["Landuse"]:
            del f["Landuse/lookup_table"]  # Overwrite if it already exists
        f["Landuse/lookup_table"] = np.bytes_(lookup_csv)  # Save as a string
        print(f"Lookup table added to {SWATplus_output}")


if __name__ == "__main__":
    path = "/data2/MyDataBase/SWATGenXAppData/LandUse/landuse_lookup.csv"
    df = pd.read_csv(path)
    lookup_csv = df.to_csv(index=False)  # Convert DataFrame to CSV string

    NAMES = os.listdir("/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12")
    NAMES.remove("log.txt")

    for NAME in NAMES:
        for ver in range(0, 6):
            add_lookup_table(NAME, ver, lookup_csv)
