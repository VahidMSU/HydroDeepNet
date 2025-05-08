from SWATGenX.NHDPlus_extract_by_VPUID import NHDPlus_extract_by_VPUID
from SWATGenX.NHDPlus_preprocessing import NHDPlus_preprocessing
from SWATGenX.gssurgo_extraction import gSSURGO_extract_by_VPUID
from SWATGenX.USGS_DEM_extraction import DEM_extract_by_VPUID
from SWATGenX.NLCD_extraction import NLCD_extract_by_VPUID_helper
import time
import os
from SWATGenX.utils import get_all_VPUIDs
from SWATGenX.PRISM_extraction import PRISMExtractor
from SWATGenX.SWATGenXConfigPars import SWATGenXPaths


if __name__ == "__main__":
    VPUIDs = get_all_VPUIDs()
    VPUIDs = ["0409"]
    for VPUID in VPUIDs:

        paths = {
            "PRISM": f"/data/SWATGenXApp/GenXAppData/PRISM/VPUID/{VPUID}/PRISM_grid.shp",
            "NLCD": f"/data/SWATGenXApp/GenXAppData/NLCD/VPUID/{VPUID}/NLCD_{VPUID}_2021_30m.tif",
            "DEM": f"/data/SWATGenXApp/GenXAppData/DEM/VPUID/{VPUID}/DEM_{VPUID}_2021_30m.tif",
            "gSSURGO": f"/data/SWATGenXApp/GenXAppData/gSSURGO/VPUID/{VPUID}/gSSURGO_{VPUID}_30m.tif",
            "unzipped_NHDPlusHR": f"/data/SWATGenXApp/GenXAppData/NHDPlusHR/VPUID/{VPUID}/unzipped_NHDPlusVPU",
            "NHDPlusHR_streams": f"/data/SWATGenXApp/GenXAppData/NHDPlusHR/VPUID/{VPUID}/streams.pkl",
        }



        if len(os.listdir(paths["unzipped_NHDPlusHR"])) == 0:
            NHDPlus_extract_by_VPUID(VPUID)
        else:
            print(f"NHDPlusHR already exists for {VPUID}")
            time.sleep(2)
        if not os.path.exists(paths["NHDPlusHR_streams"]):
            NHDPlus_preprocessing(VPUID)
        else:
            print(f"NHDPlusHR streams already exists for {VPUID}")
            time.sleep(2)
        if not os.path.exists(paths["gSSURGO"]):
            gSSURGO_extract_by_VPUID(VPUID)
        else:
            print(f"gSSURGO already exists for {VPUID}")
            time.sleep(2)
        if not os.path.exists(paths["DEM"]):
            DEM_extract_by_VPUID(VPUID)
        else:
            print(f"DEM already exists for {VPUID}")
            time.sleep(2)
        if not os.path.exists(paths["NLCD"]):
            NLCD_extract_by_VPUID_helper(VPUID, epoch="2021")
        else:
            print(f"NLCD already exists for {VPUID}")
            time.sleep(2)
        if not os.path.exists(paths["PRISM"]):
            PRISMExtractor(SWATGenXPaths, VPUID, "HUC4", "PRISM_pcp", overwrite=True).clip_PRISM_by_VPUID()
        else:
            print(f"PRISM already exists for {VPUID}")
            time.sleep(2)

    print("NHDPlus extraction and preprocessing completed for all VPUIDs")