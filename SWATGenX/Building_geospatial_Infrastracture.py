from SWATGenX.geospatial_infrastructure_builder import geospatial_infrastructure_builder
from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
from SWATGenX.utils import get_all_VPUIDs


from multiprocessing import Pool

def process_VPUID(VPUID):
    landuse_epoch = "2021"
    print(f"Building geospatial infrastructure for {VPUID}")
    geospatial_infrastructure_builder(VPUID, landuse_epoch)

if __name__ == "__main__":

    with open("/data/SWATGenXApp/codes/SWATGenX/SWATGenX/critical_errors.txt", 'w') as file:
        file.write("")
    VPUIDs = get_all_VPUIDs()
    print(VPUIDs)
    #VPUIDs = [VPUID for VPUID in VPUIDs if VPUID[:2] in ["03"]]

    with Pool(processes=5) as pool:
        print("Starting geospatial infrastructure builder")
        pool.map(process_VPUID, VPUIDs)
