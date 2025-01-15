from SWATGenX.wrapped_build_geospatial_infrastructure import wrapped_build_geospatial_infrastructure
from SWATGenX.read_VPUID import get_all_VPUIDs
from multiprocessing import Process

if __name__ == "__main__":
    VPUID = "0415"
    VPUIDs = get_all_VPUIDs()
    landuse_epoch = "2021"
    processes = []
    wrapped_build_geospatial_infrastructure(VPUID, landuse_epoch)


   # for VPUID in VPUIDs:
   #     if VPUID not in ["1506"]:
   #         continue
   #     p = Process(target=wrapped_build_geospatial_infrastructure, args=(VPUID, landuse_epoch))
   #     p.start()
   #     processes.append(p)
   # for p in processes:
   #     p.join()
