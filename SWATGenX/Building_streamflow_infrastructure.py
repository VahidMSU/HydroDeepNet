from SWATGenX.USGS_streamflow_retrieval import USGS_streamflow_retrieval_by_VPUID
from SWATGenX.utils import get_all_VPUIDs
from multiprocessing import Process
if __name__ == "__main__":

    VPUIDs = get_all_VPUIDs()
    print(VPUIDs)
    processes = []

    VPUIDs = ["0405", "0406", "0407", "0408", "0409", "0410"]

    test = False

    if test:
        VPUIDs = "0405"
        USGS_streamflow_retrieval_by_VPUID(VPUIDs)
        exit()
    from multiprocessing import Pool

    def process_VPUID(VPUID):
        #if VPUID[:2] not in ["03"]:
        #    return
        print(f"Processing VPUID: {VPUID}")
        USGS_streamflow_retrieval_by_VPUID(VPUID)

    with Pool(processes=1) as pool:
        pool.map(process_VPUID, VPUIDs)
