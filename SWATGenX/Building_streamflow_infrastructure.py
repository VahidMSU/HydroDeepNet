from SWATGenX.USGS_streamflow_retrieval import USGS_streamflow_retrieval_by_VPUID, get_all_VPUIDs
from multiprocessing import Process
if __name__ == "__main__":

    start_date = "2000-01-01"
    end_date = "2023-01-01"
    VPUIDs = get_all_VPUIDs("/data/SWATGenXApp/GenXAppData/NHDPlusData")
    processes = []

    test = False
    if test:
        VPUIDs = "0406"
        USGS_streamflow_retrieval_by_VPUID(VPUIDs, start_date, end_date)
        exit()
    from multiprocessing import Pool

    def process_VPUID(VPUID):
        #if VPUID[:2] not in ["03"]:
        #    return
        print(f"Processing VPUID: {VPUID}")
        USGS_streamflow_retrieval_by_VPUID(VPUID, start_date, end_date)

    with Pool(processes=15) as pool:
        pool.map(process_VPUID, VPUIDs)
 