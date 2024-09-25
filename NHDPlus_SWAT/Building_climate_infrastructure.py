import os
from NHDPlus_SWAT.PRISM_extraction import clip_PRISM_by_VPUID
from NHDPlus_SWAT.read_VPUID import get_all_VPUIDs
import geopandas as gpd
from multiprocessing import Process
if __name__ == "__main__":

	VPUIDs = get_all_VPUIDs()
	processes = []
	for VPUID in VPUIDs:
		if VPUID not in ["1506"]:
			continue
		p = Process(target=clip_PRISM_by_VPUID, args=(VPUID,))
		p.start()
		processes.append(p)
	for p in processes:
		p.join()
