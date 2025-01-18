import os
from SWATGenX.PRISM_extraction import clip_PRISM_by_VPUID
from SWATGenX.read_VPUID import get_all_VPUIDs
import geopandas as gpd
from multiprocessing import Process
if __name__ == "__main__":

	VPUIDs = get_all_VPUIDs()
	processes = []
	for VPUID in VPUIDs:
		if VPUID[:2] not in ["01", "02", "03", "04", "12", "18","19","08"]:
			continue
		p = Process(target=clip_PRISM_by_VPUID, args=(VPUID,))
		p.start()
		processes.append(p)
	for p in processes:
		p.join()
