# the purpose of this code is to compile the NHDPlus data into a single file
import os
import pandas as pd
import geopandas as gpd
import pandas as pd
import geopandas as gpd

#output path
state = "Michigan"
NHDPlusVAA_path   = f"/data/MyDataBase/SWATGenXAppData/NHDPlusData/{state}/NHDPlusFlowlineVAA.pkl"
NHDWaterbody_path = f"/data/MyDataBase/SWATGenXAppData/NHDPlusData/{state}/NHDWaterbody.pkl"
NHDFlowlines_path = f"/data/MyDataBase/SWATGenXAppData/NHDPlusData/{state}/NHDFlowline.pkl"

VPUIDs = ["0405", "0406", "0407", "0408", "0409", "0410","0411"]

all_NHDFlowlines = pd.DataFrame()
all_NHDFlowlinesVAA = pd.DataFrame()
all_NHDWaterbody = pd.DataFrame()

for VPUID in VPUIDs:
	NHDPlus_base = f"/data/MyDataBase/SWATGenXAppData/NHDPlusData/SWATPlus_NHDPlus/{VPUID}/unzipped_NHDPlusVPU/NHDPLUS_H_{VPUID}_HU4_GDB.gdb"
	NHDFlowlinesVAA = gpd.read_file(NHDPlus_base, layer="NHDPlusFlowlineVAA")
	NHDFlowlinesVAA['VPUID'] = VPUID	

	NHDPFlowlines = gpd.read_file(NHDPlus_base, layer="NHDFlowline")
	NHDPFlowlines['VPUID'] = VPUID
	NHDPFlowlines = NHDPFlowlines.to_crs("EPSG:4326")
	

	NHDWaterbody = gpd.read_file(NHDPlus_base, layer="NHDWaterbody")
	NHDWaterbody['VPUID'] = VPUID
	NHDWaterbody = NHDWaterbody.to_crs("EPSG:4326")
	
	
	all_NHDFlowlines = pd.concat([all_NHDFlowlines, NHDPFlowlines], ignore_index=True)
	all_NHDFlowlinesVAA = pd.concat([all_NHDFlowlinesVAA, NHDFlowlinesVAA], ignore_index=True)
	all_NHDWaterbody = pd.concat([all_NHDWaterbody, NHDWaterbody], ignore_index=True)	

os.makedirs(f"/data/MyDataBase/SWATGenXAppData/NHDPlusData/{state}", exist_ok=True)
all_NHDFlowlines.reset_index(drop=True).to_pickle(NHDFlowlines_path)
all_NHDFlowlinesVAA.reset_index(drop=True).to_pickle(NHDPlusVAA_path)
all_NHDWaterbody.reset_index(drop=True).to_pickle(NHDWaterbody_path)

#output path