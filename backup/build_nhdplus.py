import os
import pandas as pd
import geopandas as gpd
import multiprocessing

class NHDPlusCompiler:
	def __init__(self, state):
		self.state = state
		self.VPUIDs = ["0405", "0406", "0407", "0408", "0409", "0410", "0411"]
		self.base_path = f"/data/MyDataBase/SWATGenXAppData/NHDPlusData/{self.state}"
		self.geodata_paths = {
		#  "NHDPlusFlowlineVAA": os.path.join(self.base_path, "NHDPlusFlowlineVAA.pkl"),
		#  "NHDPlusCatchment": os.path.join(self.base_path, "NHDPlusCatchment.pkl"),
		#  "NHDWaterbody": os.path.join(self.base_path, "NHDWaterbody.pkl"),
		#  "NHDFlowline": os.path.join(self.base_path, "NHDFlowline.pkl"),
			"NHDPlusEROMMA": os.path.join(self.base_path, "NHDPlusEROMMA.pkl"),
		}

	def process_vpu(self, VPUID, layer_name):
		NHDPlus_base = f"/data/MyDataBase/SWATGenXAppData/NHDPlusData/SWATPlus_NHDPlus/{VPUID}/unzipped_NHDPlusVPU/NHDPLUS_H_{VPUID}_HU4_GDB.gdb"
		geodata = gpd.read_file(NHDPlus_base, layer=layer_name)
		geodata['VPUID'] = VPUID
		if layer_name not in ["NHDPlusFlowlineVAA", 'NHDPlusEROMMA']:
			geodata = geodata.to_crs("EPSG:4326")
		return (layer_name, geodata)


	def compile_data(self):
		pool = multiprocessing.Pool(50)
		results = [pool.apply_async(self.process_vpu, args=(VPUID, layer_name)) for VPUID in self.VPUIDs for layer_name in self.geodata_paths.keys()]
		pool.close()
		pool.join()

		all_geodata = {layer_name: [] for layer_name in self.geodata_paths.keys()}
		for result in results:
			layer_name, geodata = result.get()
			all_geodata[layer_name].append(geodata)

		os.makedirs(self.base_path, exist_ok=True)
		for layer_name, geodata_list in all_geodata.items():
			geodata_concat = pd.concat(geodata_list, ignore_index=True)
			geodata_concat.reset_index(drop=True).to_pickle(self.geodata_paths[layer_name])

if __name__ == "__main__":
    compiler = NHDPlusCompiler("Michigan")
    compiler.compile_data()
