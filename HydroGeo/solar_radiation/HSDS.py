## the purpose of this script is to provide a class that can be used to interact with the c API
import h5pyd
with h5pyd.File("E:/NSRDB/nsrdb_2014_full.h5") as f:
	print(f.keys())