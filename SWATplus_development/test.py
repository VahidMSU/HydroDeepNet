## read h5 data
import h5py 
from mpi4py import MPI
path = "/home/rafieiva/MyDataBase/SWATplus_development/sup/parallel.h5"

with h5py.File(path, 'r') as f:
	print(f.keys())
	array = f[' dset1']
	print(array.shape)
	print(array[:])
