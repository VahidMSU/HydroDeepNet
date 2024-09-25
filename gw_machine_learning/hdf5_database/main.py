import itertools
import os
import numpy as np
import sys
from mpi4py import MPI
import hdf5_mpi

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# generate a 2d with (1849, 1458) shape
data = np.random.uniform(0, 4, size=(10, 10)).astype(np.float32)
RESOLUTIONS = [10]
print(data)

# Call Fortran subroutine to create, write, and close the HDF5 file and dataset
for RESOLUTION, dummy in itertools.product(RESOLUTIONS, range(20,23)):
    dataset_name = 'my_dataset_name_{0}'.format(dummy)
    if rank == 0:
        print(f'Creating, writing, and closing HDF5 file: {dataset_name}')
    print(f"average: {np.average(data)} std: {np.std(data)}")
    hdf5_mpi.hdf5_operations.write_hdf5_file("ML.h5",f"{RESOLUTION}" ,dataset_name, data)

# Finalize MPI
MPI.Finalize()
