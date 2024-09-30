from mpi4py import MPI
import time 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

print(f"Hello from process {rank}")