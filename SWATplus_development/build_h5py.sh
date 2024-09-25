export HDF5_VERSION=1.14.4
export HDF5_DIR=/home/rafieiva/lib/hdf5
export MPI_DIR=/home/rafieiva/lib/openmpi
export CC=mpicc
pip uninstall -y h5py
cd h5py
python setup.py install