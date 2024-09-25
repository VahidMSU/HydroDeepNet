
### instruction for building h5py with MPI support:
### donwload hdf5 source code from https://www.hdfgroup.org/downloads/hdf5/source-code/?1713710010
## path used for openmpi
rm -rf /home/rafieiva/lib/hdf5
mkdir -p /home/rafieiva/lib/hdf5
cd /home/rafieiva/hdf5-1.14.4-2
make clean
./configure --enable-parallel --enable-shared --with-pic --prefix=/home/rafieiva/lib/hdf5 CC=mpicc
make
make install