rm -rf /home/rafieiva/lib/hdf5 
mkdir -p /home/rafieiva/lib/hdf5 
cd /home/rafieiva/hdf5-1.14.4-2
make clean 
./configure --enable-parallel --enable-shared --enable-fortran prefix=/home/rafieiva/lib/hdf5 CC=mpicc 
make 
make install