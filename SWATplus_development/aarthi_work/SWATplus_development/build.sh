rm -rf /usr/local/hdf5 
mkdir -p /usr/local/hdf5 
cd /home/rafieiva/hdf5-1.14.4-3
make clean
source /opt/intel/oneapi/setvars.sh
./configure --enable-parallel --enable-shared --enable-fortran --enable-fortran2003 --prefix=/usr/local/hdf5 CC=mpicc FC=mpiifort CXX=mpicxx LDFLAGS="-L/opt/intel/oneapi/mpi/latest/lib" LIBS="-lmpi -lmpifort" CFLAGS="-Wno-redundant-decls" FCFLAGS="-Wno-redundant-decls"
make 
make install
