source /opt/intel/oneapi/setvars.sh
rm -r build
mkdir build
cd build

mpifort -O3 -I/home/rafieiva/lib/hdf5/include -L/home/rafieiva/lib/hdf5/lib -c ../_temp.f90
mpifort -o _temp *.o -L/home/rafieiva/lib/hdf5/lib -lhdf5_fortran

./_temp