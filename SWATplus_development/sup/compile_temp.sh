# Load Intel OneAPI environment
source /opt/intel/oneapi/setvars.sh

# Setup and clean environment
rm -rf build && mkdir build
cd build
rm -f SWATplus.h5

# Compilation with proper HDF5 and MPI usage
mpifort -O3 -I/home/rafieiva/lib/hdf5/include -c ../test.f90
mpifort -o test *.o -L/home/rafieiva/lib/hdf5/lib -lhdf5_fortran -lhdf5hl_fortran -lhdf5_hl
if [ -f *.o ]; then
    # MPI execution, checking the correct usage of mpiexec arguments
    mpirun -np 4 ./test
else
    echo "Compilation failed, object files not generated."
fi
