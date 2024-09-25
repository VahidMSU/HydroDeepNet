#!/bin/bash

# Source the Intel compiler environment script
source /opt/intel/oneapi/setvars.sh

# Set the HDF5 library and include paths
HDF5_INC=/usr/local/hdf5/include
HDF5_LIB=/usr/local/hdf5/lib

# Compile the Fortran source code
ifx -I$HDF5_INC -L$HDF5_LIB -lhdf5_fortran -lhdf5 -o test_hdf5 test_hdf5.f90

# Check if the compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful."
else
    echo "Compilation failed."
fi
