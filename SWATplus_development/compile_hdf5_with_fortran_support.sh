#!/bin/bash

# Source the Intel compiler environment script
source /opt/intel/oneapi/setvars.sh

# Remove any previous installations and create a new directory for HDF5
rm -rf /usr/local/hdf5
mkdir -p /usr/local/hdf5

# Navigate to the HDF5 source directory
cd /home/rafieiva/MyDataBase/SWATplus_development/hdf5-1.14.4-2

# Clean previous build files
make clean

# Configure with the appropriate options
./configure --enable-shared --enable-fortran --prefix=/usr/local/hdf5

# Build and install
make
make install