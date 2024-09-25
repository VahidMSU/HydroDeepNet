#!/bin/bash

# Source the Intel oneAPI environment
source /opt/intel/oneapi/setvars.sh

# Clean and prepare the build directory
rm -r build
mkdir build
cd build

# Set the include path for HDF5
HDF5_INCLUDE_PATH=/home/rafieiva/lib/hdf5/include

# Compile the Fortran modules with HDF5 support
ifx -I${HDF5_INCLUDE_PATH} -O3 -c -lhdf5_fortran -lhdf5 ../SWAT_source/hru_module.f90
ifx -I${HDF5_INCLUDE_PATH} -O3 -c -lhdf5_fortran -lhdf5 ../SWAT_source/time_module.f90
ifx -I${HDF5_INCLUDE_PATH} -O3 -c -lhdf5_fortran -lhdf5 ../SWAT_source/constituent_mass_module.f90
ifx -I${HDF5_INCLUDE_PATH} -O3 -c -lhdf5_fortran -lhdf5 ../SWAT_source/*_module.f90
ifx -I${HDF5_INCLUDE_PATH} -O3 -c -lhdf5_fortran -lhdf5 ../SWAT_source/allocate_parms.f90
ifx -I${HDF5_INCLUDE_PATH} -O3 -c -lhdf5_fortran -lhdf5 ../SWAT_source/main.f90
ifx -I${HDF5_INCLUDE_PATH} -O3 -c -lhdf5_fortran -lhdf5 ../SWAT_source/command.f90
ifx -I${HDF5_INCLUDE_PATH} -O3 -c -lhdf5_fortran -lhdf5 ../SWAT_source/actions.f90
ifx -I${HDF5_INCLUDE_PATH} -O3 -c -lhdf5_fortran -lhdf5 ../SWAT_source/*.f90

# Link the object files to create the executable
ifx -o swatplus *.o -L/home/rafieiva/lib/hdf5/lib -lhdf5_fortran -lhdf5
