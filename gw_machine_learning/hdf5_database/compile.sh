#!/bin/bash
# Compile Fortran source to a shared library


f2py -c -m hdf5_mpi create_hdf5_mpi.f90 --fcompiler=gnu95 -L/home/rafieiva/lib/hdf5/lib -lhdf5_fortran -I/home/rafieiva/lib/hdf5/include -I/usr/lib64/openmpi/lib
mpiexec -np 3 python main.py 