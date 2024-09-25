# Source the Intel oneAPI environment (if not already sourced)
source /opt/intel/oneapi/setvars.sh

# Clean and prepare the build directory
rm -rf build
mkdir build
cd build

# Set the include and library paths for MPI-enabled HDF5 and MPI
MPI_INCLUDE_PATH=/opt/intel/oneapi/2024.1/include/mpi
MPI_LIB_PATH=/opt/intel/oneapi/mpi/latest/lib
HDF5_INCLUDE_PATH=/usr/local/hdf5/include
HDF5_LIB_PATH=/usr/local/hdf5/lib

# Compile Fortran source files
ifx -I${MPI_INCLUDE_PATH} -I${HDF5_INCLUDE_PATH} -O3 -c ../SWAT_source/hru_module.f90
ifx -I${MPI_INCLUDE_PATH} -I${HDF5_INCLUDE_PATH} -O3 -c ../SWAT_source/time_module.f90
ifx -I${MPI_INCLUDE_PATH} -I${HDF5_INCLUDE_PATH} -O3 -c ../SWAT_source/constituent_mass_module.f90
ifx -I${MPI_INCLUDE_PATH} -I${HDF5_INCLUDE_PATH} -O3 -c ../SWAT_source/basin_module.f90
ifx -I${MPI_INCLUDE_PATH} -I${HDF5_INCLUDE_PATH} -O3 -c ../SWAT_source/hydrograph_module.f90
ifx -I${MPI_INCLUDE_PATH} -I${HDF5_INCLUDE_PATH} -O3 -c ../SWAT_source/*_module.f90
ifx -I${MPI_INCLUDE_PATH} -I${HDF5_INCLUDE_PATH} -O3 -c ../SWAT_source/allocate_parms.f90
ifx -I${MPI_INCLUDE_PATH} -I${HDF5_INCLUDE_PATH} -O3 -c ../SWAT_source/main.f90
ifx -I${MPI_INCLUDE_PATH} -I${HDF5_INCLUDE_PATH} -O3 -c ../SWAT_source/command.f90
ifx -I${MPI_INCLUDE_PATH} -I${HDF5_INCLUDE_PATH} -O3 -c ../SWAT_source/actions.f90
ifx -I${MPI_INCLUDE_PATH} -I${HDF5_INCLUDE_PATH} -O3 -c ../SWAT_source/*.f90

# Link the object files to create the executable
ifx -g -o swatplus *.o -I/usr/local/hdf5/include -L/usr/local/hdf5/lib -lhdf5hl_fortran -lhdf5_fortran -lhdf5 -lmpi -lmpifort -lz -ldl -lm
# h5pfc -g -o swatplus *.o -lhdf5hl_fortran -lhdf5_fortran -lhdf5 -lz -ldl -lm


# Copy the executable to the desired directory (if created)
cd /home/rafieiva/MyDataBase/codes/SWATplus_development/aarthi_work/SWATplus_development/40601020807
#rm -f swatplus
cp /home/rafieiva/MyDataBase/codes/SWATplus_development/aarthi_work/SWATplus_development/build/swatplus /home/rafieiva/MyDataBase/codes/SWATplus_development/aarthi_work/SWATplus_development/40601020807

# Run the executable with MPI
mpirun -np 4 ./swatplus
