source /opt/intel/oneapi/setvars.sh
cp /home/rafieiva/MyDataBase/SWATplus_development/sup/print.prt /home/rafieiva/MyDataBase/SWATplus_example/print.prt
rm -r build
mkdir build
cd build
ifx -O3 -I/home/rafieiva/lib/hdf5/include -L/home/rafieiva/lib/hdf5/lib -c -lhdf5_fortran ../SWAT_source/hru_module.f90
ifx -O3 -I/home/rafieiva/lib/hdf5/include -L/home/rafieiva/lib/hdf5/lib -c -lhdf5_fortran ../SWAT_source/time_module.f90
ifx -O3 -I/home/rafieiva/lib/hdf5/include -L/home/rafieiva/lib/hdf5/lib -c -lhdf5_fortran ../SWAT_source/constituent_mass_module.f90
ifx -O3 -I/home/rafieiva/lib/hdf5/include -L/home/rafieiva/lib/hdf5/lib -c -lhdf5_fortran ../SWAT_source/*_module.f90
ifx -O3 -I/home/rafieiva/lib/hdf5/include -L/home/rafieiva/lib/hdf5/lib -c -lhdf5_fortran ../SWAT_source/allocate_parms.f90
ifx -O3 -I/home/rafieiva/lib/hdf5/include -L/home/rafieiva/lib/hdf5/lib -c -lhdf5_fortran ../SWAT_source/main.f90
ifx -O3 -I/home/rafieiva/lib/hdf5/include -L/home/rafieiva/lib/hdf5/lib -c -lhdf5_fortran ../SWAT_source/command.f90
ifx -O3 -I/home/rafieiva/lib/hdf5/include -L/home/rafieiva/lib/hdf5/lib -c -lhdf5_fortran ../SWAT_source/actions.f90
ifx -O3 -I/home/rafieiva/lib/hdf5/include -L/home/rafieiva/lib/hdf5/lib -c -lhdf5_fortran ../SWAT_source/*.f90
ifx -o swatplus *.o
cd /home/rafieiva/MyDataBase/SWATplus_example/
rm -r swatplus
cp /home/rafieiva/MyDataBase/SWATplus_development/build/swatplus /home/rafieiva/MyDataBase/SWATplus_example/
./swatplus