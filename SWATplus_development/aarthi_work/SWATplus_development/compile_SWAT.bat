cd D:\MyDataBase\SWAT_plus_source_code\modular_swatplus\

rmdir /S /Q build
mkdir build

cd D:\MyDataBase\SWAT_plus_source_code\modular_swatplus\build

ifx -traceback -O3 -debug -c ..\source_codes_swat_gwflow_snowdas\hru_module.f90 
ifx -traceback -O3 -debug -c ..\source_codes_swat_gwflow_snowdas\time_module.f90 
ifx -traceback -O3 -debug -c ..\source_codes_swat_gwflow_snowdas\constituent_mass_module.f90 
ifx -traceback -O3 -debug -c ..\source_codes_swat_gwflow_snowdas\*_module.f90 
ifx -traceback -O3 -debug -c ..\source_codes_swat_gwflow_snowdas\allocate_parms.f90 
ifx -traceback -O3 -debug -c ..\source_codes_swat_gwflow_snowdas\main.f90
ifx -traceback -O3 -debug -c ..\source_codes_swat_gwflow_snowdas\command.f90
ifx -traceback -O3 -debug -c ..\source_codes_swat_gwflow_snowdas\actions.f90
ifx -traceback -O3 -debug -c ..\source_codes_swat_gwflow_snowdas\*.f90
cd D:\MyDataBase\SWAT_plus_source_code\modular_swatplus\build
ifx -traceback -O3  -c ..\source_codes_swat_gwflow_snowdas\cli_pmeas.f90
ifx -traceback -O3  -c ..\source_codes_swat_gwflow_snowdas\proc_date_time.f90
ifx -traceback -O3 -debug -o swatplus_debug.exe *.obj
copy /Y "D:\MyDataBase\SWAT_plus_source_code\modular_swatplus\build\swatplus_debug.exe" "D:\MyDataBase\bin\swatplus_debug.exe"
copy /Y "D:\MyDataBase\SWAT_plus_source_code\modular_swatplus\build\swatplus_debug.exe" "D:\MyDataBase\SWATplus_by_VPUID\0407\huc12\40700040303\SWAT_MODEL\Scenarios\Default\TxtInOut\swatplus_debug.exe"
copy /Y "D:\MyDataBase\SWAT_plus_source_code\modular_swatplus\build\swatplus_debug.pdb" "D:\MyDataBase\SWATplus_by_VPUID\0407\huc12\40700040303\SWAT_MODEL\Scenarios\Default\TxtInOut\swatplus_debug.pdb"
cd D:\MyDataBase\SWATplus_by_VPUID\0407\huc12\40700040303\SWAT_MODEL\Scenarios\Default\TxtInOut
swatplus_debug.exe
