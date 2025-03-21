cd /data/SWATGenXApp/codes/MODFLOW/MODFLOW-NWT/src

# Make sure the bin directory exists
mkdir -p ../bin

# Remove all object files to start fresh
rm -f *.o

# Create a modified version of MF_NWT.f to disable SWI2
cat MF_NWT.f | sed 's/CALL GWF2SWI2AR/!CALL GWF2SWI2AR/g' | \
              sed 's/CALL GWF2SWI2AD/!CALL GWF2SWI2AD/g' | \
              sed 's/CALL GWF2SWI2FM/!CALL GWF2SWI2FM/g' | \
              sed 's/CALL GWF2SWI2BD/!CALL GWF2SWI2BD/g' | \
              sed 's/CALL GWF2SWI2DA/!CALL GWF2SWI2DA/g' > MF_NWT_noswi.f

# Fix the openspec.inc issue in Irestart.f
cat Irestart.f | sed 's/INCLUDE '\''openspec.inc'\''/USE openspec/g' > Irestart_fixed.f

# Compile module files first to create .mod files
mpifort -c openspec.F90
mpifort -c gsfmodflow.f
mpifort -c NWT1_module.f

# Compile files that depend on modules
mpifort -c NWT1_gmres.f90
mpifort -c gwfsfrmodule_NWT.f

# Compile the modified files - ensure we pass the full path to MF_NWT_noswi.f
mpifort -c Irestart_fixed.f
mpifort -c MF_NWT_noswi.f  # Main program entry point

# Compile remaining source files (excluding the original files we modified)
for file in *.f *.f90; do
  if [ "$file" != "Irestart.f" ] && [ "$file" != "MF_NWT.f" ] && [ "$file" != "MF_NWT_noswi.f" ] && [ "$file" != "Irestart_fixed.f" ] && [ "$file" != "gwf2swi27.fpp" ]; then
    mpifort -c "$file"
  fi
done

# Link all object files to create the executable - ensure MF_NWT_noswi.o is the first object
mpifort MF_NWT_noswi.o *.o -o ../bin/MODFLOW-NWT