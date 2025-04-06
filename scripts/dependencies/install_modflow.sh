#!/bin/bash#!/bin/bash
## run this only if /data/SWATGenXApp/codes/bin/modflow-nwt does not exist
set -e  # Exit immediately if a command exits with non-zero status
#run this only if /data/SWATGenXApp/codes/bin/modflow-nwt does not exist
modflow_nwt=/data/SWATGenXApp/codes/bin/modflow-nwt
if [ -f "$modflow_nwt" ]; then
    echo "MODFLOW-NWT is already installed at $modflow_nwt"
    exit 0
fi

cd /data/SWATGenXApp/codes/

wget https://water.usgs.gov/water-resources/software/MODFLOW-NWT/MODFLOW-NWT_1.3.0.zip

unzip MODFLOW-NWT_1.3.0.zip
rm MODFLOW-NWT_1.3.0.zip

cd /data/SWATGenXApp/codes/MODFLOW-NWT/build
rm -rf *
source /opt/intel/oneapi/setvars.sh

# Copy source files
cp /data/SWATGenXApp/codes/MODFLOW-NWT/src/* /data/SWATGenXApp/codes/MODFLOW-NWT/build

echo "Compiling MODFLOW-NWT..."

# First compile global module file - this is needed by many files
echo "Step 1: Compiling global module..."
for module in global.f global.f90; do
    if [ -f "$module" ]; then
        echo "  Compiling: $module"
        ifx -c "$module"
    fi
done

# Then compile other foundational modules
echo "Step 2: Compiling other foundational modules..."
for module in mach_mod.f90 modules.f90 openspec.F90 parammodule.f; do
    if [ -f "$module" ]; then
        echo "  Compiling: $module"
        ifx -c "$module"
    fi
done

# Then compile base components that other modules may need
echo "Step 3: Compiling base components..."
for base in utl7.f; do
    if [ -f "$base" ]; then
        echo "  Compiling: $base"
        ifx -c "$base"
    fi
done

# Then compile domain-specific modules
echo "Step 4: Compiling domain-specific modules..."
for module in NWT1_module.f NWT1_ilupc_mod.f90 gwfsfrmodule_NWT.f gwfuzfmodule_NWT.f gwflakmodule_NWT.f; do
    if [ -f "$module" ]; then
        echo "  Compiling: $module"
        ifx -c "$module" || echo "Warning: Failed to compile $module, continuing..."
    fi
done

# Then compile solvers
echo "Step 5: Compiling solvers..."
for solver in NWT1_xmdlib.f NWT1_xmd.f NWT1_gmres.f90 NWT1_solver.f sip7_NWT.f pcg7_NWT.f; do
    if [ -f "$solver" ]; then
        echo "  Compiling: $solver"
        ifx -c "$solver" || echo "Warning: Failed to compile $solver, continuing..."
    fi
done

# Then compile remaining base components
echo "Step 6: Compiling remaining base components..."
for base in parutl7.f gwf2bas7_NWT.f; do
    if [ -f "$base" ]; then
        echo "  Compiling: $base"
        ifx -c "$base" || echo "Warning: Failed to compile $base, continuing..."
    fi
done

# Finally compile all remaining files
echo "Step 7: Compiling remaining files..."
for file in *.f*; do
    if [ ! -f "${file%.f*}.o" ] && [ ! -f "${file%.f90}.o" ] && [ ! -f "${file%.fpp}.o" ] && [ ! -f "${file%.F90}.o" ]; then
        echo "  Compiling: $file"
        ifx -c "$file" || echo "Warning: Failed to compile $file, continuing..."
    fi
done

# Try again for any files that may have failed due to dependency issues
echo "Step 8: Retry compiling any failed files..."
for file in *.f*; do
    if [ ! -f "${file%.f*}.o" ] && [ ! -f "${file%.f90}.o" ] && [ ! -f "${file%.fpp}.o" ] && [ ! -f "${file%.F90}.o" ]; then
        echo "  Retrying: $file"
        ifx -c "$file" || echo "Warning: Failed to compile $file on retry, continuing..."
    fi
done

# Link all object files
echo "Linking object files..."
ifx -o modflow-nwt *.o || {
    echo "Error during linking. Trying an alternative approach..."
    # Try linking with specific object files first, then the rest
    ifx -o modflow-nwt mach_mod.o modules.o openspec.o NWT1_module.o *.o
}

# Clean up object files if linking was successful
if [ $? -eq 0 ]; then
    echo "Cleaning up..."
    rm -rf *.o *.mod
    echo "Compilation completed successfully!"
    echo "executable copied to /data/SWATGenXApp/codes/bin"
    cp -r modflow-nwt /data/SWATGenXApp/codes/bin
    ## remove the unzipped folder
    rm -rf /data/SWATGenXApp/codes/MODFLOW-NWT
    exit 0
else
    echo "Compilation failed at linking stage. Object files preserved for debugging."
    exit 1
fi



## End of script