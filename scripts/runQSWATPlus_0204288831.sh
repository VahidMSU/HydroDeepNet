#!/bin/bash
export PYTHONPATH=""
export QGIS_ROOT=""
export PYTHONHOME=""
export QT_PLUGIN_PATH=""
export GDAL_DATA=""
export PROJ_LIB=""
export PATH="/data/SWATGenXApp/codes/.venv/bin"
export SWATPLUS_DIR="/usr/local/share/SWATPlus"
export TAUDEM5BIN="$SWATPLUS_DIR/TauDEM5Bin"
export PATH="$TAUDEM5BIN:$PATH"

echo "Running QSWAT+"

# Add the correct QGIS Python path
export PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH

# Set QGIS installation root
export QGIS_ROOT=/usr/share/qgis

# Add additional Python paths for QGIS plugins (but keep the core bindings path intact)
export PYTHONPATH=$PYTHONPATH:$QGIS_ROOT/python:$QGIS_ROOT/python/plugins:$QGIS_ROOT/python/plugins/processing

# Set Python home (adjust to the correct Python path for QGIS if necessary)
export PYTHONHOME=/usr

# Add QGIS binaries to PATH
export PATH=$PATH:$QGIS_ROOT/bin

# Set QGIS specific environment variables
export QGIS_DEBUG=-1



# If necessary, adjust the QT plugin path
export QT_PLUGIN_PATH=$QGIS_ROOT/qtplugins
export XDG_RUNTIME_DIR="/run/user/$(id -u)"

# Change to your project directory
cd "/data/SWATGenXApp/codes/SWATGenX/SWATGenX"

# Run the Python script with Xvfb (headless display)
echo "Fixing ownership and permissions for /data/SWATGenXApp/Users/vahidr32/SWATplus_by_VPUID/0208/huc12/0204288831/SWAT_MODEL_Web_Application"
chown -R $(id -u):$(id -g) "/data/SWATGenXApp/Users/vahidr32/SWATplus_by_VPUID/0208/huc12/0204288831/SWAT_MODEL_Web_Application"
chmod -R 777 "/data/SWATGenXApp/Users/vahidr32/SWATplus_by_VPUID/0208/huc12/0204288831/SWAT_MODEL_Web_Application"
xvfb-run -a python3 -c "from QSWATPlus3_64 import runHUCProject; runHUCProject(VPUID='0208', LEVEL='huc12', NAME='0204288831', MODEL_NAME='SWAT_MODEL_Web_Application', SWATGenXPaths_swatgenx_outlet_path='/data/SWATGenXApp/Users/vahidr32/SWATplus_by_VPUID/')"
