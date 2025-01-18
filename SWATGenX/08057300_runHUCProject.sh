#!/bin/bash
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

# Change to your project directory
cd "/data/SWATGenXApp/codes/SWATGenX/SWATGenX"


# Run the Python script with Xvfb (headless display)
#xvfb-run -a python3 -c "from QSWATPlus3_64 import runHUCProject; runHUCProject(VPUID='0405', LEVEL='huc12', NAME='04101370', MODEL_NAME='SWAT_MODEL')"
xvfb-run -a python3 -c "from QSWATPlus3_64 import runHUCProject; runHUCProject(VPUID = '1203', LEVEL = 'huc12', NAME = '08057300', MODEL_NAME = 'SWAT_MODEL_Web_Application')"
