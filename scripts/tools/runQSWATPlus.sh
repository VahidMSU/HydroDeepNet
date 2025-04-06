#!/bin/bash
echo "Running QSWAT+"
# /data/SWATGenXApp/codes/scripts/runQSWATPlus.sh
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


echo "QGIS_ROOT: $QGIS_ROOT"
echo "PYTHONPATH: $PYTHONPATH"
echo "PYTHONHOME: $PYTHONHOME"
echo "PATH: $PATH"
echo "QT_PLUGIN_PATH: $QT_PLUGIN_PATH"
echo "XDG_RUNTIME_DIR: $XDG_RUNTIME_DIR"


# Change to your project directory
cd "/data/SWATGenXApp/codes/SWATGenX/SWATGenX"

# Run the Python script with Xvfb (headless display)

