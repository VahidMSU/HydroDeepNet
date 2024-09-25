#!/bin/bash
echo "Running QSWAT+"

# Set the QGIS root directory
export OSGEO4W_ROOT=/usr  # Replace this with your QGIS installation path on Linux

# Set Python environment
export PYTHONHOME=$OSGEO4W_ROOT/share/qgis/python
export PYTHONPATH=$OSGEO4W_ROOT/share/qgis/python

# QGIS binaries and Python paths
export PATH=$PATH:$OSGEO4W_ROOT/bin:$OSGEO4W_ROOT/share/qgis/python:$OSGEO4W_ROOT/share/qgis/python/plugins:$OSGEO4W_ROOT/share/qgis/python/plugins/processing

# Disable QGIS console messages
export QGIS_DEBUG=-1

# Set QGIS plugin paths
export PYTHONPATH=$PYTHONPATH:$HOME/.local/share/QGIS/QGIS3/profiles/default/python/plugins

# Set QGIS prefix and Qt plugin paths
export QGIS_PREFIX_PATH=$OSGEO4W_ROOT/share/qgis
export QT_PLUGIN_PATH=$OSGEO4W_ROOT/plugins

# Change directory to your project location
cd "/home/rafieiva/MyDataBase/codes/NHDPlus_SWAT/"

# Run the Python script
$OSGEO4W_ROOT/bin/python3 -c "from QSWATPlus3_9 import runHUCProject; runHUCProject(VPUID = '0405', LEVEL = 'huc12', NAME = '04099750', MODEL_NAME = 'SWAT_MODEL')"
