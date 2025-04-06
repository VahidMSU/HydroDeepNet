#!/bin/bash
set -e

echo "🧹 Uninstalling GDAL..."

# Remove GDAL binaries/libraries
sudo rm -rf /usr/local/bin/gdal* /usr/local/lib/libgdal* /usr/local/include/gdal /usr/local/share/gdal

# Optionally remove Python bindings
sudo rm -rf /usr/local/lib/python*/dist-packages/osgeo*

sudo ldconfig
echo "✅ GDAL uninstalled."
