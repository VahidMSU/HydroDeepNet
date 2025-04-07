#!/bin/bash
set -e

echo "🧹 Uninstalling GDAL..."

# Remove GDAL binaries/libraries
rm -rf /usr/local/bin/gdal* /usr/local/lib/libgdal* /usr/local/include/gdal /usr/local/share/gdal

# Optionally remove Python bindings
rm -rf /usr/local/lib/python*/dist-packages/osgeo*

ldconfig
echo "✅ GDAL uninstalled."
