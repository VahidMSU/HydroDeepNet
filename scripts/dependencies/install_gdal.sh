#!/bin/bash
set -e

echo "📦 Installing GDAL 3.8.4..."

# Prerequisites
apt-get update
apt-get install -y build-essential cmake g++ \
    libproj-dev libgeos-dev libsqlite3-dev libtiff-dev libcurl4-openssl-dev \
    libpng-dev libjpeg-dev libopenjp2-7-dev libwebp-dev libhdf5-dev \
    libnetcdf-dev libspatialite-dev swig python3-dev python3-numpy \
    python3-pip

# Download and build GDAL
wget https://github.com/OSGeo/gdal/releases/download/v3.8.4/gdal-3.8.4.tar.gz
tar xzf gdal-3.8.4.tar.gz
cd gdal-3.8.4
mkdir build && cd build

cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON ..
cmake --build . --parallel $(nproc)
cmake --build . --target install --parallel $(nproc)

ldconfig
cd ../..
rm -rf gdal-3.8.4 gdal-3.8.4.tar.gz

echo "Installed GDAL version: $(gdal-config --version)"
echo "✅ GDAL installed."
