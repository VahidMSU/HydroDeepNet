#!/bin/bash
set -e

echo "Removing any existing QGIS installation..."
apt-get purge -y qgis python3-qgis qgis-provider-grass qgis-plugin-grass
apt-get autoremove -y
apt-get clean

echo "Removing QGIS-related repositories..."
rm -f /etc/apt/sources.list.d/qgis.list
rm -f /etc/apt/sources.list.d/*qgis*jammy.list
rm -f /etc/apt/trusted.gpg.d/qgis.gpg

