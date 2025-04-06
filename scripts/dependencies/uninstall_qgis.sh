#!/bin/bash
set -e

echo "Removing any existing QGIS installation..."
sudo apt-get purge -y qgis python3-qgis qgis-provider-grass qgis-plugin-grass
sudo apt-get autoremove -y
sudo apt-get clean

echo "Removing QGIS-related repositories..."
sudo rm -f /etc/apt/sources.list.d/qgis.list
sudo rm -f /etc/apt/sources.list.d/*qgis*jammy.list
sudo rm -f /etc/apt/trusted.gpg.d/qgis.gpg

