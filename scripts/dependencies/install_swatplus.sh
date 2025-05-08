#!/bin/bash
# SWAT+ Installation Script
# Set the base directory

BASE_DIR="${SWAT_BASE_DIR:-$(cd "$SCRIPT_DIR/../../" && pwd)}"
INSTALL_DIR="${BASE_DIR}/swatplus_installation"


TARGET_DB_DIR="/usr/local/share/SWATPlus/Databases"
USER_DB_DIR="${HOME}/.local/share/SWATPlus/Databases"
SWATPLUS_EDITOR_DIR="/usr/local/share/SWATPlusEditor/swatplus-editor"


echo "BASE_DIR: ${BASE_DIR}"
echo "INSTALL_DIR: ${INSTALL_DIR}"
echo "TARGET_DB_DIR: ${TARGET_DB_DIR}"
echo "USER_DB_DIR: ${USER_DB_DIR}"
echo "SWATPLUS_EDITOR_DIR: ${SWATPLUS_EDITOR_DIR}"

# Create installation directory and clean it
echo "Creating and cleaning installation directory..."
mkdir -p ${INSTALL_DIR}
rm -rf ${INSTALL_DIR}/*
cd ${BASE_DIR}

# Download SWAT+ installer
echo "Downloading SWAT+ installer..."
wget https://plus.swat.tamu.edu/downloads/2.3/2.3.1/swatplus-linux-installer-2.3.1.tgz --directory-prefix=${INSTALL_DIR}
tar -xvf ${INSTALL_DIR}/swatplus-linux-installer-2.3.1.tgz --directory=${INSTALL_DIR}

# Make sure installer is executable
cd ${INSTALL_DIR}
chmod +x installforall.sh

# Run the installer
echo "Running SWAT+ installer..."
${INSTALL_DIR}/installforall.sh

# Set permissions
echo "Setting permissions..."
chown -R www-data:www-data /usr/local/share/SWATPlus
chown -R www-data:www-data /usr/share/qgis/python/plugins/QSWATPlusLinux3_64
mkdir -p /var/www/.local
chown -R www-data:www-data /var/www/.local

# Download SWAT+ Editor
echo "Downloading SWAT+ Editor..."
wget https://github.com/swat-model/swatplus-editor/archive/refs/tags/v3.0.8.tar.gz --directory-prefix=${INSTALL_DIR}
tar -xvf ${INSTALL_DIR}/v3.0.8.tar.gz --directory=${INSTALL_DIR}/

# Create the target directory for SWATPlusEditor if it doesn't exist
mkdir -p /usr/local/share/SWATPlusEditor

# Move the SWATPlusEditor to the target directory
mv ${INSTALL_DIR}/swatplus-editor-3.0.8 ${SWATPLUS_EDITOR_DIR}

# Download additional required files
echo "Downloading additional required files..."
wget --spider https://plus.swat.tamu.edu/downloads/3.0/3.0.0/swatplus_datasets.sqlite 2>/dev/null
if [ $? -ne 0 ]; then
  echo "Error: Could not access swatplus_datasets.sqlite URL"
  exit 1
fi
wget https://plus.swat.tamu.edu/downloads/3.0/3.0.0/swatplus_datasets.sqlite --directory-prefix=${INSTALL_DIR}

wget --spider https://plus.swat.tamu.edu/downloads/swatplus_wgn.zip 2>/dev/null
if [ $? -ne 0 ]; then
  echo "Error: Could not access swatplus_wgn.zip URL"
  exit 1
fi
wget https://plus.swat.tamu.edu/downloads/swatplus_wgn.zip --directory-prefix=${INSTALL_DIR}

wget --spider https://plus.swat.tamu.edu/downloads/swatplus_soils.zip 2>/dev/null
if [ $? -ne 0 ]; then
  echo "Error: Could not access swatplus_soils.zip URL"
  exit 1
fi
wget https://plus.swat.tamu.edu/downloads/swatplus_soils.zip --directory-prefix=${INSTALL_DIR}

# Extract the downloaded zip files
echo "Extracting database files..."
if [ ! -f ${INSTALL_DIR}/swatplus_wgn.zip ]; then
  echo "Error: swatplus_wgn.zip was not downloaded successfully"
  exit 1
fi
unzip ${INSTALL_DIR}/swatplus_wgn.zip -d ${INSTALL_DIR}/

if [ ! -f ${INSTALL_DIR}/swatplus_soils.zip ]; then
  echo "Error: swatplus_soils.zip was not downloaded successfully"
  exit 1
fi
unzip ${INSTALL_DIR}/swatplus_soils.zip -d ${INSTALL_DIR}/

# Create directories if they don't exist
echo "Setting up database directories..."
mkdir -p ${TARGET_DB_DIR}
mkdir -p ${USER_DB_DIR}

# Copy database files to the target directory
echo "Copying database files to system location..."
cp ${INSTALL_DIR}/swatplus_datasets.sqlite ${TARGET_DB_DIR}/
cp ${INSTALL_DIR}/swatplus_soils.sqlite ${TARGET_DB_DIR}/
cp ${INSTALL_DIR}/swatplus_wgn.sqlite ${TARGET_DB_DIR}/

# Copy for internal testing
echo "Copying database files for user testing..."
cp ${INSTALL_DIR}/swatplus_datasets.sqlite ${USER_DB_DIR}/
cp ${INSTALL_DIR}/swatplus_soils.sqlite ${USER_DB_DIR}/
cp ${INSTALL_DIR}/swatplus_wgn.sqlite ${USER_DB_DIR}/

# Set permissions for QSWATPlus files
echo "Setting permissions for QSWATPlus files..."
chmod -R 755 /usr/share/qgis/python/plugins/QSWATPlusLinux3_64

echo "SWAT+ installation completed successfully!"

### remove swatplus_installation directory
rm -rf ${INSTALL_DIR}