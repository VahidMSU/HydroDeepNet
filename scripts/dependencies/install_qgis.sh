echo "Adding official QGIS repository and key..."
sudo mkdir -p /etc/apt/keyrings
wget -qO - https://qgis.org/downloads/qgis-2023.gpg.key | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/qgis.gpg > /dev/null

echo "deb [signed-by=/etc/apt/trusted.gpg.d/qgis.gpg] https://qgis.org/ubuntu jammy main" | sudo tee /etc/apt/sources.list.d/qgis.list

echo "Updating package list..."
sudo apt-get update

echo "Installing QGIS and Python bindings..."
sudo apt-get install -y qgis python3-qgis

echo "QGIS installation complete."
