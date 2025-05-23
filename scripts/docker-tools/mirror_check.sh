#!/bin/bash

echo "===== Checking Python Version & Virtual Environment ====="
cd /data/SWATGenXApp/codes
echo "Host Python Version:"
python --version
echo "Host Virtual Environment:"
echo $VIRTUAL_ENV

container_id=$(docker ps -q)
echo "Container ID: $container_id"

echo "Container Python Version:"
docker exec -it $container_id python --version

echo "Container Virtual Environment:"
docker exec -it $container_id echo $VIRTUAL_ENV

echo "===== Checking GDAL and PROJ Versions ====="
echo "Host GDAL Version:"
gdalinfo --version
echo "Container GDAL Version:"
docker exec -it $container_id gdalinfo --version

echo "Host PROJ Version:"
proj
echo "Container PROJ Version:"
docker exec -it $container_id proj

echo "Host PROJ Library Path:"
echo $PROJ_LIB
ls -lah $PROJ_LIB
echo "Container PROJ Library Path:"
docker exec -it $container_id echo $PROJ_LIB
docker exec -it $container_id ls -lah $PROJ_LIB

#echo "===== Checking Shared Library Links ====="
#echo "Host GDAL linked libraries:"
#ldd $(which gdalinfo)
#echo "Container GDAL linked libraries:"
#docker exec -it $container_id ldd $(which gdalinfo)

echo "===== Checking Environment Variables ====="
echo "Host SWAT Environment Variables:"
printenv | grep SWAT
echo "Container SWAT Environment Variables:"
docker exec -it $container_id printenv | grep SWAT

echo "===== Checking Directory Structure ====="
echo "Host Directory Structure:"
ls -lah /data/SWATGenXApp/codes
echo "Container Directory Structure:"
docker exec -it $container_id ls -lah /data/SWATGenXApp/codes

echo "===== Checking Python GDAL Binding ====="
echo "Host GDAL Python Binding Version:"
python -c "from osgeo import gdal; print(gdal.__version__)"
echo "Container GDAL Python Binding Version:"
docker exec -it $container_id python -c "from osgeo import gdal; print(gdal.__version__)"

echo "===== Checking QGIS Version ====="
echo "Host QGIS Version:"
qgis --version 2>/dev/null || echo "QGIS not found on host"
echo "Container QGIS Version:"
docker exec -it $container_id qgis --version 2>/dev/null || echo "QGIS not found in container"

echo "===== Checking QSWATPlusLinux3_64 Plugin ====="
echo "Host QSWATPlus Plugin Structure:"
ls -la /usr/share/qgis/python/plugins/QSWATPlusLinux3_64/ 2>/dev/null || echo "QSWATPlus plugin not found on host"
echo "Container QSWATPlus Plugin Structure:"
docker exec -it $container_id ls -la /usr/share/qgis/python/plugins/QSWATPlusLinux3_64/ 2>/dev/null || echo "QSWATPlus plugin not found in container"

echo "===== Checking QSWATPlus Database Files ====="
echo "Host SWATPlus/Databases Files:"
ls -la /usr/local/share/SWATPlus/Databases/ 2>/dev/null || echo "SWATPlus database directory not found on host"
echo "Container SWATPlus/Databases Files:"
docker exec -it $container_id ls -la /usr/local/share/SWATPlus/Databases/ 2>/dev/null || echo "SWATPlus database directory not found in container"

echo "===== Checking SWATPlusEditor Installation ====="
echo "Host SWATPlusEditor Structure:"
ls -la /usr/local/share/SWATPlusEditor/ 2>/dev/null || echo "SWATPlusEditor not found on host"
echo "Container SWATPlusEditor Structure:"
docker exec -it $container_id ls -la /usr/local/share/SWATPlusEditor/ 2>/dev/null || echo "SWATPlusEditor not found in container"

echo "===== Checking User Database Files ====="
echo "Host User's SWATPlus Databases:"
ls -la ${HOME}/.local/share/SWATPlus/Databases/ 2>/dev/null || echo "User SWATPlus database directory not found on host"
echo "Container User's SWATPlus Databases:"
docker exec -it $container_id bash -c "ls -la \${HOME}/.local/share/SWATPlus/Databases/" 2>/dev/null || echo "User SWATPlus database directory not found in container"

echo "===== Checking Database File Sizes ====="
echo "Host Database File Sizes:"
du -h /usr/local/share/SWATPlus/Databases/* 2>/dev/null || echo "SWATPlus database files not found on host"
echo "Container Database File Sizes:"
docker exec -it $container_id du -h /usr/local/share/SWATPlus/Databases/* 2>/dev/null || echo "SWATPlus database files not found in container"

### inside the container, check for docker exec -it keen_panini /bin/bash
#ls -lah /data/SWATGenXApp/Users/vahidr32/SWATplus_by_VPUID/0712/huc12/05536265/SWAT_MODEL/
#ls -lah /usr/share/qgis/python/plugins/QSWATPlusLinux3_64/QSWATPlus/