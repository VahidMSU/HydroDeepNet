#!/bin/bash
echo "Running QSWAT+"
SET OSGEO4W_ROOT=C:\Program Files\QGIS 3.36.0
call "%OSGEO4W_ROOT%\bin\o4w_env.bat"
set PYTHONHOME=%OSGEO4W_ROOT%\apps\Python39
set PYTHONPATH=%OSGEO4W_ROOT%\apps\qgis\python
rem QGIS binaries
rem Important to put OSGEO4W_ROOT\bin last, not first, or PyQt.QtCore DLL load fails
set PATH=%PATH%;%OSGEO4W_ROOT%\apps\qgis\bin;%OSGEO4W_ROOT%\apps\qgis\python;%OSGEO4W_ROOT%\apps\Python39;%OSGEO4W_ROOT%\apps\Python39\Scripts;%OSGEO4W_ROOT%\apps\qt5\bin;%OSGEO4W_ROOT%\bin
rem disable QGIS console messages
set QGIS_DEBUG=-1
rem default QGIS plugins
set PYTHONPATH=%PYTHONPATH%;%OSGEO4W_ROOT%\apps\qgis\python\plugins;%OSGEO4W_ROOT%\apps\qgis\python\plugins\processing
rem user installed plugins
set PYTHONPATH=%PYTHONPATH%;%USERPROFILE%\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins
set QGIS_PREFIX_PATH=%OSGEO4W_ROOT%\apps\qgis
set QT_PLUGIN_PATH=%OSGEO4W_ROOT%\apps\qgis\qtplugins;%OSGEO4W_ROOT%\apps\qt5\plugins
cd "/data/MyDataBase/SWATGenXAppData/codes/NHDPlus_SWAT/NHDPlus_SWAT/"
"%OSGEO4W_ROOT%\bin\python3.exe" -c "from QSWATPlus3_9 import runHUCProject; runHUCProject(VPUID = '0405', LEVEL = 'huc12', NAME = '04099750', MODEL_NAME = 'SWAT_MODEL')"
