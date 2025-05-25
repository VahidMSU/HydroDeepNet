# trunk-ignore-all(black)
# -*- coding: utf-8 -*-
#/data/SWATGenXApp/codes/SWATGenX/SWATGenX/QSWATPlus3_64.py
"""
/***************************************************************************
 QSWAT
                                 A QGIS plugin
 Run HUC project
                              -------------------
        begin                : 2014-07-18
        copyright            : (C) 2014 by Chris George
        email                : cgeorge@mcmaster.ca
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
##### VAHID NOTE; We have not changed anything in the source code.
##### we just modified runHUC.py to create a predefine project.
import sys
try:
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
except ImportError:
    from SWATGenXConfigPars import SWATGenXPaths
sys.path.append(SWATGenXPaths.QSWATPlus_env_path)
from qgis.core import QgsApplication, QgsProject, QgsRasterLayer, QgsVectorLayer # type: ignore
import atexit
import sys
import os
import glob
from osgeo import gdal, ogr
from QSWATPlus.QSWATPlusMain import QSWATPlus
from QSWATPlus.delineation import Delineation
from QSWATPlus.hrus import HRUs
import traceback
from SWATGenXLogging import LoggerSetup
from QSWATPlus.QSWATUtils import QSWATUtils, FileTypes

class DummyInterface(object):
    """Dummy iface."""
    def __getattr__(self, *args, **kwargs):  # @UnusedVariable
        """Dummy function."""
        def dummy(*args, **kwargs):  # @UnusedVariable
            return self
        return dummy
    def __iter__(self):
        """Dummy function."""
        return self
    def next(self):
        """Dummy function."""
        raise StopIteration
    def layers(self):
        """Simulate iface.legendInterface().layers()."""
        return QgsProject.instance().mapLayers().values()

iface = DummyInterface()
class runHUC():

    """Run HUC14/12/10/8 project."""

    def __init__(self, projDir, logFile):
        """Initialize"""
        ## project directory
        self.projDir = projDir
        ## QSWAT plugin
        self.QSWATPlusPlugin = QSWATPlus(iface)
        ## QGIS project
        self.proj = QgsProject.instance()
        projName = os.path.split(self.projDir)[1]
        self.proj.write(self.projDir + '/{0}.qgs'.format(projName))
        assert os.path.isfile(self.projDir + '/{0}.qgs'.format(projName))
        self.proj.read(self.projDir + '/{0}.qgs'.format(projName))
        assert self.proj.fileName() == self.projDir + '/{0}.qgs'.format(projName)
        self.QSWATPlusPlugin.setupProject(self.proj, True, isHUC=False, logFile=logFile)
        self.logger = LoggerSetup(verbose=True, rewrite=False, report_path=self.projDir)
        self.logger = self.logger.setup_logger("runHUCProject")
        ## main dialogue
        self.dlg = self.QSWATPlusPlugin._odlg
        ## delineation object
        self.delin = None
        ## hrus object
        self.hrus = None
        # Prevent annoying "error 4 .shp not recognised" messages.
        # These should become exceptions but instead just disappear.
        # Safer in any case to raise exceptions if something goes wrong.
        gdal.UseExceptions()
        ogr.UseExceptions()
        self.logger.info("Set GDAL exceptions")


    def runProject(self, minHRUha):
        """Run QSWAT project."""
        gv = self.QSWATPlusPlugin._gv
        self.logger.info(f'gv {gv}')
        self.logger.info(f'gv.useGridModel {gv.useGridModel}')
        self.logger.info(f'gv.existingWshed {gv.existingWshed}')
        self.logger.info(f'ProjectDir {self.projDir}')

        assert os.path.isdir(self.projDir)
        assert os.path.isfile(self.projDir + '/{0}.qgs'.format(os.path.split(self.projDir)[1])  )

        self.delin = Delineation(gv, self.QSWATPlusPlugin._demIsProcessed)
        self.delin._dlg.tabWidget.setCurrentIndex(1)
        self.delin._dlg.selectDem.setText(
            f'{self.projDir}/Watershed/Rasters/DEM/dem.tif'
        )
        self.delin._dlg.drainStreamsButton.setChecked(True)
        self.delin._dlg.selectSubbasins.setText(
            f'{self.projDir}/Watershed/Shapes/SWAT_plus_subbasins.shp'
        )
        self.delin._dlg.selectWshed.setText(
            f'{self.projDir}/Watershed/Shapes/SWAT_plus_watersheds.shp'
        )
        self.delin._dlg.selectStreams.setText(
            f'{self.projDir}/Watershed/Shapes/SWAT_plus_streams.shp'
        )
        assert os.path.isfile(self.delin._dlg.selectDem.text())
        assert os.path.isfile(self.delin._dlg.selectSubbasins.text())
        assert os.path.isfile(self.delin._dlg.selectWshed.text())
        assert os.path.isfile(self.delin._dlg.selectStreams.text())

        self.delin._dlg.selectExistOutlets.setText('')

        self.delin._dlg.recalcButton.setChecked(False)  # want to use length field in channels shapefile
        self.delin._dlg.snapThreshold.setText('300')
        # use MPI on HUC10 and HUC8 projects
        numProc = 1
        self.delin._dlg.numProcesses.setValue(numProc)
        gv.useGridModel = False
        gv.existingWshed = True
        self.delin.runExisting()
        self.logger.info(f'gv.useGridModel {gv.useGridModel}')
        self.logger.info(f'gv.existingWshed {gv.existingWshed}')
        self.logger.info(f'ProjectDir {self.projDir}')
        lakesFile = f'{self.projDir}/Watershed/Shapes/SWAT_plus_lakes.shp'

        # Handle lakes properly - first check if file exists
        if os.path.isfile(lakesFile):
            self.logger.info(f'Lakes file found: {lakesFile}')

            # Set the file path in the UI
            self.delin._dlg.selectLakes.setText(lakesFile)
            self.delin._dlg.selectLakes.setEnabled(True)

            # Set the file path in global variables
            gv.lakeFile = lakesFile

            # Explicitly load the lakes shapefile before processing
            root = QgsProject.instance().layerTreeRoot()

            # Load lakes layer to the project if not already loaded
            lakesLayer = QSWATUtils.getLayerByFilename(root.findLayers(), lakesFile, FileTypes._LAKES, None, None, None)[0]
            if not lakesLayer:
                self.logger.info(f'Loading lakes layer from {lakesFile}')
                # Get DEM layer to use as sublayer for loading lakes
                demLayer = QSWATUtils.getLayerByFilename(root.findLayers(), gv.demFile, FileTypes._DEM, None, None, None)[0]

                # Load lakes layer explicitly
                lakesLayerName = os.path.splitext(os.path.basename(lakesFile))[0]
                lakesLayer = QgsVectorLayer(lakesFile, lakesLayerName, 'ogr')
                if lakesLayer.isValid():
                    QgsProject.instance().addMapLayer(lakesLayer)
                    self.logger.info(f'Successfully loaded lakes layer {lakesFile}')
                else:
                    self.logger.info(f'Failed to load lakes layer from {lakesFile}')

            # Mark lakes as not processed yet
            self.delin.lakesDone = False

            # Explicitly add lakes before finalizing delineation
            self.delin.addLakes()
            self.logger.info('Lakes added to the project')
        else:
            self.logger.info(f'No lakes file found at {lakesFile}')
            self.delin._dlg.selectLakes.setText('')
            self.delin._dlg.selectLakes.setEnabled(False)

        self.delin.finishDelineation()
        self.delin._dlg.close()
        self.hrus = HRUs(gv, self.dlg.reportsBox)
        self.hrus.init()
        hrudlg = self.hrus._dlg

        assert os.path.isdir(self.projDir), f'No project directory {self.projDir}'

        print(f"##### projDir: {self.projDir}")
        self.hrus.landuseFile = os.path.join(
            self.projDir, 'Watershed', 'Rasters', 'Landuse', 'landuse.tif'
        )
        self.hrus.landuseLayer = QgsRasterLayer(self.hrus.landuseFile, 'landuse')
        self.hrus.soilFile = os.path.join(self.projDir, 'Watershed', 'Rasters', 'Soil', 'soil.tif')
        self.hrus.soilLayer = QgsRasterLayer(self.hrus.soilFile, 'soil')

        self.hrus.landuseTable = 'landuse_lookup'
        self.logger.info(f'landuseFile {self.hrus.landuseFile}')
        hrudlg.SSURGOButton.setChecked(True)
        hrudlg.usersoilButton.setChecked(True)
        self.LanduseTable = SWATGenXPaths.LanduseTable
        gv.db.importCsv('landuse_lookup', "landuse", self.LanduseTable)
        gv.db.useSSURGO = True
        gv.db.slopeLimits = [3,9]
        gv.elevBandsThreshold = 500
        gv.numElevBands = 5
        hrudlg.generateFullHRUs.setChecked(True)
        self.hrus.initLanduses(self.hrus.landuseTable)
        assert os.path.isfile(self.hrus.landuseFile), f'No landuse file {self.hrus.landuseFile}'
        assert os.path.isfile(self.hrus.soilFile), f'No soil file {self.hrus.soilFile}'
        if not self.hrus.readFiles():
            hrudlg.close()
            return
        hrudlg.filterAreaButton.setChecked(True)
        hrudlg.areaButton.setChecked(True)
        hrudlg.areaVal.setText(str(minHRUha))
        self.hrus.calcHRUs()
        result = self.hrus.HRUsAreCreated()
        hrudlg.close()
        return result


def main(VPUID, LEVEL, NAME, MODEL_NAME, SWATGenXPaths_swatgenx_outlet_path):


    try:
        print(f"Running project {VPUID}")
        app = QgsApplication([], True)
        QgsApplication.initQgis()
        atexit.register(QgsApplication.exitQgis)

        base_path = f"{SWATGenXPaths_swatgenx_outlet_path}/{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}"
        model_dir = f"{base_path}/{MODEL_NAME}.qgs"

        assert os.path.exists(base_path), f"QSWATPlus Base Directory {base_path} does not exist"

        if os.path.exists(base_path):
            # Remove only files, keep directories
            for f in glob.glob(f"{base_path}/*"):
                if os.path.isfile(f):
                    os.remove(f)

        assert not os.path.exists(model_dir), f"Model directory {model_dir} already exists"

        base_directory = os.path.dirname(model_dir)
        minHRUha = 0.0
        logFile = f'{base_directory}/LogFile.txt'
        print(f"Creating project in {base_directory}")
        import time
        time.sleep(5)
        huc = runHUC(base_directory, logFile)
        huc.runProject(minHRUha)

    except Exception as e:
        print(f"Error running project: {e}")
        traceback.print_exc()

    finally:
        app.exitQgis()
        app.exit()
        del app

def runHUCProject(VPUID,LEVEL,NAME, MODEL_NAME, SWATGenXPaths_swatgenx_outlet_path):
    main(VPUID, LEVEL, NAME, MODEL_NAME, SWATGenXPaths_swatgenx_outlet_path)
