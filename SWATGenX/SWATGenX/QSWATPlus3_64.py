# trunk-ignore-all(black)
# -*- coding: utf-8 -*-
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
from qgis.core import QgsApplication, QgsProject, QgsRasterLayer # type: ignore 
import atexit
import sys
import os
import glob
from osgeo import gdal, ogr
from QSWATPlus.QSWATPlusMain import QSWATPlus  
from QSWATPlus.delineation import Delineation 
from QSWATPlus.hrus import HRUs 
import traceback

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
        self.plugin = QSWATPlus(iface)
        ## QGIS project
        self.proj = QgsProject.instance()
        projName = os.path.split(self.projDir)[1]
        self.proj.write(self.projDir + '/{0}.qgs'.format(projName))
        self.proj.read(self.projDir + '/{0}.qgs'.format(projName))
        self.plugin.setupProject(self.proj, True, isHUC=False, logFile=logFile)
        print(' %###% debug: projDir {0}'.format(self.projDir), ' ###')
        print(' %###% debug: projName {0}'.format(projName), ' ###')

        ## main dialogue
        self.dlg = self.plugin._odlg
        ## delineation object
        self.delin = None
        ## hrus object
        self.hrus = None
        # Prevent annoying "error 4 .shp not recognised" messages.
        # These should become exceptions but instead just disappear.
        # Safer in any case to raise exceptions if something goes wrong.
        gdal.UseExceptions()
        ogr.UseExceptions()

    def runProject(self, dataDir, scale, minHRUha):
        """Run QSWAT project."""
        gv = self.plugin._gv
        #print('Dem is processed is {0}'.format(self.plugin._demIsProcessed))
        self.delin = Delineation(gv, self.plugin._demIsProcessed)
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
        self.delin._dlg.selectExistOutlets.setText('')



        self.delin._dlg.recalcButton.setChecked(False)  # want to use length field in channels shapefile
        self.delin._dlg.snapThreshold.setText('300')
        # use MPI on HUC10 and HUC8 projects
        numProc = 0 if scale >= 12 else 8
        self.delin._dlg.numProcesses.setValue(numProc)
        gv.HUCDataDir = dataDir
        gv.useGridModel = False
        gv.existingWshed = True
        self.delin.runExisting()
        print(' %###% debug: gv.HUCDataDir {0}'.format(gv.HUCDataDir), ' ###')
        print(' %###% debug: gv.useGridModel {0}'.format(gv.useGridModel), ' ###')
        print(' %###% debug: gv.existingWshed {0}'.format(gv.existingWshed), ' ###')
        print(' ProjectDir is {0}'.format(self.projDir))

        lakesFile = f'{self.projDir}/Watershed/Shapes/SWAT_plus_lakes.shp'
        if os.path.isfile(lakesFile):
            self.delin._dlg.selectLakes.setText(lakesFile)
            self.delin.addLakesMap()
        self.delin.finishDelineation()
        self.delin._dlg.close()
        self.hrus = HRUs(gv, self.dlg.reportsBox)
        self.hrus.init()
        hrudlg = self.hrus._dlg
        self.hrus.landuseFile = os.path.join(
            self.projDir, 'Watershed', 'Rasters', 'Landuse', 'landuse.tif'
        )
        self.hrus.landuseLayer = QgsRasterLayer(self.hrus.landuseFile, 'landuse')
        self.hrus.soilFile = os.path.join(self.projDir, 'Watershed', 'Rasters', 'Soil', 'soil.tif')
        self.hrus.soilLayer = QgsRasterLayer(self.hrus.soilFile, 'soil')
        #landCombo = hrudlg.selectLanduseTable
        #landIndex = landCombo.findText('nlcd2001_landuses')
        #landCombo.setCurrentIndex(landIndex)
        self.hrus.landuseTable = 'landuse_lookup'
        print(' %###% debug: landuseFile {0}'.format(self.hrus.landuseFile), ' ###')
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
        print(f"self.hrus.readFiles(): {self.hrus.readFiles()}")
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


def runProject(base_directory, dataDir, scale, minHRUha):
    """Run a QSWAT+ project on directory base_directory"""
    # seems clumsy to keep opening logFile, rather than opening once and passing handle
    # but there may be many instances of this function and we want to avoid too many open files
    if os.path.isdir(base_directory):
        logFile = f'{base_directory}/LogFile.txt'
        with open(logFile, 'w') as f:
            f.write('Running project {0}\n'.format(base_directory))
        sys.stdout.write('Running project {0}\n'.format(base_directory))
        sys.stdout.flush()
        try:
            huc = runHUC(base_directory, logFile)
            if huc.runProject(dataDir, scale, minHRUha):
                with open(logFile, 'a') as f:
                    f.write('Completed project {0}\n'.format(base_directory))
            else:
                with open(logFile, 'a') as f:
                    f.write('ERROR: incomplete project {0}\n'.format(base_directory))
        except Exception:
            with open(logFile, 'a') as f:
                f.write('ERROR: exception: {0}\n'.format(traceback.format_exc()))
            sys.stdout.write('ERROR: exception: {0}\n'.format(traceback.format_exc()))
            sys.stdout.flush()

def main(VPUID,LEVEL,NAME, MODEL_NAME):
    app = QgsApplication([], True)
    QgsApplication.initQgis()
    atexit.register(QgsApplication.exitQgis)
    direc = f"{SWATGenXPaths.swatgenx_outlet_path}/{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}/{MODEL_NAME}.qgs"
    ## delete the database file if it exists
    if os.path.exists(f"{SWATGenXPaths.swatgenx_outlet_path}/{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}/"):
        ### remove all files and no directories
        files = glob.glob(f"{SWATGenXPaths.swatgenx_outlet_path}/{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}/*")
        for f in files:
            ## do not remove directories
            if os.path.isfile(f):
                os.remove(f)
    else:
        os.makedirs(f"{SWATGenXPaths.swatgenx_outlet_path}/{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}/")
    dataDir = "H:/Data"
    scale = 8
    minHRUha = 0.00
    inletId = 0

    base_directory = os.path.dirname(direc)
    print('Running project {0}'.format(base_directory))
    try:
        print(' ### debug: base_directory {0}'.format(base_directory), ' ###')
        print(' ### debug: dataDir {0}'.format(dataDir), ' ###')
        print(' ### debug: scale {0}'.format(scale), ' ###')
        print(' ### debug: minHRUha {0}'.format(minHRUha), ' ###')
        print(' ### debug: inletId {0}'.format(inletId), ' ###')

        huc = runHUC(base_directory, None)
        huc.runProject(dataDir, scale, minHRUha)
        print('Completed project {0}'.format(base_directory))
    except Exception:
        print('ERROR: exception: {0}'.format(traceback.format_exc()))

    app.exitQgis()
    app.exit()
    del app


def runHUCProject(VPUID,LEVEL,NAME, MODEL_NAME):
    main(VPUID, LEVEL, NAME, MODEL_NAME)
