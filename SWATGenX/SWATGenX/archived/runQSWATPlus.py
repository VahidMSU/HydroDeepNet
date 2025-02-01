import subprocess
import os
import sys

try:
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
except ImportError:
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths

def runQSWATPlus(VPUID, LEVEL, NAME, MODEL_NAME):
    print(f"Running QSWATPlus for {NAME}")

    # Validate QSWATPlus path
    runQSWATPlus_path = SWATGenXPaths.runQSWATPlus_path
    #assert os.path.exists(runQSWATPlus_path), f"File {runQSWATPlus_path} does not exist"

    # Set QGIS-related environment variables
    os.environ["PYTHONPATH"] = "/usr/lib/python3/dist-packages:" + os.environ.get("PYTHONPATH", "")
    os.environ["QGIS_ROOT"] = "/usr/share/qgis"
    os.environ["PYTHONPATH"] += f":{os.environ['QGIS_ROOT']}/python:{os.environ['QGIS_ROOT']}/python/plugins:{os.environ['QGIS_ROOT']}/python/plugins/processing"
    os.environ["PYTHONHOME"] = "/usr"
    os.environ["PATH"] += f":{os.environ['QGIS_ROOT']}/bin"
    os.environ["QGIS_DEBUG"] = "-1"
    os.environ["QT_PLUGIN_PATH"] = os.environ["QGIS_ROOT"] + "/qtplugins"

    # Change directory to the project folder
    os.chdir("/data/SWATGenXApp/codes/SWATGenX/SWATGenX")

    # Construct the Python command to run QSWATPlus
    python_command = (
        f"from QSWATPlus3_64 import runHUCProject; "
        f"runHUCProject(VPUID='{VPUID}', LEVEL='{LEVEL}', NAME='{NAME}', MODEL_NAME='{MODEL_NAME}')"
    )

    # Execute the command using Xvfb for headless operation
    try:
        subprocess.run(["xvfb-run", "-a", "python3", "-c", python_command], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: QSWATPlus execution failed with error code {e.returncode}", file=sys.stderr)

