#/data/SWATGenXApp/codes/SWATGenX/SWATGenX/runQSWATPlus.py

import os
import shutil

def get_python_executable():
    venv_python = "/data/SWATGenXApp/codes/.venv/bin/python"
    return venv_python

def get_real_uid_gid():
    uid = int(os.environ.get("SUDO_UID", os.getuid()))
    gid = int(os.environ.get("SUDO_GID", os.getgid()))
    return uid, gid

def runQSWATPlus(VPUID, LEVEL, NAME, MODEL_NAME, SWATGenXPaths):
    username = SWATGenXPaths.username
    print(f"runQSWATPlus: Running QSWATPlus for {NAME} for usename {username}")
    import time
    time.sleep(2)
    runQSWATPlus_path = SWATGenXPaths.runQSWATPlus_path
    if not os.path.exists(runQSWATPlus_path):
        raise FileNotFoundError(f"File {runQSWATPlus_path} does not exist")

    scripts_dir = "/data/SWATGenXApp/codes/scripts"
    new_runQSWATPlus_path = os.path.join(SWATGenXPaths.report_path, f"runQSWATPlus_{NAME}.sh")

    real_uid, real_gid = get_real_uid_gid()
    print(f"Real UID: {real_uid}, Real GID: {real_gid}")

    shutil.copyfile(runQSWATPlus_path, new_runQSWATPlus_path)

    with open(new_runQSWATPlus_path, "r") as f:
        lines = f.readlines()

    env_vars = ["PYTHONPATH", "QGIS_ROOT", "PYTHONHOME", "QT_PLUGIN_PATH", "GDAL_DATA", "PROJ_LIB", "PATH", "DISPLAY"]

    header_lines = []
    for var in env_vars:
        val = os.environ.get(var, "")
        header_lines.append(f'export {var}="{val}"')

    header_lines.append('export SWATPLUS_DIR="/usr/local/share/SWATPlus"')
    header_lines.append('export TAUDEM5BIN="$SWATPLUS_DIR/TauDEM5Bin"')
    # Add this line to ensure system binaries are available
    header_lines.append('export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$TAUDEM5BIN:$PATH"')
    #header_lines.append('export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$TAUDEM5BIN:$PATH"')

    env_header = "\n".join(header_lines) + "\n\n"

    if lines and lines[0].startswith("#!"):
        lines.insert(1, env_header)
    else:
        lines.insert(0, env_header)

    print(f"Added environment variables to script")

    python_executable = get_python_executable()
    for i, line in enumerate(lines):
        if "Run the Python script with Xvfb (headless display)" in line:
            replacement = (
                f'/usr/bin/xvfb-run -a {python_executable} -c \''
                f'from QSWATPlus3_64 import runHUCProject; '
                f'runHUCProject(VPUID="{VPUID}", LEVEL="{LEVEL}", NAME="{NAME}", MODEL_NAME="{MODEL_NAME}", '
                f'SWATGenXPaths_swatgenx_outlet_path="{SWATGenXPaths.swatgenx_outlet_path}")\'\n'
            )
            lines[i + 1] = replacement
            print(f"Replaced line for running Python script with Xvfb using {python_executable}")
            break

    with open(new_runQSWATPlus_path, "w") as f:
        f.writelines(lines)
    print(f"Written modified script to: {new_runQSWATPlus_path}")

    try:
        os.system(f"/bin/bash {new_runQSWATPlus_path}")
    except Exception as e:
        print(f"Error running QSWATPlus: {e}")
    finally:
        #os.remove(new_runQSWATPlus_path)
        print(f"Removed script: {new_runQSWATPlus_path}")

## test
if __name__ == "__main__":
    from SWATGenXConfigPars import SWATGenXPaths
    VPUID = "0408"
    LEVEL = "huc12"
    NAME = "04141000"
    MODEL_NAME = "SWAT_MODEL_Web_Application"
    SWATGenXPaths = SWATGenXPaths(username = "admin")
    runQSWATPlus(VPUID, LEVEL, NAME, MODEL_NAME, SWATGenXPaths)
