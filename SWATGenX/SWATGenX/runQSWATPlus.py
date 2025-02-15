import subprocess
import os
import sys
import shutil


def get_real_uid_gid():
    uid = int(os.environ.get("SUDO_UID", os.getuid()))
    gid = int(os.environ.get("SUDO_GID", os.getgid()))
    return uid, gid


def runQSWATPlus(VPUID, LEVEL, NAME, MODEL_NAME, SWATGenXPaths):
    print(f"runQSWATPlus: Running QSWATPlus for {NAME}")

    runQSWATPlus_path = SWATGenXPaths.runQSWATPlus_path
    if not os.path.exists(runQSWATPlus_path):
        raise FileNotFoundError(f"File {runQSWATPlus_path} does not exist")

    python_command = (
        f"from QSWATPlus3_64 import runHUCProject; "
        f"runHUCProject(VPUID='{VPUID}', LEVEL='{LEVEL}', NAME='{NAME}', MODEL_NAME='{MODEL_NAME}', "
        f"SWATGenXPaths_swatgenx_outlet_path='{SWATGenXPaths.swatgenx_outlet_path}')"
    )

    scripts_dir = "/data/SWATGenXApp/codes/scripts"
    os.makedirs(scripts_dir, exist_ok=True)

    new_runQSWATPlus_path = os.path.join(scripts_dir, f"runQSWATPlus_{NAME}.sh")

    out_dir = os.path.join(SWATGenXPaths.swatgenx_outlet_path, VPUID, LEVEL, NAME, MODEL_NAME)
    os.makedirs(out_dir, exist_ok=True)

    real_uid, real_gid = get_real_uid_gid()

    if os.path.exists(new_runQSWATPlus_path):
        os.remove(new_runQSWATPlus_path)

    shutil.copyfile(runQSWATPlus_path, new_runQSWATPlus_path)

    with open(new_runQSWATPlus_path, "r") as f:
        lines = f.readlines()

    env_vars = ["PYTHONPATH", "QGIS_ROOT", "PYTHONHOME", "QT_PLUGIN_PATH", "GDAL_DATA", "PROJ_LIB", "PATH"]
    header_lines = []
    for var in env_vars:
        val = os.environ.get(var, "")
        header_lines.append(f'export {var}="{val}"')

    header_lines.append('export SWATPLUS_DIR="/usr/local/share/SWATPlus"')
    header_lines.append('export TAUDEM5BIN="$SWATPLUS_DIR/TauDEM5Bin"')
    header_lines.append('export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$TAUDEM5BIN:$PATH"')

    env_header = "\n".join(header_lines) + "\n\n"

    if lines and lines[0].startswith("#!"):
        lines.insert(1, env_header)
    else:
        lines.insert(0, env_header)

    for i, line in enumerate(lines):
        if "Run the Python script with Xvfb (headless display)" in line:
            replacement = (
                f'echo "Fixing ownership and permissions for {out_dir}"\n'
                f'chown -R $(id -u):$(id -g) "{out_dir}"\n'
                f'chmod -R 777 "{out_dir}"\n'
                f'xvfb-run -a /data/SWATGenXApp/codes/.venv/bin/python -c \'from QSWATPlus3_64 import runHUCProject; runHUCProject(VPUID="{VPUID}", LEVEL="{LEVEL}", NAME="{NAME}", MODEL_NAME="{MODEL_NAME}", SWATGenXPaths_swatgenx_outlet_path="{SWATGenXPaths.swatgenx_outlet_path}")\'\n'
            )
            lines[i + 1] = replacement
            break

    with open(new_runQSWATPlus_path, "w") as f:
        f.writelines(lines)

    os.chown(new_runQSWATPlus_path, real_uid, real_gid)
    os.chmod(new_runQSWATPlus_path, 0o755)

    base_model_path = f"/data/SWATGenXApp/Users/menly42/SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/"
    ### correct permision
    os.makedirs(base_model_path, exist_ok=True)
    os.system(f"/bin/sudo chown -R $(id -u):$(id -g) {base_model_path}")
    os.system(f"/bin/sudo chmod -R 777 {base_model_path}")
    #subprocess.run(["/bin/bash", new_runQSWATPlus_path], check=True)
    os.system(f"/bin/bash {new_runQSWATPlus_path}") 
    