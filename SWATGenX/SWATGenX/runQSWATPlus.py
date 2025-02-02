import subprocess
import os
import sys
import shutil

def get_real_uid_gid():
    """
    Return the real user and group id.
    When run via sudo, SUDO_UID/SUDO_GID are used;
    otherwise, fall back on os.getuid()/os.getgid().
    (When running as non-root these values will be your own.)
    """
    uid = int(os.environ.get("SUDO_UID", os.getuid()))
    gid = int(os.environ.get("SUDO_GID", os.getgid()))
    return uid, gid

def runQSWATPlus(VPUID, LEVEL, NAME, MODEL_NAME, SWATGenXPaths):
    print(f"runQSWATPlus: Running QSWATPlus for {NAME}")

    # Validate that the original QSWATPlus shell script exists.
    runQSWATPlus_path = SWATGenXPaths.runQSWATPlus_path
    if not os.path.exists(runQSWATPlus_path):
        raise FileNotFoundError(f"File {runQSWATPlus_path} does not exist")

    # Construct the Python command that QSWATPlus will execute.
    python_command = (
        f"from QSWATPlus3_64 import runHUCProject; "
        f"runHUCProject(VPUID='{VPUID}', LEVEL='{LEVEL}', NAME='{NAME}', MODEL_NAME='{MODEL_NAME}', "
        f"SWATGenXPaths_swatgenx_outlet_path='{SWATGenXPaths.swatgenx_outlet_path}')"
    )

    # Ensure the scripts directory exists.
    scripts_dir = "/data/SWATGenXApp/codes/scripts"
    os.makedirs(scripts_dir, exist_ok=True)

    # Build the path to the modified shell script.
    new_runQSWATPlus_path = os.path.join(scripts_dir, f"runQSWATPlus_{NAME}.sh")

    # Ensure the SWATGenX output directory exists and is accessible.
    out_dir = os.path.join(SWATGenXPaths.swatgenx_outlet_path, VPUID, LEVEL, NAME, MODEL_NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # For a non-root run, the real UID/GID will be that of the current user.
    real_uid, real_gid = get_real_uid_gid()

    # Set ACLs on the output directory so that the user has full (rwx) permissions.
    subprocess.run(["setfacl", "-m", f"u:{real_uid}:rwx", out_dir], check=True)
    subprocess.run(["setfacl", "-m", f"g:{real_gid}:rwx", out_dir], check=True)

    # Remove any existing modified script.
    if os.path.exists(new_runQSWATPlus_path):
        os.remove(new_runQSWATPlus_path)

    # Copy the original script so we can modify it.
    shutil.copyfile(runQSWATPlus_path, new_runQSWATPlus_path)

    # Read in the contents of the copied script.
    with open(new_runQSWATPlus_path, "r") as f:
        lines = f.readlines()

    # Prepare an environment header. In addition to your base variables,
    # we export SWATPlus and TauDEM variables.
    env_vars = ["PYTHONPATH", "QGIS_ROOT", "PYTHONHOME", "QT_PLUGIN_PATH", "GDAL_DATA", "PROJ_LIB", "PATH"]
    header_lines = [f'export {var}="{os.environ.get(var, "")}"' for var in env_vars]
    header_lines.append('export SWATPLUS_DIR="/usr/local/share/SWATPlus"')
    header_lines.append('export TAUDEM5BIN="$SWATPLUS_DIR/TauDEM5Bin"')
    header_lines.append('export PATH="$TAUDEM5BIN:$PATH"')
    # (If needed, you can add additional exports here.)
    env_header = "\n".join(header_lines) + "\n\n"

    # Insert the header into the script.
    if lines and lines[0].startswith("#!"):
        lines.insert(1, env_header)
    else:
        lines.insert(0, env_header)

    # Find the marker comment in the original script where the Python command should run.
    # Replace the following line with a block that first fixes the ownership and permissions of the output.
    for i, line in enumerate(lines):
        if "Run the Python script with Xvfb (headless display)" in line:
            replacement = (
                f'echo "Fixing ownership and permissions for {out_dir}"\n'
                f'chown -R $(id -u):$(id -g) "{out_dir}"\n'
                f'chmod -R 777 "{out_dir}"\n'
                f'xvfb-run -a python3 -c "{python_command}"\n'
            )
            lines[i + 1] = replacement
            break

    # Write the modified script back to file.
    with open(new_runQSWATPlus_path, "w") as f:
        f.writelines(lines)

    # Ensure the modified script itself is owned by the current user and is executable.
    os.chown(new_runQSWATPlus_path, real_uid, real_gid)
    os.chmod(new_runQSWATPlus_path, 0o755)
    subprocess.run(["setfacl", "-m", f"u:{real_uid}:rwx", new_runQSWATPlus_path], check=True)
    subprocess.run(["setfacl", "-m", f"g:{real_gid}:rwx", new_runQSWATPlus_path], check=True)

    # Finally, execute the modified shell script.
    try:
        subprocess.run(["bash", new_runQSWATPlus_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: QSWATPlus execution failed with error code {e.returncode}", file=sys.stderr)
    finally:
        # Clean up: remove the temporary modified script.
        if os.path.exists(new_runQSWATPlus_path):
            os.remove(new_runQSWATPlus_path)

# Example usage:
# runQSWATPlus(VPUID='0712', LEVEL='huc12', NAME='05536265', MODEL_NAME='SWAT_MODEL', SWATGenXPaths=yourSWATGenXPathsObject)
