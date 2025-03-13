import os
import pwd
import grp
from dataclasses import dataclass
import pandas as pd
import geopandas as gpd
from pathlib import Path
import shutil
import tempfile
import stat


def ensure_directory_ownership(path, user="www-data", group="www-data"):
    """Ensures the specified directory and its contents are owned by the given user and group."""
    try:
        # Only try to change ownership if running as root
        if os.geteuid() == 0:
            uid = pwd.getpwnam(user).pw_uid
            gid = grp.getgrnam(group).gr_gid
            os.chown(path, uid, gid)
            for root, dirs, files in os.walk(path):
                for dir in dirs:
                    os.chown(os.path.join(root, dir), uid, gid)
                for file in files:
                    os.chown(os.path.join(root, file), uid, gid)
            print(f"Ownership of {path} and its contents changed to {user}:{group}")
        else:
            print("Skipping ownership changes (not running as root)")
    except Exception as e:
        print(f"Error changing ownership of {path}: {e}")

def create_group_writable_dir(path):
    """Creates a directory with group write permissions and setgid bit."""
    try:
        # Create directory with mode 775 (rwxrwxr-x)
        os.makedirs(path, mode=0o2775, exist_ok=True)
        # Set group write and setgid bit
        current_mode = os.stat(path).st_mode
        os.chmod(path, current_mode | 0o2775)
    except Exception as e:
        print(f"Error setting directory permissions for {path}: {e}")

def get_temp_dir():
    """Creates a temporary directory with appropriate permissions."""
    # Save current umask
    old_umask = os.umask(0)
    try:
        # Create temp dir with mode 0770
        temp_dir = tempfile.mkdtemp(prefix='swatgenx_')
        os.chmod(temp_dir, 0o770)
        return temp_dir
    finally:
        # Restore original umask
        os.umask(old_umask)

def safe_copy_or_link(src, dst):
    """Attempts to copy file, falls back to creating symlink if copy fails."""
    try:
        # Check if source and destination are the same file
        if os.path.exists(dst) and os.path.exists(src):
            try:
                if os.path.samefile(src, dst):
                    print(f"Source and destination are the same file: {src}")
                    return True, "skipped"
            except OSError:
                # Handle case where one of the files might be a broken symlink
                pass
        
        # Try to copy first
        shutil.copy2(src, dst)
        os.chmod(dst, 0o664)
        return True, "copy"
    except shutil.SameFileError:
        print(f"Source and destination are the same file: {src}")
        return True, "skipped"
    except PermissionError:
        try:
            # If copy fails, try to create a symbolic link
            if os.path.exists(dst):
                os.remove(dst)
            os.symlink(src, dst)
            return True, "link"
        except Exception as e:
            print(f"Both copy and link failed for {src}: {e}")
            return False, "failed"
    except Exception as e:
        print(f"Copy operation failed for {src}: {e}")
        try:
            # If copy fails for any other reason, try to create a symbolic link
            if os.path.exists(dst):
                os.remove(dst)
            os.symlink(src, dst)
            return True, "link"
        except Exception as e:
            print(f"Both copy and link failed for {src}: {e}")
            return False, "failed"

def fetch_streamflow_for_watershed(VPUID, LEVEL, NAME, MODEL_NAME, SWATGenXPaths):
    paths = SWATGenXPaths

    model_base = paths.construct_path(paths.swatgenx_outlet_path, VPUID, LEVEL, NAME)
    swatplus_stations_shp = Path(paths.construct_path(model_base, "streamflow_data", "stations.shp"))

    meta_data_path = paths.construct_path(paths.streamflow_path, "VPUID", VPUID, f"meta_{VPUID}.csv")
    streamflow_stations_shp = paths.construct_path(paths.streamflow_path, "VPUID", VPUID, f"streamflow_stations_{VPUID}.shp")
    swatplus_lsus2_shp = paths.construct_path(model_base, MODEL_NAME, "Watershed", "Shapes", "lsus2.shp")
    target_path = paths.construct_path(model_base, "streamflow_data")

    # Create temporary directory for processing
    temp_dir = get_temp_dir()
    print(f"Using temporary directory: {temp_dir}")

    try:
        # Read data
        stations = gpd.read_file(streamflow_stations_shp).to_crs("EPSG:4326")
        swatplus_lsus2 = gpd.read_file(swatplus_lsus2_shp).to_crs("EPSG:4326")
        meta_data = pd.read_csv(meta_data_path, dtype={"site_no": str})

        # Try to create target directory, fall back to temp dir if needed
        try:
            os.makedirs(target_path, mode=0o770, exist_ok=True)
            working_dir = target_path
        except PermissionError:
            print(f"Cannot create {target_path}, using temporary directory")
            working_dir = temp_dir

        # Spatial join
        subbasins_stations = gpd.sjoin(stations, swatplus_lsus2, how="inner", predicate="intersects")
        subbasins_stations[["site_no", "geometry"]].to_file(
            os.path.join(working_dir, "stations.shp")
        )

        # Process files
        for channel, site_no in zip(subbasins_stations["Channel"], subbasins_stations["site_no"]):
            for file_type in ['streamflow', 'streamflow_record']:
                source = paths.construct_path(
                    paths.streamflow_path, "VPUID", VPUID, f"{file_type}_{site_no}.jpeg"
                )
                if os.path.exists(source):
                    dest = os.path.join(working_dir, os.path.basename(source))
                    success, method = safe_copy_or_link(source, dest)
                    if success:
                        print(f"{file_type} data for {site_no} {method}ed to {working_dir}")
                else:
                    print(f"{file_type} JPEG for site {site_no} does not exist.")

            # Handle CSV file
            source_csv = paths.construct_path(
                paths.streamflow_path, "VPUID", VPUID, f"streamflow_{site_no}.csv"
            )
            if os.path.exists(source_csv):
                dest_csv = os.path.join(working_dir, f"{channel}_{site_no}.csv")
                success, method = safe_copy_or_link(source_csv, dest_csv)
                if success:
                    print(f"CSV data for {site_no} {method}ed to {working_dir}")
            else:
                print(f"CSV file for site {site_no} does not exist.")

        # If we used temp dir, try to move files to target
        if working_dir == temp_dir:
            try:
                if not os.path.exists(target_path):
                    os.makedirs(target_path, mode=0o770, exist_ok=True)
                for file in os.listdir(temp_dir):
                    src = os.path.join(temp_dir, file)
                    dst = os.path.join(target_path, file)
                    try:
                        shutil.move(src, dst)
                    except PermissionError:
                        print(f"Could not move {file} to target directory, leaving in temp dir")
            except Exception as e:
                print(f"Error moving files to target directory: {e}")
                print(f"Files remain in temporary directory: {temp_dir}")
                return temp_dir  # Return temp_dir path so files aren't lost

    except Exception as e:
        print(f"Error processing streamflow data: {e}")
        shutil.rmtree(temp_dir)
        raise

    return target_path

if __name__ == "__main__":
    VPUID = "0206"
    LEVEL = "huc12"
    NAME = "01583570"
    MODEL_NAME = "SWAT_MODEL"
    fetch_streamflow_for_watershed(VPUID, LEVEL, NAME, MODEL_NAME)