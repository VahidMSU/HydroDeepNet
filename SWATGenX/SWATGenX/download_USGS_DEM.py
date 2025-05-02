import os
import hashlib
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import rasterio
import h5py
import numpy as np

try:
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
except ImportError:
    from SWATGenXConfigPars import SWATGenXPaths

def calculate_file_hash(filepath):
    """Calculate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def convert_to_h5(tif_path):
    """Convert TIFF file to HDF5 format while preserving metadata."""
    h5_path = tif_path.replace('.tif', '.h5')

    # Skip if H5 file already exists
    if os.path.exists(h5_path):
        print(f"H5 file already exists: {h5_path}")
        return h5_path

    print(f"Converting {tif_path} to HDF5 format...")

    # Open the TIFF file using rasterio
    with rasterio.open(tif_path) as src:
        # Get raster data
        data = src.read(1)

        # Replace nodata values with -99999.0
        if src.nodata is not None:
            data[data == src.nodata] = -99999.0
            print(f"Replaced nodata value {src.nodata} with -99999.0")

        # Check elevation range excluding nodata values
        valid_data = data[data != -99999.0]
        if len(valid_data) > 0:  # Only check if there are valid values
            min_elev = np.min(valid_data)
            max_elev = np.max(valid_data)
            print(f"Valid elevation range: {min_elev:.2f} to {max_elev:.2f} meters")

            # Verify range is within float16 limits
            if min_elev < -65504 or max_elev > 65504:
                print(f"Warning: Elevation range exceeds float16 limits, using float32 instead")
                data = data.astype(np.float32)
            else:
                data = data.astype(np.float16)
                print("Using float16 for storage")
        else:
            print("Warning: No valid elevation data found, using float32")
            data = data.astype(np.float32)

        # Get metadata
        metadata = {
            'transform': np.array(src.transform.to_gdal(), dtype=np.float64),
            'crs': src.crs.to_string(),
            'nodata': -99999.0,  # Use new nodata value
            'width': src.width,
            'height': src.height,
            'count': src.count,
            'dtype': str(data.dtype),  # Store actual dtype used
            'driver': src.driver
        }

        # Create HDF5 file with more efficient compression
        with h5py.File(h5_path, 'w') as f:
            # Store raster data with optimized compression
            f.create_dataset(
                'elevation',
                data=data,
                compression='lzf',  # Use LZF compression which is faster and often more efficient for DEM data
                chunks=True,  # Enable chunking for better compression
                shuffle=True  # Enable byte shuffling for better compression
            )

            # Store metadata
            metadata_group = f.create_group('metadata')
            for key, value in metadata.items():
                if value is not None:
                    if isinstance(value, (np.ndarray, tuple, list)):
                        metadata_group.create_dataset(key, data=np.array(value, dtype=np.float64))
                    else:
                        # Convert string to bytes for storage
                        metadata_group.create_dataset(key, data=np.bytes_(str(value).encode('utf-8')))

    # Report file sizes
    original_size = os.path.getsize(tif_path)
    new_size = os.path.getsize(h5_path)
    compression_ratio = original_size / new_size

    print(f"Original TIFF size: {original_size/1024/1024:.2f} MB")
    print(f"New HDF5 size: {new_size/1024/1024:.2f} MB")
    print(f"Compression ratio: {compression_ratio:.2f}x")

    # Remove original TIFF file
    os.remove(tif_path)
    print(f"Conversion complete. Original TIFF file removed.")
    return h5_path

def convert_existing_tiffs(directory):
    """Convert all existing TIFF files in directory to HDF5 format."""
    tif_files = [f for f in os.listdir(directory) if f.endswith('.tif')]
    if not tif_files:
        print("No existing TIFF files found to convert.")
        return

    print(f"Found {len(tif_files)} existing TIFF files to convert to HDF5 format...")
    for tif_file in tif_files:
        tif_path = os.path.join(directory, tif_file)
        convert_to_h5(tif_path)
    print("Finished converting existing TIFF files to HDF5 format.")

def download_file(url, output_path):
    """Download a file with progress bar and verification."""
    filename = url.split('/')[-1]
    output_file = os.path.join(output_path, filename)
    temp_file = output_file + '.tmp'

    # Check if H5 file already exists
    h5_file = output_file.replace('.tif', '.h5')
    if os.path.exists(h5_file):
        print(f"H5 file already exists: {h5_file}")
        return

    # Check if TIFF file exists and is complete
    if os.path.exists(output_file):
        try:
            # Get file size from URL
            response = requests.head(url)
            remote_size = int(response.headers.get('content-length', 0))

            # Check local file size
            local_size = os.path.getsize(output_file)

            if local_size == remote_size and remote_size > 0:
                print(f"File already exists and is complete: {output_file}")
                # Convert existing complete TIFF to H5
                convert_to_h5(output_file)
                return

            print(f"File exists but is incomplete/corrupted. Redownloading: {output_file}")
        except Exception as e:
            print(f"Error checking file {output_file}: {str(e)}")
            print("Redownloading file...")

    # Download with progress bar
    print(f"Downloading {url} to {output_file}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(temp_file, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

    # Verify download
    if os.path.getsize(temp_file) == total_size:
        # Move temp file to final location
        if os.path.exists(output_file):
            os.remove(output_file)
        os.rename(temp_file, output_file)
        print(f"Successfully downloaded: {output_file}")

        # Convert to H5 format
        h5_file = convert_to_h5(output_file)
        print(f"Successfully converted to H5: {h5_file}")
    else:
        print(f"Download failed for {output_file} - file size mismatch")
        if os.path.exists(temp_file):
            os.remove(temp_file)

def download_USGS_DEM():
    DEM_13_arc_second_list = SWATGenXPaths.DEM_13_arc_second_list

    with open(DEM_13_arc_second_list, 'r') as file:
        lines = file.readlines()
        urls = [line.strip() for line in lines]

    Downloaded_CONUS_DEM_path = SWATGenXPaths.Downloaded_CONUS_DEM_path
    os.makedirs(Downloaded_CONUS_DEM_path, exist_ok=True)

    # First convert any existing TIFF files to H5
    convert_existing_tiffs(Downloaded_CONUS_DEM_path)

    # Download remaining files in parallel with progress tracking
    with ThreadPoolExecutor(max_workers=25) as executor:
        list(tqdm(
            executor.map(lambda url: download_file(url, Downloaded_CONUS_DEM_path), urls),
            total=len(urls),
            desc="Downloading DEM files"
        ))

if __name__ == "__main__":
    download_USGS_DEM()