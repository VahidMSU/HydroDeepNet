import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_origin
import glob

# Update the path to point to the CONUS directory where the HDF5 files are stored
DEM_PATH = "/data/SWATGenXApp/GenXAppData/DEM/CONUS/"

def create_tif_from_h5(h5_file, output_tif):
    """Create a temporary TIFF file from HDF5 data for verification."""
    print(f"Processing HDF5 file: {h5_file}")
    with h5py.File(h5_file, 'r') as f:
        # Get elevation data and convert to float32 for visualization
        elevation = f['elevation'][:].astype(np.float32)
        print(f"Original data type: {f['elevation'].dtype}")
        print(f"Converted to: {elevation.dtype}")
        print(f"Elevation shape: {elevation.shape}")

        # Get metadata
        metadata = {}
        for key, value in f['metadata'].items():
            if isinstance(value, h5py.Dataset):
                metadata[key] = value[()]
                if isinstance(metadata[key], bytes):
                    metadata[key] = metadata[key].decode('utf-8')

        print(f"Metadata extracted: {metadata.keys()}")
        print(f"Original dtype from metadata: {metadata.get('dtype', 'not found')}")

        # Convert nodata to float if it's a string
        nodata = metadata['nodata']
        if isinstance(nodata, str):
            try:
                nodata = float(nodata)
            except ValueError:
                nodata = None
                print(f"Warning: Could not convert nodata value '{metadata['nodata']}' to float, using None")

        # Create transform from GDAL-style transform
        transform = rasterio.Affine.from_gdal(*metadata['transform'])

        # Create a new TIFF file
        with rasterio.open(
            output_tif,
            'w',
            driver='GTiff',
            height=elevation.shape[0],
            width=elevation.shape[1],
            count=1,
            dtype=elevation.dtype,  # Now using float32
            crs=metadata['crs'],
            transform=transform
        ) as dst:
            dst.write(elevation, 1)
            if nodata is not None:
                dst.nodata = nodata

def plot_dem(tif_file, output_plot):
    """Plot DEM data with proper georeferencing."""
    print(f"Creating plot for: {tif_file}")
    with rasterio.open(tif_file) as src:
        # Read the data
        dem = src.read(1)

        # Mask nodata values for better visualization
        if src.nodata is not None:
            dem = np.ma.masked_equal(dem, src.nodata)

        # Get metadata
        transform = src.transform
        crs = src.crs
        bounds = src.bounds

        print(f"DEM bounds: {bounds}")
        print(f"DEM shape: {dem.shape}")
        print(f"DEM min/max: {dem.min()}, {dem.max()}")

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot DEM
        im = ax.imshow(
            dem,
            extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
            cmap='terrain',
            origin='upper'
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Elevation (m)')

        # Add title and labels
        ax.set_title('Digital Elevation Model')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        # Add grid
        ax.grid(True, alpha=0.3)

        # Save plot
        plt.savefig(output_plot, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {output_plot}")

def verify_metadata(h5_file):
    """Verify metadata integrity of HDF5 file."""
    print(f"Verifying metadata for: {h5_file}")
    with h5py.File(h5_file, 'r') as f:
        # Check required datasets exist
        required_datasets = ['elevation', 'metadata']
        for ds in required_datasets:
            if ds not in f:
                print(f"Error: Missing dataset '{ds}' in {h5_file}")
                return False

        # Check metadata group
        metadata = f['metadata']
        required_metadata = ['transform', 'crs', 'nodata']
        for md in required_metadata:
            if md not in metadata:
                print(f"Error: Missing metadata '{md}' in {h5_file}")
                return False

        # Check data types
        elevation = f['elevation']
        if not isinstance(elevation, h5py.Dataset):
            print(f"Error: 'elevation' is not a dataset in {h5_file}")
            return False

        # Check dimensions
        if len(elevation.shape) != 2:
            print(f"Error: Elevation data should be 2D, got shape {elevation.shape}")
            return False

        return True

def main():
    # Create output directories
    os.makedirs(os.path.join(DEM_PATH, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(DEM_PATH, 'temp_tifs'), exist_ok=True)

    # Process each HDF5 file
    h5_files = glob.glob(os.path.join(DEM_PATH, '*.h5'))
    print(f"Found {len(h5_files)} HDF5 files in {DEM_PATH}")

    for h5_file in h5_files:
        print(f"\nProcessing {os.path.basename(h5_file)}...")

        # Verify metadata
        if not verify_metadata(h5_file):
            print(f"Skipping {h5_file} due to metadata issues")
            continue

        # Create temporary TIFF for verification
        temp_tif = os.path.join(DEM_PATH, 'temp_tifs', os.path.basename(h5_file).replace('.h5', '.tif'))
        create_tif_from_h5(h5_file, temp_tif)

        # Create plot
        output_plot = os.path.join(DEM_PATH, 'plots', os.path.basename(h5_file).replace('.h5', '.png'))
        plot_dem(temp_tif, output_plot)

        print(f"Created plot: {output_plot}")

        # Clean up temporary TIFF
        os.remove(temp_tif)

if __name__ == "__main__":
    main()