import os
import rasterio
import numpy as np
from rasterio.windows import Window

path = "/data/SWATGenXApp/GenXAppData/gSSURGO/CONUS/MapunitRaster_30m.tif"

def get_unique_values_windowed(file_path, window_size=1000):
    unique_values = set()

    with rasterio.open(file_path) as src:
        # Get the total dimensions
        height = src.height
        width = src.width

        # Process the image in windows
        for i in range(0, height, window_size):
            for j in range(0, width, window_size):
                # Calculate window dimensions
                win_height = min(window_size, height - i)
                win_width = min(window_size, width - j)

                # Create window
                window = Window(j, i, win_width, win_height)

                # Read the window
                data = src.read(1, window=window)

                # Add unique values to our set
                unique_values.update(np.unique(data))

    return sorted(list(unique_values))

# Get unique values using windowed reading
unique_values = get_unique_values_windowed(path)
print(f"Found {len(unique_values)} unique values")
print("First 10 unique values:", unique_values[:10])

def inspect_raster(file_path):
    with rasterio.open(file_path) as src:
        # Print basic metadata
        print("Raster Metadata:")
        print(f"Driver: {src.driver}")
        print(f"Size: {src.width}x{src.height}")
        print(f"Number of bands: {src.count}")
        print(f"Data type: {src.dtypes[0]}")
        print(f"Coordinate system: {src.crs}")
        print(f"Transform: {src.transform}")

        # Read a small sample from the center of the image
        center_x = src.width // 2
        center_y = src.height // 2
        window_size = 100

        window = Window(
            center_x - window_size//2,
            center_y - window_size//2,
            window_size,
            window_size
        )

        data = src.read(1, window=window)
        print("\nSample data from center of image:")
        print(f"Shape: {data.shape}")
        print(f"Min value: {np.min(data)}")
        print(f"Max value: {np.max(data)}")
        print(f"Mean value: {np.mean(data)}")
        print(f"Unique values in sample: {np.unique(data)}")

        # Print first few rows and columns
        print("\nFirst 5x5 pixels from sample:")
        print(data[:5, :5])

# Inspect the raster
inspect_raster(path)
