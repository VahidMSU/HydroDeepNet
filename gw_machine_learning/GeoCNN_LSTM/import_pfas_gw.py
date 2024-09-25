import geopandas as gpd
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Read the shapefile
    shapefile_path = "/data/MyDataBase/P_locations_rasters_30m.shp"
    gdf = gpd.read_file(shapefile_path)

    # Print the first 5 rows
    print(gdf.head())
    print(f"number of rows: {len(gdf)}")
    # Plot the shapefile
    gdf.plot()
    plt.savefig("input_figs/P_locations_rasters_30m.png")
