import HydroGeoDataset as hgd
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import LineString

# Load and reproject the GeoDataFrames
bound = "/data/MyDataBase/HuronRiverPFAS/Huron_River_basin_bound.pkl"
bound = pd.read_pickle(bound).to_crs(epsg=4326)

file = "/home/rafieiva/MyDataBase/codes/PFAS_GNN/temp/WELLOGIC_WSSN_Huron.geojson"
WSSN = gpd.read_file(file, driver='GeoJSON').to_crs(epsg=4326)
print(f" WSSN: {WSSN.columns}")
WSSN.to_pickle("/data/MyDataBase/HuronRiverPFAS/WSSN_Huron.pkl")
# Extract the x and y coordinates from the geometries for plotting
bound_lines = [geom.exterior.coords.xy for geom in bound.geometry if isinstance(geom, LineString) or hasattr(geom, 'exterior')]
wssn_points = WSSN.geometry.apply(lambda geom: geom.coords.xy)

# Plotting
fig, ax = plt.subplots(figsize=(8, 8))

# Plot the boundary polygons
for x, y in bound_lines:
    ax.plot(x, y, color='black')

# Plot the well locations
for x, y in wssn_points:
    ax.scatter(x, y, color='red', s=4)

# Custom legend
#plt.xlabel('Longitude')
#plt.ylabel('Latitude')
## remove x and y axis
plt.xticks([])
plt.yticks([])
plt.title("GW Wells with WSSN in Huron River Basin")
plt.grid(alpha=0.5, linestyle='--', linewidth=0.5)
plt.legend(["Huron River Basin", "GW Wells with WSSN"])
plt.savefig("/home/rafieiva/MyDataBase/codes/PFAS_GNN/figs/WSSN_wells_Huron.png", dpi=300)
plt.close()

### extract data for the gw wells and sacve to a file
config = {
    "RESOLUTION": 250}
importer = hgd.DataImporter(config)
gdf = importer.extract_features("/data/MyDataBase/HuronRiverPFAS/WSSN_Huron.pkl")
gdf.to_pickle("/data/MyDataBase/HuronRiverPFAS/WSSN_Huron_with_features.pkl")

# Load the data
