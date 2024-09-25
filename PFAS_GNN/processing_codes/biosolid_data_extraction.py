import HydroGeoDataset as hgd
import geopandas as gpd
import pandas as pd

path = "/data/MyDataBase/HuronRiverPFAS/Huron_Biosolid_sites.geojson"
gdf = gpd.read_file(path, driver='GeoJSON')

config = {"RESOLUTION": 250}
importer = hgd.DataImporter(config)
gdf = importer.extract_features(path)
gdf.to_pickle("/data/MyDataBase/HuronRiverPFAS/Huron_Biosolid_sites_with_features.pkl")
gdf.to_file("/data/MyDataBase/HuronRiverPFAS/Huron_Biosolid_sites_with_features.geojson", driver='GeoJSON')