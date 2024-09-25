import arcpy
import geopandas as gpd
## fish sampling data
fish_path = "Z:/MyDataBase/HuronRiverPFAS/Fish_Contaminant_Monitoring_Sampling_Results_-1946005603489142699.geojson"

## read the fish sampling data
fish = gpd.read_file(fish_path, driver='GeoJSON')

fish.head()
print(list(fish.columns))
