import geopandas as gpd
from config import AgentConfig


config = { 
    "RESOLUTION": 250,
    "huc8": None,
    "video": False,
    "aggregation": "annual",
    "start_year": 2000,
    "end_year": 2003,
    'bounding_box': [-85.444332, 43.658148, -85.239256, 44.164683],
}

hydrogepdata_path = AgentConfig.USGS_governmental_path

df = gpd.read_file(hydrogepdata_path)
## layers
print(df.head())

