import pandas as pd
import os
import matplotlib.pyplot as plt
import geopandas as gpd

LEVEL = 'huc8'
for LEVEL in ['huc8','huc12']:
    NAMES = os.listdir(f"/data/MyDataBase/CIWRE-BAE/SWAT_input/{LEVEL}/")
    NAMES.remove("log.txt")
    all_shapes = []
    for NAME in NAMES:
        sub_path = f"/data/MyDataBase/CIWRE-BAE/SWAT_input/{LEVEL}/{NAME}/SWAT_gwflow_MODEL/gwflow_gis/active_domain_shape.shp"
        if not os.path.exists(sub_path):
            continue
        subbasins = gpd.read_file(sub_path).to_crs('EPSG:26990')
    
        ### only boundrary
        buffered = subbasins.buffer(100)
        dissolved = buffered.unary_union
        all_shapes.append(subbasins)
    all_shapes = pd.concat(all_shapes)
    all_shapes = gpd.GeoDataFrame(all_shapes, geometry=all_shapes['geometry'])
    all_shapes = all_shapes.to_crs(epsg=4326)
    ## plot 
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    all_shapes.plot(ax=ax)
    plt.savefig("model_bounds/{LEVEL}_model_bounds.png", dpi=300)
    all_shapes.to_pickle(f"model_bounds/{LEVEL}_model_bounds.pkl")