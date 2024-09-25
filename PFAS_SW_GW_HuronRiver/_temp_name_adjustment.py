import geopandas as gpd

# Define paths
input_path = "/data/MyDataBase/SWATGenXAppData/codes/PFAS_SW_GW_HuronRiver/input_data/PFAS_sw_data_with_features.geojson"
clip_input_feature = "/data/MyDataBase/SWATGenXAppData/codes/PFAS_SW_GW_HuronRiver/input_data/Huron_PFAS_GW_Features.geojson"
output_path = "/data/MyDataBase/SWATGenXAppData/codes/PFAS_SW_GW_HuronRiver/input_data/Huron_PFAS_SW_Features.geojson"

def clip_data_with_bbox(input_path, clip_input_feature, output_path):
    # Read the input and clip data
    input_data = gpd.read_file(input_path).to_crs("EPSG:4326")
    clip_data = gpd.read_file(clip_input_feature).to_crs("EPSG:4326")

    # Get the bounding box of the clip data
    bbox = clip_data.total_bounds  # [minx, miny, maxx, maxy]

    # Clip the input data using the bounding box
    clipped_data = input_data.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]

    # Save the clipped data to a GeoJSON file
    clipped_data.to_file(output_path, driver='GeoJSON')
    ### also save in pkl
    clipped_data.to_pickle(output_path.replace('.geojson', '.pkl'))

    return clipped_data

# Perform the clipping operation
clipped_data = clip_data_with_bbox(input_path, clip_input_feature, output_path)

# Optionally print or log the clipped data
print(clipped_data)

# Output saved to specified GeoJSON file
