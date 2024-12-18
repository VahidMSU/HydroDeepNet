from HydroGeoDataset.core import DataImporter
from HydroGeoDataset.core import get_rowcol_index_by_latlon, get_rowcol_range_by_latlon, read_h5_file, hydrogeo_dataset_dict
import h5py 
import numpy as np
import time


if __name__ == '__main__':

    path = "/data/MyDataBase/HydroGeoDataset_ML_250.h5"

    dic = hydrogeo_dataset_dict()

    print(dic['geospatial'])    
    time.sleep(10)  
    hydrogeo_dic = hydrogeo_dataset_dict(path)

    lat, lon = 42.0, -84.0
    row, col = get_rowcol_index_by_latlon(lat, lon, 250)
    for key in hydrogeo_dic['geospatial']:
        if key in ["x_250m", "y_250m", "lat_250m", "lon_250m", "mask_250m"]:
            continue
        address = f"geospatial/{key}"

        data = read_h5_file(lat=row, lon=col, address=address)
        print(f"key: {key}, data: {data:.2f}")


        min_lat, max_lat = 41.5, 42.5
        min_lon, max_lon = -84.5, -83.5
        min_row_number, max_row_number, min_col_number, max_col_number = get_rowcol_range_by_latlon(min_lat, max_lat, min_lon, max_lon)
        data = read_h5_file(address, lat_range=[min_row_number, max_row_number], lon_range=[min_col_number, max_col_number])
        print(f"key: {key}, average data: {data.mean():.2f}, median data: {np.median(data):.2f}, std data: {data.std():.2f}")
        