import pandas as pd
import os
import geopandas as gpd
import h5py
import numpy as np



def find_row_col(station_lat, station_lon):
    h5_reference = "//35.9.219.75/Data/MyDataBase/HydroGeoDataset_ML_250.h5"
    with h5py.File(h5_reference, "r") as h5_file:
        lat = h5_file["x_250m"][:]  # contains no value as -999
        lon = h5_file["y_250m"][:]  # contains no value as -999
        print(f"shape of lat: {lat.shape}, shape of lon: {lon.shape}")

        # Replace -999 with np.nan
        lat = np.where(lat == -999, np.nan, lat)
        lon = np.where(lon == -999, np.nan, lon)

        # Debug: Print min and max lat/lon after replacement
        print(f"min lat: {np.nanmin(lat)}, max lat: {np.nanmax(lat)}")
        print(f"min lon: {np.nanmin(lon)}, max lon: {np.nanmax(lon)}")

        # Flatten the arrays for easier distance calculation
        lat_flat = lat.flatten()
        lon_flat = lon.flatten()

        # Calculate the differences and find the minimum
        lat_diff = np.abs(lat_flat - station_lat)
        lon_diff = np.abs(lon_flat - station_lon)

        # Combine lat and lon differences
        combined_diff = np.sqrt(lat_diff**2 + lon_diff**2)

        # Find the index of the minimum combined difference
        min_index = np.nanargmin(combined_diff)

        # Map the flat index back to 2D indices
        row, col = np.unravel_index(min_index, lat.shape)

        return row, col

def write_on_3d_h5(data, start_date, end_date):
    # Save the numpy array in HDF5 format
    output_path = "gw_head.h5"
    with h5py.File(output_path, "w") as h5_file:
        h5_file.create_dataset("gw_head", data=data, dtype=np.uint16, compression="gzip")
        h5_file["gw_head"].attrs["unit"] = "m"
        h5_file["gw_head"].attrs["description"] = "Groundwater head data"
        h5_file["gw_head"].attrs["start_date"] = start_date.strftime("%Y-%m-%d")
        h5_file["gw_head"].attrs["end_date"] = end_date.strftime("%Y-%m-%d")
        h5_file["gw_head"].attrs["source"] = "USGS"
        h5_file["gw_head"].attrs["creator"] = "Vahid Rafiei"
        h5_file["gw_head"].attrs["email"] = "rafieiva@msu.edu"

def write_on_2d_h5(name, data, row, col, lat, lon, start_date, end_date):
    # Save the numpy array in HDF5 format
    output_path = "gw_head_2d.h5"
    open_type = "w" if not os.path.exists(output_path) else "a"
    with h5py.File(output_path, open_type) as h5_file:
        name = f"{name}_{row}_{col}"
        h5_file.create_dataset(name, data=data, dtype=np.float32)
        h5_file[name].attrs["unit"] = "m"
        h5_file[name].attrs["description"] = "Groundwater head data"
        h5_file[name].attrs["start_date"] = start_date.strftime("%Y-%m-%d")
        h5_file[name].attrs["end_date"] = end_date.strftime("%Y-%m-%d")
        h5_file[name].attrs["source"] = "USGS"
        h5_file[name].attrs["creator"] = "Vahid Rafiei"
        h5_file[name].attrs["email"] = "rafieiva@msu.edu"
        h5_file[name].attrs["lat"] = lat
        h5_file[name].attrs["lon"] = lon

    print(f"Data written to {output_path}")


if __name__ == "__main__":

    path = "/data/MyDataBase/SWATGenXAppData/groundwater_daily_stations"
    stations_path = "/data/MyDataBase/SWATGenXAppData/groundwater_daily_stations/statons_location/statons_location.shp"
    # Read stations shapefile and convert to desired CRS
    stations = gpd.read_file(stations_path).to_crs("EPSG:26990")
    # Get list of CSV files in the specified directory
    files = [f for f in os.listdir(path) if f.endswith(".csv")]
    ### create a numpy array with shape (days, 1849, 1458)
    ### days from 1/1/1990 to 12/31/2022
    import datetime
    start_date = datetime.datetime(1990, 1, 1)
    end_date = datetime.datetime(2022, 12, 31)
    days = (end_date - start_date).days + 1
    data = np.zeros((days, 1849, 1458))
    print(data.shape)
    if os.path.exists("gw_head.h5"):
        os.remove("gw_head.h5")
    if os.path.exists("gw_head_2d.h5"):
        os.remove("gw_head_2d.h5")

    for file in files:
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path)
        daily_gw_head = df.head_m.values
        print("############### length of daily_gw_head", len(daily_gw_head))
        site_no = df.site_id[0]
        station = stations[stations.site_no.isin([site_no])].to_crs("EPSG:26990")
        x = station.geometry.x.values[0]
        y = station.geometry.y.values[0]
        print("###############", site_no, x, y)

        station = stations[stations.site_no.isin([site_no])].to_crs("EPSG:4350")
        lat = station.geometry.x.values[0]
        lon = station.geometry.y.values[0]
        print("###############", site_no, x, y)

        row, col = find_row_col(x,y)
        print(row, col)

        write_on_2d_h5(site_no, daily_gw_head, row, col, lat, lon, start_date, end_date)

        for i, head in enumerate(daily_gw_head):
            data[i, row, col] = head

    write_on_3d_h5(data, start_date, end_date)
