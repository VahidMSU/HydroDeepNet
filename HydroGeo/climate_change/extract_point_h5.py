import h5py
import numpy as np
import pandas as pd

class PointDataExtractor:
    def __init__(self, h5_path, lat, lon, elev):
        self.h5_path = h5_path
        self.lat = lat
        self.lon = lon
        self.elev = elev

    def get_nearest_index(self, array, value):
        return (np.abs(array - value)).argmin()

    def extract_data(self, region, model, scenario, ensemble, time_step, time_ranges, variable):
        if variable == "pcp":
            variable = "pr"

            with h5py.File(self.h5_path, 'r') as f:
                lats = f['lat'][:]
                lons = f['lon'][:]

                lat_idx = self.get_nearest_index(lats, self.lat)
                lon_idx = self.get_nearest_index(lons, self.lon)

                print(f"Nearest lat index: {lat_idx}, value: {lats[lat_idx]}")
                print(f"Nearest lon index: {lon_idx}, value: {lons[lon_idx]}")

                all_time_series = []
                for time_range in time_ranges:
                    if time_step == "monthly":
                        group_path = f"/{region}/{model}/{scenario}/{ensemble}/{time_step}/{time_range}/{variable}_tavg"
                    else:
                        group_path = f"/{region}/{model}/{scenario}/{ensemble}/{time_step}/{time_range}/{variable}"
                    if group_path not in f:
                        raise ValueError(f"Group {group_path} does not exist in the HDF5 file.")

                    dataset = f[group_path]
                    data = dataset[:, lat_idx, lon_idx]

                    time_series = pd.date_range(start=f"1/1/{time_range.split('_')[0]}", periods=len(data), freq='D')
                    all_time_series.append(pd.DataFrame(data, index=time_series, columns=[variable]))

                    description = dataset.attrs.get("description", "No description")
                    units = dataset.attrs.get("unit", "unknown")

                    print(f"Description: {description}")
                    print(f"Units: {units}")

                    all_time_series[-1]['year'] = all_time_series[-1].index.year
                    all_time_series[-1]['day'] = all_time_series[-1].index.dayofyear
                    all_time_series[-1]['name'] = f"{model}_{scenario}_{ensemble}"
                    all_time_series[-1]['region'] = region

                all_time_series = pd.concat(all_time_series)
                nbyr = len(all_time_series['year'].unique())
                tstep = "daily"
                lat = self.lat
                lon = self.lon
                elev = self.elev

                if variable == "tas":
                    all_time_series[variable] = all_time_series[variable] - 273.15

                if variable == "pr":
                    all_time_series[variable] = all_time_series[variable] * 86400

                all_time_series[['year', 'day', variable]]

                print("Writing to SWAT format")
                with open(f"{variable}.pcp", 'w') as f:
                    f.write(f"{variable} {model} {scenario} {ensemble} {time_range} {region}\n")
                    f.write(f"nbyr tstep lat lon elev\n")
                    f.write(f"{nbyr} {tstep} {lat} {lon} {elev}\n")

                    for index, row in all_time_series.iterrows():
                        f.write(f"{row['year']} {row['day']} {row[variable]:.2f}\n")

        elif variable == 'tmp':

            with h5py.File(self.h5_path, 'r') as f:
                lats = f['lat'][:]
                lons = f['lon'][:]

                lat_idx = self.get_nearest_index(lats, self.lat)
                lon_idx = self.get_nearest_index(lons, self.lon)

                print(f"Nearest lat index: {lat_idx}, value: {lats[lat_idx]}")
                print(f"Nearest lon index: {lon_idx}, value: {lons[lon_idx]}")

                all_time_series = []
                for time_range in time_ranges:
                    if time_step == "monthly":
                        group_path = f"/{region}/{model}/{scenario}/{ensemble}/{time_step}/{time_range}/{variable}_tavg"
                    else:
                        group_path = f"/{region}/{model}/{scenario}/{ensemble}/{time_step}/{time_range}/tasmax"
                        group_path2 = f"/{region}/{model}/{scenario}/{ensemble}/{time_step}/{time_range}/tasmin"

                    if group_path not in f:
                        raise ValueError(f"Group {group_path} does not exist in the HDF5 file.")
                    if group_path2 not in f:
                        raise ValueError(f"Group {group_path2} does not exist in the HDF5 file.")

                    dataset = f[group_path]
                    dataset2 = f[group_path2]

                    data = dataset[:, lat_idx, lon_idx]
                    data2 = dataset2[:, lat_idx, lon_idx]

                    time_series = pd.date_range(start=f"1/1/{time_range.split('_')[0]}", periods=len(data), freq='D')

                    data = pd.DataFrame(data, index=time_series, columns=['tasmax'])
                    data2 = pd.DataFrame(data2, index=time_series, columns=['tasmin'])
                    data = pd.merge(data, data2, left_index=True, right_index=True)

                    all_time_series.append(pd.DataFrame(data, index=time_series, columns=['tasmax', 'tasmin']))

                    description = dataset.attrs.get("description", "No description")
                    units = dataset.attrs.get("unit", "unknown")

                    print(f"Description: {description}")
                    print(f"Units: {units}")

                    all_time_series[-1]['year'] = all_time_series[-1].index.year
                    all_time_series[-1]['day'] = all_time_series[-1].index.dayofyear
                    all_time_series[-1]['name'] = f"{model}_{scenario}_{ensemble}"
                    all_time_series[-1]['region'] = region

                all_time_series = pd.concat(all_time_series)
                nbyr = len(all_time_series['year'].unique())
                tstep = "daily"
                lat = self.lat
                lon = self.lon
                elev = self.elev

                if variable == "tmp":
                    all_time_series['tasmax'] = all_time_series['tasmax'] - 273.15
                    all_time_series['tasmin'] = all_time_series['tasmin'] - 273.15

                all_time_series[['year', 'day', 'tasmax', 'tasmin']]

                print("Writing to SWAT format")
                with open(f"{variable}.pcp", 'w') as f:
                    f.write(f"{variable} {model} {scenario} {ensemble} {time_range} {region}\n")
                    f.write(f"nbyr tstep lat lon elev\n")
                    f.write(f"{nbyr} {tstep} {lat} {lon} {elev}\n")

                    for index, row in all_time_series.iterrows():
                        f.write(f"{row['year']} {row['day']} {row['tasmin']:.2f} {row['tasmax']:.2f}\n")

if __name__ == "__main__":

    linux = False

    if linux:
        h5_path = "/data/HydroMetData/LOCA/LOCA2_extracted.h5"
    else:
        h5_path = "E:/MyDataBase/climate_change/LOCA2_MLP.h5"

    lat = 43
    lon = -84
    elev = 100
    region = "e_n_cent"
    model = "CNRM-CM6-1-HR"
    scenario = "ssp585"
    ensemble = "r1i1p1f2"

    if scenario == "historical":
        time_ranges = ["1950_2015"]
    else:
        time_ranges = ["2015_2044","2045_2074","2075_2100"]

    time_step = "daily"
    variable = "tmp"

    extractor = PointDataExtractor(h5_path, lat, lon, elev)
    data = extractor.extract_data(region, model, scenario, ensemble, time_step, time_ranges, variable)
    print(f"Extracted data for {lat}, {lon}")
    print(data)
