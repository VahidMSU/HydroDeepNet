import h5pyd
import numpy as np

def fetch_nsrdb(year, variable, NSRDB_index_SWAT):
    file_path = f'/nrel/nsrdb/v3/nsrdb_{year}.h5'
    print(f"Extracting {variable} for year {year}... ")
    with h5pyd.File(file_path, mode='r') as f:
        data = f[variable][:, NSRDB_index_SWAT]
        print(f"NSRDB data shape before getting attr: {data.shape}")
        scale = f[variable].attrs['psm_scale_factor']


        print("rescaling data....")
        data = np.divide(data, scale)
        data = np.array(data)  # data shape is (17568, len(NSRDB_index_SWAT))
        print(f"NSRDB data shape after scaling: {data.shape}")
        if variable == 'ghi':
            # convert the unit from W/m^2 (30min) to MJ/m^2/day
            data = data.reshape(-1, 48, data.shape[1])  # Reshape to (days, intervals per day, indices)
            data = data * 1800  # Multiply by interval duration in seconds to get energy in J/m²
            daily_data = data.sum(axis=1)  # Sum over intervals to get daily energy
            converter = 1 / 1e6  # Convert J to MJ
            daily_data = daily_data * converter
        elif variable in ['wind_speed', 'relative_humidity']:
            daily_data = data.reshape(-1, 48, data.shape[1]).mean(axis=1)
    print(f"scale factor: {scale}")
    return daily_data

if __name__ == "__main__":
    year = 2018
    NSRDB_index_SWAT = [974980]
    variable = 'ghi'
    data = fetch_nsrdb(year, variable, NSRDB_index_SWAT)
    print(data)
