import xarray as xr
#import matplotlib.pyplot as plt
import pandas as pd
import os
import glob


def loca2_wrapper(rst):
    """loca2_wrapper function to call the main function
    rst: dictionary containing the following
        Args:
        rst['cc_path'] (str): The base directory where the data is stored.
        rst['lat'] (float): Latitude of the location.
        rst['lon'] (float): Longitude of the location.
        rst['model'] (str): rst['model'] name.
        rst['scenario'] (str): rst['scenario'] name.
        rst['ensemble'] (str): rst['ensemble'] name.
        rst['region'] (str): rst['region'] name.
        rst['resolution'] (str): rst['resolution'] name.
        parameter_type (str, optional): Type of data. Defaults to 'pcp'.
    """


    main_preparing_data(rst)

def main_preparing_data(rst):
    """
    Processing temperature data for a given location and saves the time series.

    Returns:
        None
    """

    try:
        if rst['parameter_type'] == "tmp":
            tmp_data = []

            for variable in ["tasmax", "tasmin"]:
                rst['source'] = os.path.join(rst['cc_path'], f"{rst['model']}/{rst['region']}/{rst['resolution']}/{rst['ensemble']}/{rst['scenario']}/{variable}")
                os.chdir(rst['source'])
                files = glob.glob(rst['source'] + "/*.nc")
                to_merge_df = read_file_name(files,rst,variable)
                if len(tmp_data) == 0:
                    tmp_data = to_merge_df
                else:
                    tmp_data = pd.merge(tmp_data, to_merge_df, on=["year", "day"], how="outer")
            tmp_data = tmp_data[["year", "day", "tasmax", "tasmin"]]
            save_time_series(tmp_data, rst)

        elif rst['parameter_type'] == "pcp":

            for variable in ["pr"]:
                rst['source'] = os.path.join(rst['cc_path'], f"{rst['model']}/{rst['region']}/{rst['resolution']}/{rst['ensemble']}/{rst['scenario']}/{variable}")
                os.chdir(rst['source'])
                files = glob.glob(rst['source'] + "/*.nc")
                pcp_data = read_file_name(files,rst,variable)
                pcp_data = pcp_data[["year", "day", "pr"]]
                save_time_series(pcp_data, rst)

        message = f"Data processing for {rst['PRISM_index']} for {rst['parameter_type']} is completed"
    except Exception as e:
        ### remove the nc file
        #os.remove(rst['dataset_dir'])
        message = f"Error in processing {rst['PRISM_index']} for {rst['parameter_type']}: {e}"
    print(message)
    return message

def read_file_name(files,rst,variable):
    to_merge = []
    for file in files:
        if "monthly" in file:
            continue
        _, _, _, _, start_year, end_year, version, _ = return_file_name(file)
        file_name = f"{variable}.{rst['model']}.{rst['scenario']}.{rst['ensemble']}.{start_year}-{end_year}.{version}.{rst['region']}.nc"
        rst['dataset_dir'] = os.path.join(rst['source'], file_name)
        ts = extract_data(rst, variable, start_year, end_year)
        if variable == "pr":
            ## converting kg.m2.s-1 to mm/day
            ts["pr"] = ts["pr"] * 86400
            ## covert the data type to float with 3 decimal points
            ts["pr"] = ts["pr"].astype(float).round(3)
        if variable in ["tasmax", "tasmin"]:
            ## convert kelvin to celsius
            ts[variable] = ts[variable] - 273.15
            ## convert the data type to float with 3 decimal points
            ts[variable] = ts[variable].astype(float).round(3)
        to_merge.append(ts)

    to_merge_df = pd.concat(to_merge, axis=0)
    to_merge_df.reset_index(inplace=True)
    to_merge_df["year"] = to_merge_df["year"].astype(int)
    to_merge_df["day"] = to_merge_df["day"].astype(int)
    to_merge_df = to_merge_df.sort_values(by=["year", "day"])
    to_merge_df.reset_index(drop=True, inplace=True)

    return to_merge_df



def extract_data(rst, variable, start_year, end_year):
    """extract the data from the netcdf file
    Args:
    rst: dictionary containing the following
        rst['dataset_dir'] (str): The path to the NetCDF file.
        variable (str): The variable name.
        start_year (str): The start year.
        end_year (str): The end year."""

    if not os.path.exists(rst['dataset_dir']):
        print("dataset does not exist =  ",rst['dataset_dir'])

    ds = xr.open_dataset(rst['dataset_dir'])

    ds = ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
    ds = ds[variable]
    # Extract a time series at a specific rst['lat']/rst['lon'] location
    ts = ds.sel(lat=rst['lat'], lon=rst['lon'], method='nearest')
    # Convert to a pandas series
    ts = ts.to_series()
    ts = ts.reset_index()
    ts.columns = ['time', variable]
    ts['year'] = ts['time'].dt.year
    ts['day'] = ts['time'].dt.dayofyear
    ts = ts[['year', 'day',variable]]

    return ts


def return_file_name(file):

    file = os.path.basename(file)
    variable     = file.split(".")[0]
    model        = file.split(".")[1]
    scenario     = file.split(".")[2]
    ensemble     = file.split(".")[3]
    start_year   = file.split(".")[4].split("-")[0]
    end_year     = file.split(".")[4].split("-")[1]
    version      = file.split(".")[5]
    region       = file.split(".")[6]
    return variable, model, scenario, ensemble, start_year, end_year, version, region

# Save the pcp time series to a CSV file
def save_time_series(ts, rst):
    """
    Save the time series data to a file.

    Args:
        file_name (str): The name of the file.
        ts (DataFrame): The time series data.
        rst['lat'] (float): The latitude.
        rst['lon'] (float): The longitude.
        index (str): The index.
        rst['scenario'] (str): The rst['scenario'].
        type (str): The type.

    Returns:
        None
    """
    output_dir = rst["output_dir"]

    PRISM_index = rst["PRISM_index"]

    parameter_type = rst["parameter_type"]
    #output_dir = os.path.join(output_dir, rst['model'], rst['scenario'],parameter_type)


    ### create the directory if not exist
    os.makedirs(output_dir, exist_ok=True)

    output_name = f"{PRISM_index}.{parameter_type}"

    full_path = os.path.join(output_dir, output_name)

    ### calculate the number of years in the data
    nbyr = len(ts["year"].unique())
    header_str = f"{rst['region']}.{rst['model']}.{rst['ensemble']}\n\tnbyr\ttstep\tlat\tlon\telev\n\t{nbyr}\t0\t{rst['lat']}\t{rst['lon']}\t{rst['elev']}"

    print(f"Saving time series to {full_path}")
    with open(full_path, 'w') as f:
        f.write(header_str + '\n')


    #before saving make sure there is not gap in the data based on maximum and minimum year
    max_year = ts["year"].max()
    min_year = ts["year"].min()
    datetime = pd.date_range(f"{min_year}-01-01", f"{max_year}-12-31")
    #if there is a gap in the data fill it with -99
    if len(datetime) != len(ts):
        print(f"missing data in PRISM_index:{rst['PRISM_index']} for variable {rst['parameter_type']} between {min_year} and {max_year}")
        #filling the missing data with nan
        ts = ts.set_index(["year", "day"])
        ts = ts.reindex(datetime)
        ts = ts.reset_index()
        ts["year"] = ts["time"].dt.year
        ts["day"] = ts["time"].dt.dayofyear
        ts = ts.drop(columns=["time"])
        ts = ts.fillna(-99)
        ts = ts.reset_index()
        ts = ts.rename(columns={"index":"time"})
        ts = ts[["year", "day", "precipitation_gl"]]

    # Write the DataFrame to the same file without writing the header and index
    ts.to_csv(full_path, mode='a', header=False, index=False, sep='\t')

#"/data/LOCA2/CONUS_regions_split\\EC-Earth3-Veg/e_n_cent/0p0625deg/r1i1p1f1/ssp245/tasmax\\tasmax.EC-Earth3-Veg.ssp245.r1i1p1f1.2075-2100.LOCA_16thdeg_v20220413.e_n_cent.nc"