import h5py
import numpy as np
import os
import pandas as pd 
import re
from mpi4py import MPI
def get_soil_data(NAME= "04111379", VPUID="0000"):
    file_path = f'/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/SWAT_gwflow_MODEL/Scenarios/Default/TxtInOut/soils.sol'

    # Read the file content
    with open(file_path, 'r') as file:
        file_content = file.readlines()
        df = pd.DataFrame(columns=['soil_id', 'lyr', 'dp', 'bd', 'awc', 'soil_k', 'carbon', 'clay', 'silt'])
        row = 0
        for i, line in enumerate(file_content):
            if line.startswith('name'):
                headers = line.split()
            if re.match(r'\d', line):
                parts = line.split()
                soil_id = parts[0]
                nly = int(parts[1])
                for j in range(nly):
                    next_line = file_content[i+j+1].split()  # Corrected to access the next lines correctly
                    df.loc[row, 'soil_id'] = soil_id
                    df.loc[row, 'lyr'] = j+1
                    df.loc[row, 'dp'] = next_line[1]
                    df.loc[row, 'bd'] = next_line[2]
                    df.loc[row, 'awc'] = next_line[3]
                    df.loc[row, 'soil_k'] = next_line[4]
                    df.loc[row, 'carbon'] = next_line[5]
                    df.loc[row, 'clay'] = next_line[6]
                    df.loc[row, 'silt'] = next_line[7]
                    row += 1

    ## group by soil id and get the average of each column
    df = df.apply(pd.to_numeric, errors='ignore')
    df = df.groupby('soil_id').mean().reset_index()
    print(df)
    return df
def organize_data(pickled_data_path, NAME, VPUID):
    # Load the data
    data = pd.read_pickle(pickled_data_path)
    static_features = data.iloc[:, 0:54].columns

    categorical_features = []
    numerical_features = []

    # Separate categorical and numerical features
    for feature in static_features:
        if feature in ['model_name', 'cc_name', 'vpu_id', 'lu_mgt', 'soil', 'name']:
            categorical_features.append(feature)
            data[feature] = data[feature].astype('category')
        else:
            try:
                data[feature] = pd.to_numeric(data[feature])
                numerical_features.append(feature)
            except Exception:
                categorical_features.append(feature)
                data[feature] = data[feature].astype('category')

    num_years = 83
    years = list(range(2018, 2018 + num_years))
    months = list(range(1, 13))

    # Create an empty list to store all rows
    all_data = []

    # Expand static features for each year and month
    expanded_static = pd.concat([data[static_features]] * len(years) * len(months), ignore_index=True)

    # Create DataFrame for dynamic features and targets
    dynamic_df = []
    target_df = []

    for year in years:
        for month in months:
            print(f"Processing year {year} and month {month}...")
            # Extract dynamic features
            dynamic_feat = data[[f"precip_{year}_{month}"]].rename(columns={f"precip_{year}_{month}": "precip"})
            dynamic_df.append(dynamic_feat)

            # Extract target values
            target_values = data[[f"perc_{year}_{month}"]].rename(columns={f"perc_{year}_{month}": "target"})
            target_df.append(target_values)

    # Concatenate dynamic features and targets
    dynamic_df = pd.concat(dynamic_df, ignore_index=True)
    target_df = pd.concat(target_df, ignore_index=True)

    # Create month and year columns
    num_hru = len(data)
    month_column = months * num_hru * num_years
    year_column = [year for year in years for _ in range(num_hru * len(months))]

    # Combine all data into a single DataFrame
    all_data = pd.concat([expanded_static.reset_index(drop=True),
                          pd.Series(month_column, name="month"),
                          pd.Series(year_column, name="year"),
                          dynamic_df.reset_index(drop=True),
                          target_df.reset_index(drop=True)], axis=1)

    # Save the resulting DataFrame to an HDF5 file
    print("saving data...with shape", all_data.shape)

    pickled_data_path_final = os.path.join("/data/MyDataBase/SWAT_ML", os.path.basename(pickled_data_path).replace("SWAT_OUTPUT", "ORGANIZED"))    
    all_data.to_pickle(pickled_data_path_final)
    os.remove(pickled_data_path)

class SWAT_OUTPUT_ML:
    def __init__(self, VPUID, NAME):
        self.VPUID = VPUID
        self.NAME = NAME
        self.cc_base_path =  f'/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/climate_change_models/'
        self.hru_con = f'/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/SWAT_gwflow_MODEL/Scenarios/Default/TxtInOut/hru.con'
        self.topography =  f"/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/SWAT_gwflow_MODEL/Scenarios/Default/TxtInOut/topography.hyd"
        self.cal_file =  f"/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/best_solution_SWAT_gwflow_MODEL.txt"
        self.file_name = "SWAT_OUTPUT.h5"
        self.hru_con_df = pd.read_csv(self.hru_con, sep='\s+',  skiprows=1)[['name','gis_id','area','lat','lon','elev']] ## hru
        self.topography_df = pd.read_csv(self.topography, sep='\s+',  skiprows=1)[["name","slp","slp_len","lat_len"]].rename(columns={'name':'topo'}) ## topo
        self.cal_df = pd.read_csv(self.cal_file, sep=',', header=None, names=['par_name', 'value'])[:-1]
        self.start_year = 2015
        self.num_years = None

    def integration_and_save(self, perc, precip, soil, lu_mgt, name, topo, cc_name, NAME, VPUID):
        ## create a pandas dataframe
        df = pd.DataFrame()
        df['name'] = [x.decode('utf-8') for x in name]
        df['soil'] = [x.decode('utf-8') for x in soil]
        df['lu_mgt'] = [x.decode('utf-8') for x in lu_mgt]
        df['topo'] = [x.decode('utf-8') for x in topo]
        df['cc_name'] = cc_name
        df['model_name'] = NAME
        df['vpu_id'] = VPUID
        print(f"length of perc: {len(perc)}")
        soil_char = get_soil_data(NAME, VPUID)
        soil_char['soil_id'] = soil_char['soil_id'].astype(str)
        df = df.merge(soil_char, left_on='soil', right_on='soil_id')
        df = df.merge(self.hru_con_df, on='name')
        print(f"length of df: {len(df)}")
        df = df.merge(self.topography_df, on='topo')
        print(f"length of df: {len(df)}")

        for i in range(len(self.cal_df)):
            new_col_name = self.cal_df.iloc[i, 0]  
            new_col_value = self.cal_df.iloc[i, 1] 
            df[new_col_name] = new_col_value

        ## now merge df with perc
        df = df.merge(perc, on='name')
        print(f"length of df: {len(df)}")
        Machine_learning_data = df.merge(precip, on='name')
        print(f"length of df: {len(Machine_learning_data)}")
        Machine_learning_data.drop(columns=['name', 'gis_id', 'topo']).to_pickle(f'/data/MyDataBase/SWAT_ML/SWAT_OUTPUT_{NAME}_{cc_name}.pkl')
        print(f"length of df: {len(Machine_learning_data)}")

    def prepare_dynamic_features(self, data, var_name, name):
        
        data = pd.DataFrame(data, columns = [x.decode('utf-8') for x in name])
        data = data.T
        # Calculate the number of years based on the number of columns
        self.num_years = len(data.columns) // 12

        # Generate the target array names for each month across all years
        target_array_name = [
            f'{var_name}_{year}_{month}'
            for year in range(self.start_year + 3, self.start_year+3 + self.num_years)
            for month in range(1, 13)
        ]

        # Assign the new column names to the DataFrame
        data.columns = target_array_name
        data = data.reset_index()
        data = data.rename(columns={'index':'name'})

        return data
    
    def process_swat_output(self):
        for cc_name in os.listdir(self.cc_base_path):
            h5_path = os.path.join(self.cc_base_path, cc_name, self.file_name)
            if os.path.exists(h5_path):
                try:
                    h5 = h5py.File(h5_path, 'r')
                except Exception:
                    print(f"Error reading {h5_path}")
                    continue
                try:
                    print(h5['hru_wb/mon'].keys())
                    perc = h5['hru_wb/mon/perc'] # mon, hru
                    precip = h5['hru_wb/mon/precip']
                    soil = h5['metadata/hru/soil'] # hru
                    lu_mgt = h5['metadata/hru/lu_mgt'] # hru
                    name = h5['metadata/hru/name'] # hru
                    topo = h5['metadata/hru/topo'] # hru
                except Exception:
                    print(f"Error reading {h5_path}")
                    h5.close()
                    continue
                        
                perc = self.prepare_dynamic_features(perc, 'perc', name)

                precip = self.prepare_dynamic_features(precip, 'precip', name)
                self.integration_and_save(perc, precip, soil, lu_mgt, name, topo, cc_name, self.NAME, self.VPUID)
                pickled_data_path = f"/data/MyDataBase/SWAT_ML/SWAT_OUTPUT_{self.NAME}_{cc_name}.pkl"
                organize_data(pickled_data_path=pickled_data_path, NAME=self.NAME, VPUID=self.VPUID)

def parallel_process_swat_output(names_vpuids, rank, size):
    for i, (VPUID, NAME) in enumerate(names_vpuids):
        if i % size != rank:
            continue  # Skip NAMES not assigned to this process
        if VPUID == '0000':
            continue
        SWAT_OUTPUT_ML(VPUID, NAME).process_swat_output()
        print(f"Process {rank} completed processing {NAME} in {VPUID}")
        
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # Only the root process reads the directory
        VPUIDS = os.listdir('/data/MyDataBase/SWATplus_by_VPUID')
        names_vpuids = [(VPUID, NAME) for VPUID in VPUIDS for NAME in os.listdir(f'/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12') if NAME != 'log.txt']
        print(f"Root process distributing {len(names_vpuids)} NAMES among {size} processes")
    else:
        names_vpuids = None

    # Broadcast NAMES list to all processes
    names_vpuids = comm.bcast(names_vpuids, root=0)
    print(f"Process {rank} received {len(names_vpuids)} NAMES to process")

    # Each process processes its subset of NAMES
    parallel_process_swat_output(names_vpuids, rank, size)
    print(f"Process {rank} completed its tasks")
