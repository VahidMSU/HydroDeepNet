import pandas as pd
import geopandas as gpd
import os
import logging
import numpy as np
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

#### The aim of this script is to analyze the PFAS surface water data
#### The data is stored in a pickle file
#### The script will write the results in a csv file and a text file

def write_report(message, rep_file):
    with open(rep_file, 'a') as f:
        f.write(message + '\n')
path = "/data/MyDataBase/SWATGenXAppData/codes/PFAS_SW_GW_HuronRiver/input_data/Huron_PFAS_SW_Features.pkl"
output_file = "results/PFAS_SW_data_characteristics.csv"
rep_file = "results/PFAS_SW_data_characteristics.txt"

print("====================================")
print("Huron River PFAS Surface Water Data Analysis")
print(f"The output will be writtin in:\n1-{output_file} which contains:\n1- PFAS, 2- Nulls, 3- Zeros, 4- Positive, 5- Range, 6- Mean, 7- Median, 8- Std, 9- Unique_SiteCode\n2-{rep_file} which contains the report")
print("====================================")

if os.path.exists(rep_file):
    os.remove(rep_file)
# Print the column names
pfas_sw = pd.read_pickle(path)

#### report number of records based on LabSampleID
write_report(f"### total number of records: {len(pfas_sw)}", rep_file)
write_report(f"### number of unique LabSampleId: {pfas_sw['LabSampleId'].nunique()}", rep_file)
## report all columns
write_report(f"### columns: {list(pfas_sw.columns)}", rep_file)


def write_results(data, output_file):
    df = pd.DataFrame(data)
    ## use 2 decimal places for the floating point numbers
    df = df.round(2)

    df.to_csv(output_file, index=False, mode='a', header=not os.path.exists(output_file))

# Write average PFAS concentration in the surface water

# PFAS columns are those ending with 'Result'
PFASs = [x for x in pfas_sw.columns if x.endswith('Result')]
write_report(f"### total number of PFAS species: {len(PFASs)}", rep_file)
# Calculate PFAS, Nulls, Zeros, Positive, Range, Mean, Median, Std, Unique_SiteCode
for PFAS in PFASs:
    Nulls = pfas_sw[PFAS].isnull().sum()
    Zeros = pfas_sw[PFAS].isin([0]).sum()
    Positive = pfas_sw[PFAS][(pfas_sw[PFAS].notnull()) & (pfas_sw[PFAS] != 0)].count()
    Range = (pfas_sw[PFAS][(pfas_sw[PFAS].notnull()) & (pfas_sw[PFAS] != 0)].min(),
             pfas_sw[PFAS][(pfas_sw[PFAS].notnull()) & (pfas_sw[PFAS] != 0)].max())
    Mean = pfas_sw[PFAS][(pfas_sw[PFAS].notnull()) & (pfas_sw[PFAS] != 0)].mean()
    Median = pfas_sw[PFAS][(pfas_sw[PFAS].notnull()) & (pfas_sw[PFAS] != 0)].median()
    Std = pfas_sw[PFAS][(pfas_sw[PFAS].notnull()) & (pfas_sw[PFAS] != 0)].std()
    Unique_SiteCode = pfas_sw[pfas_sw[PFAS].notnull() & (pfas_sw[PFAS] != 0)]['SiteCode'].nunique()

    data = {
        'PFAS': [PFAS],
        'Nulls': [Nulls],
        'Zeros': [Zeros],
        'Positive': [Positive],
        'Range': [Range],
        'Mean': [Mean],
        'Median': [Median],
        'Std': [Std],
        'Unique_SiteCode': [Unique_SiteCode]
    }

    write_results(data, output_file)
