import pandas as pd
import numpy as np

# Load the data
pfas_gw = pd.read_pickle("/data/MyDataBase/SWATGenXAppData/codes/PFAS_SW_GW_HuronRiver/input_data/Huron_PFAS_GW_Features.pkl")
pfas_sw = pd.read_pickle("/data/MyDataBase/SWATGenXAppData/codes/PFAS_SW_GW_HuronRiver/input_data/Huron_PFAS_SW_Features.pkl")
pfas_sites = pd.read_pickle("/data/MyDataBase/SWATGenXAppData/codes/PFAS_SW_GW_HuronRiver/input_data/Huron_PFAS_SITE_Features.pkl")

# Display the first few rows of each dataset to understand their structure
#print(f"####################Groundwater data columns:\n {list(pfas_gw.columns)}")
#print(f"####################Surface water data columns:\n {list(pfas_sw.columns)}")
#print(f"####################Site data columns:\n {list(pfas_sites.columns)}")

# Function to calculate positive, null, and zero ratios for each industry
def calculate_pfas_ratios(pfas_data, sites_data, media_type):
    # Filter PFAS result columns
    pfas_columns = [col for col in pfas_data.columns if col.endswith('Result')]

    # Melt the PFAS data to long format
    import time
    print(f"pfas_data columns: {pfas_data.columns}")
    ### set SiteCode as index
    assert 'SiteCode' in pfas_data.columns, "SiteCode is not in the columns"
    pfas_long = pfas_data.melt(id_vars=['SiteCode'], value_vars=pfas_columns, var_name='PFAS_Compound', value_name='PFAS_Concentration')

    # Merge with site data to get industry information
    pfas_long = pfas_long.merge(sites_data[['SiteCode', 'Industry']], on='SiteCode', how='left')

    # Calculate counts for positive, null, and zero concentrations
    pfas_summary = pfas_long.groupby('Industry').agg(
        total_records=('PFAS_Concentration', 'size'),
        positive_records=('PFAS_Concentration', lambda x: (x > 0).sum()),
        null_records=('PFAS_Concentration', lambda x: x.isnull().sum()),
        zero_records=('PFAS_Concentration', lambda x: (x == 0).sum())
    ).reset_index()

    # Calculate the ratio of positive to (null + zero)
    pfas_summary['ratio_positive_to_null_zero'] = pfas_summary['positive_records'] / (pfas_summary['null_records'] + pfas_summary['zero_records'])

    # Sort by the ratio
    pfas_summary = pfas_summary.sort_values(by='ratio_positive_to_null_zero', ascending=False)

    # Save to CSV
    os.makedirs("results/Significance_analysis", exist_ok=True) # Create the directory if it doesn't exist
    pfas_summary.to_csv(f"results/Significance_analysis/{media_type}_PFAS_ratios_by_industry.csv", index=False)

    return pfas_summary

# Calculate ratios for groundwater and surface water
gw_ratios = calculate_pfas_ratios(pfas_gw, pfas_sites, 'GW')
sw_ratios = calculate_pfas_ratios(pfas_sw, pfas_sites, 'SW')

# Display the results
print("Groundwater PFAS Ratios by Industry:")
print(gw_ratios)
print("\nSurface Water PFAS Ratios by Industry:")
print(sw_ratios)
