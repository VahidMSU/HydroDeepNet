import pandas as pd
import numpy as np
import os
def write_report(message, rep_file):
    with open(rep_file, 'a') as f:
        f.write(message + '\n')

def calculate_industry_averages(output_dir, huron_river_sites, rep_file) -> None:
    # Load the dataset
    ss_sites_huron = pd.read_csv(huron_river_sites)
    if os.path.exists(rep_file):
        os.remove(rep_file)
    # Write basic dataset information
    write_report(f"## columns in the dataset: {list(ss_sites_huron.columns)}", rep_file)
    write_report(f"### total number of sites within Huron: {len(ss_sites_huron)}", rep_file)
    write_report(f"number of unique industries: {ss_sites_huron['Industry'].nunique()}", rep_file)
    write_report(f"unique industries: {np.unique(ss_sites_huron['Industry'])}", rep_file)

    # Initialize results dictionary
    results = {
        'Industry': []
    }

    # List of NHDPlus columns
    NHDPlus_columns = [
        "kriging_output_AQ_THK_1_250m", "kriging_output_TRANSMSV_1_250m",
        "melt_rate_raster_250m", 'QAMA_MILP_250m', 'kriging_output_SWL_250m',
        "snow_layer_thickness_raster_250m", "snow_water_equivalent_raster_250m",
        "snow_accumulation_raster_250m", "pden2010_ML_250m", "LC20_SlpP_220_250m",
        "LC20_Elev_220_250m"
    ]

    # Add columns to results dictionary
    for column in NHDPlus_columns:
        results[column] = []

    # Loop through each unique industry
    for industry in np.unique(ss_sites_huron['Industry']):
        write_report(f"number of sites related to {industry}: {len(ss_sites_huron[ss_sites_huron['Industry'] == industry])}", rep_file)
        results['Industry'].append(industry)

        # Loop through each NHDPlus column
        for column in NHDPlus_columns:
            # Drop -999 values
            ss_sites_huron_new = ss_sites_huron[ss_sites_huron[column] != -999]
            # Calculate the average value for the current industry and column
            average_value = ss_sites_huron_new[ss_sites_huron_new['Industry'] == industry][column].mean()
            write_report(f"average {column} for {industry}: {average_value:.2f}", rep_file)
            # Append the result to the results dictionary
            results[column].append(average_value)
        write_report("###", rep_file)

    # Create a DataFrame from the results dictionary
    results_df = pd.DataFrame(results)
    # Write or save the DataFrame as needed
    write_report(results_df.to_string(), rep_file)
    results_df.to_csv(output_dir, index=False)

if __name__ == "__main__":
    output_dir = "results/Industry_HydroGeoDataset_Averages.csv"
    huron_river_sites = "results/SS_Huron_analysis/Huron_PFAS_SITE_Features.csv"
    os.makedirs("results/SS_Huron_characteristics", exist_ok=True)
    rep_file = "results/SS_Huron_characteristics/Industry_HydroGeoDataset_Averages_report.txt"
    calculate_industry_averages(output_dir, huron_river_sites, rep_file)
