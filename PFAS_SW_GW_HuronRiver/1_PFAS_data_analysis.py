import geopandas as gpd
import pandas as pd
import numpy as np
import os
def write_report(message, rep_file):
    with open(rep_file, 'a') as f:
        f.write(message + '\n')

def analyze_pfas_data(pfas_gw_path, ori_pfas_data, rep_file, output_dir) -> None:
    """
    Analyzes PFAS data from groundwater samples and generates a report.

    Args:
        pfas_gw_path (str): Path to the PFAS groundwater data file.
        ori_pfas_data (str): Path to the original PFAS data file.
        rep_file (str): Path to the report file.
        output_dir (str): Directory to save the analysis results.

    Returns:
        None
    """
    os.makedirs("results", exist_ok=True)
    file_path = ori_pfas_data
    ori_wellogic = pd.read_excel(file_path, sheet_name='Wellogic Wells')
    ori_pfas_data = pd.read_excel(file_path, sheet_name='PFAS Sample Locations')
    ori_wssn_population = pd.read_excel(file_path, sheet_name='NTNC Well Locations')
    if os.path.exists(rep_file):
        os.remove(rep_file)
    columns = ori_pfas_data.columns

    write_report(f"### columns in the dataset: {list(columns)}", rep_file)
    write_report("### number of unique WSSN: " + str(ori_pfas_data['WSSN'].nunique()), rep_file)
    write_report("### number of unique SystemType: " + str(ori_pfas_data['SystemType'].nunique()), rep_file)
    write_report("### unique SystemType: " + str(np.unique(ori_pfas_data['SystemType'])), rep_file)
    write_report("### unique TaskCode: " + str(np.unique(ori_pfas_data['TaskCode'])), rep_file)
    write_report("### unique Township: " + str(np.unique(ori_wellogic['Township'])) + " " + str(len(np.unique(ori_wellogic['Township']))), rep_file)
    write_report("### number of unique WSSN with WellStatus = 'Active': " + str(ori_wellogic[ori_wellogic['WellStatus'] == 'ACT']['WSSN'].nunique()), rep_file)
    write_report("### number of unique WSSN with WellStatus = 'INACT': " + str(ori_wellogic[ori_wellogic['WellStatus'] == 'INACT']['WSSN'].nunique()), rep_file)
    write_report("### number of unique WSSN with WellStatus = Unknown: " + str(ori_wellogic[ori_wellogic['WellStatus'].isnull()]['WSSN'].nunique()), rep_file)

    ori_wssn_population['Population'] = ori_wssn_population['Population'].astype(float)
    write_report("### population range: " + str(np.nanmin(ori_wssn_population['Population'])) + ' - ' + str(np.nanmax(ori_wssn_population['Population'])), rep_file)
    write_report('### population median: ' + str(np.nanmedian(ori_wssn_population['Population'])), rep_file)
    write_report('### population mean: ' + str(np.nanmean(ori_wssn_population['Population'])), rep_file)
    write_report('### population std: ' + str(np.nanstd(ori_wssn_population['Population'])), rep_file)

    PFAS = ['HFPODA', 'PFBS', 'PFHxA', 'PFHxS', 'PFNA', 'PFOA',
            'PFOS', 'PF3OUdS11Cl', 'PF3ONS9Cl', 'ADONA', 'FTSA42',
            'FTSA62', 'FTSA82', 'FOSA', 'NEtFOSAA', 'NMeFOSAA',
            'PFBA', 'PFDA', 'PFDoDA', 'PFDS', 'PFHpA', 'PFHpS',
            'PFNS', 'PFPeA', 'PFPeS', 'PFTeDA', 'PFTrDA', 'PFUnDA']

    write_report(f"### total number PFAS species: {len(PFAS)}", rep_file)

    pfas_gw = pd.read_pickle(pfas_gw_path)
    columns = pfas_gw.columns
    results = []
    for pfas in PFAS:
        for column in columns:
            pfas_column = pfas + 'Result'
            if pfas_column == column:
                try:
                    Nulls = pfas_gw[pfas_column].isnull().sum()
                    Zeros = pfas_gw[pfas_column].isin([0]).sum()
                    Positive = pfas_gw[pfas_column][(pfas_gw[pfas_column].notnull()) & (pfas_gw[pfas_column] != 0)].count()
                    Range = (pfas_gw[pfas_column][(pfas_gw[pfas_column].notnull()) & (pfas_gw[pfas_column] != 0)].min(),
                             pfas_gw[pfas_column][(pfas_gw[pfas_column].notnull()) & (pfas_gw[pfas_column] != 0)].max())
                    Mean = pfas_gw[pfas_column][(pfas_gw[pfas_column].notnull()) & (pfas_gw[pfas_column] != 0)].mean()
                    Median = pfas_gw[pfas_column][(pfas_gw[pfas_column].notnull()) & (pfas_gw[pfas_column] != 0)].median()
                    Std = pfas_gw[pfas_column][(pfas_gw[pfas_column].notnull()) & (pfas_gw[pfas_column] != 0)].std()
                    Unique_WSSN = pfas_gw[pfas_gw[pfas_column].notnull() & (pfas_gw[pfas_column] != 0)]['WSSN'].nunique()

                    results.append({
                        "PFAS": pfas,
                        "Nulls": Nulls,
                        "Zeros": Zeros,
                        "Positive": Positive,
                        "Range": Range,
                        "Mean": Mean,
                        "Median": Median,
                        "Std": Std,
                        "Unique_WSSN": Unique_WSSN
                    })

                except Exception as e:
                    write_report(str(e), rep_file)

    results = pd.DataFrame(results)
    results = results.fillna('N/A')
    results.to_csv(output_dir, index=False)


if __name__ == "__main__":
    # Call the function
    pfas_gw_path = "input_data/Huron_PFAS_GW_Features.pkl"
    ori_pfas_data = "input_data/HuronRiver_PFAS_well_data.xlsx"
    os.makedirs("results/PFAS_data_analysis", exist_ok=True)
    rep_file = "results/PFAS_data_analysis/PFAS_GW_data_characteristics.txt"
    output_dir = "results/PFAS_data_analysis/PFAS_GW_data_stats.csv"

    analyze_pfas_data(pfas_gw_path, ori_pfas_data, rep_file, output_dir)
