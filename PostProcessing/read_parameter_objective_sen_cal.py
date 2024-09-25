import os
import pandas as pd
import numpy as np
def find_VPUID(station_no):
	CONUS_streamflow_data = pd.read_csv("/data/MyDataBase/CIWRE-BAE/USGS/streamflow_stations/CONUS/streamflow_stations_CONUS.csv", dtype={'site_no': str,'huc_cd': str})
	return CONUS_streamflow_data[
		CONUS_streamflow_data.site_no == station_no
	].huc_cd.values[0][:4]

def append_new_df(df, new_df):
    ## sort by column names
    df = df.reindex(sorted(df.columns), axis=1)
    new_df = new_df.reindex(sorted(new_df.columns), axis=1)

    ## check if the columns are the same
    if not df.columns.equals(new_df.columns):
        print("Columns are not the same")
        print(f"df columns: {df.columns.tolist()}")
        print(f"new_df columns: {new_df.columns.tolist()}")
        return df
    ## concatenate the new df
    df = pd.concat([df, new_df], ignore_index=True)
    return df

def load_parameters_models_performance(base_path, vpuid):
    all_data = pd.DataFrame()  # Initialize as an empty DataFrame
    names = os.listdir(f'{base_path}/{vpuid}/huc12/')
    for name in names:
        path = f"{base_path}/{vpuid}/huc12/{name}/"
        initial_points_path = os.path.join(path, "initial_points_SWAT_gwflow_MODEL.csv")
        initial_values_path = os.path.join(path, "initial_values_SWAT_gwflow_MODEL.csv")

        if not os.path.exists(initial_points_path) or not os.path.exists(initial_values_path):
            continue

        parameters = pd.read_csv(f"{path}/cal_parms_SWAT_gwflow_MODEL.cal", skiprows=1, sep="\s+")

        # Ensure there are no duplicate column names
        parameter_names = parameters['name'].tolist()
        if parameter_names.count('4_thickness_sb') > 1:
            first_index = parameter_names.index('4_thickness_sb')
            second_index = parameter_names.index('4_thickness_sb', first_index + 1)
            parameter_names[second_index] = '4_thickness_sb_2'
            parameter_names[first_index] = '4_thickness_sb_1'

        initial_points = np.loadtxt(initial_points_path, delimiter=",")
        initial_values = np.loadtxt(initial_values_path, delimiter=',')
        try:
            df = pd.DataFrame(initial_points, columns=parameter_names)
        except Exception as e:
            print(f"Error reading {initial_points_path}: {e}")
            continue
        if "4_thickness_sb_2" in df.columns:
            ### now drop 4_thinkness_sb_2 and rename 4_thickness_sb_1 to 4_thickness_sb
            df.drop(columns=['4_thickness_sb_2'], inplace=True)
            df.rename(columns={'4_thickness_sb_1': '4_thickness_sb'}, inplace=True)

        df['best_score'] = initial_values
        df['NAME'] = name
        df['VPUID'] = vpuid
        df['stage'] = 'sen'
        all_data = df if all_data.empty else append_new_df(all_data, df)
        best_scores_df = pd.read_csv(f'{path}/local_best_solution_SWAT_gwflow_MODEL.txt', sep=",")
        best_scores_df.rename(columns={'4_thickness_sb.1': '4_thickness_sb_2', '4_thickness_sb': '4_thickness_sb_1'}, inplace=True)
        best_scores_df['NAME'] = name
        best_scores_df['VPUID'] = find_VPUID(name)
        best_scores_df['stage'] = 'cal'
        ## similarly get rid of 4_thickness_sb_2 and rename 4_thickness_sb_1 to 4_thickness_sb
        if "4_thickness_sb_2" in best_scores_df.columns:
            best_scores_df.drop(columns=['4_thickness_sb_2'], inplace=True)
            best_scores_df.rename(columns={'4_thickness_sb_1': '4_thickness_sb'}, inplace=True)
            
        print(f"Number of columns in df: {len(df.columns)}", f"Number of columns in best_scores_df: {len(best_scores_df.columns)}")
        all_data = append_new_df(all_data, best_scores_df)

    if all_data.empty:
        raise ValueError("No valid data frames to concatenate")

    return all_data

if __name__ == "__main__":
    base_path = "/data/MyDataBase/SWATplus_by_VPUID"
    vpuid = "0000"
    df = load_parameters_models_performance(base_path, vpuid)
    if df is not None and not df.empty:
        df.to_csv("overal_best_performance/sensitivity_analysis_par_val.csv", index=False)
    else:
        print("No data found to process")
