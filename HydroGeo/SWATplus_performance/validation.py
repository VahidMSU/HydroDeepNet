import os
import pandas as pd

class SWATplusHistoricalReader:
    def __init__(self, base_path, output_path):
        self.base_path = base_path
        if os.path.exists(base_path):
            self.vpuids = os.listdir(base_path)
        else:
            self.vpuids = []
            print(f"Base path {base_path} does not exist.")
        self.output_path = output_path

    def read_hist_scores(self):
        all_historicals_scores = []
        print(self.vpuids)
        for vpuid in self.vpuids:
            huc12_path = os.path.join(self.base_path, vpuid, "huc12")
            if os.path.exists(huc12_path):
                names = os.listdir(huc12_path)
                if "log.txt" in names:
                    names.remove("log.txt")
                print(names)
                for name in names:
                    path = os.path.join(huc12_path, name, "historical_performance_scores.txt")
                    if os.path.exists(path):
                        historical_performance_scores = pd.read_csv(path, sep="\t", skiprows=1)
                        historical_performance_scores["VPUID"] = vpuid
                        historical_performance_scores["NAME"] = name
                        #print(historical_performance_scores.head())
                        all_historicals_scores.append(historical_performance_scores)
                    else:
                        print(f"File not found: {path}")
            else:
                print(f"Directory not found: {huc12_path}")
        if all_historicals_scores:
            self.data = pd.concat(all_historicals_scores).reset_index(drop=True)
        else:
            self.data = pd.DataFrame()

        return self.write_historical_scores(self.data)
    
    def write_historical_scores(self, historical):
        historical = historical.round(2)  # Round numerical columns to 2 decimal places
        
        historical['cc_name'] = historical['SCENARIO'].str.split('_').str[0]
        historical['ensemble'] = historical['SCENARIO'].str.split('_').str[2]
        historical = historical[["VPUID","NAME","station","time_step","cc_name","ensemble","NSE","MPE","PBIAS"]]

        historical.to_csv(os.path.join(self.output_path, "historical_scores.csv"), index=False)
    
    def process_val(self):
        self.read_hist_scores()


def process_validation_scores(base_path, input_file, model_bounds_path):
    # Load the validation scores CSV file
    file_path = os.path.join(base_path, input_file)
    validation_scores = pd.read_csv(file_path, dtype={'NSE': 'float32', 'MPE': 'float32', 'PBIAS': 'float32'})
    validation_scores['NAME'] = validation_scores['NAME'].astype(str)
    validation_scores = validation_scores.drop(columns=['SCENARIO'])
    
    # Select the best model based on max NSE
    summary_table = validation_scores.sort_values(by='NSE', ascending=False).groupby(['NAME', 'time_step', 'station']).first().reset_index()

    summary_table = summary_table.round(2)
    
    # Save the summary table to CSV files
    monthly_data = summary_table[summary_table.time_step == 'Monthly']
    monthly_data.to_csv(os.path.join(base_path, 'validation_summary_monthly.csv'), index=False)

    daily_data = summary_table[summary_table.time_step == 'Daily']
    daily_data.to_csv(os.path.join(base_path, 'validation_summary_daily.csv'), index=False)

    # Drop the 'station' column and group by 'NAME' to calculate mean
    monthly_data = monthly_data.drop(columns='station')
    daily_data = daily_data.drop(columns='station')

    watershed_monthly = monthly_data.groupby(['NAME'])[['NSE', 'MPE', 'PBIAS']].mean().reset_index()
    watershed_monthly.to_csv(os.path.join(base_path, 'watershed_monthly.csv'), index=False)

    watershed_daily = daily_data.groupby(['NAME'])[['NSE', 'MPE', 'PBIAS']].mean().reset_index()
    watershed_daily.to_csv(os.path.join(base_path, 'watershed_daily.csv'), index=False)

    # Load model_bounds
    model_bounds = pd.read_pickle(model_bounds_path).to_crs(epsg=4326)
    model_bounds['NAME'] = model_bounds['NAME'].astype("int64")
    watershed_monthly['NAME'] = watershed_monthly['NAME'].astype("int64")
    watershed_daily['NAME'] = watershed_daily['NAME'].astype("int64")

    # Merge model_bounds with validation scores
    watershed_monthly_shape = model_bounds[model_bounds.LEVEL.isin(['huc12'])].merge(watershed_monthly, on='NAME', how='left')
    watershed_monthly_shape.to_file(os.path.join(base_path, 'model_bounds_monthly'))

    watershed_daily_shape = model_bounds[model_bounds.LEVEL.isin(['huc12'])].merge(watershed_daily, on='NAME', how='left')
    watershed_daily_shape.to_file(os.path.join(base_path, 'model_bounds_daily'))

    # Average monthly and daily data
    watershed_avg = pd.merge(watershed_monthly_shape, watershed_daily_shape.drop(columns=['geometry']), on='NAME', suffixes=('_monthly', '_daily'))
    watershed_avg['NSE'] = (watershed_avg['NSE_monthly'] + watershed_avg['NSE_daily']) / 2
    watershed_avg['MPE'] = (watershed_avg['MPE_monthly'] + watershed_avg['MPE_daily']) / 2
    watershed_avg['PBIAS'] = (watershed_avg['PBIAS_monthly'] + watershed_avg['PBIAS_daily']) / 2
    watershed_avg.to_file(os.path.join(base_path, 'model_bounds_avg'))

    # Save model bounds for huc8
    model_bounds[model_bounds.LEVEL.isin(['huc8'])].to_file(os.path.join(base_path, 'model_bounds_huc8'))




class SWATplusReader:
    def __init__(self, base_path, output_path):
        self.base_path = base_path
        if os.path.exists(base_path):
            self.vpuids = os.listdir(base_path)
        else:
            self.vpuids = []
            print(f"Base path {base_path} does not exist.")
        self.output_path = output_path

    def read_val_scores(self):
        all_verifications_scores = []
        print(self.vpuids)
        for vpuid in self.vpuids:
            huc12_path = os.path.join(self.base_path, vpuid, "huc12")
            if os.path.exists(huc12_path):
                names = os.listdir(huc12_path)
                if "log.txt" in names:
                    names.remove("log.txt")
                print(names)
                for name in names:
                    path = os.path.join(huc12_path, name, "verification_performance_scores.txt")
                    if os.path.exists(path):
                        verification_performance_scores = pd.read_csv(path, sep="\t", skiprows=1)
                        verification_performance_scores["VPUID"] = vpuid
                        verification_performance_scores["NAME"] = name
                        all_verifications_scores.append(verification_performance_scores)
                    else:
                        print(f"File not found: {path}")
            else:
                print(f"Directory not found: {huc12_path}")
        if all_verifications_scores:
            self.data = pd.concat(all_verifications_scores).reset_index(drop=True)
        else:
            self.data = pd.DataFrame()

        return self.write_val_scores(self.data)
    
    def write_val_scores(self, ver_results):
        ver_results = ver_results.round(2)  # Round numerical columns to 2 decimal places
        
        ver_results = ver_results[["VPUID","NAME","station","time_step","SCENARIO","NSE","MPE","PBIAS"]]
        ver_results.to_csv(os.path.join(self.output_path, "validation_scores.csv"), index=False)
        ver_results = ver_results.drop(columns=['SCENARIO'])
        ver_results = ver_results.groupby(['VPUID', 'NAME', 'time_step', 'station']).mean().reset_index()
        ver_results = ver_results.round(2)
        ver_results.to_csv(os.path.join(self.output_path, "validation_summary_average_base_line.csv"), index=False)
    def process_val(self):
        self.read_val_scores()


def merge_validation_with_historical(OUTPUT_PATH):
    validation = f"{OUTPUT_PATH}/validation_summary_average_base_line.csv"
    historical = f"{OUTPUT_PATH}/historical_scores.csv"

    historical = pd.read_csv(historical)
    validation = pd.read_csv(validation)
    validation = validation[validation.NSE>-10]

    historical = historical.merge(validation, on=['VPUID', 'NAME', 'time_step', 'station'], suffixes=('_SWAT_LOCA2', '_SWAT_PRISM'))

    historical[historical['time_step'] == 'Monthly'].to_csv(f"{OUTPUT_PATH}/historical_scores_with_baseline_monthly.csv", index=False)
    historical[historical['time_step'] == 'Daily'].to_csv(f"{OUTPUT_PATH}/historical_scores_with_baseline_daily.csv", index=False)


if __name__ == "__main__":
    BASE_PATH = "/data/MyDataBase/SWATplus_by_VPUID"
    OUTPUT_PATH = "/home/rafieiva/MyDataBase/codes/SWAT-CONUS/HydroGeoDataBase/SWATplus_performance/results/"
    SWATplusReader(BASE_PATH, OUTPUT_PATH).process_val()   

    input_file = 'validation_scores.csv'
    model_bounds_path = '/home/rafieiva/MyDataBase/codes/SWAT-CONUS/HydroGeoDataBase/SWATplus_performance/input_shape/model_bounds.pkl'

    process_validation_scores(OUTPUT_PATH, input_file, model_bounds_path)

    SWATplusHistoricalReader(BASE_PATH, OUTPUT_PATH).process_val()

    merge_validation_with_historical(OUTPUT_PATH)
