import os
import pandas as pd

class Performance_evaluator:
    def __init__(self, base_path = "/data/MyDataBase/", LEVEL = "huc12", VPUID = "0000"):
        self.LEVEL = LEVEL
        self.VPUID = VPUID
        self.base_path = os.path.join(base_path, f"SWATplus_by_VPUID/{self.VPUID}/{self.LEVEL}/")


    # Calculates the best keys based on weighted NSE for given criteria
    def get_best_keys(self, df, ET_weight=0.1, Yield_weight=0.1, other_weight=1, number_of_best=5):
        df['weight'] = df['station'].map(lambda x: ET_weight if x == 'ET' else (Yield_weight if x == 'Yield' else other_weight))
        df['weighted_NSE'] = df['weight'] * df['NSE']
        df_grouped = df.groupby('key').sum().reset_index()
        ## dtype must be float
        df_grouped['weighted_NSE'] = df_grouped['weighted_NSE'].astype(float)        
        best_keys = df_grouped.nlargest(number_of_best, 'weighted_NSE')['key'].values
        return best_keys

    # Main function to get the best parameters for a given watershed
    def get_best_solutions(self, name):
        performance_path = f"{self.base_path}/{name}/CentralPerformance.txt"
        parameters_path = f"{self.base_path}/{name}/CentralParameters.txt"

        best_parameters = pd.read_csv(parameters_path, sep='\t')
        performance = pd.read_csv(performance_path, sep='\t')

        best_keys = self.get_best_keys(performance)

        best_parameters = best_parameters[best_parameters['key'].isin(best_keys)]


        return [dict(zip(best_parameters.columns, x)) for x in best_parameters.values]  

# Set the base path and list of watersheds
base_path = "/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/"
names = [name for name in os.listdir(base_path) if name != "log.txt"]

# Create an instance of WatershedAnalyzer and process each watershed
analyzer = Performance_evaluator()
for name in names:
    if name !="04148140":
        continue
    best_parameters = analyzer.get_best_solutions(name)
    print(f"number of best parameters: {len(best_parameters)}")
    break
 
