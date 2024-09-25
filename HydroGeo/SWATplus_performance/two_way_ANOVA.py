import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import seaborn as sns
import matplotlib.pyplot as plt

class ClimateChangeModelAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
    
    def perform_two_way_anova(self, metric):
        try:
            model = ols(f'{metric} ~ C(cc_name) + C(NAME) + C(cc_name):C(NAME)', data=self.data).fit()
            return anova_lm(model, typ=3) ## typ 1 is default and it is used for balanced data, 2 is used for unbalanced data, 3 is used for unbalanced data with unequal sample sizes
        except np.linalg.LinAlgError:
            print(f"LinAlgError: SVD did not converge for {metric}")
            return None
    
    def analyze(self):
        metrics = {
            'NSE': 'NSE_SWAT_LOCA2',
            'PBIAS': 'PBIAS_SWAT_LOCA2',
            'MPE': 'MPE_SWAT_LOCA2'
        }
        
        for name, metric in metrics.items():
            print(f"Two-way ANOVA results for {name}:")
            anova_table = self.perform_two_way_anova(metric)
            if anova_table is not None:
                print(anova_table, "\n")
    

if __name__ == "__main__":
    file_path = 'results/historical_scores_with_baseline_monthly.csv'
    analyzer = ClimateChangeModelAnalyzer(file_path)
    analyzer.analyze()
