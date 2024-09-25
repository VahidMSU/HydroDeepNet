import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

class ClimateChangeModelAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
    
    def perform_anova(self, metric):
        return stats.f_oneway(*(self.data[self.data['cc_name'] == model][metric] for model in self.data['cc_name'].unique()))
    
    def analyze(self):
        metrics = {
            'NSE': 'NSE_SWAT_LOCA2',
            'PBIAS': 'PBIAS_SWAT_LOCA2',
            'MPE': 'MPE_SWAT_LOCA2'
        }
        
        for name, metric in metrics.items():
            print(f"ANOVA results for {name}:")
            anova_result = self.perform_anova(metric)
            print(f"F-value: {anova_result.statistic}, p-value: {anova_result.pvalue}\n")


if __name__ == "__main__":
    file_path = 'results/historical_scores_with_baseline_monthly.csv'
    analyzer = ClimateChangeModelAnalyzer(file_path)
    analyzer.analyze()
