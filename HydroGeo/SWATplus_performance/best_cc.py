import pandas as pd

import matplotlib.pyplot as plt

class ClimateChangePerformanceAnalyzer:
    def __init__(self, input_file, output_path):
        self.input_file = input_file
        self.output_path = output_path

    def load_data(self):
        self.data = pd.read_csv(self.input_file)
    
    def calculate_performance_metrics(self):
        self.performance_metrics = self.data.drop(columns='ensemble').groupby(['cc_name']).agg({
            'NSE_SWAT_LOCA2': 'mean',
            'MPE_SWAT_LOCA2': 'mean',
            'PBIAS_SWAT_LOCA2': 'mean'
        }).reset_index()
    
    def rank_combinations(self):
        self.performance_metrics['NSE_rank'] = self.performance_metrics['NSE_SWAT_LOCA2'].rank(method='max') ## max is used to rank in descending order like 2, 1, 0
        self.performance_metrics['MPE_rank'] = self.performance_metrics['MPE_SWAT_LOCA2'].rank(method='min') ## min is used to rank in ascending order like 0, 1, 2
        self.performance_metrics['PBIAS_rank'] = self.performance_metrics['PBIAS_SWAT_LOCA2'].abs().rank(method='min') ## min is used to rank in ascending order like 0, 1, 2
        self.performance_metrics['overall_rank'] = self.performance_metrics[['NSE_rank', 'MPE_rank', 'PBIAS_rank']].mean(axis=1)
        self.best_performing_combinations = self.performance_metrics.sort_values('overall_rank')
    
    def save_results(self):
        self.best_performing_combinations.round(2).to_csv(f'{self.output_path}/best_cc_performing_combinations.csv', index=False)
    
    def visualize_results(self):
        top_combinations = self.best_performing_combinations

        plots = [
            ('Overal Rank', 'overall_rank', 'skyblue'),
            ('NSE Rank', 'NSE_rank', 'gray'),
            ('MPE Rank', 'MPE_rank', 'black'),
            ('PBIAS Rank', 'PBIAS_rank', 'gray'),
            ('Average NSE', 'NSE_SWAT_LOCA2', 'black'),
            ('Average MPE', 'MPE_SWAT_LOCA2', 'gray'),
            ('Average Absolute PBIAS', 'PBIAS_SWAT_LOCA2', 'black')
        ]

        for title, column, color in plots:
            fig, ax = plt.subplots(figsize=(12, 12))
            top_values = top_combinations.sort_values(
                column,
                key=abs if 'PBIAS' in title else None,
                ascending="Average NSE" in title,
            )

            ax.barh(top_values['cc_name'], top_values[column].abs() if 'PBIAS' in title else top_values[column], color=color, edgecolor='black')
            ## set xticks size
            ax.tick_params(axis='x', labelsize=15)
            ax.tick_params(axis='y', labelsize=20)
            ax.set_xlabel(title, fontsize=20)
            ax.set_ylabel('Climate Change Models', fontsize=20)
            ax.set_title(f'{title} among all SWAT+ models', fontsize=20)
            plt.tight_layout()
            plt.savefig(f'{self.output_path}/{title.lower().replace(" ", "_")}.png', dpi=300)
            plt.close()

    def run_analysis(self):
        self.load_data()
        self.calculate_performance_metrics()
        self.rank_combinations()
        self.save_results()
        self.visualize_results()

if __name__ == "__main__":
    input_file = 'results/historical_scores_with_baseline_monthly.csv'
    output_path = 'results'
    
    analyzer = ClimateChangePerformanceAnalyzer(input_file, output_path)
    analyzer.run_analysis()
