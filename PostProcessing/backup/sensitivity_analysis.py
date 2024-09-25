import matplotlib.pyplot as plt
import os
import pandas as pd
import tabulate
import seaborn as sns
import numpy as np

class SensitivityAnalysis:
    def __init__(self, model_name, level, base_dir):
        self.model_name = model_name
        self.level = level
        self.base_dir = base_dir
        self.sensitivity_dir = "Morris_Sensitivity_Analysis/"
        self.names = [name for name in os.listdir(base_dir) if name != 'log.txt']

    def fetching_best_parameters(self):
        """Fetches the best parameters for each model and returns a dataframe with the best parameters for each model"""
        all_best_par = []
        
        for name in self.names:
            source_path = os.path.join(self.base_dir, name)
            best_par_path = os.path.join(source_path, f'best_solution_{self.model_name}.txt')
            if not os.path.exists(best_par_path):
                continue
            parameters = []
            with open(best_par_path, 'r') as file:
                lines = file.readlines()
                parameters.extend(
                    [line.split(',')[0], float(line.split(',')[1][:-1])]
                    for line in lines[:-1]
                )
            best_par = pd.DataFrame(parameters, columns=['names', 'best_value'])
            best_par = best_par.assign(NAME=name)
            all_best_par.append(best_par)

        return pd.concat(all_best_par)

    def create_table_figure(self, df, filename, index_title):
        """Creates a table figure from a dataframe and saves it as a jpeg file"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        table_data = df.reset_index().values
        col_labels = [index_title] + df.columns.tolist()
        table = ax.table(cellText=table_data, colLabels=col_labels, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.auto_set_column_width(col=list(range(len(col_labels))))
        plt.savefig(filename, dpi=300)
        plt.close()

    def ranking_across_models(self, sensitivity_all):
        """Ranks the sensitivity of each parameter across all models and returns a dataframe with the average sensitivity ranking for each parameter"""
        rank_analysis = sensitivity_all.groupby('names')['RANK'].agg(['mean'])
        rank_analysis['top_five_freq'] = sensitivity_all[sensitivity_all['RANK'] <= 5].groupby('names').size()
        rank_analysis['bottom_ten_freq'] = sensitivity_all[sensitivity_all['RANK'] > int(len(np.unique(sensitivity_all['RANK'])) - 10)].groupby('names').size()
        rank_analysis.sort_values('mean', inplace=True)
        rank_analysis['mean'] = rank_analysis['mean'].round()
        
        os.makedirs(self.sensitivity_dir, exist_ok=True)
        rank_analysis.to_csv(os.path.join(self.sensitivity_dir, f"Table_Average_Sensitivity_Ranking_{self.model_name}.csv"))

        pretty_table = tabulate.tabulate(rank_analysis, headers=rank_analysis.columns, tablefmt="pretty")
        print(pretty_table)

        self.create_table_figure(rank_analysis, os.path.join(self.sensitivity_dir, f"Table_Average_Sensitivity_Ranking_{self.model_name}.jpeg"), index_title='Parameter')

        plt.figure(figsize=(15, 15))
        sensitivity_all = sensitivity_all.sort_values('RANK', ascending=False)
        sns.boxplot(y='names', x='RANK', data=sensitivity_all, color='skyblue')
        plt.title('Distribution of Sensitivity Ranks for Each Parameter', fontsize=18)
        plt.ylabel('Parameter Names', fontsize=18)
        plt.xlabel('Sensitivity Ranks', fontsize=18)
        plt.yticks(fontsize=18)
        plt.xticks(fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(self.sensitivity_dir, f"Plot_Average_Sensitivity_Ranking_{self.model_name}.jpeg"), dpi=300)
        plt.close()

        return rank_analysis

    def fetching_sensitivity_results(self):
        """Fetches the sensitivity results for each model and returns a dataframe with the sensitivity results for each model"""
        sensitivity_all = []
        for name in self.names:
            best_performance_path = os.path.join(self.base_dir, name, f'best_solution_{self.model_name}.txt')
            if os.path.exists(best_performance_path):
                try:
                    df = pd.read_csv(os.path.join(self.base_dir, name, f'morris_Si_{self.model_name}.csv'))
                    df = (df.sort_values('mu_star', ascending=False)
                            .reset_index(drop=True)
                            .assign(RANK=np.arange(1, len(df) + 1),
                                    NAME=name))
                    sensitivity_all.append(df)
                except Exception as e:
                    print(f'{name}: {e}')
            else:
                print(f'{name} best parameter&performance file does not exist')
                
        return pd.concat(sensitivity_all, ignore_index=True) if sensitivity_all else pd.DataFrame()

    def generate_sensitivity_plots(self):
        """Generates the sensitivity ranking for each parameter for each model and saves the plots as jpeg files"""
        sensitivity_all = self.fetching_sensitivity_results()
        all_best_par = self.fetching_best_parameters()
        sensitivity_all = sensitivity_all.merge(all_best_par, on=['names', 'NAME'])
        rank_analysis = self.ranking_across_models(sensitivity_all)

        for name in self.names:
            specific_df = sensitivity_all[sensitivity_all.NAME == name]
            specific_df_sorted = specific_df.sort_values('mu_star', ascending=True)
            if len(specific_df_sorted) == 0:
                continue
            plt.figure(figsize=(10, 8))
            plt.barh(specific_df_sorted['names'], specific_df_sorted['mu_star'], color='skyblue')
            plt.xlabel('Morris Sensitivity Measure ($\mu^*$)')
            plt.ylabel('Parameter Names')
            plt.title(f'{self.model_name} {name}')
            plt.tight_layout()
            os.makedirs('Morris_Sensitivity_Analysis/individual_models/', exist_ok=True)
            plt.savefig(fr'Morris_Sensitivity_Analysis/individual_models/{name}_sensitivity_mu_star_ranking_{self.model_name}.jpeg', dpi=300)
            plt.close()

if __name__ == "__main__":
    model_name = 'SWAT_gwflow_MODEL'
    level = 'huc12'
    base_dir = r'/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/'

    sensitivity_analysis = SensitivityAnalysis(model_name, level, base_dir)
    sensitivity_analysis.generate_sensitivity_plots()

    print('Sensitivity analysis plots are generated')

