import os
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
import tabulate


class SensitivityAnalysis:
    
    """Class to perform sensitivity analysis on a set of models
    Attributes:
        model_name (str): The name of the model
        level (str): The level of the model
        base_dir (str): The base directory containing the models
        sensitivity_dir (str): The directory to save sensitivity analysis results
        names (list): List of model names
        all_sen (list): List to store all sensitivity data
    
    Methods:
        load_data: Loads the sensitivity data for all models
        fetching_best_parameters: Fetches the best parameters for each model and returns a dataframe with the best parameters for each model
        create_table_figure: Creates a table figure from a dataframe and saves it as a jpeg file
        ranking_across_models: Ranks the sensitivity of each parameter across all models and returns a dataframe with the average sensitivity ranking for each parameter
        fetching_sensitivity_results: Fetches the sensitivity results for each model and returns a dataframe with the sensitivity results for each model
        analyze: Performs ranking and analysis on the loaded sensitivity data
        save_results: Saves the analyzed results to a CSV file
        plot_sensitivity: Plots a radar chart of sensitivity analysis results
        perform_mixed_effects_model: Performs a mixed-effects model analysis on the sensitivity data
        generate_sensitivity_plots: Generates the sensitivity ranking for each parameter for each model and saves the plots as jpeg files
    
    """

    def __init__(self, model_name, level, base_dir):
        self.model_name = model_name
        self.level = level
        self.base_dir = base_dir
        self.sensitivity_dir = "Morris_Sensitivity_Analysis/"
        self.names = [name for name in os.listdir(base_dir) if name != 'log.txt']
        self.all_sen = []

    def load_data(self) -> None:
        """Loads the sensitivity data for all models"""
        for name in self.names:
            morris_path = os.path.join(self.base_dir, name, f'morris_Si_{self.model_name}.csv')
            if os.path.exists(morris_path):
                morris = pd.read_csv(morris_path, index_col=0)
                morris['NAME'] = name
                morris = morris.sort_values(by='mu_star', ascending=False)
                morris['rank'] = range(1, len(morris) + 1)
                self.all_sen.append(morris)
        self.all_sen = pd.concat(self.all_sen)

    def fetching_best_parameters(self) -> pd.DataFrame:
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

    def create_table_figure(self, df, filename, index_title) -> None:
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

    def ranking_across_models(self, sensitivity_all) -> pd.DataFrame:
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

    def fetching_sensitivity_results(self) -> pd.DataFrame:
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

    def analyze(self) -> None:
        """Performs ranking and analysis on the loaded sensitivity data"""
        # Calculate total count
        total_count = self.all_sen.groupby('names').size().reset_index(name='total_count')

        # Count top 10
        top10 = self.all_sen[self.all_sen['rank'] <= 10]
        top10_count = top10.groupby('names').size().reset_index(name='top10_count')

        # Count bottom 10 (assuming rank > 27 is considered bottom 10)
        bottom10 = self.all_sen[self.all_sen['rank'] > 27]
        bottom10_count = bottom10.groupby('names').size().reset_index(name='bottom10_count')

        # Get average rank
        avg_rank = self.all_sen.groupby('names')['rank'].mean().reset_index(name='avg_rank')
        avg_rank['avg_rank'] = avg_rank['avg_rank'].astype(int)

        # Merge dataframes
        result = pd.merge(total_count, top10_count, on='names', how='left')
        result = pd.merge(result, bottom10_count, on='names', how='left')
        result = pd.merge(result, avg_rank, on='names', how='left')

        # Sort results
        result = result.sort_values(by=['top10_count', 'avg_rank', 'bottom10_count'], ascending=[False, True, True])
        result['overall_rank'] = range(1, len(result) + 1)

        self.result = result.fillna(0)  # Fill NaN values with 0 for counts

    def save_results(self, output_path) -> None:
        """Saves the analyzed results to a CSV file"""
        self.result.to_csv(output_path, index=False)
        print(f'Final table saved to {output_path}')

    def plot_sensitivity(self):
        """Plots a radar chart of sensitivity analysis results"""
        categories = list(self.result['names'])
        N = len(categories)

        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(14, 10), subplot_kw=dict(polar=True))

        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)

        plt.xticks(angles[:-1], categories, color='black', size=15)
        for label, angle in zip(ax.get_xticklabels(), angles):
            if angle in [0, pi]:
                label.set_horizontalalignment('center')
            elif 0 < angle < pi - 0.1:
                label.set_horizontalalignment('left')
                label.set_rotation_mode('anchor')
                label.set_rotation(45)
            elif pi - 0.1 < angle < pi + 0.1:
                label.set_horizontalalignment('center')
                label.set_rotation_mode('anchor')
                label.set_rotation(90)
            else:
                label.set_horizontalalignment('right')
                label.set_rotation_mode('anchor')
                label.set_rotation(270)

        ax.set_rlabel_position(30)
        plt.yticks([10, 20, 30], ["10", "20", "30"], color="grey", size=10)
        plt.ylim(0, 30)

        # Top 10 Count
        values = self.result['top10_count'].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label="Top 10 Count")
        ax.fill(angles, values, 'b', alpha=0.1)

        # Bottom 10 Count
        values = self.result['bottom10_count'].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label="Bottom 10 Count")
        ax.fill(angles, values, 'r', alpha=0.1)

        # Average Rank
        values = self.result['avg_rank'].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label="Average Rank")
        ax.fill(angles, values, 'g', alpha=0.1)

        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
        plt.title('Sensitivity Analysis of Parameters', size=20, color='darkblue', y=1.1)
        plt.tight_layout()
        os.makedirs(self.sensitivity_dir, exist_ok=True)
        plt.savefig(os.path.join(self.sensitivity_dir, 'sensitivity_plot.png'), dpi=600)

    def perform_mixed_effects_model(self) -> None:
        """Performs a mixed-effects model analysis on the sensitivity data
        """
        df_anova = self.all_sen[['names', 'NAME', 'mu_star']]
        df_anova['mu_star'] = np.multiply(df_anova['mu_star'].astype(float), 1e-6)

        df_anova['names'] = df_anova['names'].astype('category')
        df_anova['NAME'] = df_anova['NAME'].astype('category')

        common_names = set(df_anova['names'])
        for name in df_anova['NAME'].unique():
            model_params = set(df_anova[df_anova['NAME'] == name]['names'])
            common_names = common_names.intersection(model_params)

        df_anova = df_anova[df_anova['names'].isin(common_names)]
        df_anova = df_anova.replace([np.inf, -np.inf], np.nan).dropna()

        print("Snapshot of the cleaned data for mixed-effects model:")
        print(df_anova.head())
        print("Data summary after cleaning and filtering common parameters:")
        print(df_anova.describe())

        model = mixedlm("mu_star ~ names", df_anova, groups=df_anova["NAME"])
        result = model.fit()

        print(result.summary())
        with open(os.path.join(self.sensitivity_dir, 'mixed_effects_model_summary.txt'), 'w') as f:
            f.write(str(result.summary()))

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

if __name__ == '__main__':
    # Usage
    base_dir = '/data/MyDataBase/SWATplus_by_VPUID/0000/huc12'
    model_name = 'SWAT_gwflow_MODEL'
    level = 'huc12'
    os.makedirs('Morris_Sensitivity_Analysis', exist_ok=True)
    analysis = SensitivityAnalysis(model_name, level, base_dir)
    analysis.load_data()
    analysis.analyze()
    analysis.save_results('Morris_Sensitivity_Analysis/final_table.csv')
    analysis.plot_sensitivity()
    analysis.perform_mixed_effects_model()
    analysis.generate_sensitivity_plots()

    print('Sensitivity analysis complete')
