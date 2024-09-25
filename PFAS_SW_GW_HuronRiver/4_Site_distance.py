import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import os
import os
import logging

class PFASAnalysis:
    def __init__(self, sites_path, pfas_path, output_dir, media, distance=2000):
        self.sites_path = sites_path
        self.pfas_path = pfas_path
        self.output_dir = output_dir
        self.media = media
        self.ss_sites_huron = pd.read_pickle(sites_path).to_crs("EPSG:26990").sort_values('Industry').reset_index()
        self.pfas_gw_path = pd.read_pickle(pfas_path).to_crs('EPSG:26990').reset_index()
        self.results = []
        self.fig_dir = "figs/Site_distance"
        os.makedirs(self.fig_dir, exist_ok=True)
        self.distance = distance
        self.setup()
        if not os.path.exists(self.fig_dir):
            os.makedirs(self.fig_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if os.path.exists(f"{self.output_dir}/{self.media}_PFAS_analysis_report.txt"):
            os.remove(f"{self.output_dir}/{self.media}_PFAS_analysis_report.txt")
        if os.path.exists(f"{self.output_dir}/{self.media}_PFAS_summary.csv"):
            os.remove(f"{self.output_dir}/{self.media}_PFAS_summary.csv")

    def setup(self):
        setup_logging(self.output_dir)
        self.write_report(f"Number of sites: {len(self.ss_sites_huron)}")
        self.write_report(f"Number of PFAS samples: {len(self.pfas_gw_path)}")
        self.write_report(f"Distance: {self.distance}")
        self.write_report(f"Output directory: {self.output_dir}")
        self.write_report(f"Media: {self.media}")

    def write_report(self, message, file_name=None):
        if file_name is None:
            file_name = f"{self.media}_PFAS_analysis_report.txt"
        with open(f"{self.output_dir}/{file_name}", 'a') as f:
            f.write(message + '\n')

    def find_distance(self, site, distance):
        site_geom = site.geometry
        self.pfas_gw_path['distance'] = self.pfas_gw_path['geometry'].distance(site_geom)
        return self.pfas_gw_path[self.pfas_gw_path['distance'] <= distance]

    def calculate_statistics(self):
        logging.info("Calculating number of samples within distance from each site.")
        for index, row in self.ss_sites_huron.iterrows():
            site = row
            nearby_samples = self.find_distance(site, self.distance)
            num_samples = len(nearby_samples)

            stats = {}
            for column in nearby_samples.columns:
                if column.endswith('Result'):
                    if nearby_samples[column].notna().any():
                        stats[f'{column}_average'] = nearby_samples[column].mean()
                        stats[f'{column}_median'] = nearby_samples[column].median()
                        stats[f'{column}_std'] = nearby_samples[column].std()
                    else:
                        stats[f'{column}_average'] = np.nan
                        stats[f'{column}_median'] = np.nan
                        stats[f'{column}_std'] = np.nan

            result = {
                "Index": index,
                "Industry": site['Industry'],
                f"Number of Samples within {self.distance}": num_samples,
            } | stats
            self.results.append(result)
            self.write_report(f"number of samples within {self.distance} from {index}_{site['Industry']}: {num_samples}")

        self.results_df = pd.DataFrame(self.results)
        self.results_df.to_csv(f"{self.output_dir}/{self.media}_Samples_within_{self.distance}.csv", float_format='%.4f', index=False)
        logging.info(f"Number of samples within {self.distance} from each site calculated.")

    def correlation_analysis(self):  # sourcery skip: class-extract-method
        self.pfas_columns = [col for col in self.results_df.columns if col.endswith('average')]
        pfas_data = self.results_df[self.pfas_columns]
        correlation_matrix = pfas_data.corr()
        correlation_matrix.to_csv(f'{self.output_dir}/{self.media}_PFAS_concentration_correlation.csv', float_format='%.4f')

        self.write_report("Correlation between different PFAS compounds:")
        for i in range(len(self.pfas_columns)):
            for j in range(i+1, len(self.pfas_columns)):
                corr = correlation_matrix.iloc[i, j]
                if not np.isnan(corr):
                    self.write_report(f"{self.pfas_columns[i]} and {self.pfas_columns[j]}: {corr:.2f}")

        plt.figure(figsize=(14, 12))
        ### drop columns with all NaN values
        correlation_matrix = correlation_matrix.dropna(axis=0, how='all').dropna(axis=1, how='all')
        sns.heatmap(correlation_matrix, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix of average PFAS Concentrations calculated for each industry')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{self.fig_dir}/{self.media}_PFAS_concentration_correlation.png', dpi=300)
        plt.close()

    def anova_analysis(self):
        """ the anova analysis is used to determine if there is a significant difference in the average PFAS concentrations in waterwells next to different industries."""
        anova_results = []
        for PFAS_compound in self.pfas_columns:
            anova_data = {industry: self.results_df[self.results_df['Industry'] == industry][PFAS_compound].dropna() for industry in self.results_df['Industry'].unique()}
            anova_data = {k: v for k, v in anova_data.items() if len(v) > 0}

            if len(anova_data) > 1:
                anova_result = f_oneway(*anova_data.values())
                significant = anova_result.pvalue < 0.05
            else:
                significant = False

            anova_results.append({
                'PFAS_Compound': PFAS_compound,
                'Significant': significant
            })

        anova_df = pd.DataFrame(anova_results)
        anova_df.to_csv(f"{self.output_dir}/{self.media}_ANOVA_results.csv", index=False)

    def plot_concentrations(self):
        summary_stats_list = []
        for PFAS_compound in self.pfas_columns:
            valid_data = self.results_df[['Industry', PFAS_compound]].dropna()
            if valid_data.empty:
                self.write_report(f"No valid data for {PFAS_compound}, skipping boxplot generation.")
                continue

            summary_stats = valid_data.groupby('Industry').agg(
                mean_value=(PFAS_compound, 'mean')
            ).reset_index()

            summary_stats['PFAS_Compound'] = PFAS_compound
            summary_stats_list.append(summary_stats)

            plt.figure(figsize=(12, 8))
            sns.boxplot(x='Industry', y=PFAS_compound, data=valid_data)
            plt.xticks(rotation=90)
            plt.title(f'Distribution of {PFAS_compound} Concentrations by Industry')
            plt.tight_layout()
            plt.savefig(f'{self.fig_dir}/{self.media}_{PFAS_compound}_concentration_boxplot.png', dpi=300)
            plt.close()

        if summary_stats_list:
            all_summary_stats = pd.concat(summary_stats_list)
            all_summary_stats.to_csv(f"{self.output_dir}/{self.media}_summary_stats.csv", index=False)

    def pca_analysis(self):
        """Perform PCA analysis to reduce dimensionality and identify patterns."""
        pfas_data = self.results_df[self.pfas_columns].dropna()
        if pfas_data.shape[0] < 2:
            self.write_report("Not enough data for PCA analysis.")
            return

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(pfas_data)
        pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
        pca_df['Industry'] = self.results_df.loc[pfas_data.index, 'Industry']
        pca_df.to_csv(f"{self.output_dir}/{self.media}_PCA_results.csv", index=False)

        plt.figure()
        sns.scatterplot(x='PC1', y='PC2', hue='Industry', data=pca_df)
        plt.title(f'PCA of PFAS Concentrations for {self.media}')
        plt.tight_layout()
        plt.savefig(f'{self.fig_dir}/{self.media}_PCA_plot.png', dpi=300)
        plt.close()

    def regression_analysis(self):
        """Perform regression analysis to quantify relationship between PFAS concentrations and industry."""
        for PFAS_compound in self.pfas_columns:
            valid_data = self.results_df[['Industry', PFAS_compound]].dropna()
            if valid_data.empty:
                self.write_report(f"No valid data for {PFAS_compound}, skipping regression analysis.")
                continue

            X = pd.get_dummies(valid_data['Industry'], drop_first=True)
            y = valid_data[PFAS_compound]
            if len(y) < 2:
                self.write_report(f"Not enough data for regression analysis for {PFAS_compound}.")
                continue

            model = LinearRegression()
            model.fit(X, y)
            coefficients = model.coef_
            intercept = model.intercept_

            coef_df = pd.DataFrame({'Industry': X.columns, 'Coefficient': coefficients})
            coef_df['Intercept'] = intercept  ## the intercept is the value of y when x=0 which indicates the baseline value.
            coef_df['PFAS_Compound'] = PFAS_compound
            ### maximum 2 decimal points
            coef_df['Coefficient'] = coef_df['Coefficient'].round(2)
            coef_df.to_csv(f"{self.output_dir}/{self.media}_regression_coefficients_{PFAS_compound}.csv", index=False)

            self.write_report(f"Regression analysis for {PFAS_compound} completed.")

    def cluster_analysis(self):
        """Perform KMeans clustering to identify clusters of PFAS concentration profiles."""
        pfas_data = self.results_df[self.pfas_columns].dropna()
        if pfas_data.shape[0] < 2:
            self.write_report("Not enough data for clustering analysis.")
            return

        kmeans = KMeans(n_clusters=3)
        clusters = kmeans.fit_predict(pfas_data)
        self.results_df['Cluster'] = clusters
        self.results_df.to_csv(f"{self.output_dir}/{self.media}_cluster_results.csv", index=False)

        plt.figure()
        sns.scatterplot(x=self.results_df.index, y='Cluster', hue='Industry', data=self.results_df)
        plt.title(f'KMeans Clustering of PFAS Concentrations for {self.media}')
        plt.tight_layout()
        plt.savefig(f'{self.fig_dir}/{self.media}_clustering_plot.png', dpi=300)
        plt.close()

    def run_all(self):
        self.calculate_statistics()
        self.correlation_analysis()
        self.anova_analysis()
        self.plot_concentrations()
        #self.pca_analysis()
        #self.regression_analysis()
       # self.cluster_analysis()

if __name__ == "__main__":

    os.makedirs("results/Site_distance", exist_ok=True)
    for media in ['SW', 'GW']:
        analysis = PFASAnalysis(
            sites_path="input_data/Huron_PFAS_SITE_Features.pkl",
            pfas_path=f"input_data/Huron_PFAS_{media}_Features.pkl",
            output_dir="results/Site_distance",
            media=media
        )
        analysis.run_all()
