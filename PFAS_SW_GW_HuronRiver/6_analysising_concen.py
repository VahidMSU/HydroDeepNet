import arcpy
import pandas as pd
import numpy as np
import sqlite3
import os
import geopandas as gpd
from shapely.geometry import Point
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class PFASFeatureAnalysis:
    def __init__(self, pfas_gw_path, pfas_sw_path, pfas_sites_path):
        self.pfas_gw = pd.read_pickle(pfas_gw_path)
        self.pfas_sw = pd.read_pickle(pfas_sw_path)
        self.pfas_sites = pd.read_pickle(pfas_sites_path)
        self.pfas_sw_names = [p for p in self.pfas_sw.columns if p.endswith("Result")]
        self.pfas_gw_names = [p for p in self.pfas_gw.columns if p.endswith("Result")]
        self.features = [f for f in self.pfas_gw.columns if f.endswith("_250m")]
        self.huron_river = pd.read_pickle("/data/MyDataBase/SWATGenXAppData/codes/PFAS_SW_GW_HuronRiver/input_data/Huron_River_basin_bound.pkl")
        os.makedirs("/data/MyDataBase/SWATGenXAppData/codes/PFAS_SW_GW_HuronRiver/correlations", exist_ok=True)
        os.makedirs("/data/MyDataBase/SWATGenXAppData/codes/PFAS_SW_GW_HuronRiver/feature_importance", exist_ok=True)
        os.makedirs("/data/MyDataBase/SWATGenXAppData/codes/PFAS_SW_GW_HuronRiver/spatial_analysis", exist_ok=True)
    def plot_pfas_sw_gw_sites(self):
            for p in self.pfas_sw_names:  # We use PFAS names in surface water because it has more PFAS species
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                print(f"Plotting PFAS concentrations for {p}...")

                # Plot the Huron River
                self.huron_river.plot(ax=ax, color='lightgray', alpha=0.5, edgecolor='black')

                # Plot groundwater PFAS data
                if p in self.pfas_gw_names:
                    gw_data = self.pfas_gw[['Longitude', 'Latitude', p]].copy()
                    gw_data['color'] = 'black'
                    gw_data.loc[gw_data[p] == 0, 'color'] = 'gray'
                    gw_data.loc[gw_data[p] > 0, 'color'] = 'red'
                    gw_data.dropna(subset=[p], inplace=True)  # Ensure there are no NaN values
                    for _, row in gw_data.iterrows():
                        ax.scatter(row['Longitude'], row['Latitude'], color=row['color'], marker='o', label='Groundwater' if _ == 0 else "")

                # Plot surface water PFAS data
                if p in self.pfas_sw.columns:
                    sw_data = self.pfas_sw[['Longitude', 'Latitude', p]].copy()
                    sw_data['color'] = 'black'
                    sw_data.loc[sw_data[p] == 0, 'color'] = 'gray'
                    sw_data.loc[sw_data[p] > 0, 'color'] = 'blue'
                    sw_data.dropna(subset=[p], inplace=True)  # Ensure there are no NaN values
                    for _, row in sw_data.iterrows():
                        ax.scatter(row['Longitude'], row['Latitude'], color=row['color'], marker='x', label='Surface Water' if _ == 0 else "")

                # Plot PFAS sites
                self.pfas_sites.plot(ax=ax, color='red', marker='^', label='PFAS Sites', markersize=50)

                ax.set_title(f"PFAS Concentrations in Groundwater and Surface Water - {p}")
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                handles, labels = ax.get_legend_handles_labels()
                unique_labels = dict(zip(labels, handles))
                ax.legend(unique_labels.values(), unique_labels.keys())
                plt.tight_layout()

                # Create directory if it doesn't exist
                os.makedirs("/data/MyDataBase/SWATGenXAppData/codes/PFAS_SW_GW_HuronRiver/visualizations", exist_ok=True)
                plt.savefig(f"/data/MyDataBase/SWATGenXAppData/codes/PFAS_SW_GW_HuronRiver/visualizations/{p}_sw_gw_sites.png", dpi=300)
                plt.close()

            print("PFAS site plotting completed and saved.")

    def PFAS_features_corr(self):
        corr = {}
        for f in self.features:
            for p in self.pfas_sw_names:
                corr[(f, p)] = np.corrcoef(self.pfas_sw[f], self.pfas_sw[p])[0, 1]
        corr = pd.DataFrame(list(corr.items()), columns=['index', 'corr'])
        corr[['feature', 'PFAS']] = pd.DataFrame(corr['index'].tolist(), index=corr.index)
        corr.drop(columns=['index'], inplace=True)
        corr["abs_corr"] = np.abs(corr["corr"])
        corr = corr.sort_values(by="abs_corr", ascending=False)
        corr.to_csv("/data/MyDataBase/SWATGenXAppData/codes/PFAS_SW_GW_HuronRiver/correlations/PFAS_features_corr.csv", index=False)
        print("Correlation analysis completed and saved.")

    def feature_importance_analysis(self, df, target, features, output_path):
        X = df[features].dropna()
        y = df[target].dropna()
        X, y = X.align(y, join='inner', axis=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        importance = rf.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
        feature_importance_df.to_csv(output_path, index=False)

        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
        plt.title(f'Feature Importance for Predicting {target}')
        plt.savefig(output_path.replace('.csv', '.png'), dpi=300)
        plt.close()

        print(f"Feature importance analysis for {target} completed and saved.")

    def perform_feature_importance(self):
        for pfas_compound in self.pfas_sw_names:
            output_path = f"/data/MyDataBase/SWATGenXAppData/codes/PFAS_SW_GW_HuronRiver/feature_importance/feature_importance_{pfas_compound}.csv"
            self.feature_importance_analysis(self.pfas_sw, pfas_compound, self.features, output_path)
        for pfas_compound in self.pfas_gw_names:
            output_path = f"/data/MyDataBase/SWATGenXAppData/codes/PFAS_SW_GW_HuronRiver/feature_importance/feature_importance_{pfas_compound}.csv"
            self.feature_importance_analysis(self.pfas_gw, pfas_compound, self.features, output_path)

    def spatial_analysis(self, buffer_size=2000):
        pfas_sites_gdf = gpd.GeoDataFrame(self.pfas_sites, geometry=gpd.points_from_xy(self.pfas_sites.Longitude, self.pfas_sites.Latitude), crs="EPSG:4326").to_crs("EPSG:26990")

        def calculate_distance_to_nearest_site(pfas_df, sites_gdf, buffer_size=2000):
            pfas_gdf = gpd.GeoDataFrame(pfas_df, geometry=gpd.points_from_xy(pfas_df.Longitude, pfas_df.Latitude), crs="EPSG:4326").to_crs("EPSG:26990")
            pfas_gdf['min_distance'] = pfas_gdf.geometry.apply(lambda x: sites_gdf.distance(x).min())
            pfas_gdf['within_buffer'] = pfas_gdf['min_distance'] <= buffer_size
            return pfas_gdf

        pfas_gw_gdf = calculate_distance_to_nearest_site(self.pfas_gw, pfas_sites_gdf)
        pfas_sw_gdf = calculate_distance_to_nearest_site(self.pfas_sw, pfas_sites_gdf)

        plt.figure(figsize=(12, 6))
        sns.boxplot(x='within_buffer', y='PFOAResult', data=pfas_gw_gdf)
        plt.title('PFOA Concentrations in Groundwater by Proximity to Industry Sites')
        plt.savefig("/data/MyDataBase/SWATGenXAppData/codes/PFAS_SW_GW_HuronRiver/spatial_analysis/GW_PFOA_proximity.png", dpi=300)
        plt.close()

        plt.figure(figsize=(12, 6))
        sns.boxplot(x='within_buffer', y='PFOAResult', data=pfas_sw_gdf)
        plt.title('PFOA Concentrations in Surface Water by Proximity to Industry Sites')
        plt.savefig("/data/MyDataBase/SWATGenXAppData/codes/PFAS_SW_GW_HuronRiver/spatial_analysis/SW_PFOA_proximity.png", dpi=300)
        plt.close()

        print("Spatial analysis completed and saved.")

    def process(self):
        self.plot_pfas_sw_gw_sites()
       # self.PFAS_features_corr()
       # self.perform_feature_importance()
       # self.spatial_analysis()

if __name__ == "__main__":
    pfas_gw_path = "/data/MyDataBase/SWATGenXAppData/codes/PFAS_SW_GW_HuronRiver/input_data/Huron_PFAS_GW_Features.pkl"
    pfas_sw_path = "/data/MyDataBase/SWATGenXAppData/codes/PFAS_SW_GW_HuronRiver/input_data/Huron_PFAS_SW_Features.pkl"
    pfas_sites_path = "/data/MyDataBase/SWATGenXAppData/codes/PFAS_SW_GW_HuronRiver/input_data/Huron_PFAS_SITE_Features.pkl"
    PFASFeatureAnalysis(pfas_gw_path, pfas_sw_path, pfas_sites_path).process()
