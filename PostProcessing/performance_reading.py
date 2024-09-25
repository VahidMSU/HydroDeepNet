import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm

class SWATModelAnalysis:
    def __init__(self, model_name, data_path, model_bounds_path, output_path, fig_save_dir):
        self.model_name = model_name
        self.data_path = data_path
        self.model_bounds_path = model_bounds_path
        self.output_path = output_path
        self.fig_save_dir = fig_save_dir

    @staticmethod
    def get_the_number_of_simulated_stations(path):
        files = os.listdir(path)
        return len(np.unique([x.split('_')[2] for x in files if len(x.split('_')) > 2]))

    def analyze(self, vpuid):
        best_performance = []
        vpuid_path = os.path.join(self.data_path, vpuid, "huc12")
        names = os.listdir(vpuid_path)
        if 'log.txt' in names:
            names.remove('log.txt')
        for name in names:
            best_path = os.path.join(vpuid_path, name, f"best_solution_{self.model_name}.txt")
            fig_path = os.path.join(vpuid_path, name, f"calibration_figures_{self.model_name}")
            if os.path.exists(best_path) and os.path.exists(fig_path):
                print(f"Reading {best_path}")
                with open(best_path, "r") as file:
                    lines = file.readlines()
                    path = os.path.join(vpuid_path, name, f"calibration_figures_{self.model_name}")
                    try:
                        best_performance.append((name, self.model_name, float(lines[-1].split(':')[1]), self.get_the_number_of_simulated_stations(path)))
                    except Exception:
                        print(f"Error reading {best_path}")
        best_performance_df = pd.DataFrame(best_performance, columns=['NAME', 'MODEL_NAME', 'PERFORMANCE', 'NUMBER_OF_STATIONS'])
        best_performance_df['best_performance'] = best_performance_df['PERFORMANCE'] / (-2 * best_performance_df['NUMBER_OF_STATIONS'])
        best_performance_df = best_performance_df.dropna().drop(columns=['PERFORMANCE'])
        best_performance_df = best_performance_df.sort_values(by='best_performance', ascending=False)
        os.makedirs(f"{self.fig_save_dir}", exist_ok=True)
        best_performance_df.to_csv(f"{self.fig_save_dir}/best_performance_{vpuid}.csv", index=False)
        return best_performance_df

    def plot_model_bounds(self, best_performance):
        ## remove performance below 0
        best_performance = best_performance[best_performance['best_performance'] > 0]
        model_bounds = pd.read_pickle(self.model_bounds_path)
        #huc8_model_bounds = pd.read_pickle('model_bounds/huc8_model_bounds.pkl')
        county = gpd.read_file('/data/MyDataBase/Michigan_NHDPlus/COUNTY/Counties_(v17a).shp')
        county = county[county.PENINSULA == 'lower']
        model_bounds = model_bounds.merge(best_performance, on='NAME', how='left')
        model_bounds = model_bounds.dropna()
        model_bounds = model_bounds.sort_values(by='best_performance', ascending=False)
        ### sort by geometry area size
        model_bounds['AREA'] = model_bounds.to_crs("EPSG:26990").geometry.area
        model_bounds = model_bounds.sort_values(by='AREA', ascending=False)

        model_bounds = gpd.GeoDataFrame(model_bounds, geometry=model_bounds['geometry']).to_crs(epsg=4326)
        #huc8_model_bounds = huc8_model_bounds.to_crs(epsg=4326)
        county = county.to_crs(epsg=4326)
        fig, ax = plt.subplots(1, 1, figsize=(8, 10))

        def assign_cluster(value):
            if value < -1:
                return 0
            elif -1 < value <= 0:
                return 1
            elif 0 < value <= 0.15:
                return 2
            elif 0.15 < value <= 0.3:
                return 3
            elif 0.3 < value <= 0.45:
                return 4
            elif 0.45 < value < 0.6:
                return 5
            elif 0.6 < value < 1:
                return 6

        model_bounds['cluster'] = model_bounds['best_performance'].apply(assign_cluster)

        color_mapping = {
            0: 'black',
            1: 'gray',
            2: 'yellow',
            3: 'orange',
            4: 'salmon',
            5: 'red',
            6: 'darkred'
        }

        for cluster in model_bounds['cluster'].unique():
            color = color_mapping.get(cluster, 'black')
            model_bounds.loc[model_bounds['cluster'] == cluster, 'color'] = color

        fig, ax = plt.subplots(figsize=(12, 8))
        county.plot(ax=ax, color='lightgray', edgecolor='black', alpha=0.5)
        model_bounds_sorted = model_bounds.sort_values('color', ascending=False)

        for cluster in model_bounds['cluster'].unique():
            gdf_cluster = model_bounds_sorted[model_bounds_sorted['cluster'] == cluster]
            if not gdf_cluster.empty:
                color = gdf_cluster.iloc[0]['color']
                gdf_cluster.plot(ax=ax, color=color, edgecolor='black', alpha=0.8)

        cluster_ranges = model_bounds.groupby('cluster')['best_performance'].agg(["min", "max"])
        legend_labels = [f'{row["min"]:.2f} - {row["max"]:.2f}' for cluster, row in cluster_ranges.iterrows()]
        legend_patches = [mpatches.Patch(color=color_mapping[cluster], label=label) for cluster, label in zip(cluster_ranges.index, legend_labels)]

        ax.legend(handles=legend_patches, title='NSE Performance')
        ax.set_title(f'#{len(np.unique(model_bounds_sorted.NAME))} HUC12 {self.model_name}')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        os.makedirs("overal_best_performance", exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"{self.fig_save_dir}/Map_Average_Performance.jpeg", dpi=300)
        plt.close()

        

        fig, ax = plt.subplots()

        # Remove performance values less than 0
        #model_bounds = model_bounds[(model_bounds['best_performance'] > 0) & (model_bounds['best_performance'] < 1)]

        # Plot histogram
        model_bounds['best_performance'].hist(ax=ax, bins=20, color='skyblue', edgecolor='black')

        # Calculate mean and standard deviation
        mean_performance = model_bounds['best_performance'].mean()
        std_performance = model_bounds['best_performance'].std()

        # Generate x values for the normal distribution curve
        x = np.linspace(model_bounds['best_performance'].min(), model_bounds['best_performance'].max(), 100)
        # Calculate the normal distribution curve
        y = norm.pdf(x, mean_performance, std_performance) * len(model_bounds['best_performance']) * (model_bounds['best_performance'].max() - model_bounds['best_performance'].min()) / 20

        # Plot the normal distribution curve
        ax.plot(x, y, color='red', linestyle='--', linewidth=2)

        # Set titles and labels
        ax.set_title(f'Performance Distribution of {self.model_name}')
        ax.set_xlabel('NSE Performance')
        ax.set_ylabel('Frequency')

        # Set major xticks with 0.5 interval
        major_xticks = np.arange(0, 1.1, 0.5)
        ax.set_xticks(major_xticks)
        # Set minor xticks with 0.1 interval
        minor_xticks = np.arange(0, 1.1, 0.1)
        ax.set_xticks(minor_xticks, minor=True)

        # Set major xtick labels
        ax.set_xticklabels(major_xticks, rotation=45)

        # Add grid
        plt.grid(axis='both', linestyle='--', alpha=0.5)

        # Adjust layout and save the plot
        plt.tight_layout()
        plt.savefig(f"{self.fig_save_dir}/Histogram_Average_Performance.jpeg", dpi=300)
        plt.close()



        ### now plot performance vs area
        fig, ax = plt.subplots()
        model_bounds.plot.scatter(x='AREA', y='best_performance', ax=ax, color='skyblue', edgecolor='black')
        ax.set_title(f'Performance vs Area of {self.model_name}')
        ax.set_xlabel('Area (sq. km)')
        ax.set_ylabel('NSE Performance')
        plt.tight_layout()
        plt.grid(axis='both', linestyle='--', alpha=0.5)
        plt.savefig(f"{self.fig_save_dir}/Performance_vs_Area.jpeg", dpi=300)
        plt.close()


        # Define the threshold for AREA in square kilometers
        threshold = 1500 * 1e6  # 1500 km^2 in square meters

        # Separate watersheds based on the AREA threshold
        less_than_1500km2 = model_bounds[model_bounds['AREA'] <= threshold]
        more_than_1500km2 = model_bounds[model_bounds['AREA'] > threshold]

        # Plot performance vs number of stations, using AREA as the size of the points and different colors based on the threshold
        fig, ax = plt.subplots()

        # Plot watersheds with AREA <= 1500 km^2 in blue
        ax.scatter(less_than_1500km2['NUMBER_OF_STATIONS'],
                less_than_1500km2['best_performance'],
                color='blue',
                edgecolor='black',
                s=less_than_1500km2['AREA'] / 1e7,
                linewidth=0.25,
                label='Area <= 1500 km²')

        # Plot watersheds with AREA > 1500 km^2 in red
        ax.scatter(more_than_1500km2['NUMBER_OF_STATIONS'],
                more_than_1500km2['best_performance'],
                color='red',
                edgecolor='black',
                s=more_than_1500km2['AREA'] / 1e7,
                linewidth=0.25,
                label='Area > 1500 km²')

        # Create custom legend handles
        blue_patch = mpatches.Patch(color='blue', label='Area <= 1500 km²')
        red_patch = mpatches.Patch(color='red', label='Area > 1500 km²')

        # Add legends
        ax.legend(handles=[blue_patch, red_patch], loc='upper right')

        # Set titles and labels
        ax.set_title(f'Performance vs Number of Stations of {self.model_name}')
        ax.set_xlabel('Number of Stations')
        ax.set_ylabel('NSE Performance')

        # Add grid
        plt.grid(axis='both', linestyle='--', alpha=0.5)

        # Adjust layout and save the plot
        plt.tight_layout()
        plt.savefig(f"{self.fig_save_dir}/Performance_vs_Number_of_Stations.jpeg", dpi=300)
        plt.close()


        ### plot area vs number of stations and use performance as the size of the points
        fig, ax = plt.subplots()
        model_bounds.plot.scatter(x='NUMBER_OF_STATIONS', y='AREA', ax=ax, color='skyblue', edgecolor='black', s=model_bounds['best_performance']*100, linewidth=0.5)
        ax.set_title(f'Area vs Number of Stations of {self.model_name}')
        ax.set_xlabel('Number of Stations')
        ax.set_ylabel('Area (sq. km)')
        plt.tight_layout()
        plt.grid(axis='both', linestyle='--', alpha=0.5)
        plt.savefig(f"{self.fig_save_dir}/Area_vs_Number_of_Stations.jpeg", dpi=300)
        plt.close()

        ### plot

        ### now plot a 3d plot of area, number of stations and performance
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(model_bounds['NUMBER_OF_STATIONS'], model_bounds['AREA'], model_bounds['best_performance'], c='skyblue', edgecolor='black')
        ax.set_xlabel('Number of Stations')
        ax.set_ylabel('Area (sq. km)')
        ax.set_zlabel('NSE Performance')
        ax.set_title(f'3D Plot of Area, Number of Stations and Performance of {self.model_name}')
        plt.tight_layout()
        plt.grid(axis='both', linestyle='--', alpha=0.5)
        plt.savefig(f"{self.fig_save_dir}/3D_Plot_{self.model_name}.jpeg", dpi=300)
        plt.close()


    def run_analysis(self, vpuid):
        best_performance = self.analyze(vpuid)
        self.plot_model_bounds(best_performance)

if __name__ == "__main__":
    MODEL_NAME = "SWAT_gwflow_MODEL"
    DATA_PATH = "/data/MyDataBase/SWATplus_by_VPUID/"
    MODEL_BOUNDS_PATH = "/home/rafieiva/MyDataBase/codes/PostProcessing/model_bounds/model_bounds_huc12.pkl"
    OUTPUT_PATH = "/home/rafieiva/MyDataBase/codes/PostProcessing/model_bounds"
    fig_save_dir = f"overal_best_performance/{MODEL_NAME}"

    swat_analysis = SWATModelAnalysis(MODEL_NAME, DATA_PATH, MODEL_BOUNDS_PATH, OUTPUT_PATH, fig_save_dir)
    



    VPU_IDS = ["0000"]
    for vpuid in VPU_IDS:
        swat_analysis.run_analysis(vpuid)
