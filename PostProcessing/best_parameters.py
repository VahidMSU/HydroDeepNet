import matplotlib.pyplot as plt
import geopandas as gpd
import os
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
def generate_best_parameters_csv(BASE_PATH):    
    # List directory and filter out unnecessary files
    
    NAMES = os.listdir(BASE_PATH)
    NAMES.remove("log.txt")
    all_best_parameters = []

    for NAME in NAMES:
        if len(NAME) > 10:
            continue
        best_performance_path = f"{BASE_PATH}/{NAME}/best_solution_SWAT_gwflow_MODEL.txt"
        with open(best_performance_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if "4_thickness_sb" in line:
                    lines.remove(line)
                    break
            for line in lines:
                if "Final" in line:
                    lines.remove(line)
                    best_value = line.split(":")[1].strip()


                    break
            
        # Create a DataFrame from the remaining lines
        data = [line.strip().split(',') for line in lines]
        df = pd.DataFrame(data, columns=["Parameter", "Value"])
        df["Value"] = pd.to_numeric(df["Value"])
        
        # Transpose the DataFrame
        best_parameters = df.set_index("Parameter").T
        best_parameters["NAME"] = NAME  # Add a column for the NAME
        best_parameters["best_performance"] = best_value    
        all_best_parameters.append(best_parameters)

    # Concatenate all DataFrames
    all_best_parameters_df = pd.concat(all_best_parameters)

    # Save to CSV
    os.makedirs("best_parameters", exist_ok=True)
    ## drop the last column
    all_best_parameters_df = all_best_parameters_df.iloc[:, :-1]
    all_best_parameters_df.to_csv("best_parameters/best_parameters.csv", index=False)

    print("CSV file has been created successfully.")




def plot_best_parameters_with_clusters():
    # Load the model parameters
    model_parameters = pd.read_pickle("model_bounds/model_bounds_huc12.pkl")
    df = pd.read_csv("best_parameters/best_parameters.csv")

    # Ensure the NAME column is integer for both dataframes
    model_parameters['NAME'] = model_parameters['NAME'].astype(int)
    df['NAME'] = df['NAME'].astype(int)
    parameter_names = df.columns[:-1]
    
    # Merge model parameters with best parameters dataframe
    model_parameters = model_parameters.merge(df, on="NAME")

    # Ensure there's data to plot
    assert len(model_parameters.NAME) > 0, "No data to plot"
    
    for parameter in parameter_names:
        # Convert parameter values to float
        model_parameters[parameter] = model_parameters[parameter].astype(float)

        # Handle NaN values by dropping them
        parameter_data = model_parameters[['NAME', parameter]].dropna()

        # Perform k-means clustering on the parameter
        kmeans = KMeans(n_clusters=3, random_state=42)
        parameter_data['Cluster'] = kmeans.fit_predict(parameter_data[[parameter]])

        # Define cluster labels based on the range of values in each cluster
        cluster_labels = []
        for cluster in range(3):
            cluster_values = parameter_data[parameter_data['Cluster'] == cluster][parameter]
            label = f"{cluster_values.min():.2f} - {cluster_values.max():.2f}"
            cluster_labels.append(label)
        
        # Merge clustering results back into model_parameters
        model_parameters = model_parameters.drop(columns=['Cluster'], errors='ignore')  # Drop existing Cluster column if present
        model_parameters = model_parameters.merge(parameter_data[['NAME', 'Cluster']], on='NAME', how='left')

        # Plot with clustering
        fig, ax = plt.subplots(figsize=(10, 6))
        model_parameters.plot(ax=ax, color='lightgrey')  # Plot the base map without color bar

        # Overlay clusters with appropriate labels
        colors = np.array(['red', 'green', 'blue'])
        patches = []
        for cluster in range(3):
            cluster_data = model_parameters[model_parameters['Cluster'] == cluster]
            if not cluster_data.empty:
                cluster_data.plot(ax=ax, marker='o', color=colors[cluster], markersize=5)
                patch = mpatches.Patch(color=colors[cluster], label=f'Cluster {cluster + 1}: {cluster_labels[cluster]}')
                patches.append(patch)

        ax.set_title(f"{parameter} with K-Means Clustering")
        ax.set_axis_off()
        ax.legend(handles=patches, title="Clusters", loc='upper right')
        plt.tight_layout()
        plt.savefig(f"best_parameters/{parameter}_clustered.png")
        plt.close()





def best_parameters_box_plot():
    df = pd.read_csv("best_parameters/best_parameters.csv")

    parameter_names = df.columns[:-1]
    print(f"Columns: {df.columns}")
    ## group by NAME
    df = df.groupby('NAME').mean()
    ## box plot
    fig, ax = plt.subplots(figsize=(10, 6))
    df.boxplot(ax=ax)
    ax.set_title("Best Parameters")
    ax.set_xticklabels(parameter_names, rotation=90)
    plt.tight_layout()
    plt.savefig("best_parameters/best_parameters_box_plot.png")
    
if __name__ == "__main__":
    BASE_PATH = "/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/"
    generate_best_parameters_csv(BASE_PATH)
    # Call the function
    plot_best_parameters_with_clusters()
    best_parameters_box_plot()