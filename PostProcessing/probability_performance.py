import pandas as pd
import numpy as np
from scipy.stats import linregress, f_oneway
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols

def analyze_model_performance(df_models_path, performance_df_path, output_dir="overal_best_performance"):
    """
    This function analyzes the model performance by performing ANOVA tests based on various model characteristics,
    such as Total Area, HRU Count, number of rivers, number of lakes, and total lake area. It also generates a 
    scatter plot of the number of lakes versus best performance.

    Parameters:
    - df_models_path (str): Path to the CSV file containing model characteristics.
    - performance_df_path (str): Path to the CSV file containing best performance data.
    - output_dir (str): Directory where the output files (corrected CSV and plots) will be saved.

    Outputs:
    - Saves a corrected dataframe with merged performance data.
    - Prints the number of unique models in different groups.
    - Prints ANOVA test results for various model characteristics.
    - Saves a scatter plot of number of lakes versus best performance.
    """

    # Load the model characteristics and best performance data
    df_models = pd.read_csv(df_models_path).drop(columns=['best_performance'])
    best_performance = pd.read_csv(performance_df_path)
    best_performance = best_performance[best_performance.best_performance > 0]
    df_models = df_models.merge(best_performance, on="NAME", how="inner")

    # Save the corrected dataframe
    corrected_df_path = os.path.join(output_dir, "df_models_corrected.csv")
    df_models.to_csv(corrected_df_path, index=False)
    print(f"Corrected dataframe saved to: {corrected_df_path}")

    print(f"Number of unique NAME: {df_models['NAME'].nunique()}")

    # Split the data into two groups based on Total_Area threshold and perform ANOVA
    group1 = df_models[df_models['Total_Area'] > 1500 * 1e2]
    print(f"Unique NAME in group1: {group1['NAME'].nunique()}")
    group2 = df_models[df_models['Total_Area'] <= 1500 * 1e2]
    print(f"Unique NAME in group2: {group2['NAME'].nunique()}")
    anova_result = f_oneway(group1['best_performance'], group2['best_performance'])
    print(f"ANOVA test result for Total_Area groups: {anova_result}")

    # Perform ANOVA tests based on HRU_Count
    hru_group1 = df_models[df_models['HRU_Count'] > 20000]['best_performance']
    hru_group2 = df_models[df_models['HRU_Count'] <= 20000]['best_performance']
    hru_anova_result = f_oneway(hru_group1, hru_group2)
    print(f"ANOVA test result for HRU_Count groups: {hru_anova_result}")

    # Perform ANOVA tests based on n_rivers
    rivers_group1 = df_models[df_models['n_rivers'] > 1000]['best_performance']
    rivers_group2 = df_models[df_models['n_rivers'] <= 1000]['best_performance']
    rivers_anova_result = f_oneway(rivers_group1, rivers_group2)
    print(f"ANOVA test result for n_rivers groups: {rivers_anova_result}")

    # Perform ANOVA tests based on number_of_lakes
    lakes_group1 = df_models[df_models['number_of_lakes'] > 100]['best_performance']
    lakes_group2 = df_models[df_models['number_of_lakes'] <= 100]['best_performance']
    lakes_anova_result = f_oneway(lakes_group1, lakes_group2)
    print(f"ANOVA test result for number_of_lakes groups: {lakes_anova_result}")

    # Perform ANOVA tests based on total_Lake_area_sqkm
    lake_area_group1 = df_models[df_models['total_Lake_area_sqkm'] > 10]['best_performance']
    lake_area_group2 = df_models[df_models['total_Lake_area_sqkm'] <= 10]['best_performance']
    lake_area_anova_result = f_oneway(lake_area_group1, lake_area_group2)
    print(f"ANOVA test result for total_Lake_area_sqkm groups: {lake_area_anova_result}")

    # Plot the scatter of best_performance vs number_of_lakes
    fig, ax = plt.subplots()
    ax.scatter(group1['number_of_lakes'], group1['best_performance'], c='red', label='models with WA > 1500 km\u00b2')
    ax.scatter(group2['number_of_lakes'], group2['best_performance'], c='green', label='models with WA <= 1500 km\u00b2')
    ax.set_xlabel("Number of Lakes")
    ax.set_ylabel("Best Performance")
    ax.legend()
    ax.grid(alpha=0.5, linestyle='--', linewidth=0.5)

    scatter_plot_path = os.path.join(output_dir, "number_of_lakes_vs_best_performance.png")
    plt.savefig(scatter_plot_path, dpi=300)
    print(f"Scatter plot saved to: {scatter_plot_path}")


# Example usage
df_models_path = "model_characteristics/SWAT_gwflow_MODEL/df_models.csv"
performance_df_path = "overal_best_performance/SWAT_gwflow_MODEL/best_performance_0000.csv"
analyze_model_performance(df_models_path, performance_df_path)
