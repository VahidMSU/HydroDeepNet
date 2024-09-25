import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import matplotlib.patches as mpatches

# Assuming the NAMES list and processing code are the same as provided
NAMES = os.listdir("/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/")
NAMES.remove("log.txt")

# Creating the all_models DataFrame
all_models = {
    "NAME": [], "average_hours": [], "max_hours": [], "min_hours": [], "std_hours": []
}

for NAME in NAMES:
    print(f"Reading {NAME}")
    sub_path = f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/log.txt"
    all_hours = []
    with open(sub_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            if "execution terminated within" in line:
                hours = line.split("within")[1].split("hours")[0].strip()
                try:
                    all_hours.append(60 * float(hours))
                except Exception:
                    print(f"Error in {line}")
                    continue
    all_models["NAME"].append(NAME)
    all_models["average_hours"].append(np.median(all_hours))
    all_models["max_hours"].append(np.max(all_hours))
    all_models["min_hours"].append(np.min(all_hours))
    all_models["std_hours"].append(np.std(all_hours))

all_models = pd.DataFrame(all_models)

# Merging with df_model
df_model = pd.read_csv("/home/rafieiva/MyDataBase/codes/PostProcessing/model_characteristics/SWAT_gwflow_MODEL/df_models.csv")
# Print total number of HRUs
print(f"Total number of HRUs: {df_model['HRU_Count'].sum()}")
df_model["NAME"] = df_model["NAME"].astype("int64")
all_models["NAME"] = all_models["NAME"].astype("int64")
all_models = all_models.merge(df_model[['NAME', 'HRU_Count','n_rivers' ,'Total_Area']], on="NAME", how="left")

# Save the merged DataFrame
os.makedirs("compution_time", exist_ok=True)
all_models.to_csv("compution_time/compution_time.csv", index=False)

# Check for NaN or infinite values and filter them out
all_models = all_models.replace([np.inf, -np.inf], np.nan).dropna(subset=["n_rivers", "average_hours"])

# Define colors based on Total_Area. an annovative color can be like this: deep red: #8B0000, deep green: #006400
colors = np.where(all_models["Total_Area"] > 150000, '#8B0000', 'blue')

# Create custom legend entries for the colors
red_patch = mpatches.Patch(color='red', label='WA  > 1500 km\u00b2')
blue_patch = mpatches.Patch(color='blue', label='WA  <= 1500 km\u00b2')

# Plotting the scatter plot for HRU Count vs Average Hours
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
scatter1 = ax[0].scatter(all_models["HRU_Count"], all_models["average_hours"],
                        c=colors, alpha=0.5, edgecolors="w", linewidth=0.5, s=all_models["Total_Area"] / 1e3)
ax[0].legend(handles=[red_patch, blue_patch], title='Total Area')

# Adding labels and title for HRU Count plot
ax[0].set_xlabel("HRU Count")
ax[0].set_ylabel("Average execution time (minutes)")
ax[0].set_title("Average execution time vs HRU Count")
ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)

# Adding the linear regression line for HRU Count plot
x = all_models["HRU_Count"]
y = all_models["average_hours"]
if len(x) > 1 and len(y) > 1:
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    regression_line = slope * x + intercept
    print(f"HRU Count - slope: {slope}, intercept: {intercept}, r_value: {r_value}, p_value: {p_value}, std_err: {std_err}")
    ax[0].plot(x, regression_line, color='black', linewidth=1, label=f'Regression line: y = {slope:.2e}x + {intercept:.2f}', alpha=0.5)
    ax[0].legend(loc = "upper left")

# Plotting the scatter plot for n_rivers vs Average Hours
scatter2 = ax[1].scatter(all_models["n_rivers"], all_models["average_hours"],
                        c=colors, alpha=0.5, edgecolors="w", linewidth=0.5, s=all_models["Total_Area"] / 1e3)

# Adding labels and title for n_rivers plot
ax[1].set_xlabel("Number of Rivers")
ax[1].set_ylabel("Average execution time (minutes)")
ax[1].set_title("Average execution time vs Number of Rivers")
ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
ax[1].legend(handles=[red_patch, blue_patch], title='Total Area')

# Adding the linear regression line for n_rivers plot
x = all_models["n_rivers"]
y = all_models["average_hours"]
if len(x) > 1 and len(y) > 1:
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    regression_line = slope * x + intercept
    print(f"Number of Rivers - slope: {slope}, intercept: {intercept}, r_value: {r_value}, p_value: {p_value}, std_err: {std_err}")
    ax[1].plot(x, regression_line, color='black', linewidth=1, label=f'Regression line: y = {slope:.2e}x + {intercept:.2f}', alpha=0.5)
    ax[1].legend(loc = "upper left")

# Add the color legend
plt.legend(handles=[red_patch, blue_patch], title='Total Area')

# Save the plot
plt.tight_layout()
plt.savefig("compution_time/average_hours_vs_HRU_n_rivers.png", dpi=300)
plt.show()
