import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the data
path = "/data/MyDataBase/SWATGenXAppData/codes/CDL/results/all_years.csv"
cdls = pd.read_csv(path)

# Group by NAME and plot the total area with range of variation
grouped = cdls.groupby('NAME')


# Create a box plot to show the variation for each class across the years
fig, ax = plt.subplots(figsize=(20, 6))
cdls.boxplot(column='AREA', by='NAME', ax=ax, grid=False)
ax.set_xlabel('Landuse')
ax.set_ylabel('AREA (km²)')
ax.set_title('Variation of AREA for each landuse across years')
plt.suptitle('')  # Suppress the automatic 'Boxplot grouped by NAME' title
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("area_variation_boxplot.png", dpi=300)
plt.show()
