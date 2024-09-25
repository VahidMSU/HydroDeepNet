import pandas as pd
import matplotlib.pyplot as plt
import os
path = "/data/MyDataBase/SWATGenXAppData/codes/CDL/results/all_years.csv"

cdls = pd.read_csv(path)
# Group by NAME
### now plot a bar chart of the total area of each class with range of variation for each landuse based on the years
grouped = cdls.groupby('NAME')

for name, group in grouped:
    plt.figure(figsize=(10, 6))
    ## if len of group is less than 2, skip
    if len(group) < 2:
        continue
    plt.plot(group['YEAR'], group['AREA'], marker='o')
    plt.title(f'Time Series of AREA for {name}')
    plt.xlabel('Year')
    plt.ylabel('AREA (km\u00b2)')
    plt.grid(True)
    os.makedirs("time_series", exist_ok = True)
    plt.savefig(f"time_series/{name.replace('/','_').strip()}.jpeg", dpi = 300)
