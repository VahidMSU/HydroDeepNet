import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
years = np.arange(2008, 2023, 1)

all_years = []
for year in years:
# File paths
    cdls = f"/data/MyDataBase/SWATGenXAppData/CDL landuse/{year}_30m_cdls_MIP_26990_projected_clipped.tif"
    cdls_lookup = "/data/MyDataBase/SWATGenXAppData/CDL landuse/CDL_CODES.csv"
    swat_cdl = "/data/MyDataBase/SWATGenXAppData/CDL landuse/CDL_SWAT_LOOKUP.csv"
    swat_cdl = pd.read_csv(swat_cdl) # LANDUSE,SWAT_CODE

    lookup_table = pd.read_csv(cdls_lookup) #CODE,NAME
    # Open the CDL raster and read the array
    with rasterio.open(cdls) as src:
        cdls_array = src.read(1)
        no_data_value = src.nodata
        cellSize = src.res[0]*1e-3
    ## total area of the raster
    cdls_array[cdls_array == no_data_value] = 0
    total_area = np.count_nonzero(cdls_array) * cellSize**2
    print("Total area of the raster: ", round(total_area,2), "km\u00b2")
    ### area of each class
    dict = {"NAME": [], "CDL_CODE": [], "AREA": [], "PERCENTAGE": [], "YEAR": []}
    for value in np.unique(cdls_array):
        if value == 0:
            continue
        #print(f"Class {value}")
        ## if value was 176, print the number of pixels that have this value
        #print(f"Number of pixels: {np.count_nonzero(cdls_array == value)}")
        name = lookup_table.loc[lookup_table['CODE'] == value, 'NAME'].values[0]
        #print(f"Class {value} is {name}")
        area = np.count_nonzero(cdls_array == value) * cellSize**2
        #print(f"Class {value} covers an area of {round(area, 2)} km\u00b2")
        print(f"NAME: {name}, CDL_CODE: {value}, AREA: {round(area, 2)} km\u00b2, PERCENTAGE: {round(area/total_area*100, 2)}%")
        dict["NAME"].append(name)
        dict["CDL_CODE"].append(value)
        dict["AREA"].append(round(area, 2))
        dict["PERCENTAGE"].append(round(area/total_area*100, 2))
        dict["YEAR"].append(year)
    df = pd.DataFrame(dict)
    os.makedirs("results", exist_ok = True)
    df = df.merge(swat_cdl, left_on = "CDL_CODE", right_on = "LANDUSE", how = "left").drop(columns = ["LANDUSE"])
    df.to_csv(f"results/{year}_30m_cdls_MIP_26990_projected_clipped.csv", index=False)
    all_years.append(df)

    ### group by SWAT_CODE, sum and plot bar
    df_grouped = df.groupby("SWAT_CODE").sum().reset_index()
    df_grouped = df_grouped.sort_values(by = "AREA", ascending = False)
    fig, ax = plt.subplots(figsize = (10, 6))
    df_grouped.plot.bar(x = "SWAT_CODE", y = "PERCENTAGE", ax = ax)
    ax.set_xticklabels(df_grouped["SWAT_CODE"])
    ax.set_xlabel("SWAT_CODE")
    ax.set_ylabel("Percentage")
    ax.set_title("Landuse composition")
    ax.legend(["Percentage"])
    plt.grid(axis= "both", linestyle='--', linewidth=0.5)
    plt.tight_layout()

    plt.savefig(f"results/{year}_30m_cdls_MIP_26990_projected_clipped.png", dpi = 300)
    plt.close()
all_years = pd.concat(all_years)
all_years.to_csv("results/all_years.csv", index = False)
