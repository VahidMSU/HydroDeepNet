import geopandas as gpd
import pandas as pd
import numpy as np
import os
def write_report(message, rep_file):
    with open(rep_file, 'a') as f:
        f.write(message + '\n')

def analyze_huron_river_sites(rep_file, huron_river_watershed_path) -> None:
    if os.path.exists(rep_file):
        os.remove(rep_file)
    bound = gpd.read_file(huron_river_watershed_path).to_crs('EPSG:4326')
    bound = bound.dissolve()
    bound.to_pickle("input_data/Huron_River_basin_bound.pkl")
    min_lat, min_lon, max_lat, max_lon = bound.bounds.values[0]
    write_report(f"{min_lat}, {min_lon}, {max_lat}, {max_lon}", rep_file)

    ss_sites = gpd.GeoDataFrame(pd.read_pickle("input_data/Michigan_PFAS_SITE_Features.pkl"), geometry='geometry', crs='EPSG:4326')
    write_report(f"### total number of sites within Michigan: {len(ss_sites)}", rep_file)

    # Filter out the sites that are outside the watershed by clipping the bounding box using geopandas
    filtered_ss_sites = gpd.clip(ss_sites, bound)
    filtered_ss_sites.to_csv("results/SS_Huron_analysis/Huron_PFAS_SITE_Features.csv")
    write_report(f"### number of sites within the watershed: {len(filtered_ss_sites)}", rep_file)

    write_report(f"number of unique industries: {filtered_ss_sites['Industry'].nunique()}", rep_file)

    # Also write the unique industries
    write_report(f"unique industries: {np.unique(filtered_ss_sites['Industry'])}", rep_file)

    # Also write the number of sites related to each industry
    for industry in np.unique(filtered_ss_sites['Industry']):
        write_report(f"number of sites related to {industry}: {len(filtered_ss_sites[filtered_ss_sites['Industry'] == industry])}", rep_file)

# Call the function
if __name__ == "__main__":
    huron_river_watershed_path = "/data/MyDataBase/SWATGenXAppData/SWAT_input/huc8/4100013/SWAT_plus_Subbasin/SWAT_plus_Subbasin.shp"
    os.makedirs("results/SS_Huron_analysis", exist_ok=True)
    rep_file = "results/SS_Huron_analysis/Huron_River_sites_report.txt"
    analyze_huron_river_sites(rep_file, huron_river_watershed_path)
