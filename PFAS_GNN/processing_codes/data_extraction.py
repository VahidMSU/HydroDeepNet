import HydroGeoDataset as hgd
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import time 
def extraction(input_path, output_path):
    # Load the dataset
    #extract information for point locations
    config = {
        "RESOLUTION": 250}
    importer = hgd.DataImporter(config)
    gdf = importer.extract_features(input_path)
    gdf.to_pickle(output_path)
    ## save to geojson
    gdf.to_file(output_path.replace(".pkl", ".geojson"), driver='GeoJSON')
    print(f"############################## OUT FINAL IMPORTED DATA: {gdf} ##############################")
    print(f"############################## OUT FINAL IMPORTED DATA: {gdf.columns} ##############################")

def plot(gdf, bound, unconfirmed_pfas_sites, biosolids_sites):
    fig, ax = plt.subplots()
    biosolids_sites = gpd.read_file(biosolids_sites, driver='GeoJSON', crs='EPSG:4326')
    biosolids_sites.plot(ax=ax, color='green')
    suspected_sites = gpd.GeoDataFrame(pd.read_pickle(unconfirmed_pfas_sites), geometry='geometry', crs='EPSG:4326')
    suspected_sites.plot(ax=ax, color='blue')
    gdf.plot(ax=ax, color='red')
    plt.legend(["Biosolid leaching", "Suspected PFAS Sites", "Confirmed PFAS Sites"])
    bound.boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)
    ## add details  
    plt.title("Michigan confirmed PFAS Sites clipped")
    #plt.xlabel('Longitude')
    #plt.ylabel('Latitude')
    ## remove x and y axis
    plt.xticks([])
    plt.yticks([])
    plt.grid()
    ## add legend
    
    plt.savefig("figs/Michigan_confirmed_PFAS_Sites_clipped.png")
def clip(Huron_River_basin_bound, confimed_sites_with_features, output_path) -> gpd.GeoDataFrame:
    """ return huron river pfas sites clipped to the basin boundary"""
    gdf = gpd.GeoDataFrame(pd.read_pickle(confimed_sites_with_features), geometry='geometry', crs='EPSG:4326')
    bound = gpd.read_file(Huron_River_basin_bound, driver='GeoJSON', crs='EPSG:4326')
    gdf = gpd.clip(gdf, bound)
    print(f" gdf clipped: {list(gdf.Type)}")    
    #time.sleep(10)
    gdf.to_pickle(output_path)
    ## save to geojson
    gdf.to_file(output_path.replace(".pkl", ".geojson"), driver='GeoJSON')
    return gdf, bound
    


if __name__ == '__main__':
    ### now clip
    #input_path =  '/data/MyDataBase/HuronRiverPFAS/Michigan_confirmed_PFAS_Sites.geojson'
    #output_path = "/data/MyDataBase/HuronRiverPFAS/Michigan_confirmed_PFAS_Sites_with_features.pkl"
    #extraction(input_path, output_path)
    Huron_River_basin_bound =  "/data/MyDataBase/HuronRiverPFAS/Huron_River_basin_bound.geojson"
    #confimed_sites_with_features = "/data/MyDataBase/HuronRiverPFAS/Michigan_confirmed_PFAS_Sites_with_features.pkl"
    #output_path = "/data/MyDataBase/HuronRiverPFAS/Huron_confirmed_PFAS_SITE_Features.pkl"
    #unconfirmed_pfas_sites = "/data/MyDataBase/HuronRiverPFAS/Huron_PFAS_SITE_Features.pkl"
    #biosolids_sites = "/data/MyDataBase/HuronRiverPFAS/Huron_Biosolid_sites.geojson"
    #output_path = "/data/MyDataBase/HuronRiverPFAS/Huron_Biosolid_sites_with_features.pkl"
    huron_river_grids = "/data/MyDataBase/HuronRiverPFAS/Huron_River_Grid_250m.pkl"


    feature_output_path = "/data/MyDataBase/HuronRiverPFAS/Huron_River_Grid_250m_with_features.pkl" 
    extraction(huron_river_grids, feature_output_path)
