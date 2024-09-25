
import numpy as np
try:
	from MODGenX.utils import *
except:
	from utils import *
import os 
def create_error_zones_and_save(model_path , load_raster_args , ML, fitToMeter = 0.3048):
    
    RESOLUTION = load_raster_args ['RESOLUTION'] 
    raster_paths = generate_raster_paths(RESOLUTION, ML=False)

    SWL_er = fitToMeter * load_raster(raster_paths["SWL_er"], load_raster_args)
    k_horiz_1_er = fitToMeter * load_raster(raster_paths["k_horiz_1_er"], load_raster_args)
    k_horiz_2_er = fitToMeter * load_raster(raster_paths["k_horiz_2_er"], load_raster_args)
    k_vert_1_er = fitToMeter * load_raster(raster_paths["k_vert_1_er"], load_raster_args)
    k_vert_2_er = fitToMeter * load_raster(raster_paths["k_vert_2_er"], load_raster_args)
    thickness_1_er = fitToMeter * load_raster(raster_paths["thickness_1_er"], load_raster_args)
    thickness_2_er = fitToMeter * load_raster(raster_paths["thickness_2_er"], load_raster_args)

    def create_zones_based_on_average(errors):
        zones = np.zeros_like(errors)
        low_quantile = np.nanpercentile(errors, 25)
        high_quantile = np.nanpercentile(errors, 75)

        low_zone_avg = np.median(errors[errors < low_quantile])
        mid_zone_avg = np.median(errors[(errors >= low_quantile) & (errors < high_quantile)])
        high_zone_avg = np.median(errors[errors >= high_quantile])

        zones[errors < low_quantile] = low_zone_avg
        zones[(errors >= low_quantile) & (errors < high_quantile)] = mid_zone_avg
        zones[errors >= high_quantile] = high_zone_avg

        return zones


    # Create zones for each parameter
    SWL_zones = create_zones_based_on_average(SWL_er)
    k_horiz_1_zones = create_zones_based_on_average(k_horiz_1_er)
    k_horiz_2_zones = create_zones_based_on_average(k_horiz_2_er)
    k_vert_1_zones = create_zones_based_on_average(k_vert_1_er)
    k_vert_2_zones = create_zones_based_on_average(k_vert_2_er)
    thickness_1_zones = create_zones_based_on_average(thickness_1_er)
    thickness_2_zones = create_zones_based_on_average(thickness_2_er)



    # Save zone
    def save_zones_np(zones, filename):
        np.save(filename, zones)

    # Save all zones
    def save_all_zones(zones_list, filenames, model_path):
        for zone, fname in zip(zones_list, filenames):
            save_zones_np(zone, os.path.join(model_path, fname))



    # List of all zones and corresponding filenames
    zones_list = [SWL_zones, k_horiz_1_zones, k_horiz_2_zones, k_vert_1_zones, k_vert_2_zones, thickness_1_zones, thickness_2_zones]
    filenames = ["SWL_zones", "k_horiz_1_zones", "k_horiz_2_zones", "k_vert_1_zones", "k_vert_2_zones", "thickness_1_zones", "thickness_2_zones"]

    # Save zones
    save_all_zones(zones_list, filenames, model_path)
    
    
    