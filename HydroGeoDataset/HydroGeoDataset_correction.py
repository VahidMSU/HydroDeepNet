import h5py
path = "/data/MyDataBase/HydroGeoDataset_ML_250.h5"

with h5py.File(path, 'r+') as f:
    print(f"Keys: {f.keys()}")
    ## remove all datasets starting with recharge_
    for key in list(f.keys()):
        if key.startswith("recharge_"):
            del f[key]

    ### create LANDFIRE group
    if "LANDFIRE" not in f:
        f.create_group("LANDFIRE")

    ### move all files starting with LC20_ or LC22_ to LANDFIRE group
    for key in list(f.keys()):
        if key.startswith("LC20_") or key.startswith("LC22_"):
            f.move(key, f"LANDFIRE/{key}")
    
    ### create group PRISM if it does not exist
    if "PRISM" not in f:
        f.create_group("PRISM")
        f.create_group("PRISM/annual_average")

    ### move all files starting with ppt_ to PRISM/annual_average group
    for key in list(f.keys()):
        if key.startswith("ppt_"):
            f.move(key, f"PRISM/annual_average/{key}")


    ### create a group called EBK if it does not exist
    if "EBK" not in f:
        f.create_group("EBK")
    
    ### move all files starting with kriging_ to EBK group
    for key in list(f.keys()):
        if key.startswith("kriging_"):
            f.move(key, f"EBK/{key}")
    
    ### create a group called population if it does not exist
    if "population" not in f:
        f.create_group("population")
    
    ### move all files starting with pden to population group
    for key in list(f.keys()):
        if key.startswith("pden"):
            f.move(key, f"population/{key}")

    ### create a group called NHDPlus if it does not exist
    if "NHDPlus" not in f:
        f.create_group("NHDPlus")

    ### movel files start with VAMA_ or Q to NHDPlus group
    for key in list(f.keys()):
        if key.startswith("VAMA_") or key.startswith("Q") or key.startswith("GageQMA_") or key.startswith("GageIDMA_") or key.startswith("GageQ_") or key.startswith("VBMA_"):
            f.move(key, f"NHDPlus/{key}")
    ### move the 'VCMA_MILP_250m', 'VDMA_MILP_250m', 'VEMA_MILP_250m' to NHDPlus group
    for key in list(f.keys()):
        if key.startswith("VCMA_MILP_250m") or key.startswith("VDMA_MILP_250m") or key.startswith("VEMA_MILP_250m") or key.startswith('NHDPlusID_250m'):
            f.move(key, f"NHDPlus/{key}")

    ### move  'ArQNavMA_MILP_250m', 'AvgQAdjMA_MILP_250m', 'GageAdjMA_MILP_250m', to NHDPlus group
    for key in list(f.keys()):
        if key.startswith("ArQNavMA_MILP_250m") or key.startswith("AvgQAdjMA_MILP_250m") or key.startswith("GageAdjMA_MILP_250m"):
            f.move(key, f"NHDPlus/{key}")

    ### move 'HUC12_250m', 'HUC8_250m', to NHDPlus group
    for key in list(f.keys()):
        if key.startswith("HUC12_250m") or key.startswith("HUC8_250m"):
            f.move(key, f"NHDPlus/{key}")

    ### Wellogic group
    if "Wellogic" not in f:
        f.create_group("Wellogic")
    
    ### move all files starting with obs_ to Wellogic group
    for key in list(f.keys()):
        if key.startswith("obs_"):
            f.move(key, f"Wellogic/{key}")

    ### move PRISM_monthly group to PRISM group 
    if "PRISM_monthly" in f:
        if "PRISM" in f:
            if "PRISM_monthly" not in f["PRISM"]:
                f.move("PRISM_monthly", "PRISM/PRISM_monthly")
        else:
            f.move("PRISM_monthly", "PRISM")

    #### create a group called geospatial if it does not exist
    if "geospatial" not in f:
        f.create_group("geospatial")

    ### move all files starting with Soil_
    for key in list(f.keys()):
        if key.startswith("Soil_"):
            f.move(key, f"geospatial/{key}")

    ### move  'landforms_250m_250Dis', 'landuse_250m' 'geomorphons_250m_250Dis' 

    for key in list(f.keys()):
        if key.startswith("landforms_250m_250Dis") or key.startswith("landuse_250m") or key.startswith("geomorphons_250m_250Dis"):
            f.move(key, f"geospatial/{key}")

    


    ### create a group called climate_pattern if it does not exist
    if "climate_pattern" not in f:
        f.create_group("climate_pattern")

    ### move all files starting with snow_ to climate_pattern group
    for key in list(f.keys()):
        if key.startswith("snow_") or key.startswith("non_snow_") or key.startswith("snowpack_"):
            f.move(key, f"climate_pattern/{key}")


    ### move 'melt_rate_raster_250m' to climate_pattern group
    for key in list(f.keys()):
        if key.startswith("melt_rate_raster_250m"):
            f.move(key, f"climate_pattern/{key}")

    #### move 'COUNTY_250m', 'DEM_250m' to geospatial group
    for key in list(f.keys()):
        if key.startswith("COUNTY_250m") or key.startswith("DEM_250m"):
            f.move(key, f"geospatial/{key}")

    ### remove '128_128_batch_size', '256_256_batch_size', '512_512_batch_size', '64_64_batch_size', if they exist
    for key in list(f.keys()):
        if key.startswith("128_128_batch_size") or key.startswith("256_256_batch_size") or key.startswith("512_512_batch_size") or key.startswith("64_64_batch_size"):
            del f[key]

    ## move 'Aquifer_Characteristics_Of_Glacial_Drift_250m', 'BaseRaster_250m', to geospatial group
    for key in list(f.keys()):
        if key.startswith("Aquifer_Characteristics_Of_Glacial_Drift_250m") or key.startswith("BaseRaster_250m"):
            f.move(key, f"geospatial/{key}")

    ## move 'MI_geol_poly_250m', to geospatial group
    for key in list(f.keys()):
        if key.startswith("MI_geol_poly_250m"):
            f.move(key, f"geospatial/{key}")

    ### move  'Glacial_Landsystems_250m', to geospatial group
    for key in list(f.keys()):
        if key.startswith("Glacial_Landsystems_250m"):
            f.move(key, f"geospatial/{key}")

    ### move 'PETMA_MILP_250m' to climate_pattern group
    for key in list(f.keys()):
        if key.startswith("PETMA_MILP_250m"):
            f.move(key, f"climate_pattern/{key}")

    ### move 'average_temperature_raster_250m' to climate_pattern group
    for key in list(f.keys()):
        if key.startswith("average_temperature_raster_250m"):
            f.move(key, f"climate_pattern/{key}")

    ### move 'x_250m', 'y_250m',  'lat_250m', 'lon_250m', 'mask_250m' to geospatial group
    for key in list(f.keys()):
        if key.startswith("x_250m") or key.startswith("y_250m") or key.startswith("lat_250m") or key.startswith("lon_250m") or key.startswith("mask_250m"):
            f.move(key, f"geospatial/{key}")

    ### create a group called MODIS if it does not exist
    if "MODIS" not in f:
        f.create_group("MODIS")

    ## move all groups  'MOD09GQ_sur_refl_b01', 'MOD09GQ_sur_refl_b02', 'MOD13Q1_EVI', 'MOD13Q1_NDVI', 'MOD15A2H_Fpar_500m', 'MOD15A2H_Lai_500m', 'MOD16A2_ET', 'MODIS_ET', to MODIS group
    for key in list(f.keys()):
        if key.startswith("MOD09GQ_sur_refl_b01") or key.startswith("MOD09GQ_sur_refl_b02") or key.startswith("MOD13Q1_EVI") or key.startswith("MOD13Q1_NDVI") or key.startswith("MOD15A2H_Fpar_500m") or key.startswith("MOD15A2H_Lai_500m") or key.startswith("MOD16A2_ET") or key.startswith("MODIS_ET"):
            f.move(key, f"MODIS/{key}")

    ### move 'gSURRGO_swat_250m' to gssurgo group
    for key in list(f.keys()):
        if key.startswith("gSURRGO_swat_250m"):
            f.move(key, f"gssurgo/{key}")

    #move Recharge_250m to geospatial group
    for key in list(f.keys()):
        if key.startswith("Recharge_250m"):
            f.move(key, f"geospatial/{key}")

    ### remove SNODAS_250m if it exists
    for key in list(f.keys()):
        if key.startswith("SNODAS_250m"):
           del f[key]

 