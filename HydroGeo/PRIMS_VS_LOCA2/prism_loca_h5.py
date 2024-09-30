from HydroGeoDataset import DataImporter


############################## example of loading LOCA2 data ###################################

config = {
        "RESOLUTION": 250,
        "huc8": None,
        "cc_model": "ACCESS-CM2",
        "scenario": "historical",
        "ensemble": "r2i1p1f1",
        "time_range": "1950_2014",
        "cc_time_step": 'daily',  ## alternative option: monthly
        "video": True,
    }

importer = DataImporter(config)
ppt, tmax, tmin = importer.LOCA2(start=1, end=365)  ## kg m-2 s-1, K, K

pr, tmax, tmin = importer.PRISM(years=[2000])


