


### assuming the current coverage is Michigan LP

def MODFLOW_coverage(station_no):
    import os
    import pandas as pd
    import sys
    sys.path.append('/data/SWATGenXApp/codes/SWATGenX')
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths  

    CONUS_streamflow_data = pd.read_csv(SWATGenXPaths.USGS_CONUS_stations_path, dtype={'site_no': str,'huc_cd': str})
    lat = CONUS_streamflow_data.loc[CONUS_streamflow_data['site_no'] == station_no, 'dec_lat_va'].values[0]
    lon = CONUS_streamflow_data.loc[CONUS_streamflow_data['site_no'] == station_no, 'dec_long_va'].values[0]    

    ### check whether it is in Michigan LP
    if (lat > 41.696118 and lat < 47.459853 and lon > -90.418701 and lon < -82.122818):
        print('MODFLOW model is available for the requested station')   
        return True
    else:
        print('MODFLOW model is not available for the requested station')
        return False
    
if __name__ == "__main__":

    station_no = "05290000"

    print(MODFLOW_coverage(station_no))
