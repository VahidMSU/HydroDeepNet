import geopandas as gpd
import pandas as pd
import os
import matplotlib.pyplot as plt

all_wells_path = "/data/MyDataBase/SWATGenXAppData/well_info/all_wells.pkl"
gdf = gpd.GeoDataFrame(pd.read_pickle(all_wells_path), crs='EPSG:26990', geometry='geometry')
gdf.drop(columns=['geometry', 'PERMIT_NUM', 'TOWN', 'RANGE','SCRN_FLAG','NOTES','WELLCODE','WWAT_ID', 'LATITUDE',
       'LONGITUDE', 'METHD_COLL', 'ELEVATION', 'ELEV_METHD', 'WITHIN_CO',
       'WITHIN_SEC', 'LOC_MATCH', 'SEC_DIST', 'ELEV_DEM', 'ELEV_DIF','OWNER_NAME',
       'WELL_NUM', 'DRILLER_ID',
       'DRILL_METH', 'METH_OTHER', 'CASE_TYPE', 'CASE_OTHER',
       'CASE_DIA', 'CASE_DEPTH', 'SCREEN_FRM', 'SCREEN_TO',
       'LANDSYS'], inplace=True)

gdf = gdf.dropna(subset=['PMP_CPCITY'])
gdf = gdf.dropna(subset=['SWL'])
### remove outliers in pump capacity (over 99.5th percentile)
gdf = gdf[gdf.PMP_CPCITY < gdf.PMP_CPCITY.quantile(0.999)]
print(gdf.shape)
print(gdf.columns)
print(gdf.head())
print(gdf.TYPE_OTHER.unique())  #['Commercial' nan 'Public Well Type Unknown' ... 'Type II, standby well', 'Domestic/ Type III public' 'Educational Purposes']
print(gdf.WEL_STATUS.unique())  #[nan 'UNK' 'ACT' 'INACT' 'PLU' 'OTH']
# KEEP ONLY ACTIVE WELLS

gdf = gdf[gdf.WEL_STATUS == 'ACT']
print(gdf.shape)
print(gdf.WELL_TYPE.unique())  #['OTH' 'HOSHLD' 'IRRI' 'UNK' 'TY3PU' 'HEATP' 'TESTW' 'TY2PU' 'INDUS' 'HEATSU' 'TY1PU' nan 'HEATRE']
print(gdf.CONST_DATE.unique())  


# plot pump capacity with respect to static water level
#fig, ax = plt.subplots()
#gdf.plot.scatter(x='PMP_CPCITY', y='SWL', ax=ax)
#plt.show()

### plot the distribuition of pump capacity
fig, ax = plt.subplots()
gdf.PMP_CPCITY.plot.hist(ax=ax, bins=20)
plt.show()



