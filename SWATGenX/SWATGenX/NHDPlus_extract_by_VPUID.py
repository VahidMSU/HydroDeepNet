import contextlib
import os
import geopandas as gpd
#from SWATGenX.NHDPlusExtractor import *
import zipfile
import glob
import os
import utm
import pyproj

try:
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
    from SWATGenX.utils import get_all_VPUIDs
except Exception:
    from SWATGenXConfigPars import SWATGenXPaths
    from utils import get_all_VPUIDs

def NHDPlus_extract_by_VPUID(VPUID):

    # check if the processing is already done
    if os.path.exists(f'{SWATGenXPaths.extracted_nhd_swatplus_path}/{VPUID}/streams.pkl'):

        print(f'##### NHDPlus {VPUID} already extracted #####')

        return None

    extracted_nhd_path = f'{SWATGenXPaths.extracted_nhd_swatplus_path}/{VPUID}/'
    if not os.path.exists(extracted_nhd_path):
        os.makedirs(extracted_nhd_path)

    print('Extracting NHDPlus layers for preprocessing')
    nhdplus_zipped_path = "/data/SWATGenXApp/GenXAppData/NHDPlusHR/CONUS"
    nhdplus_zipped_names = os.listdir(nhdplus_zipped_path)
    nhdplus_zipped_file_path = next(
        (
            os.path.join(nhdplus_zipped_path, file)
            for file in nhdplus_zipped_names
            if f'_{VPUID}_' in file and file.endswith(".zip")
        ),
        None,
    )
    unzipped_NHDPlusVPU = os.path.join(extracted_nhd_path, 'unzipped_NHDPlusVPU')
    if not os.path.exists(unzipped_NHDPlusVPU):
        if nhdplus_zipped_file_path is None:
            raise ValueError(f"#####  No zip files found for VPUID {VPUID} #####")
        else:
            print(f'##### NHDPlus {VPUID} zipped path: {nhdplus_zipped_file_path}')



        os.makedirs(unzipped_NHDPlusVPU, exist_ok=True)

        with zipfile.ZipFile(nhdplus_zipped_file_path, 'r') as zip_ref:
            zip_ref.extractall(unzipped_NHDPlusVPU)

    print(f"NHDPlus {VPUID} unzipped path: {unzipped_NHDPlusVPU}")

    layers = ['WBDHU8', 'WBDHU12', 'NHDPlusFlowlineVAA', 'NHDFlowline', 'NHDPlusCatchment', 'NHDWaterbody']

    print(f'Extracting and filtering NHDPlus layers for {VPUID}')

    # get the dirname and find the gdb file
    nhdplus_unzipped_files = os.listdir(unzipped_NHDPlusVPU)
    gdf_path = os.path.join(unzipped_NHDPlusVPU,  next(file for file in nhdplus_unzipped_files if file.endswith('.gdb')))

    print(f'##### NHDPlus {VPUID} GeoDatabase path: {gdf_path}')



    def find_utm_proj(lon, lat):
        """
        Find a suitable UTM projection (zone) for lon and lat.

        .. warning::

        UTM is only defined between 80S and 84N. Should use UPS for those regions.

        Returns:

        pyproj.Proj in `utm` projection.
        """
        print(f'##### Finding UTM projection for {lon}, {lat} #####')
        ## send warning if is outside the range
        if lat < -80 or lat > 84:
            print(
                '########################### Warning: UTM is only defined between 80S and 84N. Should use UPS for those regions. ###########################'
            )

        _, _, zone_no, _ = utm.from_latlon(lat, lon)
        band = 'south' if lat < 0 else 'north'

        return pyproj.Proj(
            '+proj=utm +zone={zone:d} +{band} +ellps=WGS84 +datum=WGS84 +units=m +no_defs'
            .format(zone=zone_no, band=band))

    # search for utm zone
    # first read huc4 and get the centroid
    _gdf = gpd.read_file(gdf_path, layer='WBDHU4')
    _gdf = _gdf.to_crs('EPSG:4326')  # Projecting to WGS84 geographic CRS

    utm_zone = find_utm_proj(_gdf.geometry.centroid.x.values[0], _gdf.geometry.centroid.y.values[0])
    print(f'##### UTM Zone: {utm_zone} #####')

    for layer in layers:
        print(f'Extracting {layer}')

        layer_data = read_project_filter(VPUID, layer, gdf_path, utm_zone)

        layer_data.to_pickle(os.path.join(extracted_nhd_path, f'{layer}.pkl'))

        print(f'##### {layer} for {VPUID} saved to {extracted_nhd_path} #####')

    print(f'##### NHDPlus {VPUID} extraction complete #####')



def read_project_filter(VPUID, layer, gdf_path, utm_zone):
    gdf = gpd.read_file(gdf_path, layer=layer)

    fields_to_exclude = ['Resolution', 'FDate', 'Enabled', 'GNIS_ID', 'GNIS_Name', 'Shape_Length',
                        'ElevFixed', 'MaxElevRaw', 'RtnDiv', 'ToMeas', 'LevelPathI', 'InNetwork',
                        'ReachCode', 'VisibilityFilter', 'Elevation', 'Shape_Area', 'GapDistKm',
                        'DnLevelPat', 'DnLevel', 'Thinner', 'VPUIn', 'VPUOut', 'DivDASqKm', 'StatusFlag',
                        'FlowDir', 'loaddate']


    column_name_map = {
        'ftype': 'FType',
        'fcode': 'FCode',
        'nhdplusid': 'NHDPlusID',
        'uphydroseq': 'UpHydroSeq',
        'hydroseq': 'HydroSeq',
        'dnhydroseq': 'DnHydroSeq',
        'startflag': 'StartFlag',
        'terminalfl': 'TerminalFl',
        'divergence': 'Divergence',
        "streamorde": "StreamOrde",
        "elevelev": "ElevElev",
        'vpuid': 'VPUID',
        'permanent_identifier': 'Permanent_Identifier',
        "wbarea_permanent_identifier": "WBArea_Permanent_Identifier",
        'maxelevsmo': 'MaxElevSmo',
        'minelevsmo': 'MinElevSmo',
        'areasqkm': 'AreaSqKm',
        'lengthkm': 'LengthKM',
        'totdasqkm': 'TotDASqKm',
        'geometry': 'geometry'
    }

    ##map the column if exists
    for col in gdf.columns:
        if col.lower() in column_name_map:
            print(f'##### Mapping {col} to {column_name_map[col.lower()]} #####')
            gdf.rename(columns={col: column_name_map[col.lower()]}, inplace=True)

    gdf = gdf[[col for col in gdf.columns if col not in fields_to_exclude]]

    # Convert large numeric fields to strings
    large_numeric_fields = ['NHDPlusID', 'HydroSeq', 'DnHydroSeq', 'UpHydroSeq']
    for field in large_numeric_fields:
        if field in gdf.columns:
            gdf[field] = gdf[field].astype(str)

    if 'VAA' in layer:
        with contextlib.suppress(Exception):
            gdf = gdf.drop(columns='geometry')
    else:
        gdf = gdf.to_crs(f"{utm_zone}")
        print(f'##### {layer} for {VPUID} projected to {utm_zone} #####')

    if 'WBD' in layer:
        gdf.columns = gdf.columns.str.lower()
        gdf['VPUID'] = VPUID

    # if huc12 or huc8 or huc4 are in capital letters, convert them to lower case
    if 'huc12' in gdf.columns:
        gdf.rename(columns={'huc12': 'huc12'}, inplace=True)
    if 'huc8' in gdf.columns:
        gdf.rename(columns={'huc8': 'huc8'}, inplace=True)
    if 'huc4' in gdf.columns:
        gdf.rename(columns={'huc4': 'huc4'}, inplace=True)

    if 'tohuc' in gdf.columns:
        gdf.rename(columns={'tohuc': 'tohuc'}, inplace=True)

    if layer == 'WBDHU8':
        selected_columns = ['huc8', 'name', 'VPUID', 'geometry']
        gdf = gdf[~gdf['huc8'].isna()][selected_columns].explode(index_parts=False)
        gdf['area'] = gdf['geometry'].area
        gdf = gdf.sort_values(by='area', ascending=False).drop_duplicates(subset=["huc8"]).drop(columns='area')

    if layer == 'WBDHU12':
        selected_columns = ['huc12','tohuc',  'name', 'VPUID', 'geometry']
        gdf = gdf[~gdf['huc12'].isna()][selected_columns].explode(index_parts=False)
        gdf['area'] = gdf['geometry'].area
        gdf = gdf.sort_values(by='area', ascending=False).drop_duplicates(subset=["huc12"]).drop(columns='area')

    return gdf



if __name__ == "__main__":

    #VPUIDs = get_all_VPUIDs()
    #print("VPUIDs", VPUIDs)
    VPUIDs = ["0405"]
    for VPUID in VPUIDs:
        NHDPlus_extract_by_VPUID(VPUID)
