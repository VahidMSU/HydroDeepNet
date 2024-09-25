import contextlib
import geopandas as gpd
import pandas as pd
import os
import numpy as np
import os




def start_extracting(BASE_PATH, list_of_HUC,  LEVEL, VPUID):
    """
    The start_extracting function takes three parameters:
    - BASE_PATH: Directory path for loading stream data.
    - GEOGRAPHIC_EXTENT: A map specifying filtering criteria.
    - LEVEL: The criteria name to be used from the GEOGRAPHIC_EXTENT for filtering streams.

    The function performs the following operations:
    1. It loads stream data from a pickled GeoDataFrame located in the directory specified by 'BASE_PATH'.
    2. Filters the streams based on the LEVEL from GEOGRAPHIC_EXTENT.
    3. Creates initial subbasins using 'huc12' and 'Subbasin_level_1' as criteria for unique identifiers.
    4. Refines these initial subbasins according to the one-outlet rule.
    5. Checks for any missing subbasin values.
    6. Sets zeros for HydroSeqs at outlets and headwaters.

    It returns the processed and refined streams GeoDataFrame.
    """


    def creating_unique_subbasin(df, first_criteria, second_criteria):
        """
        Create unique subbasin identifiers based on two criteria.
        """
        df['Subbasin_updated'] = pd.factorize(df[first_criteria].astype(str) + "_" + df[second_criteria].astype(str))[0] + 1
        print('Subbasin IDs created based on combination of', first_criteria, 'and', second_criteria)
        with contextlib.suppress(Exception):
            df = df.drop(columns='Subbasin')

        df = df.rename(columns={'Subbasin_updated':'Subbasin'})
        print('Subbasin IDs updated based on combination of', first_criteria, 'and', second_criteria)
        return df

    def find_upstreams(hydroseq, streams_df):
        """
        Recursive function to find all upstream segments.

        Parameters:
        - hydroseq: The HydroSeq of the segment you want to find upstreams for.
        - streams_df: The streams dataframe.

        Returns:
        - A list of HydroSeq IDs that are upstream of the provided HydroSeq.
        """
        upstreams = streams_df[streams_df.DnHydroSeq == hydroseq].HydroSeq.values
        all_upstreams = list(upstreams)
        for u in upstreams:
            all_upstreams.extend(find_upstreams(u, streams_df))
        return all_upstreams

    def create_subbasins(streams_df):
        streams_df['Subbasin_sub'] = 0
        next_subbasin_sub_id = 1
        streams_df.sort_values(by='HydroSeq', inplace=True)

        while True:
            subbasin_outlet_count = {}

            # Group by subbasin and count outlets
            for subbasin, subbasin_df in streams_df.groupby('Subbasin'):
                outlets_counter = 0
                processed_DnHydroSeqs = set()

                for DnHydroSeq, downstream_df in subbasin_df.groupby('DnHydroSeq'):
                    if DnHydroSeq in processed_DnHydroSeqs:
                        continue

                    downstream_subbasin_values = streams_df[streams_df.HydroSeq == DnHydroSeq].Subbasin.values

                    if downstream_subbasin_values.size > 0 and subbasin != downstream_subbasin_values[0]:
                        outlets_counter += 1
                    elif DnHydroSeq not in streams_df['HydroSeq'].values:
                        outlets_counter += 1

                    if outlets_counter > 1:
                        upstream_segments = find_upstreams(DnHydroSeq, streams_df)
                        streams_df.loc[streams_df.HydroSeq.isin(upstream_segments), 'Subbasin_sub'] = next_subbasin_sub_id
                        next_subbasin_sub_id += 1
                        processed_DnHydroSeqs.update(upstream_segments)

                subbasin_outlet_count[subbasin] = outlets_counter
            print(streams_df.columns)
            streams_df = creating_unique_subbasin(streams_df, 'Subbasin', 'Subbasin_sub')

            # Verify that no subbasins have more than one outlet
            multiple_outlets = [k for k, v in subbasin_outlet_count.items() if v > 1]
            if not multiple_outlets:
                break

        return streams_df



    # Load data and select the area of interest
    os.makedirs(os.path.join(BASE_PATH,'SWATplus_by_VPUID',VPUID, LEVEL), exist_ok=True)

    streams = gpd.GeoDataFrame(pd.read_pickle(os.path.join(BASE_PATH,  f'NHDPlusData/SWATPlus_NHDPlus/{VPUID}/streams.pkl' )))
    problematic_watersheds = [60001800008141, 60002600018915]
    streams = streams[~streams.NHDPlusID.isin(problematic_watersheds)].reset_index(drop=True)
    problematic_huc12 = [40900010502,40900040502,40900040603,40900040602,40900040703, 40900040503, 40900040601,40900040501,40900010502,40900010501,40900010504,40900010503]  ### these are those huc12 outside the scope of the reseaech
    streams = streams[~streams.huc12.isin(problematic_huc12)].reset_index(drop=True)
    streams = streams[~streams.huc12.isna()].reset_index(drop=True)
    print('Number of streams loaded:', streams.shape[0])

    streams = streams[streams[LEVEL].isin(list_of_HUC)].reset_index(drop=True)
    print('Number of streams extracted:', streams.shape[0])

    # Create initial subbasins
    streams = creating_unique_subbasin(streams, 'huc12', 'Subbasin_level_1')
    print('Initial number of subbasins:', streams['Subbasin'].nunique())

    # Refine subbasins based on the one-outlet rule
    streams = create_subbasins(streams)
    print('Final number of subbasins:', streams['Subbasin'].nunique())



    def setting_zero_for_outlets_and_headwaters(df):
        df['DnHydroSeq'] = np.where(df.DnHydroSeq.isin(df.HydroSeq), df.DnHydroSeq, 0)
        df['UpHydroSeq'] = np.where(df.UpHydroSeq.isin(df.HydroSeq), df.UpHydroSeq, 0)
        df['DnHydroSeq'] = df['DnHydroSeq'].fillna(0).astype('Int64')
        df['UpHydroSeq'] = df['UpHydroSeq'].fillna(0).astype('Int64')
        return df

    streams = setting_zero_for_outlets_and_headwaters(streams)

    return streams




def incorporate_lakes(BASE_PATH, streams, VPUID):
    """ Once the streams are processed, we will include lakes. We only include those lakes that are connected to streams and have area more than 0.1 skqm.
    The incorporate_lakes function updates the streams GeoDataFrame to include relevant lake information (LakeId and Permenant_Identifier to use it later to connect Lakes with Streams).
    - First, it loads and filters lakes based on their area, only considering those larger than a given criteria (default 1 SqKm).
    - Next, the function assigns unique LakeIds based on Permanent Identifiers.
    - Finally, it merges the modified lake data with the streams GeoDataFrame, updating the streams with corresponding LakeIds.

    The function returns the modified streams GeoDataFrame with added lake information.
    """
    def load_and_clean_lakes(VPUID, criteria=0.1):
        Lakes_path = os.path.join(BASE_PATH,f'NHDPlusData/SWATPlus_NHDPlus/{VPUID}/NHDWaterbody.pkl')
        Lakes = gpd.GeoDataFrame(pd.read_pickle(Lakes_path), geometry='geometry')
        print('Total number of lakes added', len(Lakes))
        Lakes ['LakeId'] = Lakes.Permanent_Identifier


        criteria = 0.1
        print(f"################# minimum area of lakes to be considered: {criteria} SqKm ################")
        Lakes = Lakes[Lakes.AreaSqKm>criteria].reset_index(drop=True)
        Lakes = Lakes.reset_index(drop=True)
        print(f'Lakes more than {criteria} SqKm', len(Lakes))

        return Lakes

    def assign_new_unique_values(df, column_name, new_columns,correspondings):
        """
        Assign new unique integer values starting from 0 for the unique values in a column or series of columns.
        """
        unique_values = df[column_name].unique()
        value_map = {value: idx for idx, value in enumerate(unique_values)}
        c=0
        for new_column_name  in new_columns:
            df[new_column_name] = df[correspondings[c]].map(value_map)
            c=1+c
        return df

    def adding_lake_ids(streams, VPUID):
        Lakes= load_and_clean_lakes(VPUID)

        streams = streams.merge(Lakes[['LakeId','Permanent_Identifier']], left_on='WBArea_Permanent_Identifier',
                                right_on='Permanent_Identifier', how='left')
        Lakes=Lakes[Lakes.Permanent_Identifier.isin(streams.WBArea_Permanent_Identifier)].reset_index(drop=True)
        return streams

    return adding_lake_ids(streams, VPUID)




def include_lakes_in_streams(streams):
    """
    include_lakes_in_streams function identifies and tags lake inlets, outlets, and main lakes within a stream network.
    - LakeIn: Stream segments that directly flow into a lake.
    - LakeOut: Stream segments that flow out from a lake.
    - LakeMain: The main lake associated with each LakeOut, identified by the maximum stream order.
    - LakeWithin: Stream segments that are within a lake boundary.


    NOTE: Remember that with this code, LakeOut and LakeIn will are not be inside the lakes polygones unless there is a special case where the outlet of the lake is the outlet of the basin.

    """

    # Initialize new columns with -9999  (I found this an effective approach for handling nan and null values)
    print('################################# Unique LAKEID',streams.LakeId.unique())
    streams['LakeId'] = streams['LakeId'].fillna(-9999)
    streams['LakeIn'] = -9999
    streams['LakeOut'] = -9999
    streams['LakeMain'] = -9999
    streams['LakeWithin'] = streams['LakeId'].fillna(-9999)  ## initially considering all lakes having lakeId as LakeWithin. We will modify if later


    # Traverse downstream from each headwater
    print('start traversing downstream to find LakeIn ids')
    print('NUMBER OF HEAD WATERS:',len(streams[streams['UpHydroSeq'] == 0]))

    downstream_dict = pd.Series(streams.DnHydroSeq.values, index=streams.HydroSeq).to_dict()
    lake_dict = pd.Series(streams.LakeId.values, index=streams.HydroSeq).to_dict()


    # Update LakeIn for each stream based on downstream LakeId
    for hydroseq, dn_hydro_seq in downstream_dict.items():
        if dn_hydro_seq == 0:  # Skip cases where 'DnHydroSeq' is 0
            continue
        if lake_dict.get(dn_hydro_seq, -9999) != -9999 and lake_dict.get(hydroseq, -9999) == -9999:
            streams.loc[streams['HydroSeq'] == hydroseq, 'LakeIn'] = lake_dict[dn_hydro_seq]

    def identify_lake_out(streams):
        # Group streams by LakeId, ignoring the -9999 group
        grouped_streams = streams[streams['LakeId'] != -9999].groupby('LakeId')

        for lake_id, group in grouped_streams:
            # Handle special case: outlet of the watershed with only one stream having LakeId
            if len(group) == 1:
                single_row = group.iloc[0]
                dn_hydro_seq = single_row['DnHydroSeq']
                if dn_hydro_seq == 0:  # This is the outlet of the watershed
                    streams.loc[group.index, ['LakeId', 'LakeIn', 'LakeWithin']] = -9999
                    continue

            # Handle special case where there is only one inlet and one flowline within the lake
            if len(group) == 1:
                single_row = group.iloc[0]
                dn_hydro_seq = single_row['DnHydroSeq']
                dn_row = streams.loc[streams['HydroSeq'] == dn_hydro_seq]
                if not dn_row.empty and dn_row['LakeId'].iloc[0] == -9999:
                    streams.loc[streams['HydroSeq'] == dn_hydro_seq, 'LakeOut'] = lake_id
                    streams.loc[streams['HydroSeq'] == dn_hydro_seq, 'LakeWithin'] = -9999  ## so no more LakeWithin when we have LakeOut
                continue  # Skip the rest of the loop for this special case

            for idx, row in group.iterrows():
                hydro_seq = row['HydroSeq']

                # Get upstream and downstream HydroSeq
                up_hydro_seq = row['UpHydroSeq']
                dn_hydro_seq = row['DnHydroSeq']

                # Retrieve corresponding upstream and downstream rows
                up_row = streams.loc[streams['HydroSeq'] == up_hydro_seq]
                dn_row = streams.loc[streams['HydroSeq'] == dn_hydro_seq]

                # Apply the condition to determine if it's a LakeOut
                if not dn_row.empty and dn_row['LakeId'].iloc[0] == -9999:
                    streams.loc[streams['HydroSeq'] == dn_hydro_seq, 'LakeOut'] = lake_id
                    streams.loc[streams['HydroSeq'] == dn_hydro_seq, 'LakeWithin'] = -9999

    # Then run the function identify_lake_out to populate LakeOut
    print('start finding the lakeOuts')
    identify_lake_out(streams)
    streams.replace(-9999, np.nan, inplace=True)
    # Identify main lake for each LakeOut
    lakes_without_outlets=streams[~streams.LakeId.isin(streams.LakeOut)].LakeId.unique()
    if len(lakes_without_outlets)>0:
        ### Special case: Sometimes two lakes adjacent to each other have different name, resulting in upstream lake become without LakeOut. In this condition we change the Name of upstream lake to downstream lake
        ## getting the HydroSeq of Lakes without lakeOut
        for lwo in lakes_without_outlets:
            downstreamLake_HydroSeq=streams[streams.LakeId.isin([lwo])].DnHydroSeq.values
            downstreamLake_LakeId = streams[streams.HydroSeq.isin(downstreamLake_HydroSeq)].LakeId.unique()
            if len(downstreamLake_LakeId)>0:
                print("##################",downstreamLake_LakeId,"###################")
                for i in ['LakeId', 'LakeWithin', 'LakeIn']:
                    for j in downstreamLake_LakeId:
                        streams[i] = np.where(streams[i].isin([lwo]), j, streams[i])

        ### Special case: sometimes the outlet of the lake is the actually the outlet of the basin.
        ### In this case we need to change the LakeWithin and LakeId where DnHydroSeq is nan
        streams['LakeOut']=np.where((streams.LakeId.isin(lakes_without_outlets)) & (streams.DnHydroSeq==0) , streams['LakeId'], streams['LakeOut'])
        streams['LakeId']=np.where((streams.LakeId.isin(lakes_without_outlets)) & (streams.DnHydroSeq==0) , np.nan, streams['LakeId'])
        streams['LakeWithin']=np.where((streams.LakeWithin.isin(lakes_without_outlets)) & (streams.DnHydroSeq==0) , np.nan, streams['LakeId'])

        #### final check
        lakes_without_outlets=streams[~streams.LakeId.isin(streams.LakeOut)].LakeId.unique()
        if list(lakes_without_outlets):
            print(
                '#######################################################################################################################'
            )
            print(f'######## ERROR ERROR ERROR     THE FOLLOWING LAKES DOES NOT HAVE OUTLETS:  {list(lakes_without_outlets)}   ERROR ERROR ERROR #####'  )
            print(
                '#######################################################################################################################'
            )
        streams['LakeId']=np.where(streams.LakeId.isin(lakes_without_outlets), np.nan, streams['LakeId'])
        streams['LakeIn']=np.where(streams.LakeIn.isin(lakes_without_outlets), np.nan, streams['LakeIn'])
        streams['LakeWithin']=np.where(streams.LakeWithin.isin(lakes_without_outlets), np.nan, streams['LakeWithin'])

    lakes_without_inlets=streams[~streams.LakeId.isin(streams.LakeIn)].LakeId.unique()
    if len(lakes_without_inlets)>0:
        print(f' %%%%%%%   THE FOLLOWING LAKES DOES NOT HAVE INTLET:  {list(lakes_without_inlets)}    %%%%%%%%%%%   '  )
    idx_max_stream_order = streams.groupby('LakeOut')['StreamOrde'].idxmax()
    streams.loc[idx_max_stream_order, 'LakeMain'] = streams.loc[idx_max_stream_order, 'LakeOut']
    # Remap Lake IDs
    unique_lake_ids = streams['LakeId'].dropna().unique()
    lake_id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_lake_ids, 1)}

    lakes_with_in_but_no_out = streams[~streams.LakeId.isin(streams.LakeOut)]
    streams.loc[lakes_with_in_but_no_out [lakes_with_in_but_no_out.DnHydroSeq==0].index, 'LakeOut'] = streams.loc[lakes_with_in_but_no_out [lakes_with_in_but_no_out.DnHydroSeq==0].index, 'LakeWithin']
    streams.loc[lakes_with_in_but_no_out [lakes_with_in_but_no_out.DnHydroSeq==0].index, 'LakeMain'] = streams.loc[lakes_with_in_but_no_out [lakes_with_in_but_no_out.DnHydroSeq==0].index, 'LakeWithin']
    streams.loc[lakes_with_in_but_no_out [lakes_with_in_but_no_out.DnHydroSeq==0].index, 'LakeWithin'] = np.nan

    # changing dtype
    for col in ['LakeId', 'LakeIn', 'LakeOut', 'LakeMain', 'LakeWithin']:
        streams[col] = streams[col].map(lake_id_mapping).astype('Int64')
    return streams


def write_output(BASE_PATH, streams, LEVEL ,NAME, VPUID, EPSG, MODEL_NAME):
    ######  Writing watersheds, streams and lakes based on QSWAT+ requirments
    ########### Final Stage, when we reach here, we are done with modifying streams
    ########## THis is still does not include RES ids


    class LakesProcessor:
        def __init__(self, streams, BASE_PATH, VPUID):
            self.streams = streams
            self.BASE_PATH = BASE_PATH
            self.VPUID = VPUID
        def loading_and_adding_lake_ids(self):
            Lakes_path = os.path.join(self.BASE_PATH, f'NHDPlusData/SWATPlus_NHDPlus/{VPUID}/NHDWaterbody.pkl')
            Lakes = gpd.GeoDataFrame(pd.read_pickle(Lakes_path), geometry='geometry')
            print('NHDPluIDs of lakes renamed to LakeId')
            print('lakes are loaded')

            Lakes = Lakes.merge(self.streams[['LakeId', 'WBArea_Permanent_Identifier']], right_on='WBArea_Permanent_Identifier', left_on='Permanent_Identifier', how='inner')
            print('LakeId added to streams based on the WBArea_Permanent_Identifier of streams and Permanent_Identifier of lakes')
            print('number of lakes after merging streams and lake based on Permanent_Identifier and WBArea_Permanent_Identifier',len(Lakes))

            Lakes = Lakes.dropna(subset='LakeId').reset_index(drop=True)
            Lakes ['LakeId']=Lakes ['LakeId'].astype('Int64')
            print('lakes name after merging with streams (for debugging purposes, "LakesProcessor" function )',Lakes['LakeId'].unique)
            Lakes = Lakes.drop_duplicates(subset='geometry')

            return Lakes

        def report_area(self, df, title):
            print(f'\nREPORTING {title} AREAS:\n')
            print("Max area: {:,.2f}".format(df.area.max()))
            print("95th area: {:,.2f}".format(df.area.quantile(0.95)))
            print("50th area: {:,.2f}".format(df.area.quantile(0.5)))
            print("2.5th area: {:,.2f}".format(df.area.quantile(0.025)))
            print("Min area: {:,.2f}".format(df.area.min()))

        @staticmethod
        def map_waterbody(row):
            ftype = row['FType']
            if ftype == 436:
                return 1  # Reservoir
            elif ftype == 466:
                return 3  # Wetlands
            elif ftype == 361:
                return 4  # Playa
            else:
                return 2  # anything else including lakes, ponds, estuaries, .....

        def format_SWAT_plus_lakes(self, Lakes):
            print('\nmapping waterbodies\n')
            Lakes = Lakes.assign(RES=Lakes.apply(self.map_waterbody, axis=1))
            Lakes['LakeId'] = Lakes['LakeId'].astype('Int64')
            Lakes['RES'] = Lakes['RES'].astype('Int64')

            swatplus_shape_path = os.path.join(self.BASE_PATH, 'SWATplus_by_VPUID' ,f'{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}/Watershed/Shapes/')
            os.makedirs(swatplus_shape_path, exist_ok=True)
            swatplus_lakes_path = os.path.join(swatplus_shape_path, 'SWAT_plus_lakes.shp')
            Lakes[['LakeId', 'RES', 'geometry']].dissolve('LakeId').to_file(swatplus_lakes_path)

            print(f"Lakes shapefile is created and saved in {swatplus_lakes_path}")
            print('Final Number of lakes:',len(Lakes))
            self.report_area(Lakes, title='LAKES')

            SWAT_lakes=gpd.read_file(swatplus_lakes_path)

            if streams[(~streams.LakeMain.isin(SWAT_lakes.LakeId)) & (~streams.LakeMain.isna())].empty:
                print('All lakes have main outlets')
            if streams[(~streams.LakeOut.isin(SWAT_lakes.LakeId)) & (~streams.LakeOut.isna())].empty:
                print('All lakes outlets repeated in LakeId')
            if streams[(~streams.LakeIn.isin(SWAT_lakes.LakeId)) & (~streams.LakeIn.isna())].empty:
                print('All lakes inlets repeated in LakeId')
            if streams[(~streams.LakeWithin.isin(SWAT_lakes.LakeId)) & (~streams.LakeWithin.isna())].empty:
                print('All lakes Within repeated in LakeId')

        def process_lakes(self):
            Lakes = self.loading_and_adding_lake_ids()
            if Lakes.empty:
                print('#########################  WARNING:        No lakes to process. Exiting function   ###############################')
                return
            self.format_SWAT_plus_lakes(Lakes)

    # Usage example:
    lakes_processor = LakesProcessor(streams, BASE_PATH, VPUID)
    lakes_processor.process_lakes()


    def report_area(df, title):
        print(f'\nREPORTING {title} AREAS:\n')
        print("Max area:", df.area.max())
        print("95th area:", df.area.quantile(0.95))
        print("50th area:", df.area.quantile(0.5))
        print("2.5th area:", df.area.quantile(0.025))
        print("Min area:", df.area.min())



    def creating_subbasins_shapefile_parallel(df, VPUID):
        """Use ThreadPoolExecutor to parallelize the dissolve operation."""

        df=df[df.VPUID == VPUID][['Subbasin', 'geometry']].dissolve('Subbasin').reset_index()

        return df


    def inserting_watershed_keys(subbasins,watersheds):
        watershed_keys=watersheds[['Subbasin','huc12','huc8','Subbasin_level_1']].drop_duplicates()
        df = subbasins.merge(watershed_keys, on='Subbasin')
        print('Subbasins number are added to the subbasins shapefile\n')
        return df

    def assign_new_unique_values(df, column_name, new_columns,correspondings):
        """
        Assign new unique integer values starting from 1 for the unique values in a column or series of columns.
        """
        # Create a mapping of unique values to new integer values
        unique_values = df[column_name].unique()
        value_map = {value: idx+1 for idx, value in enumerate(unique_values)}
        # Apply the mapping to the column
        c=0
        for new_column_name  in new_columns:
            df[new_column_name] = df[correspondings[c]].map(value_map)
            c=1+c
        return df

    def process_SWAT_plus_watersheds(watersheds, streams_SWAT):
        """
        prepare SWAT PLUS WATERSHEDS BASED ON processed SWAT PLUS STREAMS
        """
        result_df = watersheds[['NHDPlusID','geometry']].merge(streams_SWAT[['NHDPlusID','WSNO']], on='NHDPlusID')
        result_df = result_df.rename(columns={'WSNO':'PolygonId'}).drop(columns='NHDPlusID')[['PolygonId','geometry']]
        return result_df


    def process_SWAT_plus_subbasins(subbasins, streams_SWAT):
        """
        prepare SWAT PLUS WATERSHEDS BASED ON processed SWAT PLUS STREAMS
        """
        result_df = subbasins[['Subbasin','geometry']].merge(streams_SWAT[['Subbasin','BasinNo']], on='Subbasin')
        result_df=result_df.rename(columns={'Subbasin':'PolygonId'})
        result_df = result_df.rename(columns={'BasinNo':'Subbasin'})[['PolygonId','Subbasin','geometry']]
        result_df = result_df.drop_duplicates(subset='PolygonId').reset_index(drop=True)
        result_df['PolygonId'] = result_df['Subbasin']
        return result_df

    def formating_stream_data_type_and_saving(df):
        df = df.copy()  # Work on a copy to avoid SettingWithCopyWarning
        df['DSLINKNO'] = df['DSLINKNO'].astype('Int64')
        df['LINKNO'] = df['LINKNO'].astype('Int64')
        df['Length'] = df['Length'].astype(float)
        df['Drop'] = df['Drop'].astype(float)
        df['BasinNo'] = df['BasinNo'].astype('Int64')
        df['WSNO'] = df['WSNO'].astype('Int64')

        df['LakeId'] = df['LakeId'].astype('Int64')
        df['LakeWithin'] = df['LakeWithin'].astype('Int64')
        df['LakeIn'] = df['LakeIn'].astype('Int64')
        df['LakeOut'] = df['LakeOut'].astype('Int64')
        df['LakeMain'] = df['LakeMain'].astype('Int64')

        swatplus_shape_path = os.path.join(BASE_PATH,'SWATplus_by_VPUID' ,f'{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}/Watershed/Shapes/')
        os.makedirs(swatplus_shape_path, exist_ok=True)
        swatplus_streams_path = os.path.join(swatplus_shape_path,'SWAT_plus_streams.shp')
        df[['DSLINKNO','LINKNO','Length','Drop','BasinNo','WSNO','LakeId','LakeWithin','LakeIn','LakeOut','LakeMain','geometry']].reset_index(drop=True)
        df.to_file(swatplus_streams_path)
        #df[['DSLINKNO','LINKNO','Length','Drop','BasinNo','WSNO','geometry']].to_file(BASE_PATH+'NHDPlusData/_SWAT_plus_streams_')

        print('\nSWAT+ streams shapefile is created')

    def formating_watersheds_data_type_and_saving(df):
        df = df.copy()  # Work on a copy to avoid SettingWithCopyWarning
        df.PolygonId = df.PolygonId.astype('Int64')
        swatplus_shape_path = os.path.join(BASE_PATH,'SWATplus_by_VPUID' ,f'{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}/Watershed/Shapes/')
        swatplus_watersheds_path = os.path.join(swatplus_shape_path,'SWAT_plus_watersheds.shp')
        df[['PolygonId','geometry']].reset_index(drop=True).to_file(swatplus_watersheds_path)
        print('\nSWAT+ watersheds shapefile is created')

    def formating_basins_data_type_and_saving(df):
        df = df.copy()  # Work on a copy to avoid SettingWithCopyWarning
        df.PolygonId = df.PolygonId.astype('Int64')
        df.Subbasin = df.Subbasin.astype('Int64')
        swatplus_shape_path = os.path.join(BASE_PATH,'SWATplus_by_VPUID' ,f'{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}/Watershed/Shapes/')
        swatplus_subbasins_path = os.path.join(swatplus_shape_path,'SWAT_plus_subbasins.shp')
        df[['PolygonId','Subbasin','geometry']].reset_index(drop=True).to_file(swatplus_subbasins_path)
        print('\nSWAT+ subbasins shapefile is created')

    watersheds = gpd.GeoDataFrame(pd.read_pickle(os.path.join(BASE_PATH,f'NHDPlusData/SWATPlus_NHDPlus/{VPUID}/watersheds.pkl')), geometry='geometry', crs=EPSG)

    watersheds = watersheds.merge(streams[['Subbasin','huc12','huc8','Subbasin_level_1','NHDPlusID']]).reset_index(drop=True)
    report_area(watersheds,title='Watersheds')
    subbasins = creating_subbasins_shapefile_parallel(watersheds, VPUID)  ### step 3: dissolving watersheds based on subbasins and create subbasin shapefile
    subbasins = inserting_watershed_keys(subbasins,watersheds)  ####### adding other HUC8, HUC12 and ultimate drangage area to the subbasins shapefile
    print(' Lakes unique LakeIn Ids. This is for debugging purposes (write_output function before renaming streams columns):',streams.LakeIn.unique())

    streams = assign_new_unique_values(streams, column_name="NHDPlusID", new_columns=["WSNO"], correspondings=['NHDPlusID']) #### WSNO: indexing streams based on NHDPlusID
    streams = assign_new_unique_values(streams, column_name="HydroSeq", new_columns=["LINKNO",'DSLINKNO'], correspondings=['HydroSeq','DnHydroSeq'])  ### LINKNO, DSLINKNO: indexing based on HydroSeq adn DnHydroSeq
    streams = assign_new_unique_values(streams, column_name="Subbasin", new_columns=["BasinNo"], correspondings=['Subbasin'])  ### BasinNo: Indexing based on the subbasins number that we created
    streams ['DSLINKNO'] = np.where(streams.DSLINKNO.isna(), -1, streams.DSLINKNO)

    streams_SWAT=streams[["WSNO","LINKNO",'DSLINKNO','BasinNo', 'Drop','Length','NHDPlusID','Subbasin','LakeId','LakeWithin','LakeIn','LakeOut','LakeMain','geometry']]
    watersheds_SWAT = process_SWAT_plus_watersheds(watersheds, streams_SWAT)
    print('SWAT Plus watersheds are created based on the proccesesed streams data')
    subbasins_SWAT = process_SWAT_plus_subbasins(subbasins, streams_SWAT)
    print('SWAT Plus subbasins are created based on the proccesesed streams data')
    formating_stream_data_type_and_saving(streams_SWAT)
    print('SWAT Plus streams are created')
    formating_watersheds_data_type_and_saving(watersheds_SWAT)
    print('SWAT Plus watersheds are created')
    formating_basins_data_type_and_saving(subbasins_SWAT)
    print('SWAT Plus Subbasins are created')
    print("writing process completed")


def writing_swatplus_cli_files(BASE_PATH, VPUID, LEVEL, NAME):
    SWAT_MODEL_PRISM_path = os.path.join(BASE_PATH,f'SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/PRISM/')
    ## get the name of files
    files = os.listdir(SWAT_MODEL_PRISM_path)
    tmp_files = [file for file in files if 'tmp' in file]
    pcp_files = [file for file in files if 'pcp' in file]
    with open(os.path.join(SWAT_MODEL_PRISM_path, 'tmp.cli'), 'w') as f:
        f.write(f' climate data written on {NAME}\n')
        f.write(f' pcp files\n')
        for file in tmp_files:
            f.write(file + '\n')
    with open(os.path.join(SWAT_MODEL_PRISM_path, 'pcp.cli'), 'w') as f:
        f.write(f' climate data written on {NAME}\n')
        f.write(f' tmp files:\n')
        for file in pcp_files:
            f.write(file + '\n')

def creating_modified_inputs(BASE_PATH,VPUID,  LEVEL, NAME, MODEL_NAME):

    def calculate_outlets_and_warn(streams, subbasins):
        # Create a new column in subbasins GeoDataFrame to hold the number of outlets
        subbasins['num_outlets'] = 0

        # Calculate the number of outlets for each subbasin
        for subbasin in subbasins['Subbasin'].unique():
            subbasin_streams = streams[streams['BasinNo'] == subbasin]
            outlets_counter = 0
            processed_DnHydroSeqs = set()

            for DnHydroSeq in subbasin_streams['DSLINKNO'].unique():
                if DnHydroSeq in processed_DnHydroSeqs:
                    continue

                downstream_subbasin_values = streams[streams['LINKNO'] == DnHydroSeq]['BasinNo'].values

                if downstream_subbasin_values.size > 0 and subbasin != downstream_subbasin_values[0]:
                    outlets_counter += 1
                elif DnHydroSeq not in streams['LINKNO'].values:
                    outlets_counter += 1  # DnHydroSeq does not repeat anywhere.

                processed_DnHydroSeqs.add(DnHydroSeq)

            # Store the number of outlets in the subbasins DataFrame
            subbasins.loc[subbasins['Subbasin'] == subbasin, 'num_outlets'] = outlets_counter

            # Check for subbasins with more than one outlet and issue warning
            if outlets_counter > 1:
                print(f"ERROR ########### ERROR ################: Subbasin {subbasin} has more than one outlet ######### ERROR #################  ERROR")

        # Return the modified subbasins DataFrame
        return subbasins
    swatplus_shape_path = os.path.join(BASE_PATH,'SWATplus_by_VPUID' ,f'{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}/Watershed/Shapes/')
    swatplus_subbasins_path = os.path.join(swatplus_shape_path,'SWAT_plus_subbasins.shp')
    swatplus_watersheds_path = os.path.join(swatplus_shape_path,'SWAT_plus_watersheds.shp')
    swatplus_streams_path = os.path.join(swatplus_shape_path, 'SWAT_plus_streams.shp')
    subbasins= gpd.read_file(swatplus_subbasins_path) ## columns: PolygonId, Subbasin
    watersheds= gpd.read_file(swatplus_watersheds_path)  ## column PolygonId
    streams= gpd.read_file(swatplus_streams_path)      ## columns: 'DSLINKNO' (donstream id), 'LINKNO' (current stream id), 'BasinNo', 'WSNO'


    ### CHECKING ALL watersheds and streams being each other
    if watersheds [~watersheds.PolygonId.isin(streams.WSNO)].empty: print('all watersheds have a stream')
    if subbasins [~subbasins.Subbasin.isin(streams.BasinNo)].empty: print('all subbasinss have a stream')
    if streams [~streams.WSNO.isin(watersheds.PolygonId)].empty: print('all streams have a watershed')
    if streams [~streams.BasinNo.isin(subbasins.Subbasin)].empty: print('all streams have a basin')
    lakes_with_in_but_no_out = streams[~streams.LakeId.isin(streams.LakeOut)]
    if len(lakes_with_in_but_no_out)>0:
        print("THERE ARE LAKES WITH NO OUTLET")
    else:
        print('ALL LAKES HAVE OUTLET')
    #subbasins = calculate_outlets_and_warn(streams, subbasins)  #### you will see warning if there is more than one outlet for any subbasin
    subbasins['Area']= subbasins.area
    print('subbasins area stats (sqkm):')
    print((subbasins.area * 1e-6).describe().round(2))


    watersheds['Area']= watersheds.area
    print('watersheds area stats (sqkm):')
    print((watersheds.area * 1e-6).describe().round(2))


    watersheds_m = watersheds.merge(streams[['WSNO','LINKNO','DSLINKNO','BasinNo']], left_on='PolygonId', right_on='WSNO')
    print('Number of initial subbasins:', len(watersheds_m.BasinNo.unique()))
    subbasins_number = watersheds_m.BasinNo.unique()
    threshold = 250*250*1000
    old_new_subbasins = {}

    for sub in subbasins_number:
        area = watersheds_m[watersheds_m.BasinNo == sub].geometry.area.sum()
        if area < threshold:
            watersheds_candidate = watersheds_m[watersheds_m.BasinNo == sub].reset_index(drop=True)
            watersheds_others = watersheds_m[~watersheds_m.BasinNo.isin([sub])].reset_index(drop=True)
            downstream_basin = watersheds_others[watersheds_others.LINKNO.isin(watersheds_candidate.DSLINKNO)].BasinNo.unique()

            if len(downstream_basin) == 0:
                if area < threshold / 5:
                    print(f'################### WARNING: THE SUBBASIN {sub} IS ISOLATED & ITS AREA < THRESHOLD ({threshold/5}sqm:)')
            else:
                # downstream_basin contains only one unique value.
                old_new_subbasins[sub] = downstream_basin[0]
                watersheds_m['BasinNo'] = np.where(watersheds_m.BasinNo == sub, downstream_basin[0], watersheds_m['BasinNo'])


    watersheds_m = watersheds_m.reset_index(drop=True)

    unique_basins = sorted(watersheds_m['BasinNo'].unique())
    new_basin_mapping = {old: new for new, old in enumerate(unique_basins, start=1)}

    # Update BasinNo in watersheds_m
    watersheds_m['BasinNo'] = watersheds_m['BasinNo'].map(new_basin_mapping)


    print('Number of final subbasins:', len(watersheds_m.BasinNo.unique()))
    print('max number of subbasinss:', watersheds_m.BasinNo.max())

    streams_m= streams.drop(columns='BasinNo')
    streams_m = streams_m.merge(watersheds_m[['WSNO','BasinNo']], on='WSNO', how='inner')
    streams_m = streams_m.reset_index(drop=True)

    #print('number of unique Basin in streams',streams_m.BasinNo.unique().shape[0])
    print('number of new streams',streams_m.shape[0])
    print('number of new watersheds',watersheds_m.shape[0])

    print('number of unique basins in new streams',len(streams_m.BasinNo.unique()))
    print('number of  unique basins in  new watersheds',len(watersheds_m.BasinNo.unique()))


    print('number of unique basins in new streams',streams_m.BasinNo.max())
    print('number of  unique basins in  new watersheds',watersheds_m.BasinNo.max())


    # Dissolve the watersheds based on BasinNo
    new_subbasins = watersheds_m[['BasinNo','geometry']].dissolve(by='BasinNo', as_index=False)
    new_subbasins.rename(columns={'BasinNo':'PolygonId'}, inplace=True)
    new_subbasins['Subbasin']= new_subbasins['PolygonId']


    new_subbasins = calculate_outlets_and_warn(streams_m, new_subbasins)  #### you will see warning if there is more than one outlet for any subbasin
    print('subbasins area stats  (sqkm):')
    print((new_subbasins.area * 1e-6).describe().round(2))

    swatplus_shape_path = os.path.join(BASE_PATH,'SWATplus_by_VPUID',f"{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}/Watershed/Shapes")
    swatplus_subbasins_path = os.path.join(swatplus_shape_path,'SWAT_plus_subbasins.shp')
    swatpplus_streams_path = os.path.join(swatplus_shape_path,'SWAT_plus_streams.shp')
    swatplus_watersheds_path = os.path.join(swatplus_shape_path,'SWAT_plus_watersheds.shp')
    new_subbasins[['PolygonId','geometry']].to_file(swatplus_subbasins_path)
    streams_m.to_file(swatpplus_streams_path)
    watersheds_m.rename(columns={'BasinNo':'Subbasin'}, inplace=True)
    watersheds_m[['PolygonId','Subbasin','geometry']].to_file(swatplus_watersheds_path)
    print('success')

    watersheds_m['Area']= watersheds_m.area
    print('watersheds area stats  (sqkm):')
    print((watersheds_m.area * 1e-6).describe().round(2))

    return streams_m
