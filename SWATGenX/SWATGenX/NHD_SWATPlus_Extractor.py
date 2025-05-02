import contextlib
import geopandas as gpd
import pandas as pd
import os
import numpy as np
import os
from SWATGenX.SWATGenXLogging import LoggerSetup


class NHD_SWATPlus_Extractor:
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
    def __init__(self, SWATGenXPaths, list_of_HUC, LEVEL, VPUID, MODEL_NAME, NAME):
        self.SWATGenXPaths = SWATGenXPaths
        self.BASE_PATH = SWATGenXPaths.database_dir
        self.list_of_HUC = list_of_HUC
        self.LEVEL = LEVEL
        self.VPUID = VPUID
        self.MODEL_NAME = MODEL_NAME
        self.NAME = NAME
        self.report_path = SWATGenXPaths.report_path
        self.logger = LoggerSetup(report_path=self.report_path, rewrite=False, verbose=True)
        self.logger = self.logger.setup_logger("NHDPlusExtractor")
        self.no_value = -9999
        self.swatplus_shape_path = f"{SWATGenXPaths.swatgenx_outlet_path}/{self.VPUID}/{self.LEVEL}/{self.NAME}/{self.MODEL_NAME}/Watershed/Shapes"
        self.swatplus_subbasins_path = os.path.join(self.swatplus_shape_path,'SWAT_plus_subbasins.shp')
        self.swatplus_streams_path = os.path.join(self.swatplus_shape_path,'SWAT_plus_streams.shp')
        self.swatplus_watersheds_path = os.path.join(self.swatplus_shape_path,'SWAT_plus_watersheds.shp')
        self.streams_pickle_path = f'{SWATGenXPaths.extracted_nhd_swatplus_path}/{self.VPUID}/streams.pkl'
        self.lakes_pickle_path = f'{SWATGenXPaths.extracted_nhd_swatplus_path}/{self.VPUID}/NHDWaterbody.pkl'


    def creating_unique_subbasin(self, df, first_criteria, second_criteria):
        """
        Create unique subbasin identifiers based on two criteria.
        """
        df['Subbasin_updated'] = pd.factorize(df[first_criteria].astype(str) + "_" + df[second_criteria].astype(str))[0] + 1
        self.logger.info(f"Subbasin IDs created based on combination of {first_criteria} and {second_criteria}")
        with contextlib.suppress(Exception):
            df = df.drop(columns='Subbasin')

        df = df.rename(columns={'Subbasin_updated':'Subbasin'})
        self.logger.info(f"Subbasin IDs updated based on combination of {first_criteria} and {second_criteria}")
        return df

    def find_upstreams(self, hydroseq, streams_df):
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
            all_upstreams.extend(self.find_upstreams(u, streams_df))
        return all_upstreams

    def create_subbasins(self, streams_df):
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
                        upstream_segments = self.find_upstreams(DnHydroSeq, streams_df)
                        streams_df.loc[streams_df.HydroSeq.isin(upstream_segments), 'Subbasin_sub'] = next_subbasin_sub_id
                        next_subbasin_sub_id += 1
                        processed_DnHydroSeqs.update(upstream_segments)

                subbasin_outlet_count[subbasin] = outlets_counter
            self.logger.info(f"Subbasin outlet count: {subbasin_outlet_count}")
            streams_df = self.creating_unique_subbasin(streams_df, 'Subbasin', 'Subbasin_sub')

            # Verify that no subbasins have more than one outlet
            multiple_outlets = [k for k, v in subbasin_outlet_count.items() if v > 1]
            if not multiple_outlets:
                break

        return streams_df

    def setting_zero_for_outlets_and_headwaters(self,df):
        df['DnHydroSeq'] = np.where(df.DnHydroSeq.isin(df.HydroSeq), df.DnHydroSeq, 0)
        df['UpHydroSeq'] = np.where(df.UpHydroSeq.isin(df.HydroSeq), df.UpHydroSeq, 0)
        df['DnHydroSeq'] = df['DnHydroSeq'].fillna(0).astype('Int64')
        df['UpHydroSeq'] = df['UpHydroSeq'].fillna(0).astype('Int64')
        return df

    def adding_lake_ids(self, streams):
        Lakes= self.load_and_clean_lakes()

        streams = streams.merge(Lakes[['LakeId','Permanent_Identifier']], left_on='WBArea_Permanent_Identifier',
                                right_on='Permanent_Identifier', how='left')
        Lakes=Lakes[Lakes.Permanent_Identifier.isin(streams.WBArea_Permanent_Identifier)].reset_index(drop=True)
        return streams

    def load_and_clean_lakes(self, criteria=0.1):
        Lakes_path = f'{self.SWATGenXPaths.extracted_nhd_swatplus_path}/{self.VPUID}/NHDWaterbody.pkl'
        Lakes = gpd.GeoDataFrame(pd.read_pickle(Lakes_path), geometry='geometry')
        self.logger.info(f"Total number of lakes added: {len(Lakes)}")
        Lakes ['LakeId'] = Lakes.Permanent_Identifier


        criteria = 0.1
        self.logger.info(f"################# minimum area of lakes to be considered: {criteria} SqKm ################")
        Lakes = Lakes[Lakes.AreaSqKm>criteria].reset_index(drop=True)
        Lakes = Lakes.reset_index(drop=True)
        self.logger.info(f"Lakes more than {criteria} SqKm: {len(Lakes)}")

        return Lakes

    def assign_new_unique_values(self, df, column_name, new_columns,correspondings):
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

    def incorporate_lakes(self, streams):
        """ Once the streams are processed, we will include lakes. We only include those lakes that are connected to streams and have area more than 0.1 skqm.
        The incorporate_lakes function updates the streams GeoDataFrame to include relevant lake information (LakeId and Permenant_Identifier to use it later to connect Lakes with Streams).
        - First, it loads and filters lakes based on their area, only considering those larger than a given criteria (default 1 SqKm).
        - Next, the function assigns unique LakeIds based on Permanent Identifiers.
        - Finally, it merges the modified lake data with the streams GeoDataFrame, updating the streams with corresponding LakeIds.

        The function returns the modified streams GeoDataFrame with added lake information.
        """

        return self.adding_lake_ids(streams)


    def extract_initial_streams(self):
        # Load data and select the area of interest
        os.makedirs(os.path.join(self.SWATGenXPaths.swatgenx_outlet_path, self.VPUID, self.LEVEL), exist_ok=True)
        streams = gpd.GeoDataFrame(pd.read_pickle(self.streams_pickle_path), geometry='geometry')
        streams = streams[~streams.huc12.isna()].reset_index(drop=True)
        self.logger.info(f"Number of streams loaded: {streams.shape[0]}")
        self.logger.info(f"list of huc12 requested: {self.list_of_HUC}")
        self.logger.info(f"stream['LEVEL']:{self.LEVEL}")
        streams = streams[streams["huc12"].isin(self.list_of_HUC)].reset_index(drop=True)
        self.logger.info(f"Number of streams extracted: {streams.shape[0]}")

        # Create initial subbasins
        streams = self.creating_unique_subbasin(streams, 'huc12', 'Subbasin_level_1')
        print('Initial number of subbasins:', streams['Subbasin'].nunique())

        # Refine subbasins based on the one-outlet rule
        streams = self.create_subbasins(streams)
        self.logger.info(f"Final number of subbasins: {streams['Subbasin'].nunique()}")

        streams = self.setting_zero_for_outlets_and_headwaters(streams)

        return streams


    def loading_and_adding_lake_ids(self, streams):
        Lakes = gpd.GeoDataFrame(pd.read_pickle(self.lakes_pickle_path), geometry='geometry')
        self.logger.info('NHDPluIDs of lakes renamed to LakeId')
        self.logger.info('lakes are loaded')

        Lakes = Lakes.merge(streams[['LakeId', 'WBArea_Permanent_Identifier']], right_on='WBArea_Permanent_Identifier', left_on='Permanent_Identifier', how='inner')

        Lakes = Lakes.dropna(subset='LakeId').reset_index(drop=True)
        Lakes ['LakeId']=Lakes ['LakeId'].astype('Int64')
        Lakes = Lakes.drop_duplicates(subset='geometry')

        return Lakes

    def report_area(self, df, title):
        self.logger.info(f'\nREPORTING {title} AREAS:\n')
        self.logger.info("Max area: {:,.2f}".format(df.area.max()))
        self.logger.info("95th area: {:,.2f}".format(df.area.quantile(0.95)))
        self.logger.info("50th area: {:,.2f}".format(df.area.quantile(0.5)))
        self.logger.info("2.5th area: {:,.2f}".format(df.area.quantile(0.025)))
        self.logger.info("Min area: {:,.2f}".format(df.area.min()))

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

    def format_SWAT_plus_lakes(self, Lakes, streams):
        self.logger.info('\nmapping waterbodies\n')
        Lakes = Lakes.assign(RES=Lakes.apply(self.map_waterbody, axis=1))
        swatplus_lakes_path = (
            self._extracted_from_formating_stream_data_type_and_saving_4(
                Lakes, 'LakeId', 'RES', 'SWAT_plus_lakes.shp'
            )
        )
        Lakes[['LakeId', 'RES', 'geometry']].dissolve('LakeId').to_file(swatplus_lakes_path)

        self.logger.info(f"Lakes shapefile is created and saved in {swatplus_lakes_path}")
        self.logger.info(f'Final Number of lakes: {len(Lakes)}')
        self.report_area(Lakes, title='LAKES')

        SWAT_lakes = gpd.read_file(swatplus_lakes_path)

        if streams[(~streams.LakeMain.isin(SWAT_lakes.LakeId)) & (~streams.LakeMain.isna())].empty:
            self.logger.info('All lakes have main outlets')
        if streams[(~streams.LakeOut.isin(SWAT_lakes.LakeId)) & (~streams.LakeOut.isna())].empty:
            self.logger.info('All lakes outlets repeated in LakeId')
        if streams[(~streams.LakeIn.isin(SWAT_lakes.LakeId)) & (~streams.LakeIn.isna())].empty:
            self.logger.info('All lakes inlets repeated in LakeId')
        if streams[(~streams.LakeWithin.isin(SWAT_lakes.LakeId)) & (~streams.LakeWithin.isna())].empty:
            self.logger.info('All lakes Within repeated in LakeId')

    def process_lakes(self, streams):
        Lakes = self.loading_and_adding_lake_ids(streams)
        if Lakes.empty:
            self.logger.info('#########################  WARNING:        No lakes to process. Exiting function   ###############################')
            return
        self.format_SWAT_plus_lakes(Lakes, streams)


    def process_SWAT_plus_subbasins(self, subbasins, streams_SWAT):
        """
        prepare SWAT PLUS WATERSHEDS BASED ON processed SWAT PLUS STREAMS
        """
        result_df = subbasins[['Subbasin','geometry']].merge(streams_SWAT[['Subbasin','BasinNo']], on='Subbasin')
        result_df=result_df.rename(columns={'Subbasin':'PolygonId'})
        result_df = result_df.rename(columns={'BasinNo':'Subbasin'})[['PolygonId','Subbasin','geometry']]
        result_df = result_df.drop_duplicates(subset='PolygonId').reset_index(drop=True)
        result_df['PolygonId'] = result_df['Subbasin']
        return result_df

    def formating_stream_data_type_and_saving(self,df):
        df = df.copy()  # Work on a copy to avoid SettingWithCopyWarning

        # Convert large numeric fields to strings
        large_numeric_fields = ['NHDPlusID', 'HydroSeq', 'DnHydroSeq', 'UpHydroSeq']
        for field in large_numeric_fields:
            if field in df.columns:
                df[field] = df[field].astype(str)

        # Convert other numeric fields
        df['DSLINKNO'] = df['DSLINKNO'].astype('Int64')
        df['LINKNO'] = df['LINKNO'].astype('Int64')
        df['Length'] = df['Length'].astype(float)
        df['Drop'] = df['Drop'].astype(float)
        df['BasinNo'] = df['BasinNo'].astype('Int64')
        df['WSNO'] = df['WSNO'].astype('Int64')

        df['LakeId'] = df['LakeId'].astype('Int64')
        df['LakeWithin'] = df['LakeWithin'].astype('Int64')
        df['LakeIn'] = df['LakeIn'].astype('Int64')
        swatplus_streams_path = (
            self._extracted_from_formating_stream_data_type_and_saving_4(
                df, 'LakeOut', 'LakeMain', 'SWAT_plus_streams.shp'
            )
        )
        df[['DSLINKNO','LINKNO','Length','Drop','BasinNo','WSNO','LakeId','LakeWithin','LakeIn','LakeOut','LakeMain','geometry']].reset_index(drop=True)
        df.to_file(swatplus_streams_path)
        #df[['DSLINKNO','LINKNO','Length','Drop','BasinNo','WSNO','geometry']].to_file(BASE_PATH+'NHDPlusHR/_SWAT_plus_streams_')

        self.logger.info('\nSWAT+ streams shapefile is created')

    # TODO Rename this here and in `format_SWAT_plus_lakes` and `formating_stream_data_type_and_saving`
    def _extracted_from_formating_stream_data_type_and_saving_4(self, arg0, arg1, arg2, arg3):
        arg0[arg1] = arg0[arg1].astype('Int64')
        arg0[arg2] = arg0[arg2].astype('Int64')

        os.makedirs(self.swatplus_shape_path, exist_ok=True)
        return os.path.join(self.swatplus_shape_path, arg3)

    def formating_watersheds_data_type_and_saving(self, df):
        df = df.copy()  # Work on a copy to avoid SettingWithCopyWarning
        df.PolygonId = df.PolygonId.astype('Int64')
        df[['PolygonId','geometry']].reset_index(drop=True).to_file(self.swatplus_watersheds_path)
        self.logger.info('\nSWAT+ watersheds shapefile is created')

    def formating_basins_data_type_and_saving(self, df):
        df = df.copy()  # Work on a copy to avoid SettingWithCopyWarning
        df.PolygonId = df.PolygonId.astype('Int64')
        df.Subbasin = df.Subbasin.astype('Int64')
        df[['PolygonId','Subbasin','geometry']].reset_index(drop=True).to_file(self.swatplus_subbasins_path)
        self.logger.info('\nSWAT+ subbasins shapefile is created')



    def report_area(self, df, title):
        self.logger.info(f'\nREPORTING {title} AREAS:\n')
        self.logger.info(f"Max area: {df.area.max():,.2f}")
        self.logger.info(f"95th area: {df.area.quantile(0.95):,.2f}")
        self.logger.info(f"50th area: {df.area.quantile(0.5):,.2f}")
        self.logger.info(f"2.5th area: {df.area.quantile(0.025):,.2f}")
        self.logger.info(f"Min area: {df.area.min():,.2f}")



    def creating_subbasins_shapefile_parallel(self, df):
        """Use ThreadPoolExecutor to parallelize the dissolve operation."""

        df=df[df.VPUID == self.VPUID][['Subbasin', 'geometry']].dissolve('Subbasin').reset_index()

        return df


    def inserting_watershed_keys(self, subbasins,watersheds):
        watershed_keys=watersheds[['Subbasin','huc12','huc8','Subbasin_level_1']].drop_duplicates()
        df = subbasins.merge(watershed_keys, on='Subbasin')
        self.logger.info('Subbasins number are added to the subbasins shapefile\n')
        return df


    def process_SWAT_plus_watersheds(self, watersheds, streams_SWAT):
        """
        prepare SWAT PLUS WATERSHEDS BASED ON processed SWAT PLUS STREAMS
        """
        result_df = watersheds[['NHDPlusID','geometry']].merge(streams_SWAT[['NHDPlusID','WSNO']], on='NHDPlusID')
        result_df = result_df.rename(columns={'WSNO':'PolygonId'}).drop(columns='NHDPlusID')[['PolygonId','geometry']]
        return result_df


    def write_output(self, streams,EPSG):
        ######  Writing watersheds, streams and lakes based on QSWAT+ requirments
        ########### Final Stage, when we reach here, we are done with modifying streams
        ########## THis is still does not include RES ids

        self.process_lakes(streams)


        watersheds = gpd.GeoDataFrame(pd.read_pickle(f'{self.SWATGenXPaths.extracted_nhd_swatplus_path}/{self.VPUID}/watersheds.pkl'), geometry='geometry', crs=EPSG)

        watersheds = watersheds.merge(streams[['Subbasin','huc12','huc8','Subbasin_level_1','NHDPlusID']]).reset_index(drop=True)
        self.report_area(watersheds,title='Watersheds')
        subbasins = self.creating_subbasins_shapefile_parallel(watersheds)  ### step 3: dissolving watersheds based on subbasins and create subbasin shapefile
        subbasins = self.inserting_watershed_keys(subbasins,watersheds)  ####### adding other HUC8, HUC12 and ultimate drangage area to the subbasins shapefile
        self.logger.info(f'Lakes unique LakeIn Ids. This is for debugging purposes (write_output function before renaming streams columns): {streams.LakeIn.unique()}')

        streams = self.assign_new_unique_values(streams, column_name="NHDPlusID", new_columns=["WSNO"], correspondings=['NHDPlusID']) #### WSNO: indexing streams based on NHDPlusID
        streams = self.assign_new_unique_values(streams, column_name="HydroSeq", new_columns=["LINKNO",'DSLINKNO'], correspondings=['HydroSeq','DnHydroSeq'])  ### LINKNO, DSLINKNO: indexing based on HydroSeq adn DnHydroSeq
        streams = self.assign_new_unique_values(streams, column_name="Subbasin", new_columns=["BasinNo"], correspondings=['Subbasin'])  ### BasinNo: Indexing based on the subbasins number that we created
        streams ['DSLINKNO'] = np.where(streams.DSLINKNO.isna(), -1, streams.DSLINKNO)

        streams_SWAT=streams[["WSNO","LINKNO",'DSLINKNO','BasinNo', 'Drop','Length','NHDPlusID','Subbasin','LakeId','LakeWithin','LakeIn','LakeOut','LakeMain','geometry']]
        watersheds_SWAT = self.process_SWAT_plus_watersheds(watersheds, streams_SWAT)
        self.logger.info('SWAT Plus watersheds are created based on the proccesesed streams data')
        subbasins_SWAT = self.process_SWAT_plus_subbasins(subbasins, streams_SWAT)
        self.logger.info('SWAT Plus subbasins are created based on the proccesesed streams data')
        self.formating_stream_data_type_and_saving(streams_SWAT)
        self.logger.info('SWAT Plus streams are created')
        self.formating_watersheds_data_type_and_saving(watersheds_SWAT)
        self.logger.info('SWAT Plus watersheds are created')
        self.formating_basins_data_type_and_saving(subbasins_SWAT)
        self.logger.info('SWAT Plus Subbasins are created')
        self.logger.info("writing process completed")


    def include_lakes_in_streams(self, streams):
        """
        include_lakes_in_streams function identifies and tags lake inlets, outlets, and main lakes within a stream network.
        - LakeIn: Stream segments that directly flow into a lake.
        - LakeOut: Stream segments that flow out from a lake.
        - LakeMain: The main lake associated with each LakeOut, identified by the maximum stream order.
        - LakeWithin: Stream segments that are within a lake boundary.


        NOTE: Remember that with this code, LakeOut and LakeIn will are not be inside the lakes polygones unless there is a special case where the outlet of the lake is the outlet of the basin.

        """

        # Initialize new columns with self.no_value  (I found this an effective approach for handling nan and null values)
        self.logger.info(f'################################# Unique LAKEID {streams.LakeId.unique()}')
        streams['LakeId'] = streams['LakeId'].fillna(self.no_value).infer_objects(copy=False)
        streams['LakeIn'] = self.no_value
        streams['LakeOut'] = self.no_value
        streams['LakeMain'] = self.no_value
        streams['LakeWithin'] = streams['LakeId'].fillna(self.no_value)  ## initially considering all lakes having lakeId as LakeWithin. We will modify if later


        # Traverse downstream from each headwater
        self.logger.info('start traversing downstream to find LakeIn ids')
        self.logger.info(f"NUMBER OF HEAD WATERS: {len(streams[streams['UpHydroSeq'] == 0])}")

        downstream_dict = pd.Series(streams.DnHydroSeq.values, index=streams.HydroSeq).to_dict()
        lake_dict = pd.Series(streams.LakeId.values, index=streams.HydroSeq).to_dict()


        # Update LakeIn for each stream based on downstream LakeId
        for hydroseq, dn_hydro_seq in downstream_dict.items():
            if dn_hydro_seq == 0:  # Skip cases where 'DnHydroSeq' is 0
                continue
            # Ensure the LakeIn column is of type object
            streams['LakeIn'] = streams['LakeIn'].astype(object)

            # Check and assign LakeIn value
            if lake_dict.get(dn_hydro_seq, self.no_value) != self.no_value and lake_dict.get(hydroseq, self.no_value) == self.no_value:
                streams.loc[streams['HydroSeq'] == hydroseq, 'LakeIn'] = lake_dict[dn_hydro_seq]
        def identify_lake_out(streams):
            # Group streams by LakeId, ignoring the self.no_value group
            grouped_streams = streams[streams['LakeId'] != self.no_value].groupby('LakeId')

            for lake_id, group in grouped_streams:
                # Handle special case: outlet of the watershed with only one stream having LakeId
                if len(group) == 1:
                    single_row = group.iloc[0]
                    dn_hydro_seq = single_row['DnHydroSeq']
                    if dn_hydro_seq == 0:  # This is the outlet of the watershed
                        streams.loc[group.index, ['LakeId', 'LakeIn', 'LakeWithin']] = self.no_value
                        continue

                # Handle special case where there is only one inlet and one flowline within the lake
                if len(group) == 1:
                    single_row = group.iloc[0]
                    dn_hydro_seq = single_row['DnHydroSeq']
                    dn_row = streams.loc[streams['HydroSeq'] == dn_hydro_seq]

                    # Ensure the LakeOut and LakeWithin columns are of type object
                    streams['LakeOut'] = streams['LakeOut'].astype(object)
                    streams['LakeWithin'] = streams['LakeWithin'].astype(object)

                    # Check if dn_row is not empty and LakeId is self.no_value
                    if not dn_row.empty and dn_row['LakeId'].iloc[0] == self.no_value:
                        # Assign lake_id to LakeOut for the matching HydroSeq
                        streams.loc[streams['HydroSeq'] == dn_hydro_seq, 'LakeOut'] = lake_id
                        # Assign self.no_value to LakeWithin for the matching HydroSeq
                        streams.loc[streams['HydroSeq'] == dn_hydro_seq, 'LakeWithin'] = self.no_value  # so no more LakeWithin when we have LakeOut
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
                    # Ensure the LakeOut and LakeWithin columns are of type object
                    streams['LakeOut'] = streams['LakeOut'].astype(object)
                    streams['LakeWithin'] = streams['LakeWithin'].astype(object)

                    # Check if dn_row is not empty and LakeId is self.no_value
                    if not dn_row.empty and dn_row['LakeId'].iloc[0] == self.no_value:
                        # Assign lake_id to LakeOut for the matching HydroSeq
                        streams.loc[streams['HydroSeq'] == dn_hydro_seq, 'LakeOut'] = lake_id
                        # Assign self.no_value to LakeWithin for the matching HydroSeq
                        streams.loc[streams['HydroSeq'] == dn_hydro_seq, 'LakeWithin'] = self.no_value

        # Then run the function identify_lake_out to populate LakeOut
        self.logger.info('start finding the lakeOuts')
        identify_lake_out(streams)
        #streams.replace(self.no_value, np.nan, inplace=True)
        streams = streams.replace(self.no_value, np.nan).infer_objects(copy=False)
        # Identify main lake for each LakeOut
        lakes_without_outlets=streams[~streams.LakeId.isin(streams.LakeOut)].LakeId.unique()
        if len(lakes_without_outlets)>0:
            ### Special case: Sometimes two lakes adjacent to each other have different name, resulting in upstream lake become without LakeOut. In this condition we change the Name of upstream lake to downstream lake
            ## getting the HydroSeq of Lakes without lakeOut
            for lwo in lakes_without_outlets:
                downstreamLake_HydroSeq=streams[streams.LakeId.isin([lwo])].DnHydroSeq.values
                downstreamLake_LakeId = streams[streams.HydroSeq.isin(downstreamLake_HydroSeq)].LakeId.unique()
                if len(downstreamLake_LakeId)>0:
                    self.logger.info("##################",downstreamLake_LakeId,"###################")
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
                self.logger.error(f'######## THE FOLLOWING LAKES DOES NOT HAVE OUTLETS:  {list(lakes_without_outlets)} #####')
            streams['LakeId']=np.where(streams.LakeId.isin(lakes_without_outlets), np.nan, streams['LakeId'])
            streams['LakeIn']=np.where(streams.LakeIn.isin(lakes_without_outlets), np.nan, streams['LakeIn'])
            streams['LakeWithin']=np.where(streams.LakeWithin.isin(lakes_without_outlets), np.nan, streams['LakeWithin'])

        lakes_without_inlets=streams[~streams.LakeId.isin(streams.LakeIn)].LakeId.unique()
        if len(lakes_without_inlets)>0:
            self.logger.warning(f' %%%%%%%  THE FOLLOWING LAKES DOES NOT HAVE INTLET:  {list(lakes_without_inlets)}    %%%%%%%%%%%   '  )
        idx_max_stream_order = streams.groupby('LakeOut')['StreamOrde'].idxmax()
        streams['LakeMain'] = streams['LakeMain'].astype(object)

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

    def calculate_outlets_and_warn(self, streams, subbasins):
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
                self.logger.error(f"ERROR ########### ERROR ################: Subbasin {subbasin} has more than one outlet ######### ERROR #################  ERROR")

        # Return the modified subbasins DataFrame
        return subbasins
    def creating_modified_inputs(self):

        subbasins= gpd.read_file(self.swatplus_subbasins_path) ## columns: PolygonId, Subbasin
        watersheds= gpd.read_file(self.swatplus_watersheds_path)  ## column PolygonId
        streams= gpd.read_file(self.swatplus_streams_path)      ## columns: 'DSLINKNO' (donstream id), 'LINKNO' (current stream id), 'BasinNo', 'WSNO'


        ### CHECKING ALL watersheds and streams being each other
        if watersheds [~watersheds.PolygonId.isin(streams.WSNO)].empty: self.logger.info('all watersheds have a stream')
        if subbasins [~subbasins.Subbasin.isin(streams.BasinNo)].empty: self.logger.info('all subbasinss have a stream')
        if streams [~streams.WSNO.isin(watersheds.PolygonId)].empty: self.logger.info('all streams have a watershed')
        if streams [~streams.BasinNo.isin(subbasins.Subbasin)].empty: self.logger.info('all streams have a basin')
        lakes_with_in_but_no_out = streams[~streams.LakeId.isin(streams.LakeOut)]
        if len(lakes_with_in_but_no_out)>0:
            self.logger.error("THERE ARE LAKES WITH NO OUTLET")
        else:
            self.logger.info('ALL LAKES HAVE OUTLET')
        self.report_watersheds_stats(
            subbasins, 'subbasins area stats (sqkm):'
        )
        self.report_watersheds_stats(
            watersheds, 'watersheds area stats (sqkm):'
        )
        watersheds_m = watersheds.merge(streams[['WSNO','LINKNO','DSLINKNO','BasinNo']], left_on='PolygonId', right_on='WSNO')
        self.logger.info(f'Number of initial subbasins: {len(watersheds_m.BasinNo.unique())}')
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
                        self.logger.warning(f'################### WARNING: THE SUBBASIN {sub} IS ISOLATED & ITS AREA < THRESHOLD ({threshold/5}sqm:)')
                else:
                    # downstream_basin contains only one unique value.
                    old_new_subbasins[sub] = downstream_basin[0]
                    watersheds_m['BasinNo'] = np.where(watersheds_m.BasinNo == sub, downstream_basin[0], watersheds_m['BasinNo'])


        watersheds_m = watersheds_m.reset_index(drop=True)

        unique_basins = sorted(watersheds_m['BasinNo'].unique())
        new_basin_mapping = {old: new for new, old in enumerate(unique_basins, start=1)}

        # Update BasinNo in watersheds_m
        watersheds_m['BasinNo'] = watersheds_m['BasinNo'].map(new_basin_mapping)


        self.logger.info(f'Number of final subbasins: { len(watersheds_m.BasinNo.unique())}')
        self.logger.info(f'max number of subbasinss: {watersheds_m.BasinNo.max()}')

        streams_m= streams.drop(columns='BasinNo')
        streams_m = streams_m.merge(watersheds_m[['WSNO','BasinNo']], on='WSNO', how='inner')
        streams_m = streams_m.reset_index(drop=True)

        #self.logger.info('number of unique Basin in streams',streams_m.BasinNo.unique().shape[0])
        self.logger.info(f'number of new streams {streams_m.shape[0]}')
        self.logger.info(f'number of new watersheds {watersheds_m.shape[0]}')

        self.logger.info(f'number of unique basins in new streams {len(streams_m.BasinNo.unique())}')
        self.logger.info(f'number of  unique basins in  new watersheds {len(watersheds_m.BasinNo.unique())}')


        self.logger.info(f'number of unique basins in new streams {streams_m.BasinNo.max()}')
        self.logger.info(f'number of  unique basins in  new watersheds {watersheds_m.BasinNo.max()}')


        # Dissolve the watersheds based on BasinNo
        new_subbasins = watersheds_m[['BasinNo','geometry']].dissolve(by='BasinNo', as_index=False)
        new_subbasins.rename(columns={'BasinNo':'PolygonId'}, inplace=True)
        new_subbasins['Subbasin']= new_subbasins['PolygonId']


        new_subbasins = self.calculate_outlets_and_warn(streams_m, new_subbasins)  #### you will see warning if there is more than one outlet for any subbasin
        self.logger.info('subbasins area stats  (sqkm):')
        self.logger.info((new_subbasins.area * 1e-6).describe().round(2))


        new_subbasins[['PolygonId','geometry']].to_file(self.swatplus_subbasins_path)
        streams_m.to_file(self.swatplus_streams_path)
        watersheds_m.rename(columns={'BasinNo':'Subbasin'}, inplace=True)
        watersheds_m[['PolygonId','Subbasin','geometry']].to_file(self.swatplus_watersheds_path)
        self.logger.info('success')

        self.report_watersheds_stats(
            watersheds_m, 'watersheds area stats  (sqkm):'
        )
        return streams_m

    def report_watersheds_stats(self, arg0, arg1):
        #subbasins = self.calculate_outlets_and_warn(streams, subbasins)  #### you will see warning if there is more than one outlet for any subbasin
        arg0['Area'] = arg0.area
        self.logger.info(arg1)
        self.logger.info((arg0.area * 1e-6).describe().round(2))



def writing_swatplus_cli_files(SWATGenXPaths, VPUID, LEVEL, NAME):
    SWAT_MODEL_PRISM_path = f'{SWATGenXPaths.swatgenx_outlet_path}/{VPUID}/{LEVEL}/{NAME}/PRISM/'
    ## get the name of files
    files = os.listdir(SWAT_MODEL_PRISM_path)
    tmp_files = [file for file in files if file.endswith('.tmp')]
    pcp_files = [file for file in files if file.endswith('.pcp')]
    ## remove tmp.cli and pcp.cli
    if os.path.exists(os.path.join(SWAT_MODEL_PRISM_path, 'tmp.cli')):
        os.remove(os.path.join(SWAT_MODEL_PRISM_path, 'tmp.cli'))
    if os.path.exists(os.path.join(SWAT_MODEL_PRISM_path, 'pcp.cli')):
        os.remove(os.path.join(SWAT_MODEL_PRISM_path, 'pcp.cli'))
    ## write the cli files
    with open(os.path.join(SWAT_MODEL_PRISM_path, 'tmp.cli'), 'w') as f:
        write_cli_files(
            f, NAME, ' pcp files\n', tmp_files
        )
    with open(os.path.join(SWAT_MODEL_PRISM_path, 'pcp.cli'), 'w') as f:
        write_cli_files(
            f, NAME, ' tmp files:\n', pcp_files
        )

def write_cli_files(f, NAME, arg2, arg3):
    f.write(f' climate data written on {NAME}\n')
    f.write(f'{arg2}')
    for file in arg3:
        f.write(file + '\n')