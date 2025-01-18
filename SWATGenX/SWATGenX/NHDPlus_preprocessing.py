import os
import geopandas as gpd
import pandas as pd
import datetime
import numpy as np
import concurrent.futures
from tqdm import tqdm

def loginfo(logfile, info):
    with open(logfile, 'a') as f:
        f.write(info + '\n')
        print(info)
    print(info)

def generate_hydroseq_upstream_dict(streams):
    """ Generate a dictionary of HydroSeqs and their upstream HydroSeqs."""
    def move_upstream(df, start_hydroseq, segments_dict):
        """Traverse upstream and return a list of HydroSeqs, considering stream splits."""
        upstream_seqs = {start_hydroseq}
        to_process = [start_hydroseq]
        processed_hydroseqs = set()
        c = 0

        while to_process:
            c += 1
            current_hydroseq = to_process.pop()

            if current_hydroseq in processed_hydroseqs:
                print(f"Detected a circular reference involving HydroSeq {current_hydroseq}")
                return list(upstream_seqs)
            processed_hydroseqs.add(current_hydroseq)

            segments = segments_dict.get(current_hydroseq, [])
            for segment in segments:
                up_hydroseq = segment['HydroSeq']
                if up_hydroseq not in upstream_seqs:
                    upstream_seqs.add(up_hydroseq)
                    to_process.append(up_hydroseq)

            if c > len(df):
                print('number of upstream search exceeded the number of total streams')

        return list(upstream_seqs)

    def process_hydroseq(hydroseq):
        return hydroseq, move_upstream(reduced_streams, hydroseq, segments_dict)

    reduced_streams = streams[['DnHydroSeq', 'HydroSeq']]
    segments_dict = reduced_streams.groupby('DnHydroSeq').apply(lambda x: x.to_dict('records')).to_dict()

    hydroseq_upstream_dict = {}
    hydroseq_values = streams[streams.TerminalFl == 1].HydroSeq.values

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for hydroseq, upstreams in executor.map(process_hydroseq, hydroseq_values):
            hydroseq_upstream_dict[hydroseq] = upstreams

    return hydroseq_upstream_dict

def adding_subbasins_level_one(df,outlets_upstream_dict):
    df ['Subbasin_level_1'] =np.nan
    sub=0
    for HydroSeq in df[df.TerminalFl == 1].HydroSeq.values:
        sub=sub+1
        upstreams=outlets_upstream_dict[HydroSeq]
        df ['Subbasin_level_1'] = np.where(df.HydroSeq.isin(upstreams), sub, df.Subbasin_level_1)
    print('basins level one (one basin for each outlet) are added')
    return(df)

def process_huc_and_merge(watersheds, WBDHU, streams, huc_level):
    """
    Add HUC basins number (either HUC8 or HUC12) to streams based on watershed centroids.
    
    Parameters:
    - watersheds: GeoDataFrame of watersheds.
    - WBDHU: GeoDataFrame of HUC (either WBDHU8 or WBDHU12).
    - streams: DataFrame of streams.
    - huc_level: Either 'huc8' or 'huc12'.
    
    Returns:
    - DataFrame of streams merged with the appropriate HUC.
    """
    
    print(f'Number of STREAMS before merging with {huc_level}:', len(streams))
    print(f'Number of WATERSHEDS before processing with {huc_level}:', len(watersheds))

    # Calculate centroids of watersheds
    watersheds_centroids = watersheds.copy()
    watersheds_centroids['geometry'] = watersheds['geometry'].centroid

    # Spatially join watersheds' centroids to WBDHU to get corresponding HUC basins
    huc_columns = ['geometry', huc_level]
    if huc_level == 'huc12':
        huc_columns.append('tohuc')

    watersheds_huc = gpd.sjoin(watersheds_centroids, WBDHU[huc_columns], how='inner', predicate='within')
    print(f'Number of watersheds after joining with {huc_level}:', len(watersheds_huc))

    # Drop duplicates based on NHDPlusID, keeping the first occurrence
    watersheds_huc_clean = watersheds_huc.drop_duplicates(subset='NHDPlusID', keep='first')
    print(
        'Number of watersheds after dropping duplicates:',
        len(watersheds_huc_clean),
    )

    # Merge the data back to streams
    merge_columns = ['NHDPlusID', huc_level]
    if huc_level == 'huc12':
        merge_columns.append('tohuc')

    streams_f = streams.merge(watersheds_huc_clean[merge_columns], on='NHDPlusID', how='left')
    print(f'Number of streams after merging with {huc_level}:', len(streams_f))
    print(f'{huc_level.upper()} basins are added')

    return streams_f

def converting_length_unit_of_streams(df):
    """converts the length unit from kilometer to meter."""
    df['LengthKM']=df['LengthKM']*1000  ### WE CONVERT KILOMETER UNIT IN NHDPLUS FOR LENGTH TO METER 
    df=df.rename(columns={'LengthKM':'Length'})
    print('length (m) column is added')
    return(df)

def calculating_drop(df):
    """Calculates the drop in elevation for each stream segment."""
    df['Drop']=0.01*(df['MaxElevSmo'] - df['MinElevSmo'])  ### WE CONVERT CENTIMETER UNIT IN NHDPLUS FOR MIN AND MAX ELEVATION TO METER 
    print('Drop (m) column is calculated as the difference betweeen maximum and minimum elevation')
    df=df.drop(columns=['MaxElevSmo','MinElevSmo'])
    return(df)

def removing_second_divergence(df):
    
    """we remove the streams that have divergence 2. This is because the divergence 2 is a result of circular hydrographs"""
    
    a=len(df)
    df = df[df.Divergence!=2].reset_index(drop=True)    
    b=len(df)
    print('Number of removed streams due to being second divergence',a-b)
    return(df)

def remove_streams_without_drainage(df, df2):
    """"we remove the streams that does not have corresponding drainage area simply by checking the NHDPlusID of streams and catchments"""
    a=len(df)
    df = df[df.NHDPlusID.isin(df2.NHDPlusID)].reset_index(drop=True)
    b=len(df)
    print('Number of removed streams due to lack of drainage',a-b)
    return(df)

    
def remove_coastal_lines(df):
    """ we remove the coastal lines. The coastal lines are results of using watershed boundary walls in NHDPlus and considered error since these are not streams."""
    a=len(df)
    df = df[~df['Permanent_Identifier'].str.startswith('C')].reset_index(drop=True)
    b=len(df)
    print('Number of flowlines removed due to being coastal lines',a-b)
    df=df.drop(columns=['Permanent_Identifier'])
    return(df)

def setting_zero_for_outlets_and_headwaters(df):
    """ setting 0 for UpHydroSeq and DnHydroSeq for headwaters and outlets."""
    df['DnHydroSeq'] = np.where(df.DnHydroSeq.isin(df.HydroSeq),df.DnHydroSeq, 0)
    df['UpHydroSeq'] = np.where(df.UpHydroSeq.isin(df.HydroSeq),df.UpHydroSeq, 0)
    df ['DnHydroSeq'] = df['DnHydroSeq'].fillna(0).astype('int64')
    df ['UpHydroSeq'] = df['UpHydroSeq'].fillna(0).astype('int64')
    return(df)


def resetting_start_terminate_flags(df):
    """ resetting start and terminal flags. This is because the start and terminal flags are not correctly set in NHDPlus."""
    df ['StartFlag'] = np.where(df.UpHydroSeq==0,1,0)
    df ['TerminalFl'] = np.where(df.DnHydroSeq==0,1,0)
    print('start and terminal flags are reset')
    print('number of outlets:', len(df[df.TerminalFl==1]) )
    print('number of headwaters:', len(df[df.StartFlag==1]) )
    return(df)

def save_divergence_2_streams(df,df2, output_base):
    divergence_2_streams = df[df.Divergence == 2]
    df2 = df2.merge(divergence_2_streams[['NHDPlusID','huc12','huc8']], on='NHDPlusID')
    output_path = os.path.join(output_base,"Divergence2Streams.shp")
    df2.to_file(output_path)
    print(f"Divergence 2 streams saved to {output_path}")


def removing_isolated_streams(df,watersheds, save_path):
    """### remove isolated channels WHERE BOTH UpHydroSeq and DnHydroSeq are
    ZERO (not be found in any other HydroSeq). THIS WILL EFFECTIVELY REMOVE ALL ISOLATED STREAMS
    """
    # Identify isolated streams
    isolated_streams = df[(df.UpHydroSeq == 0) & (df.DnHydroSeq == 0)]
    # Print the number of removed isolated streams
    num_removed = len(isolated_streams)
    print('Number of removed isolated streams:', num_removed)
    save_file(
        save_path, 'isolated_streams/isolated_streams.shp', isolated_streams
    )
    # Save the isolated streams to a file
    isolated_watersheds = watersheds.merge(isolated_streams[['NHDPlusID']], on='NHDPlusID')
    save_file(
        save_path,
        'isolated_watersheds/isolated_watersheds.shp',
        isolated_watersheds,
    )
    # Remove isolated streams from the main dataframe
    df = df[~((df.UpHydroSeq == 0) & (df.DnHydroSeq == 0))].reset_index(drop=True)
    return df

def save_file(save_path, arg1, arg2):
    # Save the isolated streams to a file
    isolated_stream_path = os.path.join(save_path, arg1)
    ## remove if the file already exists
    os.remove(isolated_stream_path) if os.path.exists(isolated_stream_path) else None
    os.makedirs(os.path.dirname(isolated_stream_path), exist_ok=True)
    arg2.to_file(isolated_stream_path)

def setting_data_type(streams):
    
    streams=streams.dropna(subset='huc8')
    streams=streams.dropna(subset='huc12')
    streams = streams.dropna(subset=['tohuc'])
    streams = streams.dropna(subset=['HydroSeq'])
    streams = streams.dropna(subset=['DnHydroSeq'])
    streams = streams.dropna(subset=['UpHydroSeq'])
    streams = streams.dropna(subset=['NHDPlusID'])
    
    streams['HydroSeq'] = streams.HydroSeq.astype('int64', errors='ignore')
    streams['DnHydroSeq'] = streams.DnHydroSeq.astype('int64', errors='ignore')
    streams['UpHydroSeq'] = streams.UpHydroSeq.astype('int64', errors='ignore')
    streams['NHDPlusID'] = streams.NHDPlusID.astype('int64', errors='ignore')
    streams['huc12'] = streams.huc12.astype('int64', errors='ignore')
    streams['huc8'] = streams.huc8.astype('int64', errors='ignore')
    return streams

def creating_watershed_dissolving_divergence (watersheds):
    
    divergence_2 = watersheds[watersheds.Divergence == 2]
    # Step 2: Create a mapping of HydroSeq to DnHydroSeq for these rows
    hydroseq_dn_mapping = divergence_2.set_index('HydroSeq')['DnHydroSeq'].to_dict()
    # Step 3 and 4: For each HydroSeq, merge its geometry with the corresponding downstream geometry
    for hydroseq, dn_hydroseq in tqdm(hydroseq_dn_mapping.items(), total=len(hydroseq_dn_mapping)):
        # Find the downstream watershed
        downstream_watershed_idx = watersheds[watersheds.HydroSeq == dn_hydroseq].index
        if not downstream_watershed_idx.empty:
            # Union the geometries
            downstream_geom = watersheds.loc[downstream_watershed_idx, 'geometry'].values[0]
            upstream_geom = divergence_2[divergence_2.HydroSeq == hydroseq].geometry.values[0]
            merged_geom = downstream_geom.union(upstream_geom)
            # Update the geometry of the downstream watershed
            watersheds.at[downstream_watershed_idx[0], 'geometry'] = merged_geom
    # Step 5: Remove the original watershed entries where Divergence is 2
    watersheds = watersheds[watersheds.Divergence != 2]

    return watersheds


def NHDPlus_preprocessing(VPUID):
    database_name = f'SWATPlus_NHDPlus/{VPUID}'
    outpath = os.path.join(SWATGenXPaths.NHDPlus_path,f'{database_name}/streams.pkl')
    if not os.path.exists(outpath):
        execute_processes(database_name)
    else:
        print(f'#### NHDPlus for {VPUID} is already processed ####')

def execute_processes(database_name):
    report_path = os.path.join(SWATGenXPaths.NHDPlus_path, database_name)

    logger = LoggerSetup(
        report_path=report_path,
        verbose=True, rewrite=True).setup_logger("NHDPlus_preprocessing")

    logger.info(f"Start time: {datetime.datetime.now()}")

    stage = "loading data"
    #loginfo(logfile, f"stage: {stage}")
    logger.info(f"stage: {stage}")
    WBDHU12 = gpd.GeoDataFrame(pd.read_pickle(os.path.join(SWATGenXPaths.NHDPlus_path, f'{database_name}/WBDHU12.pkl')))
    WBDHU8 = gpd.GeoDataFrame(pd.read_pickle(os.path.join(SWATGenXPaths.NHDPlus_path, f'{database_name}/WBDHU8.pkl')))
    NHDPlusCatchment = gpd.GeoDataFrame(pd.read_pickle(os.path.join(SWATGenXPaths.NHDPlus_path, f'{database_name}/NHDPlusCatchment.pkl')))
    NHDWaterbody = gpd.GeoDataFrame(pd.read_pickle(os.path.join(SWATGenXPaths.NHDPlus_path, f'{database_name}/NHDWaterbody.pkl')))
    NHDFlowline = gpd.GeoDataFrame(pd.read_pickle(os.path.join(SWATGenXPaths.NHDPlus_path, f'{database_name}/NHDFlowline.pkl')))
    NHDPlusFlowlineVAA = gpd.GeoDataFrame(pd.read_pickle(os.path.join(SWATGenXPaths.NHDPlus_path, f'{database_name}/NHDPlusFlowlineVAA.pkl')))
    logger.info('Data loaded')
    logger.info(f"NHDPlusFlowlineVAA columns: {NHDPlusFlowlineVAA.columns}")
    logger.info(f"NHDPlusCatchment columns: {NHDPlusCatchment.columns}")
    logger.info(f"NHDFlowline columns: {NHDFlowline.columns}")
    logger.info(f"NHDWaterbody columns: {NHDWaterbody.columns}")
    logger.info(f"WBDHU8 columns: {WBDHU8.columns}")



    ### in some cases the VPUID is in lower case, we need to change to upper case


    NHDPlusCatchment = NHDPlusCatchment.merge(NHDPlusFlowlineVAA.drop(columns=['VPUID','AreaSqKm']), on='NHDPlusID')
    NHDFlowline = NHDFlowline.merge(NHDPlusFlowlineVAA.drop(columns=['VPUID']), on='NHDPlusID')

    streams = NHDFlowline.copy()[
                                [
                                'NHDPlusID','StreamOrde','UpHydroSeq','HydroSeq',
                                'DnHydroSeq','WBArea_Permanent_Identifier',
                                'StartFlag','TerminalFl','Divergence','VPUID',
                                'Permanent_Identifier','MaxElevSmo','MinElevSmo',
                                'AreaSqKm','LengthKM','TotDASqKm', 'geometry'
                                ]
                                ]


    watersheds = NHDPlusCatchment.copy()[['NHDPlusID','UpHydroSeq','HydroSeq','DnHydroSeq','StartFlag','TerminalFl','Divergence','VPUID','geometry']]
    # process dat huc8 and huc12"
    stage = "processing huc8 and huc12"
    logger.info(f"stage: {stage}, \nlength of streams before processing: {len(streams)}")
    streams = process_huc_and_merge(watersheds, WBDHU8, streams, 'huc8')
    streams = process_huc_and_merge(watersheds, WBDHU12, streams, 'huc12')

    stage = "resetting length unit"
    logger.info(f"stage: {stage}, \nlength of streams before processing: {len(streams)}")   
    streams = converting_length_unit_of_streams(streams)
    streams = calculating_drop(streams)

    stage = "removing second divergence"
    logger.info(f"stage: {stage}, \nlength of streams before processing: {len(streams)}")   
    save_divergence_2_streams(streams,watersheds,report_path)
    streams = removing_second_divergence(streams)
    streams = remove_streams_without_drainage(streams,watersheds)
    streams = remove_coastal_lines(streams)
    streams = setting_zero_for_outlets_and_headwaters(streams)

    stage = "removing isolated streams"
    logger.info(f"stage: {stage}, \nlength of streams before processing: {len(streams)}")
    isolated_path = os.path.join(
        SWATGenXPaths.NHDPlus_path, database_name)
    streams = removing_isolated_streams(streams,watersheds, isolated_path)

    stage = "resetting start and terminal flags"
    logger.info(f"stage: {stage}, \nlength of streams before processing: {len(streams)}")
    streams = resetting_start_terminate_flags(streams)
    outlets_upstream_dict = generate_hydroseq_upstream_dict(streams)
    streams = adding_subbasins_level_one(streams,outlets_upstream_dict)

    stage = "setting data type"
    logger.info(f"stage: {stage}, \nlength of streams before processing: {len(streams)}")   
    streams = setting_data_type(streams)

    stage = "removing huc12 with nan"
    #loginfo(logfile, f"stage: {stage}, \nlength of streams before processing: {len(streams)}")
    logger.info(f"stage: {stage}, \nlength of streams before processing: {len(streams)}")   
    streams = streams[~streams.huc12.isna()].reset_index(drop=True)

    stage = "creating watersheds"
    logger.info(f"stage: {stage}, \nlength of streams before processing: {len(streams)}")   
    watersheds = creating_watershed_dissolving_divergence (watersheds)
    ###### SAVING THE PROCESSES STREAMS AND WATERSHEDS TO PICKLE FORMAT FOR FURTHER analysis

    stage = "saving the processed data"
    logger.info(f"stage: {stage}")
    watersheds.to_pickle(os.path.join(SWATGenXPaths.NHDPlus_path,f'{database_name}/watersheds.pkl'))
    streams.to_pickle(os.path.join(SWATGenXPaths.NHDPlus_path,f'{database_name}/streams.pkl'))

    # save the log file
    logger.info(f"End time: {datetime.datetime.now()}")
    
try:
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
    from SWATGenX.SWATGenXLogging import LoggerSetup
    from SWATGenX.utils import get_all_VPUIDs
except Exception:
    from SWATGenXConfigPars import SWATGenXPaths
    from SWATGenXLogging import LoggerSetup
    from utils import get_all_VPUIDs

def check_NHDPlus_preprocessed_by_VPUID(VPUID):
    output_names = ['watersheds.pkl', 'streams.pkl']
    for output_name in output_names:
        if not os.path.exists(os.path.join(SWATGenXPaths.extracted_nhd_swatplus_path, VPUID, output_name)):
            raise ValueError(f'#### NHDPlus for {VPUID} is not preprocessed ####')    
    print(f'#### NHDPlus for {VPUID} is preprocessed ####')


if __name__ == "__main__":
    VPUID = '0202'
    VPUIDs = get_all_VPUIDs()

    for VPUID in VPUIDs:
        try:
            check_NHDPlus_preprocessed_by_VPUID(VPUID)
        except Exception:
            NHDPlus_preprocessing(VPUID)
