import os
import geopandas as gpd
import numpy as np
"""

Description:
Abstract: The data in these six files is derived from Wellogic, the EGLE statewide ground water database. The six files combined contain information on over 575,000 spatially verified water well records. The six files are intended to provide water well information for wells in counties clustered by geographic region: Upper Peninsula, Northern Lower Peninsula, East Central Lower Peninsula, West Central Lower Peninsula, Southwest Lower Peninsula and the South Central – Southeastern Lower Peninsula. The files are constructed to be easily merged, containing the same number and type of attribute fields. Although the derived data in these files represents the best readily available data, the six files do not represent a complete database of all wells or well records in existence. Beginning January 1, 2000 virtually 100% of new wells constructed are accounted for in Wellogic, however for wells older than 2000 the rate of inclusion varies from county to county, and may be considerably lower. Further, there is a quality control check on location that may exclude a limited number of wells from Wellogic from the six files made available on this site. The locational data also has varying degrees of accuracy; ranging from precise GPS point collection to address geocoding, but there may also be erroneous locations regardless of collection method that have not been corrected as of yet.  Refer to the METHD_COLL field to determine each individual record’s potential locational accuracy. Field codes described below.------------------------------------------------------------------------------------------Field Definitions:WELLID : Wellogic ID number (unique identifying number, first 2 digits represent county number)PERMIT_NUM : Well permit number as assigned by local health departmentWELL_TYPE : Type of wellOTH = OtherHEATP = Heat pumpHOSHLD = HouseholdINDUS = IndustrialIRRI = IrrigationTESTW = Test wellTY1PU = Type I publicTY2PU = Type II publicTY3PU = Type III publicTYPE_OTHER : Type of well if WELL_TYPE is 'OTH'WEL_STATUS : Status of wellOTH = OtherACT = ActiveINACT = InactivePLU = Plugged/AbandonedSTATUS_OTH : Status of well if WEL_STATUS is 'OTH' WSSN : Water Supply Serial Number, only if public wellWELL_NUM : Individual well number/name, only if public wellDRILLER_ID : Water Well Drilling Contractor Registration Number as assigned by State of Michigan DRILL_METH : Method used to drill the well boreholeOTH = OtherAUGBOR = Auger/BoredCABTOO = Cable ToolCASHAM = Casing HammerDRIVEN = Driven HandHOLROD = Hollow RodJETTIN = JettedMETH_OTHER : Method used to drill if DRILL_METH is 'OTH'CASE_TYPE : Well casing typeOTH  = OtherUNK  = UnknownPVCPLA  = PVC PlasticSTEBLA  = Steel-blackSTEGAL  = Steel-GalvanizedCASE_OTHER : Well casing type is CASE_TYPE is 'OTH'CASE_DIA : Well Casing Diameter (in inches)CASE_DEPTH : Depth of Casing (in feet) SCREEN_FRM : Depth of top of screen (in feet)SCREEN_TO : Depth of bottom of screen (in feet)SWL : Depth of Static Water Level (in feet)FLOWING : Naturally flowing well (Y or N)AQ_TYPE : Aquifer typeDRIFT   = Well draws water from the glacial driftROCK    = Well draws water from the bedrockDRYHOL  = Dry hole, well did not produce waterUNK     = UnknownTEST_DEPTH : Depth of drawdown when the well was developed (in feet)TEST_HOURS : Duration of pumping when the well was developed (in hours)TEST_RATE : Rate of water flow when the well was developed (in Gallons per Minute)TEST_METHD : Method used to develop the wellUNK     = UnknownOTH     = OtherAIR     = AirBAIL    = BailerPLUGR   = PlungerTSTPUM  = Test Pump TEST_OTHER : Method used to develop the well if TEST_METHD is 'OTH'GROUT : Whether the well was grouted or notPMP_CPCITY : Capacity of the pump installed in the well (in Gallons per minute)METHD_COLL : Method of collection of the latitude/longitude coordinates001 = Address Matching-House Number002 = Address Matching-Street Centerline004 = Address Matching-Nearest Intersection012 = GPS Carrier Phase Static Relative Position Tech.013 = GPS Carrier Phase Kinematic Relative Position Tech.014 = GPS Code Measurement Differential (DGPS)015 = GPS Precise Positioning Service016 = GPS Code Meas. Std. Positioning Service  SA Off017 = GPS Std. Positioning Service SA On018 = Interpolation-Map019 = Interpolation-Aerial Photo020 = Interpolation-Satellite Photo025 = Classical Surveying Techniques027 = Section centroid028 = TownRange centroid036 = Quarter-Quarter-Quarter centroidELEV_METHD : Method of collection of the elevation003 = GPS Code Measurement Differential (DGPS)005 = GPS Code Meas. Std. Positioning Svc.  SA Off007 = Classical Surveying Techniques014 = Topographic Map InterpolationOTH = OtherUNK = UnknownWITHIN_CO: Whether the well is within the stated countyWITHIN_SEC: Whether the well is within the stated land survey sectionLOC_MATCH: Whether the well is within the stated Tier/RangeSEC_DIST: Whether the well point is within 200 feet of the stated land survey sectionELEV_DEM: Elevation in feet above mean sea levelELEV_DIF: Absolute difference, in feet, between ELEVATION and ELEV_DEMLANDSYS: The Land System Group polygon that the well falls withinDEPTH_FLAG:1: WELL_DEPTH = 02: WELL_DEPTH &lt; 25ft or WELL_DEPTH &gt; 1000ftELEV_FLAG:1: ELEVATION (Wellogic Field) =02: ELEVATION (Wellogic Field) &lt; 507ft OR &gt; 1980ft3: ELEVATION (Wellogic Field) &lt; DEM min OR &gt; DEM max4: ELEV_DIF &gt; 20 ftSWL_FLAG:1: SWL = 02: SWL &gt;= WELL_DEPTH in a Bedrock well OR SWL &gt;= SCREEN_BOT in a Glacial well3: SWL &gt; 900ftSPC_CPCITY: Specific Capacity = (TEST_RATE / TEST_DEPTH). Only calculated if TEST_METHD = BAIL, PLUGR or TSTPUMAQ_CODE:N: No Lithology Record associated with the well recordB: Blank (AQTYPE = null) noted among the strataD: Drift (Glacial) WellR: Rock WellU: Unknown Lithology noted among the strata* PROCESSING NOTE – This evaluation reads the [AQTYPE] field for each stratum from the LITHOLOGY table, beginning at the top and looping down to each subjacent stratum. If the previous stratum = ‘R’ AND the bottommost stratum = ‘R’, then [AQ_CODE] is set to ‘R’. If  the previous stratum = ‘R’ AND the next stratum = ‘D’, then [AQ_CODE] is set to ‘D’ and [AQ_FLAG] is set to ‘L’. If aType = ‘R’ AND screendepth &gt; 0 R’ AND  screendepth &lt;= welldepth, then [AQ_CODE] is set to ‘D’ and [AQ_FLAG] is set to ‘S’. If aType = ‘R’ AND welldepth &lt;= topofrock, then [AQ_CODE] is set to ‘D’ and [AQ_FLAG] is set to ‘D’.&lt;o:p&gt;&lt;/o:p&gt;ROCK_TOP: Depth below land surface in ftAQ_THK_1: Glacial Wells only; Summed thickness of all drift strata from SCREEN_BOT upward to:A) the bottom of a qualifying confining unit (&gt;= 5ft CM or &gt;= 10ft PCM)elseB) the SWL (if SWL &gt;0 and SWL &lt; SCREEN_BOT and SWL &lt;= 99elseC) the ground surface* PROCESSING NOTE – If [SCREEN_BOT] &lt;= layerbottom for seqnum1 (the bottommost layer), AND seqnum1 = PCM OR seqnum1 = CM (i.e., the screen bottoms out in a PCM or CM unit), the analysis for a qualifying confining unit moves up to the next strata in order to prevent this circumstance from generating an [AQ_THK_1] = 0. The code performs a “layer-clip” function in the case where [SCREEN_BOT] is within a PCM or CM strata, but the [SCREEN_TOP] is in an AQ or MAQ strata, counting upward from layerbottom of bottommost AQ or MAQ strata in the screened interval, rather than counting upward from [SCREEN_BOT], which is the normal logic.&lt;o:p&gt;&lt;/o:p&gt;TOPAQ: Top of AquiferBOTAQ: Bottom of AquiferWWAT_ID: The number assigned to a well that is a large quantity withdrawalAQ_FLAG:U: Unknown lithologies (i.e., Hyd. Cond. = 999999, so no values are calculated for B_H_COND, B_V_COND, or B_TRANS)D: Depth Condition - WELL_DEPTH &lt;= ROCK_TOP so AQ_CODE was set to &quot;D&quot;L: Lithology Problem (a Glacial stratum was encountered below a Rock stratum)S: Bottommost stratum is 'R' and SCREEN_TO &gt; 0 and SCREEN_TO &lt;= WELL_DEPTH, so AQ_CODE was set to 'D' (this traps for Glacial wells that were &quot;over-bored&quot; into the rock, but a screen was set in the glacial package)FOR GLACIAL WELLS ONLYAQ_THK_1: Glacial Wells only; Summed thickness of all drift strata (i.e., AQ, MAQ, PCM, and CM) from SCREEN_BOT upward to:A) the SWL (if SWL &gt; 0 and SWL &lt; SCREEN_BOT and SWL &lt;= 900)elseB) the ground surfaceH_COND_1: &quot;Confined&quot; equivalent horizontal hydraulic conductivity for sediments in a glacial well (uses AQ_THK_1)H_COND_2: &quot;Unconfined&quot; equivalent horizontal hydraulic conductivity for sediments in a glacial well (uses AQ_THK_2)V_COND_1: &quot;Confined&quot; equivalent vertical hydraulic conductivity for sediments in a glacial well (uses AQ_THK_1)V_COND_2: &quot;Unconfined&quot; equivalent vertical hydraulic conductivity for sediments in a glacial well (uses AQ_THK_2)TRANSMSV_1: &quot;Confined&quot; equivalent transmissivity for sediments in a glacial well (H_COND_1 * AQ_THK_1)TRANSMSV_2: &quot;Unconfined&quot; equivalent transmissivity for sediments in a glacial well (H_COND_2 * AQ_THK_2)FOR GLACIAL PORTION OF BEDROCK WELLSAQ_THK_D: The summed thickness of a drift aquifer from bedrock to ground surfaceH_COND_D: Equivalent horizontal hydraulic conductivity for glacial sediments in a bedrock well (uses AQ_THK_D)V_COND_D: Equivalent vertical hydraulic conductivity for glacial sediments in a bedrock well (uses AQ_THK_D)TRANS_D: Equivalent transmissivity for glacial sediments in a bedrock well (H_COND_D * AQ_THK_D)FOR BEDROCK PORTION OF BEDROCK WELLSB_AQ_THK: Summed thickness of all bedrock strata (i.e. AQ, MAQ, PCM, and CM) from WELL_DEPTH upward to ROCK_TOPB_H_COND: Equivalent horizontal hydraulic conductivity for Bedrock Units (uses B_AQ_THK)B_V_COND: Equivalent vertical hydraulic conductivity for Bedrock Units (uses B_AQ_THK)B_TRANS: Equivalent transmissivity for Bedrock Units (B_H_COND * B_AQ_THK)------------------------------------------------------------------------------------------NOTE: This data download does not contain the associated lithology files. Download this data on our Wellogic Water Wells by County website. If you have questions concerning this data, please contact: Wellogic@Michigan.gov


"""
def clean_observations():
    """
    Clean the observations.geojson file to remove invalid, unknown, and 
    unreasonable values based on Wellogic dataset descriptions.
    """

    # Define file paths
    path_original = "/data/SWATGenXApp/GenXAppData/observations/observations_original.geojson"
    path_cleaned = "/data/SWATGenXApp/GenXAppData/observations/observations.geojson"

    # Load the original observations file
    gdf = gpd.read_file(path_original)

    # Print CRS and column information
    print(f"CRS: {gdf.crs}")
    print("\nColumns in observations.geojson:")
    print(gdf.columns)
    print("\nData types in observations.geojson:")
    print(gdf.dtypes)
    print(f"\nRange of SWL before filtering: {np.ptp(gdf['SWL'])}")

    # Define required columns
    required_columns = ['WELLID', 'SWL', 'ELEV_DEM', 'PMP_CPCITY', 'WEL_STATUS', 'AQ_TYPE', 'geometry']

    # Retain only required columns
    gdf = gdf[required_columns]

    assert 'col' not in gdf.columns, "Column 'col' should not be present in the dataset"
    assert 'row' not in gdf.columns, "Column 'row' should not be present in the dataset"    
    
    for col in required_columns:
        assert col in gdf.columns, f"Column {col} is missing from the dataset"

    # Filtering step: Drop invalid and unreasonable values
    # 1. Remove rows where SWL is NaN
    gdf = gdf.dropna(subset=['SWL'])

    # 2. Remove known placeholders or missing values
    invalid_values = [999999, -999999, -9999, 99999, -9999]  # Common placeholders for missing data
    gdf = gdf[~gdf['SWL'].isin(invalid_values)]
    gdf = gdf[~gdf['ELEV_DEM'].isin(invalid_values)]
    gdf = gdf[~gdf['PMP_CPCITY'].isin(invalid_values)]

    # 3. Apply domain-specific constraints based on Wellogic dataset description

    # Remove unreasonable static water levels (SWL)
    gdf = gdf[(gdf['SWL'] > 0) & (gdf['SWL'] < 900)]  # SWL = 0 or >900 ft is flagged

    # Remove unreasonable elevation values
    gdf = gdf[(gdf['ELEV_DEM'] > 507) & (gdf['ELEV_DEM'] < 1980)]  # Flagged if <507 ft or >1980 ft

    # Remove unrealistic pump capacities (assuming typical well capacities are < 5000 GPM)
    gdf = gdf[(gdf['PMP_CPCITY'] > 0) & (gdf['PMP_CPCITY'] < 5000)]  

    # 4. Filter based on Well Status (WEL_STATUS)
    # Remove unknown ('OTH') or abandoned ('PLU') wells
    valid_well_statuses = ['ACT', 'INACT']  # Active and Inactive wells only
    gdf = gdf[gdf['WEL_STATUS'].isin(valid_well_statuses)]

    # 5. Filter based on Aquifer Type (AQ_TYPE)
    # Remove dry holes ('DRYHOL') and unknown aquifers ('UNK')
    invalid_aquifers = ['DRYHOL', 'UNK']
    gdf = gdf[~gdf['AQ_TYPE'].isin(invalid_aquifers)]

    # Display updated statistics
    print(f"\nCRS after filtering: {gdf.crs}")
    print("\nColumns retained in cleaned dataset:")
    print(gdf.columns)
    print("\nData types after cleaning:")
    print(gdf.dtypes)
    print(f"Range of SWL after filtering: {np.ptp(gdf['SWL'])}")

    # Save the cleaned dataset
    gdf.to_file(path_cleaned, driver='GeoJSON')
    print(f"Cleaned observations.geojson file has been written to {path_cleaned}")

if __name__ == "__main__":
    clean_observations()
