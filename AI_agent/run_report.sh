#!/bin/bash
echo "Welcome to the Interactive Report Generator"
echo "------------------------------------------"

# Function to prompt for yes/no with default
ask_yes_no() {
    local prompt="$1"
    local default="$2"
    local response
    
    if [[ "$default" == "y" ]]; then
        read -p "$prompt [Y/n]: " response
        response=${response:-y}
    else
        read -p "$prompt [y/N]: " response
        response=${response:-n}
    fi
    
    [[ "$response" =~ ^[Yy]$ ]]
}

# Get the basic report type
echo "Select the report type to generate:"
select REPORT_TYPE in prism nsrdb modis cdl groundwater gov_units gssurgo snodas climate_change all quit; do
    case $REPORT_TYPE in
        quit)
            echo "Exiting."
            exit 0
            ;;
        *)
            echo "You selected: $REPORT_TYPE"
            break
            ;;
    esac
done

# Prompt for report-specific information first
case $REPORT_TYPE in
    prism)
        echo -e "\n==== PRISM Climate Data Report Configuration ===="
        echo "This report analyzes temperature, precipitation, and other climate variables."
        ;;
    nsrdb)
        echo -e "\n==== NSRDB Solar Radiation Report Configuration ===="
        echo "This report analyzes solar radiation, wind speed, and relative humidity."
        ;;
    modis)
        echo -e "\n==== MODIS Satellite Data Report Configuration ===="
        echo "This report analyzes vegetation indices, land surface temperature, and land cover."
        ;;
    cdl)
        echo -e "\n==== Cropland Data Layer (CDL) Report Configuration ===="
        echo "This report analyzes crop types, distributions, and agricultural patterns."
        ;;
    groundwater)
        echo -e "\n==== Groundwater Report Configuration ===="
        echo "This report analyzes well data, groundwater levels, and aquifer properties."
        ;;
    gov_units)
        echo -e "\n==== Governmental Units Report Configuration ===="
        echo "This report analyzes administrative boundaries and jurisdictions."
        ;;
    gssurgo)
        echo -e "\n==== gSSURGO Soil Report Configuration ===="
        echo "This report analyzes soil types, properties, and classifications."
        ;;
    snodas)
        echo -e "\n==== SNODAS Snow Data Report Configuration ===="
        echo "This report analyzes snow depth, snow water equivalent, and snow cover."
        ;;
    climate_change)
        echo -e "\n==== Climate Change Analysis Report Configuration ===="
        echo "This report analyzes climate projections and compares historical vs. future climate."
        ;;
    all)
        echo -e "\n==== Comprehensive Report Configuration ===="
        echo "This will generate all report types for your region of interest."
        ;;
esac

# Get geographic area of interest
echo -e "\n==== Geographic Area Configuration ===="
echo "Please specify how you would like to define your area of interest."

PS3="Select method: "
select METHOD in "County Selection" "Manual Coordinates" "Use Defaults"; do
    case $METHOD in
        "County Selection")
            # Get list of states from Python utility
            echo -e "\nGetting list of states..."
            STATES=$(python -c "from utils.get_county_bbox import get_state_codes; print(','.join(get_state_codes()))" 2>/dev/null)
            IFS=',' read -ra STATE_ARRAY <<< "$STATES"
            
            # Display state selection menu
            echo -e "\nSelect a state:"
            select STATE_CODE in "${STATE_ARRAY[@]}"; do
                if [[ -n "$STATE_CODE" ]]; then
                    echo "You selected: $STATE_CODE"
                    break
                else
                    echo "Invalid selection. Please try again."
                fi
            done
            
            # Get list of counties for the selected state
            echo -e "\nGetting list of counties for $STATE_CODE..."
            COUNTIES=$(python -c "from utils.get_county_bbox import get_counties_by_state; print(','.join(get_counties_by_state('$STATE_CODE')))" 2>/dev/null)
            IFS=',' read -ra COUNTY_ARRAY <<< "$COUNTIES"
            
            # Display county selection menu
            echo -e "\nSelect a county:"
            select COUNTY_NAME in "${COUNTY_ARRAY[@]}"; do
                if [[ -n "$COUNTY_NAME" ]]; then
                    echo "You selected: $COUNTY_NAME County, $STATE_CODE"
                    break
                else
                    echo "Invalid selection. Please try again."
                fi
            done
            
            # Get the bounding box for the selected county
            echo -e "\nRetrieving bounding box for $COUNTY_NAME County, $STATE_CODE..."
            
            # Create a temporary Python script to get bbox and handle errors
            TEMP_SCRIPT=$(mktemp)
            cat > "$TEMP_SCRIPT" << EOF
from utils.get_county_bbox import get_bounding_box
import sys
import json

try:
    bbox = get_bounding_box('$COUNTY_NAME', '$STATE_CODE')
    if bbox:
        print(json.dumps(bbox))
    else:
        print("")
except Exception as e:
    print("")
EOF
            
            # Run the script and capture the output
            BBOX_JSON=$(python "$TEMP_SCRIPT" 2>/dev/null)
            rm "$TEMP_SCRIPT"
            
            # Check if we got valid output
            if [[ -z "$BBOX_JSON" ]]; then
                echo "Error: Could not retrieve bounding box for $COUNTY_NAME County, $STATE_CODE."
                echo "Using default values instead."
                MIN_LON=-85.444332
                MIN_LAT=43.158148
                MAX_LON=-84.239256
                MAX_LAT=44.164683
            else
                # Parse the JSON string to extract values
                MIN_LON=$(echo $BBOX_JSON | python -c "import sys, json; print(json.load(sys.stdin)['min_lon'])")
                MIN_LAT=$(echo $BBOX_JSON | python -c "import sys, json; print(json.load(sys.stdin)['min_lat'])")
                MAX_LON=$(echo $BBOX_JSON | python -c "import sys, json; print(json.load(sys.stdin)['max_lon'])")
                MAX_LAT=$(echo $BBOX_JSON | python -c "import sys, json; print(json.load(sys.stdin)['max_lat'])")
                
                echo "Successfully retrieved bounding box coordinates."
            fi
            ;;
            
        "Manual Coordinates")
            echo "Please specify the bounding box coordinates for your area of interest."
            echo "These define the rectangular region to analyze (decimal degrees)."

            read -p "Enter minimum longitude (Western boundary, default -85.444332): " MIN_LON
            MIN_LON=${MIN_LON:--85.444332}

            read -p "Enter minimum latitude (Southern boundary, default 43.158148): " MIN_LAT
            MIN_LAT=${MIN_LAT:-43.158148}

            read -p "Enter maximum longitude (Eastern boundary, default -84.239256): " MAX_LON
            MAX_LON=${MAX_LON:--84.239256}

            read -p "Enter maximum latitude (Northern boundary, default 44.164683): " MAX_LAT
            MAX_LAT=${MAX_LAT:-44.164683}
            ;;
            
        "Use Defaults")
            echo "Using default coordinates for central Michigan."
            MIN_LON=-85.444332
            MIN_LAT=43.158148
            MAX_LON=-84.239256
            MAX_LAT=44.164683
            ;;
    esac
    break
done

echo -e "\nAnalyzing region: $MIN_LON,$MIN_LAT to $MAX_LON,$MAX_LAT"

# Time period configuration for reports that need it
if [[ $REPORT_TYPE != "gov_units" ]]; then
    echo -e "\n==== Time Period Configuration ===="
    echo "Please specify the time range for your analysis."

    read -p "Enter start year (default 2010): " START_YEAR
    START_YEAR=${START_YEAR:-2010}

    read -p "Enter end year (default 2020): " END_YEAR
    END_YEAR=${END_YEAR:-2020}

    echo -e "\nAnalyzing period: $START_YEAR to $END_YEAR"
fi

# Data resolution configuration
echo -e "\n==== Data Resolution Configuration ===="
echo "Higher resolution provides more detail but takes longer to process."

read -p "Enter resolution in meters (default 250): " RESOLUTION
RESOLUTION=${RESOLUTION:-250}

# Temporal aggregation only for reports that use it
AGGREGATION="monthly"  # Default value
if [[ $REPORT_TYPE == "prism" || $REPORT_TYPE == "nsrdb" || $REPORT_TYPE == "modis" || 
      $REPORT_TYPE == "snodas" || $REPORT_TYPE == "climate_change" || $REPORT_TYPE == "all" ]]; then
    echo -e "\nTemporal aggregation determines how data is summarized over time."
    PS3="Select temporal aggregation: "
    select AGGREGATION_OPTION in daily monthly seasonal annual; do
        if [[ -n "$AGGREGATION_OPTION" ]]; then
            AGGREGATION="$AGGREGATION_OPTION"
            echo "Selected aggregation: $AGGREGATION"
            break
        else
            echo "Invalid selection. Please choose a number from the list."
        fi
    done
fi

# Climate change-specific input if needed
if [[ $REPORT_TYPE == "climate_change" || $REPORT_TYPE == "all" ]]; then
    echo -e "\n==== Climate Change Analysis Configuration ===="
    echo "This requires historical and future time periods for comparison."
    
    echo -e "\nHistorical period configuration:"
    read -p "Enter historical start year (default 2000): " HIST_START
    HIST_START=${HIST_START:-2000}
    
    read -p "Enter historical end year (default 2014): " HIST_END
    HIST_END=${HIST_END:-2014}
    
    echo -e "\nFuture period configuration:"
    read -p "Enter future start year (default 2045): " FUT_START
    FUT_START=${FUT_START:-2045}
    
    read -p "Enter future end year (default 2060): " FUT_END
    FUT_END=${FUT_END:-2060}
    
    echo -e "\nClimate model configuration:"
    echo "Available models: ACCESS-CM2, CMCC-ESM2, MPI-ESM1-2-HR, CESM2, NorESM2-MM"
    read -p "Enter climate model (default ACCESS-CM2): " CC_MODEL
    CC_MODEL=${CC_MODEL:-ACCESS-CM2}
    
    read -p "Enter ensemble member (default r1i1p1f1): " CC_ENSEMBLE
    CC_ENSEMBLE=${CC_ENSEMBLE:-r1i1p1f1}
    
    echo -e "\nClimate scenario (emissions scenario):"
    echo "Options: ssp126 (low emissions), ssp245 (moderate), ssp585 (high emissions)"
    read -p "Enter climate scenario (default ssp245): " CC_SCENARIO
    CC_SCENARIO=${CC_SCENARIO:-ssp245}
    
    echo -e "\nComparing $HIST_START-$HIST_END to $FUT_START-$FUT_END under $CC_SCENARIO scenario"
fi

# Initialize empty options variables
EXTRA_PRISM_OPTIONS=""
EXTRA_NSRDB_OPTIONS=""
EXTRA_MODIS_OPTIONS=""
EXTRA_SNOW_OPTIONS=""



# Processing mode configuration
echo -e "\n==== Processing Mode Configuration ===="
echo "Sequential processing is slower but uses less memory."
echo "Parallel processing is faster but uses more memory."

if ask_yes_no "Run sequentially?" "n"; then
    SEQ="--sequential"
else
    SEQ=""
fi

# Output directory configuration
echo -e "\n==== Output Configuration ===="
read -p "Enter output directory name (default: reports_output/$REPORT_TYPE): " OUTPUT_DIR_NAME
OUTPUT_DIR_NAME=${OUTPUT_DIR_NAME:-"reports_output/$REPORT_TYPE"}

# Create directory
OUTPUT_DIR="$OUTPUT_DIR_NAME"
mkdir -p "$OUTPUT_DIR"
echo "Output will be saved to: $OUTPUT_DIR"

# Build the command - without using line breaks or backslashes
CMD="python report_generator.py --type $REPORT_TYPE --output $OUTPUT_DIR --min-lon $MIN_LON --min-lat $MIN_LAT --max-lon $MAX_LON --max-lat $MAX_LAT --resolution $RESOLUTION"

# Add time period parameters for reports that need them
if [[ $REPORT_TYPE != "gov_units" ]]; then
    CMD="$CMD --start-year $START_YEAR --end-year $END_YEAR"
fi

# Add aggregation parameter only for reports that need it
if [[ $REPORT_TYPE == "prism" || $REPORT_TYPE == "nsrdb" || $REPORT_TYPE == "modis" || 
      $REPORT_TYPE == "snodas" || $REPORT_TYPE == "climate_change" || $REPORT_TYPE == "all" ]]; then
    CMD="$CMD --aggregation $AGGREGATION"
fi

# Add report-specific options
if [[ -n "$EXTRA_PRISM_OPTIONS" ]]; then
    CMD="$CMD $EXTRA_PRISM_OPTIONS"
fi

if [[ -n "$EXTRA_NSRDB_OPTIONS" ]]; then
    CMD="$CMD $EXTRA_NSRDB_OPTIONS"
fi

if [[ -n "$EXTRA_MODIS_OPTIONS" ]]; then
    CMD="$CMD $EXTRA_MODIS_OPTIONS"
fi

if [[ -n "$EXTRA_SNOW_OPTIONS" ]]; then
    CMD="$CMD $EXTRA_SNOW_OPTIONS"
fi

# Add climate change options if needed
if [[ $REPORT_TYPE == "climate_change" || $REPORT_TYPE == "all" ]]; then
    CMD="$CMD --hist-start-year $HIST_START --hist-end-year $HIST_END --fut-start-year $FUT_START --fut-end-year $FUT_END --cc-model $CC_MODEL --cc-ensemble $CC_ENSEMBLE --cc-scenario $CC_SCENARIO"
fi

# Add sequential flag if needed
if [[ -n "$SEQ" ]]; then
    CMD="$CMD $SEQ"
fi

# Review and run
echo -e "\n==== Command Review ===="
echo "The following command will be executed:"
echo -e "\n$CMD\n"

if ask_yes_no "Would you like to proceed?" "y"; then
    echo -e "\nStarting report generation process..."
    echo "This may take some time depending on the selected options."
    echo "-----------------------------------------------------"
    eval "$CMD"
    
    # Check exit status
    if [ $? -eq 0 ]; then
        echo -e "\n✅ Report generation completed successfully!"
        echo "Output is available in: $OUTPUT_DIR"
    else
        echo -e "\n❌ Report generation encountered errors."
        echo "Please check the logs for more information."
    fi
else
    echo "Operation cancelled."
fi
