#!/bin/bash

base_url="https://water.usgs.gov/nwaa-data/data/water-use/"
download_dir="./downloaded_data"

# Create download directory if it doesn't exist
mkdir -p "$download_dir"

# List files directly in the for loop instead of using an array
for file in \
    "wu-irrigation-cu/combined_wu-irrigation-cu_CONUS_200001-202012_long.csv" \
    "wu-irrigation-cu/irrcutot/irrcutot_wu-irrigation-cu_CONUS_200001-202012_long.csv" \
    "wu-irrigation-cu/irrcutot/irrcutot_wu-irrigation-cu_CONUS_200001-202012_wide.csv" \
    "wu-irrigation-cu/irrcutot/irrcutot_wu-irrigation-cu_CONUS_200910-202009_mean_map.pdf" \
    "wu-irrigation-wd/combined_wu-irrigation-wd_CONUS_200001-202012_long.csv" \
    "wu-irrigation-wd/irrcutot/irrwdtot_wu-irrigation-wd_CONUS_200001-202012_long.csv" \
    "wu-irrigation-wd/irrcutot/irrwdtot_wu-irrigation-wd_CONUS_200001-202012_wide.csv" \
    "wu-irrigation-wd/irrcutot/irrwdtot_wu-irrigation-wd_CONUS_200910-202009_mean_map.pdf" \
    "wu-public-supply-cu/combined_wu-public-supply-cu_CONUS_200901-202012_long.csv" \
    "wu-public-supply-cu/pscutot/pscutot_wu-public-supply-cu_CONUS_200901-202012_long.csv" \
    "wu-public-supply-cu/pscutot/pscutot_wu-public-supply-cu_CONUS_200901-202012_wide.csv" \
    "wu-public-supply-cu/pscutot/pscutot_wu-public-supply-cu_CONUS_200910-202009_mean_map.pdf" \
    "wu-public-supply-wd/combined_wu-public-supply-wd_CONUS_200001-202012_long.csv" \
    "wu-public-supply-wd/pswdtot/pswdtot_wu-public-supply-wd_CONUS_200001-202012_long.csv" \
    "wu-public-supply-wd/pswdtot/pswdtot_wu-public-supply-wd_CONUS_200001-202012_wide.csv"
do
    # Skip empty entries
    if [ -z "$file" ]; then
        continue
    fi
    
    # Extract filename from path
    filename=$(basename "$file")
    # Create subdirectory structure if needed
    subdir=$(dirname "$file")
    mkdir -p "$download_dir/$subdir"
    
    # Full path where file will be saved
    local_path="$download_dir/$file"
    
    # Check if file already exists
    if [ -f "$local_path" ]; then
        echo "File already exists: $local_path - skipping"
    else
        echo "Downloading: $file"
        # Create the directory structure if it doesn't exist
        mkdir -p "$(dirname "$local_path")"
        
        # Use wget with error handling
        if wget -q --spider "${base_url}${file}"; then
            wget -P "$download_dir/$subdir" "${base_url}${file}"
            echo "Downloaded successfully: $file"
        else
            echo "File not found on server: ${base_url}${file}"
        fi
    fi
done

echo "Download process completed."