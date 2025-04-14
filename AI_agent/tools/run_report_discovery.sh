#!/bin/bash

# Colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Add the tools directory to PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Function to print colored text
print_color() {
    color=$1
    text=$2
    echo -e "${color}${text}${NC}"
}

# Function to check if Python modules are available
check_dependencies() {
    print_color $BLUE "Checking dependencies..."
    if ! python3 -c "import dir_discover" 2>/dev/null; then
        print_color $RED "Error: Cannot find dir_discover module."
        print_color $YELLOW "Make sure you are in the correct directory and virtual environment is activated."
        exit 1
    fi
}

# Function to get available reports
get_reports() {
    reports=$(python3 -c "
import dir_discover
try:
    reports = dir_discover.discover_reports(silent=True, recursive=True)
    if not reports:
        print('NO_REPORTS')
    else:
        for report in sorted(reports.keys()):
            print(report)
except Exception as e:
    print(f'ERROR: {str(e)}')
")

    if [[ $reports == NO_REPORTS ]]; then
        print_color $YELLOW "No reports found in the specified directory."
        exit 1
    elif [[ $reports == ERROR* ]]; then
        print_color $RED "${reports}"
        exit 1
    fi
    echo "$reports"
}

# Function to get available groups for a report
get_groups() {
    local report=$1
    groups=$(python3 -c "
import dir_discover
try:
    reports = dir_discover.discover_reports(silent=True, recursive=True)
    if '$report' in reports:
        groups = sorted(reports['$report']['groups'].keys())
        if not groups:
            print('NO_GROUPS')
        else:
            for group in groups:
                print(group)
    else:
        print('NO_REPORT')
except Exception as e:
    print(f'ERROR: {str(e)}')
")

    if [[ $groups == NO_GROUPS ]]; then
        print_color $YELLOW "No groups found in report $report"
        return 1
    elif [[ $groups == NO_REPORT ]]; then
        print_color $RED "Report $report not found"
        return 1
    elif [[ $groups == ERROR* ]]; then
        print_color $RED "${groups}"
        return 1
    fi
    echo "$groups"
}

# Function to get files of a specific type in a group
get_files_by_type() {
    local report=$1
    local group=$2
    local ext=$3
    python3 -c "
import dir_discover
try:
    reports = dir_discover.discover_reports(silent=True, recursive=True)
    if '$report' in reports and '$group' in reports['$report']['groups']:
        files = reports['$report']['groups']['$group']['files'].get('$ext', {})
        for filename in sorted(files.keys()):
            print(filename)
except Exception as e:
    print(f'ERROR: {str(e)}')
"
}

# Function to get available file types in a group
get_file_types() {
    local report=$1
    local group=$2
    python3 -c "
import dir_discover
try:
    reports = dir_discover.discover_reports(silent=True, recursive=True)
    if '$report' in reports and '$group' in reports['$report']['groups']:
        for ext in sorted(reports['$report']['groups']['$group']['files'].keys()):
            print(ext)
except Exception as e:
    print(f'ERROR: {str(e)}')
"
}

# Function to run specific reader for a single file
run_reader_single_file() {
    local reader=$1
    local report=$2
    local group=$3
    local filename=$4
    local ext=$5
    local recreate_db="True"  # Python boolean value
    
    case $reader in
        "text")
            print_color $BLUE "Running text reader for $filename..."
            python3 -c "
import dir_discover
from text_reader import text_reader
try:
    reports = dir_discover.discover_reports(silent=True, recursive=True)
    file_info = reports['$report']['groups']['$group']['files']['$ext']['$filename']
    response = text_reader(file_info['path'])
    print(response)
except Exception as e:
    print(f'Error: {str(e)}')
"
            ;;
        "json")
            print_color $BLUE "Running JSON reader for $filename..."
            python3 -c "
import dir_discover
from json_reader import json_reader
try:
    reports = dir_discover.discover_reports(silent=True, recursive=True)
    if reports['$report']['config']:
        json_reader(reports['$report']['config'])
except Exception as e:
    print(f'Error: {str(e)}')
"
            ;;
        "image")
            print_color $BLUE "Running image reader for $filename..."
            python3 -c "
import dir_discover
from image_reader import image_reader
try:
    reports = dir_discover.discover_reports(silent=True, recursive=True)
    file_info = reports['$report']['groups']['$group']['files']['$ext']['$filename']
    image_reader(file_info['path'])
except Exception as e:
    print(f'Error: {str(e)}')
"
            ;;
        "csv")
            print_color $BLUE "Running CSV reader for $filename..."
            python3 -c "
import dir_discover
from csv_reader import csv_reader
try:
    reports = dir_discover.discover_reports(silent=True, recursive=True)
    file_info = reports['$report']['groups']['$group']['files']['$ext']['$filename']
    csv_reader(file_info['path'], recreate_db=$recreate_db)
except Exception as e:
    print(f'Error: {str(e)}')
"
            ;;
    esac
}

# Function to run specific reader for all files in a group
run_reader_group() {
    local reader=$1
    local report=$2
    local group=$3
    local recreate_db="True"  # Python boolean value
    
    case $reader in
        "text")
            print_color $BLUE "Running text reader for all text files..."
            python3 -c "
import dir_discover
from text_reader import text_reader
try:
    reports = dir_discover.discover_reports(silent=True, recursive=True)
    report_data = reports['$report']['groups']['$group']['files']
    for ext in ['.md', '.txt']:
        if ext in report_data:
            for filename, file_info in report_data[ext].items():
                print(f'\nAnalyzing {filename}...')
                response = text_reader(file_info['path'])
                print(response)
except Exception as e:
    print(f'Error: {str(e)}')
"
            ;;
        "json")
            print_color $BLUE "Running JSON reader..."
            python3 -c "
import dir_discover
from json_reader import json_reader
try:
    reports = dir_discover.discover_reports(silent=True, recursive=True)
    if reports['$report']['config']:
        json_reader(reports['$report']['config'])
except Exception as e:
    print(f'Error: {str(e)}')
"
            ;;
        "image")
            print_color $BLUE "Running image reader for all images..."
            python3 -c "
import dir_discover
from image_reader import image_reader
try:
    reports = dir_discover.discover_reports(silent=True, recursive=True)
    report_data = reports['$report']['groups']['$group']['files']
    if '.png' in report_data:
        for filename, file_info in report_data['.png'].items():
            print(f'\nAnalyzing {filename}...')
            image_reader(file_info['path'])
except Exception as e:
    print(f'Error: {str(e)}')
"
            ;;
        "csv")
            print_color $BLUE "Running CSV reader for all CSV files..."
            python3 -c "
import dir_discover
from csv_reader import csv_reader
try:
    reports = dir_discover.discover_reports(silent=True, recursive=True)
    report_data = reports['$report']['groups']['$group']['files']
    if '.csv' in report_data:
        for filename, file_info in report_data['.csv'].items():
            print(f'\nAnalyzing {filename}...')
            csv_reader(file_info['path'], recreate_db=$recreate_db)
except Exception as e:
    print(f'Error: {str(e)}')
"
            ;;
        "combined")
            print_color $BLUE "Running combined reader..."
            python3 -c "
import dir_discover
from combine_reader import combined_reader
try:
    reports = dir_discover.discover_reports(silent=True, recursive=True)
    combined_reader(reports, '$report', '$group', recreate_db=$recreate_db)
except Exception as e:
    print(f'Error: {str(e)}')
"
            ;;
    esac
}

# File selection menu
file_selection_menu() {
    local report=$1
    local group=$2
    local reader=$3
    local ext=$4
    
    while true; do
        print_color $GREEN "\nAvailable files of type $ext in $group:"
        readarray -t files < <(get_files_by_type "$report" "$group" "$ext")
        
        if [ ${#files[@]} -eq 0 ]; then
            print_color $RED "No $ext files found in group $group"
            return
        fi
        
        # Get the source directory for each file to display alongside the filename
        for i in "${!files[@]}"; do
            file_path=$(python3 -c "
import dir_discover
try:
    reports = dir_discover.discover_reports(silent=True, recursive=True)
    file_info = reports['$report']['groups']['$group']['files']['$ext']['${files[$i]}']
    print(file_info['path'])
except Exception as e:
    print('ERROR')
")
            if [[ $file_path != ERROR ]]; then
                dir_path=$(dirname "$file_path")
                # Get relative path from group directory
                relpath=${dir_path#*"$group"/}
                if [[ "$relpath" != "$dir_path" ]]; then  # Only show if it's in a subdirectory
                    echo "[$i] ${files[$i]} (in $relpath)"
                else
                    echo "[$i] ${files[$i]}"
                fi
            else
                echo "[$i] ${files[$i]}"
            fi
        done
        echo "[a] Process all files"
        echo "[b] Back to reader selection"
        echo "[q] Quit"
        
        read -p "Select a file number, 'a' for all, 'b' for back, or 'q' to quit: " choice
        
        if [[ $choice == "q" ]]; then
            exit 0
        elif [[ $choice == "b" ]]; then
            return
        elif [[ $choice == "a" ]]; then
            run_reader_group "$reader" "$report" "$group"
            return
        elif [[ $choice =~ ^[0-9]+$ ]] && [ $choice -lt ${#files[@]} ]; then
            selected_file=${files[$choice]}
            run_reader_single_file "$reader" "$report" "$group" "$selected_file" "$ext"
            return
        else
            print_color $RED "Invalid selection"
        fi
    done
}

# Reader menu
reader_menu() {
    local report=$1
    local group=$2
    while true; do
        print_color $GREEN "\nAvailable Readers:"
        print_color $YELLOW "Available file types in $group: $(get_file_types "$report" "$group")"
        
        # Check if CSV files exist before showing CSV reader option
        local has_csv=false
        if [[ $(get_file_types "$report" "$group") =~ ".csv" ]]; then
            has_csv=true
        fi
        
        echo "[1] Text Reader (for .md and .txt files)"
        echo "[2] JSON Reader (for config files)"
        echo "[3] Image Reader (for .png files)"
        if $has_csv; then
            echo "[4] CSV Reader (for .csv files)"
        fi
        echo "[5] Combined Reader (analyzes all available files)"
        echo "[b] Back to groups"
        echo "[q] Quit"
        
        read -p "Select a reader number, 'b' for back, or 'q' to quit: " choice
        
        case $choice in
            1) 
                if [[ $(get_file_types "$report" "$group") =~ ".md" ]]; then
                    file_selection_menu "$report" "$group" "text" ".md"
                fi
                if [[ $(get_file_types "$report" "$group") =~ ".txt" ]]; then
                    file_selection_menu "$report" "$group" "text" ".txt"
                fi
                ;;
            2) run_reader_group "json" "$report" "$group" ;;
            3) 
                if [[ $(get_file_types "$report" "$group") =~ ".png" ]]; then
                    file_selection_menu "$report" "$group" "image" ".png"
                fi
                ;;
            4) 
                if $has_csv; then
                    file_selection_menu "$report" "$group" "csv" ".csv"
                else
                    print_color $RED "No CSV files available in this group"
                fi
                ;;
            5) run_reader_group "combined" "$report" "$group" ;;
            "b") return ;;
            "q") exit 0 ;;
            *) print_color $RED "Invalid selection" ;;
        esac
    done
}

# Main menu
main_menu() {
    while true; do
        print_color $GREEN "\nAvailable Reports:"
        readarray -t reports < <(get_reports)
        
        if [ ${#reports[@]} -eq 0 ]; then
            print_color $RED "No reports found"
            exit 1
        fi
        
        for i in "${!reports[@]}"; do
            echo "[$i] ${reports[$i]}"
        done
        echo "[q] Quit"
        
        read -p "Select a report number or 'q' to quit: " choice
        
        if [[ $choice == "q" ]]; then
            exit 0
        fi
        
        if [[ $choice =~ ^[0-9]+$ ]] && [ $choice -lt ${#reports[@]} ]; then
            selected_report=${reports[$choice]}
            group_menu "$selected_report"
        else
            print_color $RED "Invalid selection"
        fi
    done
}

# Group menu
group_menu() {
    local report=$1
    while true; do
        print_color $GREEN "\nAvailable Groups for $report:"
        readarray -t groups < <(get_groups "$report")
        
        if [ ${#groups[@]} -eq 0 ]; then
            print_color $RED "No groups found in report $report"
            return
        fi
        
        for i in "${!groups[@]}"; do
            echo "[$i] ${groups[$i]}"
        done
        echo "[b] Back to reports"
        echo "[q] Quit"
        
        read -p "Select a group number, 'b' for back, or 'q' to quit: " choice
        
        if [[ $choice == "q" ]]; then
            exit 0
        elif [[ $choice == "b" ]]; then
            return
        elif [[ $choice =~ ^[0-9]+$ ]] && [ $choice -lt ${#groups[@]} ]; then
            selected_group=${groups[$choice]}
            reader_menu "$report" "$selected_group"
        else
            print_color $RED "Invalid selection"
        fi
    done
}

# Check dependencies before starting
check_dependencies

# Start the script
print_color $GREEN "Welcome to Report Analysis Tool"
main_menu 