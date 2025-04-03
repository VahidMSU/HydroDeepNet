import os

cli_path = "/data/SWATGenXApp/Users/admin/SWATplus_by_VPUID/0408/huc12/04141000/PRISM/hmd.cli"
dir_path = os.path.dirname(cli_path)

def check_file(file_path):
    print(f"Checking file: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    print(f"Total lines: {len(lines)}")
    
    # Simulating the exact logic from import_weather.py
    non_empty_lines = [line for line in lines if line.strip()]
    print(f"Non-empty lines: {len(non_empty_lines)}")
    
    if len(non_empty_lines) > 0:
        try:
            last_line = non_empty_lines[len(non_empty_lines)-1].strip().split()
            print(f"Last line split into: {last_line}")
            
            # This is the exact operation that causes the error
            if len(last_line) >= 2:
                year = int(last_line[0])
                day = int(last_line[1])
                print(f"Year: {year}, Day: {day}")
            else:
                print("WARNING: Last line has fewer than 2 elements!")
        except Exception as e:
            print(f"ERROR: {str(e)}")

# Check each station file referenced in the cli file
with open(cli_path, 'r') as f:
    lines = f.readlines()
    
# Skip the first two header lines
station_files = [line.strip() for line in lines[2:] if line.strip()]
print(f"Found {len(station_files)} station files in {cli_path}")

for station_file in station_files:
    station_path = os.path.join(dir_path, station_file)
    print("\n" + "="*50)
    check_file(station_path)