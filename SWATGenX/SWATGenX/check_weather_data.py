import os
import datetime

def check_weather_station_files(directory, start_year=2000, end_year=2020):
    errors = []

    # Define valid ranges for each variable type
    VALID_RANGES = {
        'pcp': {'min': 0, 'max': 2000},  # precipitation in mm, max ~2000mm per day is extreme but possible
        'tmp': {'tmin': -50, 'tmax': 60},  # temperature in Celsius
        'hmd': {'min': 0, 'max': 100},  # relative humidity in percentage
        'slr': {'min': 0, 'max': 50},   # solar radiation in MJ/m2, typical max ~40 MJ/m2/day
        'wnd': {'min': 0, 'max': 100}    # wind speed in m/s, extreme winds ~100 m/s
    }

    # Compute expected number of daily entries and generate expected dates
    expected_dates = set()
    current_date = datetime.datetime(start_year, 1, 1)
    end_date = datetime.datetime(end_year, 12, 31)
    while current_date <= end_date:
        year = current_date.year
        day_of_year = current_date.timetuple().tm_yday
        expected_dates.add((year, day_of_year))
        current_date += datetime.timedelta(days=1)

    expected_days = len(expected_dates)

    for fname in os.listdir(directory):
        if not fname.endswith((".hmd", ".pcp", ".tmp", ".slr", ".wnd")):
            continue

        path = os.path.join(directory, fname)
        file_type = fname.split('.')[-1]  # Get file extension for variable type
        
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip()]
        
        # Basic file checks
        if len(lines) == 0:
            errors.append((fname, "File is empty"))
            continue

        if len(lines) < 4:
            errors.append((fname, f"File has only {len(lines)} lines"))
            continue

        # Check station info line
        station_info = lines[2].split()
        if len(station_info) < 5:
            errors.append((fname, f"Line 3 has < 5 values: {station_info}"))
            continue
        try:
            float(station_info[2])  # lat
            float(station_info[3])  # lon
        except ValueError:
            errors.append((fname, "Latitude/Longitude not convertible to float"))

        # Extract and validate all dates and values
        found_dates = set()
        prev_date = None
        
        for i, line in enumerate(lines[3:], start=4):
            parts = line.split()
            
            # Check minimum number of values
            min_values = 4 if file_type == 'tmp' else 3  # tmp files have tmax and tmin
            if len(parts) < min_values:
                errors.append((fname, f"Line {i} has < {min_values} values: {line}"))
                continue
                
            try:
                year = int(parts[0])
                day = int(parts[1])
                
                # Value range validation based on file type
                if file_type == 'tmp':
                    # Temperature files have both max and min
                    try:
                        tmax = float(parts[2])
                        tmin = float(parts[3])
                        if tmax < VALID_RANGES['tmp']['tmin'] or tmax > VALID_RANGES['tmp']['tmax']:
                            errors.append((fname, f"Line {i} has invalid tmax value: {tmax}°C"))
                        if tmin < VALID_RANGES['tmp']['tmin'] or tmin > VALID_RANGES['tmp']['tmax']:
                            errors.append((fname, f"Line {i} has invalid tmin value: {tmin}°C"))
                        if tmin > tmax:
                            errors.append((fname, f"Line {i} has tmin ({tmin}°C) greater than tmax ({tmax}°C)"))
                    except ValueError:
                        errors.append((fname, f"Line {i} has invalid temperature values: {parts[2]}, {parts[3]}"))
                else:
                    # Other files have single values
                    try:
                        value = float(parts[2])
                        range_check = VALID_RANGES[file_type]
                        if value < range_check['min'] or value > range_check['max']:
                            errors.append((fname, f"Line {i} has value {value} outside valid range "
                                               f"[{range_check['min']}, {range_check['max']}]"))
                    except ValueError:
                        errors.append((fname, f"Line {i} has invalid value: {parts[2]}"))
                
                # Date validation
                if not (start_year <= year <= end_year):
                    errors.append((fname, f"Line {i} year {year} outside valid range {start_year}-{end_year}"))
                    continue
                    
                try:
                    date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day-1)
                except ValueError:
                    errors.append((fname, f"Line {i} has invalid date: year={year}, day={day}"))
                    continue
                
                # Check sequence
                if prev_date is not None:
                    expected_next = prev_date + datetime.timedelta(days=1)
                    if date != expected_next:
                        errors.append((fname, f"Line {i} date {date.date()} breaks sequence, expected {expected_next.date()}"))
                
                date_key = (year, day)
                if date_key in found_dates:
                    errors.append((fname, f"Line {i} duplicate date: year={year}, day={day}"))
                found_dates.add(date_key)
                prev_date = date
                
            except ValueError as e:
                errors.append((fname, f"Line {i} has invalid date format: {str(e)}"))

        # Check for missing dates
        missing_dates = expected_dates - found_dates
        if missing_dates:
            sorted_missing = sorted(missing_dates)
            if len(sorted_missing) > 5:
                sample = sorted_missing[:5]
                errors.append((fname, f"Missing {len(missing_dates)} dates, first 5: " + 
                             ", ".join(f"{y}-{d}" for y, d in sample)))
            else:
                errors.append((fname, "Missing dates: " + 
                             ", ".join(f"{y}-{d}" for y, d in sorted_missing)))

        # Check total number of data lines
        data_lines = lines[3:]
        if len(data_lines) != expected_days:
            errors.append((fname, f"Expected {expected_days} daily entries, found {len(data_lines)}"))

    # Report
    if not errors:
        print("✅ weather data check passed.")
    else:
        print("❌ Detected issues in weather data:")
        for fname, msg in errors:
            print(f"  - {fname}: {msg}")

    
    ### raise value error if check fails and return
    if errors:
        raise ValueError("Weather data check failed. See above for details.")
    else:
        print("All files passed checks.")
        # Example usage
        return True
    

# Example usage:
if __name__ == "__main__":

    check_weather_station_files("/data/SWATGenXApp/Users/admin/SWATplus_by_VPUID/0406/huc12/04127200/PRISM")
