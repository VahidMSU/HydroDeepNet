import os
import datetime

def check_weather_station_files(directory, start_year=2000, end_year=2020):
    errors = []

    # Compute expected number of daily entries
    expected_days = sum(
        366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365
        for y in range(start_year, end_year + 1)
    )

    for fname in os.listdir(directory):
        if not fname.endswith((".hmd", ".pcp", ".tmp", ".slr", ".wnd")):
            continue

        path = os.path.join(directory, fname)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip()]
        
        # Check 1: file not empty
        if len(lines) == 0:
            errors.append((fname, "File is empty"))
            continue

        # Check 2: file must have at least 4 lines
        if len(lines) < 4:
            errors.append((fname, f"File has only {len(lines)} lines"))
            continue

        # Check 3: line 3 must contain lat/lon/elev
        station_info = lines[2].split()
        if len(station_info) < 5:
            errors.append((fname, f"Line 3 has < 5 values: {station_info}"))
            continue
        try:
            float(station_info[2])  # lat
            float(station_info[3])  # lon
        except ValueError:
            errors.append((fname, "Latitude/Longitude not convertible to float"))

        # Check 4: line 4 must contain year and julian day
        begin_data = lines[3].split()
        if len(begin_data) < 2:
            errors.append((fname, f"Line 4 has < 2 values: {begin_data}"))
            continue
        try:
            datetime.datetime(int(begin_data[0]), 1, 1) + datetime.timedelta(days=int(begin_data[1]) - 1)
        except Exception as e:
            errors.append((fname, f"Invalid start date on line 4: {str(e)}"))

        # Check 5: last line should contain valid date
        last_parts = lines[-1].split()
        if len(last_parts) < 2:
            errors.append((fname, "Last line has < 2 values"))
            continue
        try:
            datetime.datetime(int(last_parts[0]), 1, 1) + datetime.timedelta(days=int(last_parts[1]) - 1)
        except Exception as e:
            errors.append((fname, f"Invalid end date on last line: {str(e)}"))

        # Check 6: number of data lines matches expected
        data_lines = lines[3:]
        if len(data_lines) != expected_days:
            errors.append((fname, f"Expected {expected_days} daily entries, found {len(data_lines)}"))

    # Report
    if not errors:
        print("✅ All files passed checks.")
    else:
        print("❌ Detected issues:")
        for fname, msg in errors:
            print(f"  - {fname}: {msg}")

# Example usage:
check_weather_station_files("/data/SWATGenXApp/Users/admin/SWATplus_by_VPUID/0406/huc12/04127200/PRISM")
