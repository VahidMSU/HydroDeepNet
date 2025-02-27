def log_query_info(query_info):
    """Log structured information about a query."""
    print("\n--- Query Information ---")
    print(f"County: {query_info.get('county')}")
    print(f"State: {query_info.get('state')}")
    print(f"Years: {query_info.get('years')}")
    print(f"Analysis Type: {query_info.get('analysis_type')}")
    print(f"Focus: {query_info.get('focus')}")
    print("------------------------\n")

def log_data_structure(data):
    """Log structure of retrieved data."""
    if not data:
        print("No data retrieved")
        return
        
    print("\n--- Retrieved Data Structure ---")
    if 'config' in data:
        print(f"Config: {type(data['config']).__name__} with {len(data['config'])} keys")
        
    if 'climate' in data and data['climate']:
        try:
            pr_data, tmax_data, tmin_data = data['climate']
            print(f"Climate Data: precipitation[{len(pr_data)}], tmax[{len(tmax_data)}], tmin[{len(tmin_data)}]")
        except:
            print(f"Climate Data: {type(data['climate']).__name__}")
            
    if 'landcover' in data and data['landcover']:
        print(f"Land Cover Data: {type(data['landcover']).__name__} with {len(data['landcover'])} years")
        year_sample = next(iter(data['landcover'].keys()))
        print(f"  Example Year ({year_sample}): {len(data['landcover'][year_sample])} land cover categories")
    print("------------------------------\n")

def log_landcover_data(landcover_data, years_requested):
    """Log detailed information about landcover data"""
    print("\n--- Landcover Data Debug ---")
    
    if not landcover_data:
        print("Landcover data is None or empty")
        print("--------------------------\n")
        return
        
    print(f"Keys in landcover_data: {list(landcover_data.keys())}")
    
    # Check if we have data for the requested years
    years_found = []
    for year in years_requested:
        year_str = str(year)
        if year in landcover_data:
            years_found.append(year)
            print(f"Year {year} found as integer key")
        elif year_str in landcover_data:
            years_found.append(year) 
            print(f"Year {year} found as string key")
            
    years_missing = [year for year in years_requested if year not in years_found]
    if years_missing:
        print(f"Years not found: {years_missing}")
        
    # Show sample data for one year to check structure
    if landcover_data:
        sample_year = list(landcover_data.keys())[0]
        print(f"\nSample data structure for year {sample_year}:")
        sample_data = landcover_data[sample_year]
        print(f"Keys in year data: {list(sample_data.keys())}")
        
        # Check for key categories
        agricultural_keys = [key for key in sample_data.keys() 
                            if key not in ['Total Area', 'unit', 'county', 'state', 
                                          'Open Water', 'Developed/Open Space', 
                                          'Developed/Low Intensity', 'Developed/Med Intensity',
                                          'Developed/High Intensity', 'Barren', 'Deciduous Forest',
                                          'Evergreen Forest', 'Mixed Forest', 'Shrubland',
                                          'Woody Wetlands', 'Herbaceous Wetlands']]
        
        print("\nTop 5 agricultural categories:")
        ag_items = [(key, sample_data[key]) for key in agricultural_keys]
        ag_items.sort(key=lambda x: x[1], reverse=True)
        
        for key, value in ag_items[:5]:
            print(f"- {key}: {value}")
            
    print("--------------------------\n")

def debug_query_info(query, query_info):
    """Debug parsing of query information"""
    print("\n--- Query Parsing Debug ---")
    print(f"Original query: '{query}'")
    
    if not query_info:
        print("Failed to parse query")
        print("------------------------\n")
        return
        
    print(f"Parsed county: {query_info.get('county')}")
    print(f"Parsed state: {query_info.get('state')}")
    print(f"Parsed years: {query_info.get('years')}")
    print(f"Analysis type: {query_info.get('analysis_type')}")
    print(f"Focus: {query_info.get('focus')}")
    
    # Look for potential issues
    if 'years' in query_info and query_info['years']:
        for year in query_info['years']:
            if year < 1900 or year > 2100:
                print(f"WARNING: Unusual year value: {year}")
                
    print("------------------------\n")

def timing_log(name, start_time, end_time):
    """Log timing information for operations."""
    duration = end_time - start_time
    print(f"Operation '{name}' completed in {duration:.2f} seconds")
