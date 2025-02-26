from agent import chat_with_deepseek, analyze_year_chunk
from get_county_bbox import get_bounding_box
from utils import cdl_trends
from prism import PRISM_Dataset
import re
import numpy as np
from coordinator import AgentCoordinator

class QuerySession:
    def __init__(self):
        self.last_county = None
        self.last_state = None

def analyze_specific_request(query, landcover_data):
    """Analyze specific data requests about the last analyzed region."""
    # Common agricultural categories
    ag_categories = [
        'Corn', 'Soybeans', 'Wheat', 'Alfalfa', 'Other Hay/Non Alfalfa',
        'Grassland/Pasture', 'Oats', 'Barley', 'Sorghum', 'Sugarbeets',
        'Dry Beans', 'Potatoes', 'Peas', 'Sweet Corn'
    ]
    
    if 'total' in query.lower() and 'agricultural' in query.lower():
        total_ag = sum(landcover_data.get(crop, 0) for crop in ag_categories)
        return f"\nTotal agricultural land: {total_ag:.2f} hectares"
    
    return None

def parse_query(query, session):
    """Enhanced query parsing with context memory and specific data requests."""
    # Check for specific data analysis requests first
    if session.last_county and any(word in query.lower() for word in ['total', 'percentage', 'how much', 'what is']):
        return {
            'county': session.last_county,
            'state': session.last_state,
            'years': session.last_years,
            'valid': True,
            'analysis_request': query
        }
    
    # First, strip out any command words that might be misinterpreted as county names
    query = re.sub(r'^(?:analyze|tell\sme\sabout|what\sare|show)\s+', '', query.strip(), flags=re.IGNORECASE)
    
    # Check for follow-up patterns
    followup_patterns = [
        r'(?i)(?:what|how) about (?:year[s]? )?(\d{4}(?:\s*(?:to|-)\s*\d{4})?)',
        r'(?i)(?:and|for) (?:year[s]? )?(\d{4}(?:\s*(?:to|-)\s*\d{4})?)',
        r'(?i)same (?:county|area|region) (?:in|for) (?:year[s]? )?(\d{4}(?:\s*(?:to|-)\s*\d{4})?)'
    ]
    
    # Check if this is a follow-up question
    for pattern in followup_patterns:
        match = re.search(pattern, query)
        if match and session.last_county and session.last_state:
            years = parse_years(match.group(1))
            return {
                'county': session.last_county,
                'state': session.last_state,
                'years': years,
                'valid': True
            }
    
    # If not a follow-up, proceed with regular parsing
    # Multiple patterns to match different query formats
    patterns = [
        # Pattern 1: "<county> county, <state>"
        r'(?i)([A-Za-z]+)\s+(?:County|county|COUNTY)?,?\s+([A-Za-z]+)',
        # Pattern 2: "<county>, <state>"
        r'(?i)([A-Za-z]+),\s*([A-Za-z]+)',
        # Pattern 3: "<county> <state>"
        r'(?i)([A-Za-z]+)\s+([A-Za-z]+)',
    ]
    
    # Try each pattern
    county = state = None
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            county, state = match.groups()
            break
    
    # Look for years with better pattern matching
    year_pattern = r'(?:19|20)\d{2}(?:\s*(?:to|[-])\s*(?:19|20)\d{2})?'
    years_match = re.search(year_pattern, query)
    
    if years_match:
        year_str = years_match.group(0)
        if 'to' in year_str or '-' in year_str:
            start_year, end_year = map(int, re.split(r'\s*(?:to|[-])\s*', year_str))
            years = list(range(start_year, end_year + 1))
        else:
            years = [int(year_str)]
    else:
        years = [2008, 2009, 2010]  # default to 3 years
    
    # Properly capitalize county and state names
    if county:
        county = county.title()
    if state:
        state = state.title()
        # Fix common state misspellings
        state_corrections = {
            'Michigam': 'Michigan',
            # Add more corrections as needed
        }
        state = state_corrections.get(state, state)
    
    if county and state:
        # Update session memory
        session.last_county = county
        session.last_state = state
        return {
            'county': county,
            'state': state,
            'years': years,
            'valid': True
        }
    
    return {'valid': False}

def parse_years(year_str):
    """Parse year string into list of years."""
    if 'to' in year_str or '-' in year_str:
        start_year, end_year = map(int, re.split(r'\s*(?:to|[-])\s*', year_str))
        return list(range(start_year, end_year + 1))
    return [int(year_str)]

def get_data_for_county(county, state, years):
    """Retrieve all necessary data for a county."""
    bbox = get_bounding_box(county, state)
    if not all(x is not None for x in bbox):
        print(f"Could not find bounding box for {county} County, {state}")
        return None

    min_lon, min_lat, max_lon, max_lat = bbox
    print(f"Found bounding box: {min_lon:.4f}, {min_lat:.4f}, {max_lon:.4f}, {max_lat:.4f}")

    config = {
        "RESOLUTION": 250,
        "huc8": None,
        "video": False,
        "aggregation": "annual",
        "start_year": min(years),
        "end_year": max(years),
        'bounding_box': [min_lon, min_lat, max_lon, max_lat],
    }

    try:
        # Get PRISM climate data
        prism_dataset = PRISM_Dataset(config)
        climate_data = prism_dataset.get_spatial_average_over_time()
        if not all(x is not None and len(x) > 0 for x in climate_data):
            print("Warning: Could not retrieve complete climate data")
            return None

        # Get CDL data
        landcover_data = cdl_trends(config)
        if not landcover_data:
            print("Warning: Could not retrieve land cover data")
            return None

        return {
            'config': config,
            'climate': climate_data,
            'landcover': landcover_data
        }
    except Exception as e:
        print(f"Error retrieving data: {e}")
        return None

def generate_response(query_data, county_data):
    """Generate AI response based on the data."""
    if not county_data['landcover']:
        return "I couldn't find land cover data for the specified period."

    years = query_data['years']
    pr_prism, tmax_prism, tmin_prism = county_data['climate']
    
    year_indices = [y - min(years) for y in years]
    analysis = analyze_year_chunk(
        county_data['landcover'],
        years,
        pr_prism[year_indices],
        tmax_prism[year_indices],
        tmin_prism[year_indices]
    )
    
    return analysis

def interactive_session():
    print("Welcome to the Multi-Agent County Analysis System!")
    print("\nYou can ask questions like:")
    print("- 'Analyze Mecosta County, Michigan'")
    print("- 'How has the climate affected farming in Mecosta County?'")
    print("- 'What were the main crops in 2010?'")
    print("\nType 'quit' to exit.")

    coordinator = AgentCoordinator()

    while True:
        query = input("\nWhat would you like to know? > ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
            
        if not query:
            continue

        response = coordinator.process_query(query)
        print("\nAnalysis:")
        print(response)

if __name__ == "__main__":
    interactive_session()
