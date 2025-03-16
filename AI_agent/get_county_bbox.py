import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Default county data path - adjust as needed
DEFAULT_COUNTY_DATA_PATH = Path(__file__).parent.parent / "data" / "county_boundaries.json"

def get_bounding_box(county_name=None, state_code=None, county_fips=None):
    """
    Get the bounding box coordinates for a specific county.
    
    Args:
        county_name (str): Name of the county
        state_code (str): Two-letter state code
        county_fips (str): County FIPS code
        
    Returns:
        tuple: (min_lon, min_lat, max_lon, max_lat) or None if not found
    """
    if not any([county_name, county_fips]):
        logger.warning("No county name or FIPS code provided")
        return None, None, None, None
    
    # Path to county boundaries data file
    data_path = os.environ.get("COUNTY_DATA_PATH", DEFAULT_COUNTY_DATA_PATH)
    
    # Fallback to common counties if data file doesn't exist
    common_counties = {
        "Ingham": {"MI": (-84.8, 42.4, -84.2, 42.8)},
        "Kent": {"MI": (-85.8, 42.8, -85.2, 43.2)},
        "Oakland": {"MI": (-83.7, 42.4, -83.1, 42.9)},
        "Wayne": {"MI": (-83.5, 42.1, -82.9, 42.5)},
        "Washtenaw": {"MI": (-84.0, 42.1, -83.4, 42.6)},
        "Mecosta": {"MI": (-85.6, 43.4, -85.1, 43.8)},
        "Barry": {"MI": (-85.5, 42.4, -85.0, 42.8)},
        "Isabella": {"MI": (-85.0, 43.5, -84.4, 43.9)},
        "Gratiot": {"MI": (-84.9, 43.1, -84.3, 43.5)},
        "Eaton": {"MI": (-85.0, 42.4, -84.5, 42.8)},
        "Clinton": {"MI": (-84.9, 42.8, -84.3, 43.2)}
    }
    
    try:
        # First try to load from data file if it exists
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                counties_data = json.load(f)
                
            for county in counties_data:
                if county_name and state_code and county['name'].lower() == county_name.lower() and county['state'].lower() == state_code.lower():
                    return (county['bbox'][0], county['bbox'][1], county['bbox'][2], county['bbox'][3])
                elif county_fips and county.get('fips') == county_fips:
                    return (county['bbox'][0], county['bbox'][1], county['bbox'][2], county['bbox'][3])
                
        # Fall back to hardcoded common counties
        if county_name and state_code and county_name in common_counties and state_code in common_counties[county_name]:
            return common_counties[county_name][state_code]
            
        # If we get here, the county wasn't found
        logger.warning(f"County not found: {county_name}, {state_code}")
        
        # Return a default bounding box for Michigan as fallback
        return (-86.5, 41.7, -82.4, 45.8)  # Michigan state bounding box
        
    except Exception as e:
        logger.error(f"Error getting county bounding box: {e}")
        # Return a default US bounding box
        return (-125.0, 24.0, -66.0, 49.0)


def get_state_bbox(state_code):
    """
    Get the bounding box coordinates for a state.
    
    Args:
        state_code (str): Two-letter state code
        
    Returns:
        tuple: (min_lon, min_lat, max_lon, max_lat) or None if not found
    """
    state_bboxes = {
        "MI": (-86.5, 41.7, -82.4, 45.8),  # Michigan
        "OH": (-84.8, 38.4, -80.5, 42.0),  # Ohio
        "IN": (-88.1, 37.8, -84.8, 41.8),  # Indiana
        "IL": (-91.5, 37.0, -87.5, 42.5),  # Illinois
        "WI": (-92.9, 42.5, -86.8, 47.1),  # Wisconsin
        # Add more states as needed
    }
    
    if not state_code:
        logger.warning("No state code provided")
        return None, None, None, None
        
    state_code = state_code.upper()
    if state_code in state_bboxes:
        return state_bboxes[state_code]
        
    logger.warning(f"State not found: {state_code}")
    # Return continental US bounding box as fallback
    return (-125.0, 24.0, -66.0, 49.0)

if __name__ == "__main__":
    min_lon, min_lat, max_lon, max_lat = get_bounding_box("Mecosta", "MI")
    print(f"Bounding box for Mecosta County, MI: {min_lon, min_lat, max_lon, max_lat}")
