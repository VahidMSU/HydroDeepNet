from agent import analyze_year_chunk
from AI_agent.get_county_bbox import get_bounding_box
from AI_agent.AI_agent.cdl import cdl_trends
from AI_agent.prism import PRISM_Dataset
from conversation_handler import ConversationalAgent
from coordinator import AgentCoordinator
import re


class QuerySession:
    def __init__(self):
        self.last_county = None
        self.last_state = None
        self.last_years = None  # Added to store years for follow-up queries

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
    
    # If no county is found, return invalid
    if not county:
        return {'valid': False, 'message': "Please specify a county name."}
    
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

    # Initialize result structure with None values
    result = {
        'config': config,
        'climate': None,
        'landcover': None
    }

    try:
        # Get PRISM climate data
        try:
            prism_dataset = PRISM_Dataset(config)
            climate_data = prism_dataset.get_spatial_average_over_time()
            if all(x is not None and len(x) > 0 for x in climate_data):
                result['climate'] = climate_data
                print("Successfully retrieved climate data")
            else:
                print("Warning: Could not retrieve complete climate data")
        except Exception as climate_e:
            print(f"Error retrieving climate data: {climate_e}")
            # Continue execution to at least try getting landcover data

        # Get CDL data
        try:
            landcover_data = cdl_trends(config)
            if landcover_data:
                result['landcover'] = landcover_data
                print("Successfully retrieved landcover data")
            else:
                print("Warning: Could not retrieve land cover data")
        except Exception as landcover_e:
            print(f"Error retrieving landcover data: {landcover_e}")

        # Return whatever data we were able to retrieve
        return result if (result['climate'] is not None or result['landcover'] is not None) else None
        
    except Exception as e:
        print(f"Error retrieving data: {e}")
        return None

def generate_response(query_data, county_data):
    """Generate AI response based on the data."""
    # Determine what type of data is being requested
    analysis_type = query_data.get('analysis_type', 'crop')
    
    # If asking about climate but we don't have climate data
    if analysis_type == 'climate' and not county_data.get('climate'):
        return "I'm sorry, but I don't have climate data available for this county and time period."
        
    # If asking about crops but we don't have landcover data
    if analysis_type == 'crop' and not county_data.get('landcover'):
        return "I couldn't find land cover data for the specified period."
        
    # If we have both types of data or the requested type is available, continue with analysis
    years = query_data['years']
    
    if analysis_type == 'climate' and county_data.get('climate'):
        pr_prism, tmax_prism, tmin_prism = county_data['climate']
        
        # Format climate data for response
        year_indices = [y - min(years) for y in years if y - min(years) < len(pr_prism)]
        if not year_indices:
            return "I don't have climate data for the specific years requested."
            
        climate_info = []
        for i in year_indices:
            year = min(years) + i
            climate_info.append(f"Year {year}:\n- Precipitation: {float(pr_prism[i]):.1f} mm\n"
                               f"- Average maximum temperature: {float(tmax_prism[i]):.1f}°C\n"
                               f"- Average minimum temperature: {float(tmin_prism[i]):.1f}°C")
                               
        return "\n\n".join(climate_info)
    
    # For crop analysis or combined analysis, use the existing approach
    if county_data.get('climate') and county_data.get('landcover'):
        pr_prism, tmax_prism, tmin_prism = county_data['climate']
        year_indices = [y - min(years) for y in years if y - min(years) < len(pr_prism)]
        return analyze_year_chunk(
            county_data['landcover'],
            years,
            pr_prism[year_indices],
            tmax_prism[year_indices],
            tmin_prism[year_indices]
        )
    elif county_data.get('landcover'):
        # If we only have landcover data, analyze that alone
        from AI_agent.specialized_agents import AnalysisAgent
        analyzer = AnalysisAgent()
        context = {'analysis_type': 'crop', 'focus': 'pattern', 'requested_years': years}
        return analyzer.process(county_data, context)
    
    return "I couldn't find sufficient data to answer your query."

def process_query(self, query):
    """Process user query in a more conversational manner."""
    # First, understand the query
    query_info = self.understand_query(query)
    if not query_info:
        return "I'm sorry, I couldn't understand your query. Could you please specify the county and year you're interested in?"
    
    # Fix any year extraction issues
    if 'years' in query_info:
        fixed_years = []
        for year in query_info['years']:
            # Fix common year extraction errors like '20' instead of '2010' 
            if year < 100:  # This is likely an error
                if year > 20:  # Probably 19xx
                    fixed_years.append(1900 + year)
                else:  # Probably 20xx
                    fixed_years.append(2000 + year)
            else:
                fixed_years.append(year)
        query_info['years'] = fixed_years
        print(f"Fixed years: {fixed_years}")
    
    # Update conversation context
    self.context.update_context(query_info)
    
    # Get the data
    data = self.retrieve_data(query_info)
    if not data:
        return f"I'm sorry, I couldn't find data for {query_info.get('county')} County in {', '.join(map(str, query_info.get('years', [])))}."
    
    # Generate the response
    response = self.generate_response(query, query_info, data)
    
    # Add proactive insights
    response = self.enhance_response_with_insights(response, query_info, data)
    
    return response

def interactive_agent(query):
    """
    Entry point for the chatbot API to process user queries.
    This function handles queries and returns responses as strings.
    
    Args:
        query (str): The user's query text
        
    Returns:
        str: The agent's response to the query
    """

    try:
        # Check for crop-specific queries that need to be handled precisely
        query_lower = query.lower()
        if 'major crop' in query_lower or 'main crop' in query_lower:
            print("Detected crop pattern question, ensuring proper handling")
        
        # Use the AgentCoordinator for enhanced processing
        coordinator = AgentCoordinator()
        response = coordinator.process_query(query)
        return response
        
    except Exception as e:
        # Log the error but return a user-friendly message
        print(f"Error in interactive_agent: {e}")
        import traceback
        print(traceback.format_exc())
        return "I'm sorry, I encountered an error while processing your request. Please try a different query or contact support if the issue persists."

def interactive_session():
    """Run an interactive session with the conversational agent."""
    print("Welcome to the Agricultural Data Analysis Assistant!")
    print("\nYou can ask me about:")
    print("- Crop patterns in different counties")
    print("- Climate data for specific regions")
    print("- Agricultural trends over time")
    print("\nExamples:")
    print("- What's the major crop in Ingham County?")
    print("- How was the climate in Mecosta County in 2015?")
    print("- Tell me about soybean cultivation in Barry County from 2010-2015")
    print("\nType 'quit' to exit.")
    
    # Use coordinator instead of ConversationalAgent
    coordinator = AgentCoordinator()
    
    while True:
        query = input("\nWhat would you like to know? > ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Thank you for using the Agricultural Data Analysis Assistant. Goodbye!")
            break
            
        if not query:
            continue
            
        print("\nAnalyzing your query...")
        try:
            response = coordinator.process_query(query)
            print("\nAnalysis:")
            print(response)
        except Exception as e:
            print(f"\nError processing query: {e}")
            import traceback
            traceback.print_exc()
            print("\nPlease try a different query.")
        
if __name__ == "__main__":
    interactive_session()
