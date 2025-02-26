from agent import chat_with_deepseek
from model_selector import ModelSelector
import json
import re

def extract_query_info(query, model=None):
    """Use LLM to understand the query and extract structured information."""
    # First check for common query patterns to avoid using LLM when not needed
    county = extract_county(query)
    years = extract_years(query)
    
    if county and "major crop" in query.lower():
        print("Direct pattern match: Major crop query")
        return {
            "county": county,
            "state": "Michigan",
            "years": years or [2010],
            "analysis_type": "crop",
            "focus": "pattern"
        }
        
    # Handle session context if provided
    session_context = None
    if isinstance(model, dict) and 'session' in model:
        session_context = model.get('session')
        model = None
    
    # Check for follow-up questions with year references
    if session_context and session_context.get('last_county'):
        # Pattern to match questions like "what about year 2014" or "and 2016?"
        followup_patterns = [
            r'(?:what|how) about (?:year[s]? )?(\d{4}(?:\s*(?:to|-)\s*\d{4})?)',
            r'(?:and|for) (?:year[s]? )?(\d{4}(?:\s*(?:to|-)\s*\d{4})?)',
            r'(?:in|during) (?:year[s]? )?(\d{4}(?:\s*(?:to|-)\s*\d{4})?)'
        ]
        
        for pattern in followup_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                years = parse_years(match.group(1))
                return {
                    "county": session_context.get('last_county'),
                    "state": session_context.get('last_state', 'Michigan'),
                    "years": years,
                    "analysis_type": session_context.get('last_analysis_type', 'crop'),
                    "focus": session_context.get('last_focus', 'pattern')
                }
    
    # First check for specific keywords
    crop_keywords = ['crop', 'crops', 'agriculture', 'farming', 'farm', 'major', 'grow']
    is_crop_query = any(keyword in query.lower() for keyword in crop_keywords)

    # Extract years using more flexible pattern
    if not years:  # Only extract years if not already done
        year_pattern = r'(?:19|20)\d{2}'
        years = [int(year) for year in re.findall(year_pattern, query)]
        if len(years) == 2:
            years = list(range(min(years), max(years) + 1))
        elif not years:
            years = [2010]
    
    # Use county if already extracted, otherwise try to extract it
    if not county:
        # Extract county name with improved pattern matching
        county_pattern = r'([A-Za-z]+)\s+(?:County|county)'
        county_match = re.search(county_pattern, query)
        if not county_match:
            # Try alternative pattern without 'County' keyword
            alt_pattern = r'(?:in|for|of)\s+([A-Za-z]+)(?:\s+(?:County|county|in|,|\s))'
            county_match = re.search(alt_pattern, query)
        
        if not county_match and not session_context:
            return None
            
        county = county_match.group(1).title() if county_match else session_context.get('last_county')
    
    # Check for major crop query pattern specifically
    major_crop_pattern = r'(?:what|which)\s+(?:is|are)\s+(?:the\s+)?(?:main|major|primary|dominant|most\s+common)\s+crop'
    if re.search(major_crop_pattern, query.lower()):
        return {
            "county": county,
            "state": "Michigan",
            "years": years,
            "analysis_type": "crop",
            "focus": "pattern",
            "query": query  # Store original query for context
        }
    
    # Check for more specific crop keywords to set analysis_type more precisely
    major_crop_pattern = r'(?:main|major|primary|dominant)\s+crop'
    crop_percentage_pattern = r'(?:percentage|proportion|share|how much)'
    specific_crop_pattern = r'(?:corn|soy|wheat|barley|oats)'
    
    # Determine analysis type more precisely
    if re.search(major_crop_pattern, query.lower()):
        return {
            "county": county,
            "state": "Michigan",
            "years": years,
            "analysis_type": "crop",
            "focus": "pattern"
        }
    elif re.search(crop_percentage_pattern, query.lower()):
        return {
            "county": county,
            "state": "Michigan",
            "years": years,
            "analysis_type": "crop_percentage",
            "focus": "pattern"
        }
    elif re.search(specific_crop_pattern, query.lower()):
        # Extract the specific crop
        crop_match = re.search(specific_crop_pattern, query.lower())
        crop_focus = crop_match.group(0).title() if crop_match else "pattern"
        return {
            "county": county,
            "state": "Michigan", 
            "years": years,
            "analysis_type": "crop",
            "focus": crop_focus
        }
    elif is_crop_query:
        return {
            "county": county,
            "state": "Michigan",
            "years": years,
            "analysis_type": "crop",
            "focus": "pattern"
        }
    
    # Check for climate-specific keywords
    climate_keywords = ['climate', 'weather', 'temperature', 'precipitation', 'rainfall']
    is_climate_query = any(keyword in query.lower() for keyword in climate_keywords)
    
    # Handle climate queries directly
    if is_climate_query:
        return {
            "county": county,
            "state": "Michigan",
            "years": years,
            "analysis_type": "climate",
            "focus": "pattern"
        }
    
    # Otherwise, proceed with LLM analysis
    prompt = f"""
    Extract structured information from this query: "{query}"
    For climate queries, set analysis_type="climate".
    For year-specific queries, include only the requested year(s).
    
    Return JSON with:
    {{
        "county": "<County name>",
        "state": "<State name>",
        "years": [<year numbers>],
        "analysis_type": "<crop|climate|both>",
        "focus": "<pattern|specific crop name|climate aspect>"
    }}
    If you cannot determine the county, return {{"error": "county_not_found"}}
    """
    
    # Use simple model for query extraction
    if not model:
        model = ModelSelector.get_model_for_task("query_analysis")
        
    try:
        response = chat_with_deepseek(prompt, model=model).strip()
        # Remove any markdown code block markers
        response = response.replace('```json', '').replace('```', '').strip()
        
        # Handle climate-specific queries without county info
        if 'climate' in query.lower() and not response.startswith('{'):
            # No default county - require explicitly specified county
            return None
            
        result = json.loads(response)
        if result.get("error") == "county_not_found":
            return None
            
        return result
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        # Don't provide a default county anymore
        return None

def validate_query_info(query_info):
    """Validate and clean up the extracted information."""
    if not query_info or not isinstance(query_info, dict):
        return None
    
    required_fields = ['county', 'state', 'years', 'analysis_type']
    if not all(field in query_info for field in required_fields):
        return None
    
    # Ensure years is a list of integers
    if isinstance(query_info['years'], list):
        query_info['years'] = [int(y) for y in query_info['years'] if str(y).isdigit()]
    else:
        query_info['years'] = [2010]  # default year
    
    return query_info

def parse_years(year_str):
    """Parse year string into list of years."""
    if 'to' in year_str or '-' in year_str:
        start_year, end_year = map(int, re.split(r'\s*(?:to|[-])\s*', year_str))
        return list(range(start_year, end_year + 1))
    return [int(year_str)]

def extract_county(query):
    """Extract county name using regex patterns."""
    patterns = [
        r'([A-Za-z]+)\s+(?:County|county)',
        r'in\s+([A-Za-z]+)(?:\s+County|county|\s)',
        r'for\s+([A-Za-z]+)(?:\s+County|county|\s)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            return match.group(1).title()
    
    return None

def extract_years(query):
    """Extract years from query text."""
    year_pattern = r'(?:19|20)\d{2}'
    years = [int(year) for year in re.findall(year_pattern, query)]
    if len(years) == 2:
        years = list(range(min(years), max(years) + 1))
    return years if years else None


if __name__ == "__main__":
    query = "Analyze climate or crop pattern for Mecosta County in 2010"
    query_info = extract_query_info(query)
    query_info = validate_query_info(query_info)
    print(query_info)
    # Expected output: {'county': 'Mecosta', 'state': 'Michigan', 'years': [2010], 'analysis_type': 'both', 'focus': 'pattern'}
