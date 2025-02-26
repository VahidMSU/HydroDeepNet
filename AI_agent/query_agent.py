from agent import chat_with_deepseek
import json

def extract_query_info(query):
    """Use LLM to understand the query and extract structured information."""
    prompt = f"""
    As a geographic query understanding expert, extract information from this query:
    Query: "{query}"
    
    Common patterns to detect:
    - If asking about "crop pattern(s)" or "land use" -> set analysis_type="crop" and focus="pattern"
    - If asking about specific crops -> set analysis_type="crop" and focus=<crop name>
    - If about climate/weather -> set analysis_type="climate"
    - If about both -> set analysis_type="both"
    
    Return ONLY a JSON object with these fields (and nothing else):
    {{
        "county": "string (County name without 'County' suffix)",
        "state": "string (default to 'Michigan' if not mentioned)",
        "years": [integers],
        "analysis_type": "crop" or "climate" or "both",
        "focus": "pattern" or specific crop name or climate aspect
    }}
    """
    
    try:
        response = chat_with_deepseek(prompt)
        # Clean up the response to ensure it's valid JSON
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:-3]  # Remove ```json and ``` markers
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
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
        query_info['years'] = [2008, 2009, 2010]  # default years
    
    return query_info
