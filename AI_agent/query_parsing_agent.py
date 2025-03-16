import re
import json
from agent import chat_with_deepseek
from model_selector import ModelSelector

class QueryParsingAgent:
    """
    A dedicated agent that uses AI to parse natural language queries into structured information.
    Replaces mechanical regex-based parsing with AI-based understanding.
    """
    
    def __init__(self):
        self.model = ModelSelector.get_model_for_task("query_analysis")
        self.default_years = [2010]
        self.common_crop_types = {
            'corn': 'Corn', 
            'soybeans': 'Soybeans',
            'soy': 'Soybeans', 
            'wheat': 'Wheat', 
            'alfalfa': 'Alfalfa',
            'hay': 'Other Hay/Non Alfalfa',
            'barley': 'Barley', 
            'oats': 'Oats'
        }
    
    def parse_query(self, query_text, session_context=None):
        """
        Parse a natural language query into structured information using an AI model.
        
        Args:
            query_text (str): The user's query text
            session_context (dict, optional): Previous conversation context
            
        Returns:
            dict: Structured information extracted from the query
        """
        # Check for direct crop pattern questions first to avoid misclassification
        if self._is_crop_pattern_question(query_text):
            county = self._extract_county(query_text)
            if county:
                years = self._extract_years(query_text) or self.default_years
                return {
                    'county': county,
                    'state': 'Michigan',
                    'years': years,
                    'analysis_type': 'crop',
                    'focus': 'pattern',
                    'query_type': 'crop'  # Explicitly mark as crop query
                }
        
        # Check if this is a follow-up question
        if session_context and self._is_followup_question(query_text):
            return self._handle_followup(query_text, session_context)
        
        # Formulate the prompt for the AI
        prompt = self._create_parsing_prompt(query_text, session_context)
        
        # Get response from the AI
        try:
            response = chat_with_deepseek(prompt, model=self.model)
            
            # Extract the JSON object from the response
            query_info = self._extract_json_from_response(response)
            
            # Validate and clean the extracted information
            if query_info:
                return self._validate_and_complete_info(query_info, session_context)
            else:
                return self._fallback_parsing(query_text, session_context)
                
        except Exception as e:
            print(f"Error in AI query parsing: {e}")
            return self._fallback_parsing(query_text, session_context)
    
    def _is_crop_pattern_question(self, query_text):
        """Check if query is specifically about crop patterns"""
        query_lower = query_text.lower()
        return (('major crop' in query_lower or 'main crop' in query_lower) and 
                ('county' in query_lower or 'counties' in query_lower))
    
    def _extract_county(self, query):
        """Extract county name from query text"""
        county_match = re.search(r'([A-Za-z]+)\s+County', query, re.IGNORECASE)
        if county_match:
            return county_match.group(1).title()
        return None
    
    def _extract_years(self, query):
        """Extract years from query text"""
        years = [int(year) for year in re.findall(r'(19|20)\d{2}', query)]
        return years if years else None
    
    def _create_parsing_prompt(self, query_text, session_context):
        """Create a prompt for the AI to parse the query."""
        base_prompt = f"""
        You are a specialized agricultural query parser. Extract structured information from this user query:
        
        "{query_text}"
        
        Return ONLY a valid JSON object with these fields:
        {{
            "county": "County name (or null if not mentioned)",
            "state": "State name (default to Michigan if not specified)",
            "years": [list of years mentioned, or [2010] if none specified],
            "analysis_type": "crop" or "climate" or "both" based on what's being asked,
            "focus": "pattern" for general crop patterns, or specific crop name if asking about one crop
        }}
        
        Follow these rules:
        1. If "major crop" is mentioned, set focus to "pattern" 
        2. If specific crops like corn, soybeans, wheat are mentioned, set focus to that crop name
        3. For climate questions, set analysis_type to "climate"
        4. For crop questions, set analysis_type to "crop"
        5. If both climate and crops are mentioned, set analysis_type to "both"
        6. Extract all mentioned years, defaulting to [2010] if none mentioned
        7. Return only the JSON, no other text
        """
        
        # Add context information if available
        if session_context:
            context_info = f"""
            Previous conversation context:
            - Last county discussed: {session_context.get('last_county', 'None')}
            - Last state discussed: {session_context.get('last_state', 'Michigan')}
            - Last years discussed: {session_context.get('last_years', [])}
            - Last analysis type: {session_context.get('last_analysis_type', 'None')}
            
            Use this context to resolve references like "there", "that county", "last year", etc.
            """
            base_prompt += context_info
            
        return base_prompt
    
    def _extract_json_from_response(self, response):
        """Extract and parse the JSON object from the AI's response."""
        try:
            # First try to parse the whole response as JSON
            return json.loads(response)
        except json.JSONDecodeError:
            # If that fails, try to extract a JSON object using regex
            json_match = re.search(r'\{[^{]*"county".*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                # Fix common JSON formatting issues
                json_str = json_str.replace("'", '"')
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    print("Found JSON-like structure but failed to parse it")
            else:
                print("No JSON object found in response")
        return None
    
    def _validate_and_complete_info(self, query_info, session_context):
        """Validate the extracted information and fill in missing values."""
        # Ensure required fields exist
        if not query_info:
            return None
            
        # If county is missing but we have context, use that
        if not query_info.get('county') and session_context and session_context.get('last_county'):
            query_info['county'] = session_context['last_county']
            
        # Default state to Michigan if not specified
        if not query_info.get('state'):
            query_info['state'] = 'Michigan'
            
        # Ensure years is a list and contains valid years
        if not query_info.get('years'):
            query_info['years'] = self.default_years
        else:
            # Handle potential string values
            if isinstance(query_info['years'], str):
                try:
                    query_info['years'] = [int(query_info['years'])]
                except ValueError:
                    query_info['years'] = self.default_years
            
            # Fix any 2-digit years (convert 20 to 2020, etc.)
            fixed_years = []
            for year in query_info['years']:
                year = int(year)
                if year < 100:
                    if year > 50:
                        fixed_years.append(1900 + year)
                    else:
                        fixed_years.append(2000 + year)
                else:
                    fixed_years.append(year)
            query_info['years'] = fixed_years
            
        # Standardize crop names
        if query_info.get('focus') and query_info['focus'] != 'pattern':
            focus_lower = query_info['focus'].lower()
            for crop_key, standard_name in self.common_crop_types.items():
                if crop_key in focus_lower:
                    query_info['focus'] = standard_name
                    break
        
        # Ensure we have valid County name format
        if query_info.get('county'):
            query_info['county'] = query_info['county'].replace(' County', '').replace(' county', '').title()
            
        return query_info
    
    def _handle_followup(self, query_text, session_context):
        """Handle follow-up questions using the session context."""
        # Extract only the new information from the follow-up
        prompt = f"""
        This is a follow-up question to a previous conversation:
        "{query_text}"
        
        Previous county discussed: {session_context.get('last_county', 'None')}
        Previous state discussed: {session_context.get('last_state', 'Michigan')}
        Previous years discussed: {session_context.get('last_years', [])}
        
        Extract ONLY the new information provided in this follow-up question. Return a JSON with only the fields that are mentioned or implied in this follow-up.
        For example, if the follow-up only mentions a different year, just return {{"years": [the new year]}}.
        If it's asking about the same things but for a different county, return {{"county": "new county name"}}.
        """
        
        try:
            response = chat_with_deepseek(prompt, model=self.model)
            new_info = self._extract_json_from_response(response) or {}
            
            # Start with the previous context and update with new information
            result = {
                'county': session_context.get('last_county'),
                'state': session_context.get('last_state', 'Michigan'),
                'years': session_context.get('last_years', self.default_years),
                'analysis_type': session_context.get('last_analysis_type', 'crop'),
                'focus': session_context.get('last_focus', 'pattern')
            }
            
            # Update with any new information
            result.update(new_info)
            
            return self._validate_and_complete_info(result, session_context)
            
        except Exception as e:
            print(f"Error handling follow-up: {e}")
            # Fall back to using the existing context
            return {
                'county': session_context.get('last_county'),
                'state': session_context.get('last_state', 'Michigan'),
                'years': session_context.get('last_years', self.default_years),
                'analysis_type': session_context.get('last_analysis_type', 'crop'),
                'focus': session_context.get('last_focus', 'pattern')
            }
    
    def _fallback_parsing(self, query_text, session_context):
        """
        Fallback method when AI parsing fails. This is a minimal implementation
        that checks for county mentions and defaults to session context.
        """
        # Very basic county extraction
        county_match = re.search(r'([A-Za-z]+)\s+County', query_text, re.IGNORECASE)
        county = county_match.group(1).title() if county_match else None
        
        # Very basic year extraction
        years = [int(year) for year in re.findall(r'(19|20)\d{2}', query_text)]
        
        # Determine if this is likely a climate query
        is_climate = any(word in query_text.lower() for word in 
                         ['climate', 'weather', 'temperature', 'rain', 'precipitation'])
        
        # Determine if this is specifically about crops
        is_crop = any(word in query_text.lower() for word in 
                      ['crop', 'agriculture', 'farm', 'corn', 'soybean'])
        
        # Check for "major crop" pattern specifically
        is_crop_pattern = ('major crop' in query_text.lower() or 
                          'main crop' in query_text.lower())
        
        # Use session context if available
        if not county and session_context and session_context.get('last_county'):
            county = session_context['last_county']
            
        if not years and session_context and session_context.get('last_years'):
            years = session_context['last_years']
        elif not years:
            years = self.default_years
            
        # Construct a basic result
        result = {
            'county': county,
            'state': 'Michigan',
            'years': years,
            'analysis_type': 'climate' if is_climate else 'crop',
            'focus': 'pattern'
        }
        
        # Add explicit query_type for crop pattern questions
        if is_crop_pattern:
            result['query_type'] = 'crop'
            
        return result if county else None
    
    def _is_followup_question(self, query_text):
        """Determine if a query is likely a follow-up question."""
        followup_indicators = [
            r'what about',
            r'how about',
            r'and (in|for)',
            r'what if',
            r'compare',
            r'^and ',
            r'there',
            r'that county',
            r'those years',
            r'instead',
        ]
        return any(re.search(pattern, query_text.lower()) for pattern in followup_indicators)


if __name__ == "__main__":
    # Test the query parsing agent
    query_parser = QueryParsingAgent()
    
    # Test query with context
    context = {
        'last_county': 'Washtenaw',
        'last_state': 'Michigan',
        'last_years': [2010, 2011],
        'last_analysis_type': 'crop',
        'last_focus': 'Corn'
    }
    query = "What about 2012 for corn in that county?"
    print(query_parser.parse_query(query, context))
    
    # Test a follow-up question
    query = "How about 2013?"
    print(query_parser.parse_query(query, context))
    
    # Test a new query
    query = "What was the weather like in 2015 in Wayne County?"
    print(query_parser.parse_query(query))
    
    # Test a more complex query
    query = "Compare the corn patterns in 2010 and 2011 for Washtenaw and Wayne counties."
    print(query_parser.parse_query(query))