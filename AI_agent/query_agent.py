from conversation_handler import chat_with_deepseek
from AI_agent.base_agent import BaseAgent
import json
import re

class BaseQueryAgent(BaseAgent):
    """A base agent for query parsing that uses LLM for understanding queries."""
    
    def __init__(self):
        super().__init__("QueryParser", "Understands and parses user queries", "query_analysis")
        
    def process(self, query, context=None):
        
        """
        Process a user query to extract structured information.
        
        Args:
            query (str): The user's natural language query
            context (dict, optional): Previous conversation context
            
        Returns:
            dict: Structured information extracted from the query
        """
        # Check if this is a general information request
        if self._is_general_info_request(query):
            county = self._extract_county(query)
            if county:
                return {
                    "county": county,
                    "state": "Michigan",
                    "query_type": "general_info",
                    "years": [2010]  # Default year
                }
        
        # Check if we have session context for follow-up questions
        session_context = None
        if context and 'session' in context:
            session_context = context['session']
            
        # First try direct pattern matching for common query types
        query_info = self._try_direct_parsing(query, session_context)
        if query_info:
            return query_info
            
        # Otherwise use AI-based parsing
        return self._parse_with_llm(query, session_context)
    
    def _try_direct_parsing(self, query, session=None):
        """Try to parse common query patterns without using LLM"""
        query_lower = query.lower()
        
        # Check for follow-up questions when we have context
        if session and session.get('last_county'):
            if self._is_followup_question(query):
                years = self._extract_years(query)
                return {
                    "county": session.get('last_county'),
                    "state": session.get('last_state', 'Michigan'),
                    "years": years or session.get('last_years', [2010]),
                    "analysis_type": self._extract_analysis_type(query) or session.get('last_analysis_type', 'crop'),
                    "focus": self._extract_focus(query) or session.get('last_focus', 'pattern')
                }
        
        # Extract core information
        county = self._extract_county(query)
        years = self._extract_years(query)
        
        # Check for major crop pattern - a very common query
        if county and "major crop" in query_lower:
            return {
                "county": county,
                "state": "Michigan",
                "years": years or [2010],
                "analysis_type": "crop",
                "focus": "pattern"
            }
        
        # For other queries that specify a county, extract what we can
        if county:
            return {
                "county": county,
                "state": "Michigan", 
                "years": years or [2010],
                "analysis_type": self._extract_analysis_type(query) or "crop",
                "focus": self._extract_focus(query) or "pattern"
            }
            
        return None
    
    def _parse_with_llm(self, query, session=None):
        """Parse the query using LLM"""
        # Create a prompt that includes context if available
        context_str = ""
        if session:
            context_str = f"""
            Previous conversation context:
            - Last county discussed: {session.get('last_county', 'None')}
            - Last state discussed: {session.get('last_state', 'Michigan')}
            - Last years discussed: {session.get('last_years', [])}
            
            Consider this context when resolving references like "there", "that county", "last year", etc.
            """
            
        prompt = f"""
        Extract structured information from this query: "{query}"
        
        {context_str}
        
        For climate queries (containing terms like weather, temperature, rainfall, climate), set analysis_type="climate".
        For crop queries (containing terms like agriculture, farm, crop, corn, soy), set analysis_type="crop".
        For queries mentioning both, set analysis_type="both".
        
        For year-specific queries, include only the requested year(s).
        For queries about specific crops (like corn, soybeans), set focus to that crop name.
        For general crop pattern questions, set focus="pattern".
        
        Return JSON with:
        {{
            "county": "<County name>",
            "state": "<State name>",
            "years": [<year numbers>],
            "analysis_type": "<crop|climate|both>",
            "focus": "<pattern|specific crop name>"
        }}
        
        Return only the JSON object without any additional text.
        """
        
        try:
            response = chat_with_deepseek(prompt, model=self.model)
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0).replace("'", '"')
                query_info = json.loads(json_str)
                
                # Validate and clean up the results
                return self._validate_and_clean(query_info, session)
            else:
                print(f"No valid JSON found in response: {response[:100]}...")
                return self._fallback_parsing(query, session)
                
        except Exception as e:
            print(f"Error in LLM parsing: {e}")
            return self._fallback_parsing(query, session)
    
    def _validate_and_clean(self, query_info, session=None):
        """Validate and clean the extracted information"""
        if not query_info:
            return None
            
        # Ensure we have a county
        if not query_info.get('county') and session:
            query_info['county'] = session.get('last_county')
            
        # If we still don't have a county, we can't process the query
        if not query_info.get('county'):
            return None
            
        # Default state to Michigan
        if not query_info.get('state'):
            query_info['state'] = 'Michigan'
            
        # Fix county formatting
        query_info['county'] = query_info['county'].replace(' County', '').replace(' county', '').title()
            
        # Ensure years is a list of valid years
        if not query_info.get('years'):
            query_info['years'] = [2010]  # Default year
        elif isinstance(query_info.get('years'), str):
            # Handle string years like "2010"
            query_info['years'] = [int(query_info['years'])]
        else:
            # Convert all years to integers and handle 2-digit years
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
            
        # Default focus to pattern if not specified
        if not query_info.get('focus'):
            query_info['focus'] = 'pattern'
            
        # Default analysis_type to crop if not specified
        if not query_info.get('analysis_type'):
            query_info['analysis_type'] = 'crop'
            
        return query_info
    
    def _fallback_parsing(self, query, session=None):
        """Fallback method for when LLM parsing fails"""
        county = self._extract_county(query)
        years = self._extract_years(query)
        analysis_type = self._extract_analysis_type(query)
        focus = self._extract_focus(query)
        
        # If fallback extraction also fails, try to use session data
        if not county and session:
            county = session.get('last_county')
            
        if not years and session:
            years = session.get('last_years')
        elif not years:
            years = [2010]
            
        if not analysis_type:
            analysis_type = 'crop'
            
        if not focus:
            focus = 'pattern'
            
        if county:  # Only return if we have a county
            return {
                "county": county,
                "state": "Michigan",
                "years": years,
                "analysis_type": analysis_type,
                "focus": focus
            }
            
        return None
    
    def _is_followup_question(self, query):
        """Check if a query is a follow-up question"""
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
        return any(re.search(pattern, query.lower()) for pattern in followup_indicators)
    
    def _extract_county(self, query):
        """Extract county name from query using regex patterns"""
        patterns = [
            r'([A-Za-z]+)\s+County',
            r'([A-Za-z]+)\s+county',
            r'in\s+([A-Za-z]+)(?:\s+County|\s+county|\s|,)',
            r'for\s+([A-Za-z]+)(?:\s+County|\s+county|\s|,)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1).title()
        
        return None
    
    def _extract_years(self, query):
        """Extract years from query text"""
        # Look for full years (4 digits)
        year_pattern = r'(?:19|20)\d{2}'
        years = [int(year) for year in re.findall(year_pattern, query)]
        
        if len(years) == 2:
            # Check if this might be a range like "2010 to 2015"
            range_pattern = r'(\d{4})\s*(?:to|-)\s*(\d{4})'
            range_match = re.search(range_pattern, query)
            if range_match:
                start, end = map(int, range_match.groups())
                return list(range(start, end + 1))
        
        return years
    
    def _extract_analysis_type(self, query):
        """Determine the type of analysis being requested"""
        query_lower = query.lower()
        
        climate_keywords = ['climate', 'weather', 'temperature', 'rainfall', 'precipitation', 'rain']
        crop_keywords = ['crop', 'agriculture', 'farming', 'corn', 'soy', 'wheat', 'planting']
        
        has_climate = any(keyword in query_lower for keyword in climate_keywords)
        has_crop = any(keyword in query_lower for keyword in crop_keywords)
        
        if has_climate and has_crop:
            return 'both'
        elif has_climate:
            return 'climate'
        elif has_crop:
            return 'crop'
            
        return None
    
    def _extract_focus(self, query):
        """Extract focus (specific crop or pattern) from query"""
        query_lower = query.lower()
        
        # Check for specific crops
        crop_patterns = {
            r'\bcorn\b': 'Corn',
            r'\bsoy(?:bean)?s?\b': 'Soybeans',
            r'\bwheat\b': 'Wheat',
            r'\boat(?:s)?\b': 'Oats',
            r'\bbarley\b': 'Barley',
            r'\balfalfa\b': 'Alfalfa',
            r'\bhay\b': 'Other Hay/Non Alfalfa'
        }
        
        for pattern, crop in crop_patterns.items():
            if re.search(pattern, query_lower):
                return crop
                
        # Check if it's a pattern question
        if re.search(r'(?:major|main|dominant|primary)\s+crop', query_lower):
            return 'pattern'
            
        return None

    def _is_general_info_request(self, query):
        """Check if this is a general county information request."""
        query_lower = query.lower()
        
        # Patterns indicating general info requests
        general_patterns = [
            r'tell\s+(?:me\s+)?about\s+',
            r'describe\s+',
            r'explain\s+',
            r'what\s+is\s+',
            r'information\s+(?:about|on)\s+',
            r'overview\s+of\s+',
            r'facts\s+about\s+'
        ]
        
        # Check if any pattern matches and the query ends with "county"
        for pattern in general_patterns:
            if re.search(pattern, query_lower) and re.search(r'\bcounty\b', query_lower):
                return True
                
        # Also detect standalone county names with "tell me about" or similar phrases
        if re.search(r'(tell|explain|describe|what)\s+(me\s+)?(about\s+)?([A-Za-z]+)\s+county', query_lower):
            return True
            
        return False

# For backward compatibility
def extract_query_info(query, model=None):
    agent = BaseQueryAgent()
    context = {'session': model} if isinstance(model, dict) and 'session' in model else None
    return agent.process(query, context)
    
def validate_query_info(query_info):
    """Legacy function for backward compatibility"""
    if not query_info:
        return None
        
    required_fields = ['county', 'state', 'years', 'analysis_type']
    if not all(field in query_info for field in required_fields):
        return None
        
    return query_info


if __name__ == "__main__":
    query = "Analyze climate or crop pattern for Mecosta County in 2010"
    query_info = extract_query_info(query)
    query_info = validate_query_info(query_info)
    print(query_info)
    # Expected output: {'county': 'Mecosta', 'state': 'Michigan', 'years': [2010], 'analysis_type': 'both', 'focus': 'pattern'}
