
from agent import chat_with_deepseek
from get_county_bbox import get_bounding_box
# from cdl import cdl_trends
from query_parsing_agent import QueryParsingAgent

    
import json
import re
class ConversationContext:
    """Maintains context across multiple conversation turns."""
    def __init__(self):
        self.session = {
            'counties_discussed': [],
            'years_discussed': [],
            'topics_discussed': [],
            'last_county': None,
            'last_state': 'Michigan',
            'last_years': [],
            'last_analysis_type': None
        }
        self.data_cache = {}  # Cache to avoid re-fetching data
    
    def update_context(self, query_info):
        """Update conversation context with new information."""
        if query_info.get('county'):
            if self.session['last_county'] != query_info['county']:
                self.session['counties_discussed'].append(query_info['county'])
            self.session['last_county'] = query_info['county']
        
        if query_info.get('state'):
            self.session['last_state'] = query_info['state']
            
        if query_info.get('years'):
            self.session['last_years'] = query_info['years']
            for year in query_info['years']:
                if year not in self.session['years_discussed']:
                    self.session['years_discussed'].append(year)
                    
        if query_info.get('analysis_type'):
            self.session['last_analysis_type'] = query_info['analysis_type']
            if query_info['analysis_type'] not in self.session['topics_discussed']:
                self.session['topics_discussed'].append(query_info['analysis_type'])
    
    def get_cache_key(self, county, state, year):
        """Generate a cache key for data lookup."""
        return f"{county}_{state}_{year}"
    
    def get_cached_data(self, county, state, year):
        """Get data from cache if available."""
        key = self.get_cache_key(county, state, year)
        return self.data_cache.get(key)
    
    def cache_data(self, county, state, year, data):
        """Store data in cache for future reference."""
        key = self.get_cache_key(county, state, year)
        self.data_cache[key] = data


class ConversationalAgent:
    """A more conversational approach to the agricultural data analysis."""
    
    def __init__(self):
        self.context = ConversationContext()
        self.query_parser = QueryParsingAgent()
        
    def process_query(self, query):
        """Process user query in a more conversational manner."""
        # First, understand the query using the AI parser
        query_info = self.query_parser.parse_query(query, self.context.session)
        
        if not query_info:
            return "I'm sorry, I couldn't understand your query. Could you please specify the county and year you're interested in?"
        
        # Fix any year extraction issues if needed
        if 'years' in query_info:
            # This is now handled by the QueryParsingAgent
            print(f"Using years: {query_info['years']}")
        
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
    
    def understand_query(self, query):
        """Use a more flexible approach to understand the query."""
        # If this seems like a follow-up question, use context
        if self._is_followup_question(query) and self.context.session['last_county']:
            return {
                'county': self.context.session['last_county'],
                'state': self.context.session['last_state'],
                'years': self._extract_years_from_followup(query) or self.context.session['last_years'],
                'analysis_type': self._detect_analysis_type(query) or self.context.session['last_analysis_type']
            }
        
        # Try direct pattern matching first for common queries like "major crop"
        county = self._extract_county(query)
        years = self._extract_years(query)
        
        if county and "major crop" in query.lower():
            print(f"Detected direct pattern: Major crop query for {county}")
            return {
                'county': county,
                'state': 'Michigan',
                'years': years or [2010],
                'analysis_type': 'crop',
                'focus': 'pattern'
            }
        
        # Otherwise, extract information from the full query
        prompt = f"""
        Extract the following information from this query: "{query}"
        
        Return ONLY a JSON object with these fields:
        {{
            "county": "County name (or null if not mentioned)",
            "state": "State name (default to Michigan if not specified)",
            "years": [list of years mentioned, or [2010] if none specified],
            "analysis_type": "crop" or "climate" or "both" based on what's being asked,
            "focus": "specific crop name" if asking about a specific crop, otherwise "pattern"
        }}
        
        Example: For "What's the major crop in Ingham county in 2015?", return:
        {{"county": "Ingham", "state": "Michigan", "years": [2015], "analysis_type": "crop", "focus": "pattern"}}
        
        Do not include any explanation or text outside the JSON object.
        """
        
        try:
            response = chat_with_deepseek(prompt, task="query_analysis")
            # Extract just the JSON part
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                # Fix common JSON formatting issues
                json_str = json_str.replace("'", '"')
                query_info = json.loads(json_str)
                
                # Validate the extracted info
                if not query_info.get('county'):
                    # Fall back to regex extraction
                    raise ValueError("No county found in JSON response")
                    
                return query_info
            else:
                raise ValueError("No JSON object found in response")
                
        except Exception as e:
            print(f"Error parsing query: {e}")
            # Fall back to regex-based extraction for critical fields
            analysis_type = self._detect_analysis_type(query) or 'crop'
            focus = 'pattern'
            
            # Check if specific crop is mentioned
            crop_keywords = ['corn', 'soybean', 'wheat', 'barley', 'oats']
            for crop in crop_keywords:
                if crop in query.lower():
                    focus = crop.title()
                    break
            
            if not county:
                return None
                
            print(f"Using regex fallback: {county}, {years}, {analysis_type}, {focus}")
            return {
                'county': county,
                'state': 'Michigan',
                'years': years or [2010],
                'analysis_type': analysis_type,
                'focus': focus
            }
    
    def _is_followup_question(self, query):
        """Check if this is a follow-up question."""
        followup_indicators = [
            r'what about',
            r'how about',
            r'and (in|for)',
            r'what if',
            r'compare',
            r'^and ',
        ]
        return any(re.search(pattern, query.lower()) for pattern in followup_indicators)
    
    def _extract_county(self, query):
        """Extract county name using regex."""
        patterns = [
            r'([A-Za-z]+)\s+County',
            r'([A-Za-z]+)\s+county',
            r'in\s+([A-Za-z]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1).title()
        
        return None
    
    def _extract_years(self, query):
        """Extract years from query."""
        year_pattern = r'(19|20)\d{2}'
        years = [int(year) for year in re.findall(year_pattern, query)]
        return years
    
    def _extract_years_from_followup(self, query):
        """Extract years from a follow-up question."""
        years = self._extract_years(query)
        if years:
            return years
            
        # Check for relative time references
        if re.search(r'next year', query, re.IGNORECASE) and self.context.session['last_years']:
            return [max(self.context.session['last_years']) + 1]
            
        if re.search(r'previous|last year', query, re.IGNORECASE) and self.context.session['last_years']:
            return [min(self.context.session['last_years']) - 1]
            
        return None
    
    def _detect_analysis_type(self, query):
        """Detect what type of analysis is being requested."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['climate', 'weather', 'temperature', 'precipitation', 'rainfall']):
            return 'climate'
            
        if any(word in query_lower for word in ['crop', 'agriculture', 'farm', 'grow', 'plant', 'corn', 'soybean', 'wheat']):
            return 'crop'
            
        return None
    
    
    def generate_response(self, query, query_info, data):
        """Generate a response based on the query and data."""
        analysis_type = query_info.get('analysis_type', 'crop')
        
        if analysis_type == 'climate':
            return self._generate_climate_response(query, query_info, data)
        elif analysis_type == 'crop':
            return self._generate_crop_response(query, query_info, data)
        else:
            return self._generate_combined_response(query, query_info, data)
    
    def _generate_climate_response(self, query, query_info, data):
        """Generate a response about climate data."""
        county = query_info['county']
        state = query_info['state']
        years = query_info['years']
        
        climate_data = data.get('climate')
        if not climate_data:
            return f"I don't have climate data for {county} County for the requested years."
            
        pr_data, tmax_data, tmin_data = climate_data
        
        # Format climate data into readable text
        climate_text = []
        for i, year in enumerate(years):
            climate_text.append(
                f"Year {year}:\n"
                f"- Precipitation: {float(pr_data[i]):.1f} mm\n"
                f"- Average high temperature: {float(tmax_data[i]):.1f}°C\n"
                f"- Average low temperature: {float(tmin_data[i]):.1f}°C\n"
            )
        
        climate_summary = "\n".join(climate_text)
        
        # Ask the LLM to analyze this data
        prompt = f"""
        Analyze this climate data for {county} County, {state} and explain its implications for agriculture:
        
        {climate_summary}
        
        Provide:
        1. A conversational explanation of the temperature and precipitation patterns
        2. How these conditions likely affected agriculture in the region
        3. Any notable weather patterns or extremes
        
        Make your response conversational but informative, as if you're explaining to a farmer.
        """
        
        return chat_with_deepseek(prompt, task="climate_analysis")
    
    def _generate_crop_response(self, query, query_info, data):
        """Generate a response about crop data."""
        county = query_info['county']
        state = query_info['state']
        years = query_info['years']
        focus = query_info.get('focus', 'pattern')
        
        # Debug the data we received
        print(f"Generating crop response for {county} County, years {years}")
        
        landcover_data = data.get('landcover')
        if not landcover_data:
            print("No landcover data found")
            return f"I don't have crop data for {county} County for the requested years."
        
        # Debug the data keys
        print(f"Landcover data keys: {list(landcover_data.keys())}")
        
        # If asking about a specific crop
        if focus != 'pattern':
            return self._analyze_specific_crop(query, landcover_data, focus, years)
        
        # Extract crop data for the requested years
        crop_data = {}
        for year in years:
            year_str = str(year)
            if year in landcover_data:
                crop_data[year] = landcover_data[year]
                print(f"Found data for year {year} (integer key)")
            elif year_str in landcover_data:
                crop_data[year] = landcover_data[year_str]
                print(f"Found data for year {year} (string key)")
            else:
                # Try to find the closest available year
                available_years = [int(y) if isinstance(y, str) and y.isdigit() else y 
                                  for y in landcover_data.keys() if str(y).isdigit()]
                if available_years:
                    closest_year = min(available_years, key=lambda y: abs(int(y) - year))
                    print(f"Using {closest_year} as fallback for {year}")
                    if isinstance(closest_year, str):
                        crop_data[year] = landcover_data[closest_year]
                    else:
                        crop_data[year] = landcover_data[closest_year]
        
        print(f"Found crop data for years: {list(crop_data.keys())}")
        
        if not crop_data:
            print("No matching crop data found after trying all keys")
            return f"I don't have crop data for {county} County for the requested years."
        
        # Format crop data into readable text
        crop_text = []
        for year, year_data in crop_data.items():
            # Get top crops for this year
            # Filter out non-crop categories
            non_crop_categories = ['Total Area', 'unit', 'county', 'state', 
                                  'Open Water', 'Developed/Open Space', 
                                  'Developed/Low Intensity', 'Developed/Med Intensity', 
                                  'Developed/High Intensity', 'Barren', 'Deciduous Forest', 
                                  'Evergreen Forest', 'Mixed Forest', 'Shrubland', 
                                  'Woody Wetlands', 'Herbaceous Wetlands']
            
            crops = [(crop, area) for crop, area in year_data.items() 
                    if crop not in non_crop_categories]
            crops.sort(key=lambda x: x[1], reverse=True)
            
            crop_list = []
            for crop, area in crops[:5]:  # Top 5 crops
                crop_list.append(f"- {crop}: {area:.1f} hectares")
                
            crop_text.append(
                f"Year {year}:\n"
                f"Total agricultural area: {year_data.get('Total Area', 0):.1f} hectares\n"
                f"Top crops:\n" + "\n".join(crop_list)
            )
        
        crop_summary = "\n".join(crop_text)
        
        # If it's a question about "major crop" specifically, be direct
        if "major" in query.lower() and "crop" in query.lower():
            major_crop_response = []
            for year, year_data in crop_data.items():
                # Filter out non-crop categories for accurate major crop determination
                non_crop_categories = ['Total Area', 'unit', 'county', 'state', 
                                      'Open Water', 'Developed/Open Space', 
                                      'Developed/Low Intensity', 'Developed/Med Intensity', 
                                      'Developed/High Intensity', 'Barren', 'Deciduous Forest', 
                                      'Evergreen Forest', 'Mixed Forest', 'Shrubland', 
                                      'Woody Wetlands', 'Herbaceous Wetlands']
                
                crops = [(crop, area) for crop, area in year_data.items() 
                        if crop not in non_crop_categories and not crop.startswith('Developed/')]
                
                if not crops:
                    continue
                    
                # Find the major crop
                major_crop, area = max(crops, key=lambda x: x[1])
                percentage = (area / year_data.get('Total Area', area)) * 100
                
                major_crop_response.append(
                    f"The major crop in {county} County in {year} was {major_crop} "
                    f"({area:.1f} hectares, {percentage:.1f}% of agricultural land)."
                )
            
            if major_crop_response:
                return "\n".join(major_crop_response)
        
        # Ask the LLM to analyze the crop data
        prompt = f"""
        Analyze this crop data for {county} County, {state}:
        
        {crop_summary}
        
        Provide:
        1. A conversational explanation of the major crops and land use patterns
        2. Any notable trends or changes across the years (if multiple years)
        3. Brief context about what this means for agriculture in this region
        
        Make your response conversational but informative.
        """
        
        return chat_with_deepseek(prompt, task="crop_analysis")
    
    def _analyze_specific_crop(self, query, landcover_data, crop_name, years):
        """Analyze data for a specific crop."""
        # Find the best matching crop name in the data
        all_crops = set()
        for year_data in landcover_data.values():
            all_crops.update(crop for crop in year_data.keys() 
                           if crop not in ['Total Area', 'unit', 'county', 'state'])
        
        # Find the closest match to the requested crop
        matched_crop = None
        for crop in all_crops:
            if crop_name.lower() in crop.lower():
                matched_crop = crop
                break
                
        if not matched_crop:
            return f"I couldn't find data for {crop_name} in the requested years."
            
        # Extract data for this crop across years
        crop_data = []
        for year in years:
            year_str = str(year)
            if year in landcover_data or year_str in landcover_data:
                key = year if year in landcover_data else year_str
                year_data = landcover_data[key]
                
                crop_area = year_data.get(matched_crop, 0)
                total_area = year_data.get('Total Area', 0)
                percentage = (crop_area / total_area) * 100 if total_area > 0 else 0
                
                crop_data.append({
                    'year': year,
                    'area': crop_area,
                    'total': total_area,
                    'percentage': percentage
                })
        
        if not crop_data:
            return f"I couldn't find data for {matched_crop} in the requested years."
            
        # Format crop data into readable text
        crop_text = []
        for data in crop_data:
            crop_text.append(
                f"Year {data['year']}:\n"
                f"- Area: {data['area']:.1f} hectares\n"
                f"- Percentage of agricultural land: {data['percentage']:.1f}%\n"
            )
        
        # Calculate change if we have multiple years
        if len(crop_data) > 1:
            first = crop_data[0]
            last = crop_data[-1]
            change = last['area'] - first['area']
            pct_change = (change / first['area']) * 100 if first['area'] > 0 else 0
            
            crop_text.append(
                f"\nChange from {first['year']} to {last['year']}:\n"
                f"- Absolute change: {change:.1f} hectares\n"
                f"- Relative change: {pct_change:+.1f}%\n"
            )
        
        crop_summary = "\n".join(crop_text)
        
        # Ask the LLM to analyze the crop data
        prompt = f"""
        Analyze this data for {matched_crop} in the requested years:
        
        {crop_summary}
        
        Provide:
        1. A conversational explanation of the {matched_crop} cultivation pattern
        2. Context about why these patterns might exist (climate, markets, etc.)
        3. Brief information about the importance of {matched_crop} in this region
        
        Make your response conversational but informative.
        """
        
        return chat_with_deepseek(prompt, task="crop_analysis")
    
    def _generate_combined_response(self, query, query_info, data):
        """Generate a response combining climate and crop data."""
        county = query_info['county']
        state = query_info['state']
        
        # Generate individual responses
        climate_response = self._generate_climate_response(query, query_info, data)
        crop_response = self._generate_crop_response(query, query_info, data)
        
        # Combine with a synthesis prompt
        prompt = f"""
        Synthesize these two analyses for {county} County, {state} into a unified, 
        conversational response that explains the relationship between climate and crops:
        
        CLIMATE ANALYSIS:
        {climate_response}
        
        CROP ANALYSIS:
        {crop_response}
        
        Your response should:
        1. Flow naturally between climate and crop information
        2. Highlight likely connections between climate conditions and crop patterns
        3. Be conversational and informative
        
        Avoid simply concatenating the two analyses - integrate them thoughtfully.
        """
        
        return chat_with_deepseek(prompt, task="synthesis")
    
    def enhance_response_with_insights(self, response, query_info, data):
        """Add proactive insights to the response based on context."""
        # Check if this is a repeat visit to the same county
        county = query_info['county']
        if county in self.context.session['counties_discussed'] and len(self.context.session['counties_discussed']) > 1:
            # Add comparative insight
            prompt = f"""
            The user has previously asked about other counties besides {county}. Add a brief additional paragraph to the
            end of this response comparing {county} County to the other counties they've asked about.
            
            Previous counties discussed: {', '.join(self.context.session['counties_discussed'])}
            
            Current response:
            {response}
            
            Add a brief, conversational comparative paragraph starting with "Compared to other counties you've asked about..."
            """
            response = chat_with_deepseek(prompt, task="synthesis")
        
        # Check if we have multiple years to potentially suggest trend analysis
        years = query_info.get('years', [])
        if len(years) == 1 and len(self.context.session['years_discussed']) > 3:
            available_years = sorted(self.context.session['years_discussed'])
            suggestion = f"\n\nIf you'd like to see how these patterns have changed over time, I could also analyze trends from {min(available_years)} to {max(available_years)} for this county. Just ask!"
            response += suggestion
        
        return response


# Test the conversational agent
if __name__ == "__main__":
    agent = ConversationalAgent()
    
    # Test a general information query
    query = "What's the major crop in Ingham county in 2015?"
    response = agent.process_query(query)
    print(response)
    
    # Test a climate analysis query
    query = "What was the climate like in Washtenaw county in 2010?"
    response = agent.process_query(query)
    print(response)
    
    # Test a combined analysis query
    query = "What was the major crop in Washtenaw county in 2010 and how was the climate?"
    response = agent.process_query(query)
    print(response)
    
    # Test a follow-up question
    query = "How about 2011?"
    response = agent.process_query(query)
    print(response)
    
    # Test a more complex query
    query = "What's the major crop in Washtenaw county in 2010 and how was the climate? Compare to Ingham county."
    response = agent.process_query(query)
    print(response)
    
    # Test a specific crop query
    query = "What's the major crop in Washtenaw county in 2010?"
    response = agent.process_query(query)
    print(response)
    
    # Test a more conversational query
    query = "What was the climate like in Washtenaw county in 2010 and how did it affect agriculture?"
    response = agent.process_query(query)
    print(response)
    
    # Test a more conversational query with a follow-up
    query = "What was the climate like in Washtenaw county in 2010?"
    response = agent.process_query(query)
    print(response)
    
    query = "How about 2011?"
    response = agent.process_query(query)
    print(response)
    
    query = "What was the major crop in Washtenaw county in 2010?"
    response = agent.process_query(query)
    print(response)
    
    query = "What was the major crop in Washtenaw county in 2011?"
    response = agent.process_query(query)
    print(response)
    
    query = "What was the major crop in Washtenaw county in 2012?"
    response = agent.process_query(query)
    print(response)
    
    query = "What was the major crop in Washtenaw county in 2013?"
    response = agent.process_query(query)
    print(response)