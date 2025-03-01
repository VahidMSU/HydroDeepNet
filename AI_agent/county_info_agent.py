import os
import geopandas as gpd
import pandas as pd
import json
from AI_agent.base_agent import BaseAgent
from AI_agent.get_county_bbox import get_bounding_box
from conversation_handler import chat_with_deepseek


class CountyInfoAgent(BaseAgent):
    """
    Agent responsible for providing general information about counties.
    Handles queries like "explain Ingham county" or "tell me about Mecosta county".
    """
    
    def __init__(self):
        super().__init__("CountyInfo", "Provides general information about counties", "county_info")
        self.county_cache = {}
        
        # Try to load pre-cached county info if available
        cache_path = os.path.join(os.path.dirname(__file__), 'data', 'county_info_cache.json')
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    self.county_cache = json.load(f)
            except:
                print("Could not load county cache")
    
    def process(self, query_info, context=None):
        """
        Process a general county information request.
        
        Args:
            query_info (dict): Information about the query, including county name
            context (dict): Additional context for processing
            
        Returns:
            str: General information about the requested county
        """
        county_name = query_info.get('county')
        state_name = query_info.get('state', 'Michigan')
        
        if not county_name:
            return "I need a county name to provide information."
            
        # Check cache first
        cache_key = f"{county_name.lower()}_{state_name.lower()}"
        if cache_key in self.county_cache:
            print(f"Using cached information for {county_name} County")
            return self.county_cache[cache_key]
            
        # Collect basic county data
        county_data = self._collect_county_data(county_name, state_name)
        
        # Generate a comprehensive response
        response = self._generate_county_overview(county_name, state_name, county_data)
        
        # Cache the response for future use
        self.county_cache[cache_key] = response
        
        return response
    
    def _collect_county_data(self, county_name, state_name):
        """Collect basic data about the county."""
        county_data = {
            "bbox": None,
            "area_sq_km": None,
            "population": None,
            "agricultural_data": {}
        }
        
        # Get bounding box
        try:
            bbox = get_bounding_box(county_name, state_name)
            if bbox[0] is not None:
                county_data["bbox"] = bbox
                
                # Calculate area (very approximate)
                min_lon, min_lat, max_lon, max_lat = bbox
                width_km = abs(max_lon - min_lon) * 111.32 * (abs(min_lat + max_lat)/2)
                height_km = abs(max_lat - min_lat) * 110.574
                county_data["area_sq_km"] = width_km * height_km
        except Exception as e:
            print(f"Error getting bounding box: {e}")
        
        # Try to get population data from Census API (this is a placeholder)
        county_data["population"] = self._get_county_population(county_name, state_name)
        
        # For agricultural data, we'd use a similar approach to our crop data retrieval
        # but for now we'll leave this for the LLM to fill in
        
        return county_data
    
    def _get_county_population(self, county_name, state_name):
        """
        Get county population data.
        In a real implementation, this would connect to Census API or a local database.
        """
        # This is a placeholder - in a real implementation, you'd use an API or database
        # Common Michigan county populations (2020 census)
        populations = {
            "ingham": 284900,
            "mecosta": 43453,
            "kent": 657974,
            "oakland": 1274395,
            "wayne": 1793561,
            "washtenaw": 372258
        }
        
        return populations.get(county_name.lower(), "Unknown")
    
    def _generate_county_overview(self, county_name, state_name, county_data):
        """Generate a comprehensive overview of the county using the LLM."""
        # Create a prompt for the LLM with all the data we have
        area_str = f"{county_data['area_sq_km']:.1f} square kilometers" if county_data.get('area_sq_km') else "Unknown"
        population_str = f"{county_data['population']:,}" if isinstance(county_data.get('population'), (int, float)) else "Unknown"
        
        prompt = f"""
        Create a comprehensive overview of {county_name} County, {state_name}.
        
        Known facts:
        - Area: {area_str}
        - Population: {population_str}
        
        Please include:
        1. Geographic location within {state_name}
        2. Major cities and towns
        3. Economic overview (major industries, particularly agriculture)
        4. Brief history of the county
        5. Notable features (universities, natural landmarks, etc.)
        6. Agricultural importance and common crops
        
        Format this as a conversational response to someone asking about this county.
        Ensure accuracy and focus on agricultural relevance when applicable.
        """
        
        return chat_with_deepseek(prompt, task="county_info")
