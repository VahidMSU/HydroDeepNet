from base_agent import BaseAgent
from agent import chat_with_deepseek
import json

class QueryAnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__("QueryAnalyzer", "Specializes in understanding user queries")
    
    def process(self, query, context=None):
        from query_agent import extract_query_info
        
        # Extract information from query
        query_info = extract_query_info(query)
        if not query_info:
            return None
            
        # Ensure state is set
        if not query_info.get('state'):
            query_info['state'] = 'Michigan'
            
        # Clean up county name
        if query_info.get('county'):
            query_info['county'] = (query_info['county']
                                  .replace(' county', '')
                                  .replace(' County', '')
                                  .title())
            
        # Handle corn/cron typos
        if 'cron' in query.lower():
            query_info['crop'] = 'Corn'
            query_info['analysis_type'] = 'crop_percentage'
        
        return query_info

class DataRetrievalAgent(BaseAgent):
    def __init__(self):
        super().__init__("DataRetriever", "Handles data retrieval from various sources")
    
    def process(self, query_info, context=None):
        from get_county_bbox import get_bounding_box
        from prism import PRISM_Dataset
        from utils import cdl_trends
        
        # Fix county name formatting
        county = query_info['county'].replace(' county', '').replace(' County', '').title()
        state = query_info['state'].title()
        
        bbox = get_bounding_box(county, state)
        if not bbox[0]:
            return None
            
        config = {
            "RESOLUTION": 250,
            "aggregation": "annual",
            "start_year": int(min(query_info['years'])),  # Ensure integers
            "end_year": int(max(query_info['years'])),
            "bounding_box": bbox
        }
        
        try:
            climate_data = PRISM_Dataset(config).get_spatial_average_over_time()
            landcover_data = cdl_trends(config)
            
            if not climate_data or not landcover_data:
                return None
                
            return {
                'climate': climate_data,
                'landcover': landcover_data,
                'config': config
            }
        except Exception as e:
            print(f"Error retrieving data: {e}")
            return None

class AnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__("Analyzer", "Performs in-depth analysis of retrieved data")
        self.crop_categories = {
            'major_crops': ['Corn', 'Soybeans', 'Wheat', 'Winter Wheat', 'Spring Wheat'],
            'forage': ['Alfalfa', 'Other Hay/Non Alfalfa', 'Grassland/Pasture'],
            'specialty': ['Sweet Corn', 'Dry Beans', 'Potatoes', 'Sugarbeets'],
            'other': ['Oats', 'Barley', 'Rye', 'Sorghum']
        }
    
    def process(self, data, context=None):
        if not (data['climate'] and data['landcover']):
            return "Insufficient data for analysis"
            
        analysis_type = context.get('analysis_type', 'both')
        
        if analysis_type == 'climate':
            return self.analyze_climate(data)
        elif analysis_type == 'crop':
            return self.analyze_crops(data, context.get('focus'))
        else:
            return self.analyze_full_data(data)
    
    def analyze_climate(self, data):
        """Analyze climate patterns."""
        pr_data, tmax_data, tmin_data = data['climate']
        years = range(data['config']['start_year'], data['config']['end_year'] + 1)
        
        # Format yearly climate data
        yearly_data = []
        for i, year in enumerate(years):
            yearly_data.append(
                f"Year {year}:\n"
                f"- Average Precipitation: {float(pr_data[i]):.2f} mm\n"
                f"- Maximum Temperature: {float(tmax_data[i]):.2f}°C\n"
                f"- Minimum Temperature: {float(tmin_data[i]):.2f}°C"
            )
        
        yearly_summary = "\n".join(yearly_data)
        prompt = f"""
        Analyze the climate patterns for {min(years)} to {max(years)}:
        
        Climate Data:
        {yearly_summary}
        
        Provide a clear explanation of:
        1. Temperature trends and variations across years
        2. Precipitation patterns and changes
        3. Notable weather events or patterns
        4. Potential impacts on agriculture
        
        Keep the response focused on climate patterns and their implications.
        Use clear, concise language and highlight significant changes or patterns.
        """

        return chat_with_deepseek(prompt)
    
    def analyze_crops(self, data, crop_focus=None):
        """Analyze crop patterns."""
        if crop_focus and crop_focus.lower() != 'pattern':
            return self.analyze_specific_crop(data, crop_focus)
            
        # General crop pattern analysis
        year_data = data['landcover'][list(data['landcover'].keys())[0]]
        if 'Total Area' not in year_data:
            return "Could not analyze crop patterns - no area data available"
        
        total_area = year_data['Total Area']
        year = list(data['landcover'].keys())[0]
        
        # Calculate category totals
        category_totals = {}
        for category, crops in self.crop_categories.items():
            area = sum(year_data.get(crop, 0) for crop in crops)
            if area > 0:
                category_totals[category] = {
                    'area': area,
                    'percentage': (area / total_area) * 100,
                    'crops': [
                        f"{crop}: {year_data.get(crop, 0):.1f} ha" 
                        for crop in crops 
                        if year_data.get(crop, 0) > 0
                    ]
                }
        
        # Format response
        response = [f"Crop Pattern Analysis for {year}:"]
        response.append(f"\nTotal Agricultural Area: {total_area:.1f} hectares\n")
        
        for category, data in category_totals.items():
            if data['area'] > 0:
                response.append(f"{category.replace('_', ' ').title()}:")
                response.append(f"Total Area: {data['area']:.1f} ha ({data['percentage']:.1f}%)")
                response.append("Breakdown:")
                response.extend([f"- {crop}" for crop in data['crops']])
                response.append("")
        
        return "\n".join(response)
    
    def analyze_specific_crop(self, data, crop_focus):
        """Analyze a specific crop."""
        # ... existing specific crop analysis code ...

    def analyze_crop_percentage(self, data, crop_name):
        """Analyze percentage of a specific crop."""
        year_data = data['landcover'][list(data['landcover'].keys())[0]]
        if 'Total Area' not in year_data:
            return f"Could not calculate percentage for {crop_name}"
            
        total_area = year_data['Total Area']
        crop_area = year_data.get(crop_name, 0)
        percentage = (crop_area / total_area) * 100
        
        return f"{crop_name} covers {percentage:.1f}% of agricultural land ({crop_area:.1f} hectares out of {total_area:.1f} total hectares)"
    
    def analyze_full_data(self, data):
        """Full analysis of climate and land cover data."""
        pr_data, tmax_data, tmin_data = data['climate']
        years = range(data['config']['start_year'], data['config']['end_year'] + 1)
        
        prompt = f"""
        Analyze this agricultural region's data:
        
        Climate Data:
        - Precipitation: {pr_data.tolist()}
        - Max Temperature: {tmax_data.tolist()}
        - Min Temperature: {tmin_data.tolist()}
        
        Land Cover Data:
        {json.dumps(data['landcover'], indent=2)}
        
        Provide insights on:
        1. Land use patterns and changes
        2. Climate impacts on agriculture
        3. Notable trends and correlations
        """
        
        return chat_with_deepseek(prompt)

class SynthesisAgent(BaseAgent):
    def __init__(self):
        super().__init__("Synthesizer", "Creates final, coherent responses")
    
    def process(self, analysis_results, context=None):
        prompt = f"""
        Create a clear, concise summary of this analysis:
        {analysis_results}
        
        Focus on:
        1. Key findings
        2. Practical implications
        3. Notable patterns
        Use natural, conversational language.
        """
        return chat_with_deepseek(prompt)
