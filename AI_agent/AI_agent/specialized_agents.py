from conversation_handler import chat_with_deepseek
from get_county_bbox import get_bounding_box
from prism import PRISM_Dataset
#from cdl import cdl_trends
from base_agent import BaseAgent
from query_parsing_agent import QueryParsingAgent
from data_cache import get_cached_data, cache_data
from debug_utils import timed_function, log_error
import json
import numpy as np

class QueryAnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__("QueryAnalyzer", "Specializes in understanding user queries", "query_analysis")
        self.parser = QueryParsingAgent()
    
    @timed_function
    def process(self, query, context=None):
        """Use the AI parser to extract query information"""
        session = context.get('session') if context else None
        
        try:
            # Use the AI parser
            query_info = self.parser.parse_query(query, session)
            
            if not query_info:
                return {
                    'error': True,
                    'message': "Could not identify a county name in your query. Please include the county name followed by 'County' (e.g., 'Mecosta County')."
                }
                
            return query_info
        except Exception as e:
            log_error("Error in query analysis", e)
            return {
                'error': True,
                'message': f"Error analyzing query: {str(e)}"
            }


class DataRetrievalAgent(BaseAgent):
    def __init__(self):
        super().__init__("DataRetriever", "Handles data retrieval from various sources", "data_retrieval")
    @timed_function
    def process(self, query_info, context=None):
        try:
            # Fix county name formatting
            county = query_info['county'].replace(' county', '').replace(' County', '').title()
            state = query_info['state'].title()
            
            # Create cache parameters for lookup
            cache_params = {
                'county': county,
                'state': state,
                'start_year': int(min(query_info['years'])),
                'end_year': int(max(query_info['years'])),
                'analysis_type': query_info.get('analysis_type', 'crop')
            }
            
            # Check if we have this data in cache
            cached_data = get_cached_data('county_data', cache_params)
            if cached_data:
                print(f"Using cached data for {county} County, {state}")
                return cached_data
            
            # If not in cache, retrieve the data
            bbox = get_bounding_box(county, state)
            if not bbox[0]:
                log_error(f"Failed to get bounding box for {county} County, {state}")
                return None
                
            config = {
                "RESOLUTION": 250,
                "aggregation": "annual",
                "start_year": int(min(query_info['years'])),
                "end_year": int(max(query_info['years'])),
                "bounding_box": bbox,
                "county": county,  # Add county to config
                "state": state     # Add state to config
            }
            
            try:
                # Determine which data types are needed based on the query type
                analysis_type = query_info.get('analysis_type', 'crop')
                
                # Initialize data containers
                climate_data = None
                landcover_data = None
                
                # Only get climate data if needed (climate or both type queries)
                if analysis_type in ['climate', 'both']:
                    climate_data = PRISM_Dataset(config).get_spatial_average_over_time()
                    
                # Get landcover data if needed (crop or both type queries)
                if analysis_type in ['crop', 'both', 'crop_percentage']:
                    landcover_data = cdl_trends(config)
                    if landcover_data:
                        # Add county and state info to landcover data for reference
                        for year_data in landcover_data.values():
                            if isinstance(year_data, dict):
                                year_data['county'] = county
                                year_data['state'] = state
                    
                # If we couldn't get data for the requested years, try nearby years as fallback
                if (analysis_type in ['crop', 'both', 'crop_percentage'] and landcover_data is None):
                    # Try to get data for nearby years
                    for offset in [1, -1, 2, -2]:
                        fallback_config = config.copy()
                        fallback_config["start_year"] = int(min(query_info['years'])) + offset
                        fallback_config["end_year"] = int(max(query_info['years'])) + offset
                        
                        print(f"Trying fallback years {fallback_config['start_year']}-{fallback_config['end_year']}")
                        try:
                            landcover_data = cdl_trends(fallback_config)
                            if landcover_data:
                                print(f"Using fallback data from years {fallback_config['start_year']}-{fallback_config['end_year']}")
                                # Add county and state info
                                for year_data in landcover_data.values():
                                    if isinstance(year_data, dict):
                                        year_data['county'] = county
                                        year_data['state'] = state
                                break
                        except Exception as fallback_e:
                            print(f"Fallback retrieval attempt failed: {fallback_e}")
                    
                # Validate that we got the data we need
                if (analysis_type in ['climate', 'both'] and climate_data is None) or \
                   (analysis_type in ['crop', 'both', 'crop_percentage'] and landcover_data is None):
                    log_error(f"Failed to retrieve required data for {county} County, {state}")
                    return None
                    
                # Create the result
                result = {
                    'climate': climate_data,
                    'landcover': landcover_data,
                    'config': config
                }
                
                # Cache the result for future use
                cache_data('county_data', cache_params, result, 
                          f"{county} County, {state} ({config['start_year']}-{config['end_year']})")
                
                return result
                
            except Exception as e:
                log_error(f"Error retrieving data for {county} County, {state}", e)
                return None
        except Exception as e:
            log_error("Error in data retrieval process", e)
            return None

class AnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__("Analyzer", "Performs in-depth analysis of retrieved data", "analysis")
        self.crop_categories = {
            'major_crops': ['Corn', 'Soybeans', 'Wheat', 'Winter Wheat', 'Spring Wheat'],
            'forage': ['Alfalfa', 'Other Hay/Non Alfalfa', 'Grassland/Pasture'],
            'specialty': ['Sweet Corn', 'Dry Beans', 'Potatoes', 'Sugarbeets'],
            'other': ['Oats', 'Barley', 'Rye', 'Sorghum']
        }
    
    def process(self, data, context=None):
        # For climate-only analysis, we only need climate data
        analysis_type = context.get('analysis_type', 'both')
        
        if analysis_type == 'climate':
            if not data.get('climate'):
                return "Insufficient climate data for analysis"
            return self.analyze_climate(data)
            
        # For crop-specific analysis
        if analysis_type in ['crop', 'crop_percentage']:
            if not data.get('landcover'):
                return "Insufficient crop data for analysis"
                
            # Check for specific crop queries like "what is the major crop"
            if ('major' in context.get('query', '').lower() or 
                'main' in context.get('query', '').lower() or 
                context.get('focus') == 'pattern'):
                
                year = context.get('requested_years', [2010])[0]  # Get the first requested year
                print(f"Processing major crop analysis for {year}")
                return self.get_major_crop_for_year(data['landcover'], year)
            else:
                # Get the specific years requested in the query
                requested_years = context.get('requested_years', list(data['landcover'].keys()))
                return self.analyze_crops(data, context.get('focus'), requested_years)
                
        # For combined analysis, need both
        if not (data.get('climate') and data.get('landcover')):
            return "Insufficient data for analysis"
            
        return self.analyze_full_data(data)
    
    def get_major_crop_for_year(self, landcover_data, year):
        """Get the major crop for a specific year with a simple direct response."""
        # Convert year to string if it exists as string key
        year_key = str(year) if str(year) in landcover_data else year
        
        # If the year isn't directly available, try to find the closest year
        if year_key not in landcover_data:
            available_years = [int(y) if isinstance(y, str) and y.isdigit() else y 
                              for y in landcover_data.keys() if str(y).isdigit()]
            if available_years:
                year_key = min(available_years, key=lambda y: abs(int(y) - year))
                print(f"Using {year_key} as closest available year to {year}")
            else:
                return f"No crop data available for the year {year} or nearby years"
        
        year_data = landcover_data[year_key]
        
        # Filter to only consider actual crops (exclude non-crop categories)
        non_crop_categories = ['Total Area', 'unit', 'county', 'state', 
                             'Open Water', 'Developed/Open Space', 
                             'Developed/Low Intensity', 'Developed/Med Intensity', 
                             'Developed/High Intensity', 'Barren', 'Deciduous Forest', 
                             'Evergreen Forest', 'Mixed Forest', 'Shrubland', 
                             'Woody Wetlands', 'Herbaceous Wetlands']
        
        all_crops = {}
        for category, crop_list in self.crop_categories.items():
            for crop in crop_list:
                if crop in year_data:
                    all_crops[crop] = year_data[crop]
        
        # Also check for any crops not in our categories
        for crop, area in year_data.items():
            if (crop not in all_crops and 
                crop not in non_crop_categories and 
                isinstance(area, (int, float)) and 
                area > 0):
                all_crops[crop] = area
        
        if not all_crops:
            return f"No crop data available for {year}"
            
        # Find the crop with the largest area
        major_crop = max(all_crops.items(), key=lambda x: x[1])
        
        # Calculate percentage
        total_area = year_data.get('Total Area', sum(all_crops.values()))
        percentage = (major_crop[1] / total_area) * 100 if total_area else 0
        
        # Format a simple response
        response = (
            f"The major crop in {year} was {major_crop[0]} covering {major_crop[1]:.1f} hectares "
            f"({percentage:.1f}% of the total agricultural area).\n\n"
            f"Other significant crops included:\n"
        )
        
        # List other top crops
        other_crops = sorted(all_crops.items(), key=lambda x: x[1], reverse=True)[1:4]  # Get next 3 top crops
        for crop, area in other_crops:
            crop_pct = (area / total_area) * 100 if total_area else 0
            response += f"- {crop}: {area:.1f} hectares ({crop_pct:.1f}%)\n"
            
        return response
    
    def analyze_climate(self, data):
        """Analyze climate patterns."""
        pr_data, tmax_data, tmin_data = data['climate']
        config = data['config']
        target_years = config.get('requested_years', range(config['start_year'], config['end_year'] + 1))
        
        # Format yearly climate data
        yearly_data = []
        for i, year in enumerate(target_years):
            if i < len(pr_data):
                yearly_data.append(
                    f"Year {year}:\n"
                    f"- Annual Precipitation: {float(pr_data[i]):.2f} mm\n"
                    f"- Average Maximum Temperature: {float(tmax_data[i]):.2f}°C\n"
                    f"- Average Minimum Temperature: {float(tmin_data[i]):.2f}°C"
                )
        
        if not yearly_data:
            return "No climate data available for the requested years."
            
        yearly_summary = "\n".join(yearly_data)
        
        # Get county and state from config
        county = config.get('county')
        if not county:
            return "County information not found in the data"
        
        state = config.get('state', 'Michigan')
        
        prompt = f"""
        Analyze these climate patterns for {county} County, {state}:
        
        {yearly_summary}
        
        Provide a clear explanation of:
        1. Temperature characteristics (average high/low, extremes)
        2. Precipitation patterns
        3. Agricultural implications of these conditions
        4. How these conditions compare to typical patterns for the region
        
        Keep the response focused on climate patterns and their practical implications 
        for agriculture in this area.
        """
        return chat_with_deepseek(prompt, task="climate_analysis")
    
    def analyze_crops(self, data, crop_focus=None, requested_years=None):
        """Analyze crop patterns for multiple years."""
        if crop_focus and crop_focus.lower() != 'pattern':
            return self.analyze_specific_crop(data, crop_focus, requested_years)
        
        # Get landcover data for the requested years only
        landcover_data = data['landcover']
        if requested_years:
            # Convert to string keys if needed
            str_years = [str(y) for y in requested_years]
            available_years = [y for y in landcover_data.keys() if str(y) in str_years or int(y) in requested_years]
        else:
            available_years = list(landcover_data.keys())
        
        # Sort years for consistent output
        available_years = sorted(available_years)
        
        if not available_years:
            return "No land cover data available for the requested years."
        
        # Process each year individually
        yearly_analyses = []
        for year in available_years:
            year_data = landcover_data[year]
            if 'Total Area' not in year_data:
                yearly_analyses.append(f"Year {year}: Could not analyze crop patterns - no area data available")
                continue
            
            total_area = year_data['Total Area']
            
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
            
            # Format response for this year
            year_response = [f"Crop Pattern Analysis for Year {year}:"]
            year_response.append(f"\nTotal Agricultural Area: {total_area:.1f} hectares\n")
            
            for category, cat_data in category_totals.items():
                if cat_data['area'] > 0:
                    year_response.append(f"{category.replace('_', ' ').title()}:")
                    year_response.append(f"Total Area: {cat_data['area']:.1f} ha ({cat_data['percentage']:.1f}%)")
                    year_response.append("Breakdown:")
                    year_response.extend([f"- {crop}" for crop in cat_data['crops']])
                    year_response.append("")
            
            yearly_analyses.append("\n".join(year_response))
        
        # If we have multiple years, also add a summary
        if len(available_years) > 1:
            summary = self.generate_multi_year_crop_summary(landcover_data, available_years)
            yearly_analyses.append(summary)
            
        return "\n\n" + "\n\n".join(yearly_analyses)
    
    def generate_multi_year_crop_summary(self, landcover_data, years):
        """Generate a summary of crop trends across multiple years."""
        # Find top crops across all years
        all_crops = {}
        for year in years:
            for crop, area in landcover_data[year].items():
                if crop != 'Total Area' and crop != 'unit':
                    if crop not in all_crops:
                        all_crops[crop] = []
                    all_crops[crop].append(area)
        
        # Calculate averages for major crops only
        major_crops = []
        for crop in self.crop_categories['major_crops']:
            if crop in all_crops:
                avg_area = np.mean(all_crops[crop])
                major_crops.append((crop, avg_area))
        
        # Sort by average area
        major_crops.sort(key=lambda x: x[1], reverse=True)
        
        # Format summary
        summary = [f"Summary for Years {min(years)}-{max(years)}:"]
        summary.append("\nMajor Crops by Average Area:")
        for crop, avg in major_crops[:5]:  # Top 5 major crops
            year_values = [f"{year}: {landcover_data[year].get(crop, 0):.1f} ha" for year in years]
            summary.append(f"- {crop}: {avg:.1f} ha average ({', '.join(year_values)})")
        
        # Calculate overall trend
        if len(years) > 1:
            summary.append("\nYear-to-Year Changes in Major Crop Areas:")
            for i, year in enumerate(years[:-1]):
                next_year = years[i + 1]
                for crop, _ in major_crops[:3]:  # Top 3 crops only
                    current = landcover_data[year].get(crop, 0)
                    future = landcover_data[next_year].get(crop, 0)
                    change = future - current
                    pct_change = (change / current * 100) if current > 0 else 0
                    summary.append(f"- {crop} from {year} to {next_year}: {change:.1f} ha ({pct_change:+.1f}%)")
            
        return "\n".join(summary)

    def analyze_specific_crop(self, data, crop_focus, requested_years=None):
        """Analyze a specific crop across multiple years."""
        landcover_data = data['landcover']
        
        # Filter for requested years if specified
        if requested_years:
            str_years = [str(y) for y in requested_years]
            years = [y for y in landcover_data.keys() if str(y) in str_years or int(y) in requested_years]
        else:
            years = list(landcover_data.keys())
            
        years = sorted(years)
        
        # Find closest matching crop name (case insensitive)
        crop_name = None
        all_crops = set()
        for year_data in landcover_data.values():
            all_crops.update(crop for crop in year_data.keys() if crop not in ['Total Area', 'unit'])
        
        for crop in all_crops:
            if crop_focus.lower() in crop.lower():
                crop_name = crop
                break
                
        if not crop_name:
            return f"Could not find data for crop '{crop_focus}' in the available years."
            
        # Collect data for each year
        crop_data = []
        for year in years:
            if year in landcover_data:
                total_area = landcover_data[year].get('Total Area', 0)
                crop_area = landcover_data[year].get(crop_name, 0)
                percentage = (crop_area / total_area * 100) if total_area > 0 else 0
                crop_data.append({
                    'year': year, 
                    'area': crop_area,
                    'percentage': percentage,
                    'total': total_area
                })
        
        if not crop_data:
            return f"No data available for {crop_name} in the requested years."
            
        # Format response
        response = [f"Analysis of {crop_name} for years {min(years)} to {max(years)}:"]
        for data in crop_data:
            response.append(
                f"\nYear {data['year']}:\n"
                f"- Area: {data['area']:.1f} hectares\n"
                f"- Percentage of agricultural land: {data['percentage']:.1f}%\n"
                f"- Total agricultural area: {data['total']:.1f} hectares"
            )
            
        # Calculate trend if we have multiple years
        if len(crop_data) > 1:
            first_year = crop_data[0]['area']
            last_year = crop_data[-1]['area']
            change = last_year - first_year
            pct_change = (change / first_year * 100) if first_year > 0 else 0
            
            response.append(
                f"\nOverall Change ({crop_data[0]['year']} to {crop_data[-1]['year']}):\n"
                f"- Absolute change: {change:.1f} hectares\n"
                f"- Percentage change: {pct_change:+.1f}%"
            )
        
        return "\n".join(response)
    
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
        
        return chat_with_deepseek(prompt, task="analysis")

class SynthesisAgent(BaseAgent):
    def __init__(self):
        super().__init__("Synthesizer", "Creates final, coherent responses", "synthesis")
    
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
        return chat_with_deepseek(prompt, task="synthesis")
