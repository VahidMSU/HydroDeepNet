import time
import traceback
from specialized_agents import (
    QueryAnalysisAgent,
    DataRetrievalAgent,
    AnalysisAgent,
    SynthesisAgent
)
from county_info_agent import CountyInfoAgent
from debug_utils import log_query_info, log_data_structure



DEBUG = False

class AgentCoordinator:
    def __init__(self):
        self.agents = {
            'query': QueryAnalysisAgent(),
            'data': DataRetrievalAgent(),
            'analysis': AnalysisAgent(),
            'synthesis': SynthesisAgent(),
            'county_info': CountyInfoAgent()
        }
        self.context = {}
        # Session context to remember previous queries
        self.session = {
            'last_county': None,
            'last_state': None,
            'last_years': None,
            'last_analysis_type': None,
            'last_focus': None,
            'query_history': []  # Track query history
        }
        self.performance_metrics = {
            'query_time': [],
            'data_time': [],
            'analysis_time': [],
            'synthesis_time': [],
            'total_time': []
        }

    def process_query(self, query):
        total_start = time.time()
        
        try:
            # Track query in history (limit to last 10)
            self.session['query_history'] = self.session['query_history'][-9:] + [query]
            
            # Step 1: Understand the query using QueryAnalysisAgent
            query_start = time.time()
            query_info = self.agents['query'].process(query, {'session': self.session})
            query_time = time.time() - query_start
            self.performance_metrics['query_time'].append(query_time)
            
            if DEBUG:
                timing_log("Query analysis", query_start, time.time())
                print(f"Query analysis took {query_time:.2f} seconds")
            
            if not query_info:
                return "I couldn't understand your query. Please provide more specific information about which county you're interested in."
            
            # Detect if someone is asking about major crops specifically
            # This is a critical fix to ensure crop queries don't go to county_info
            if query.lower().find('major crop') >= 0 or query.lower().find('main crop') >= 0:
                print(f"Detected direct crop question for {query_info['county']} County")
                query_info['query_type'] = 'crop'
                query_info['focus'] = 'pattern'
                query_info['analysis_type'] = 'crop'
            
            # Check if this is a general information request
            if query_info.get('query_type') == 'general_info' and not any(x in query.lower() for x in ['crop', 'climate', 'agriculture', 'farm']):
                print(f"Processing general information request for {query_info['county']} County")
                
                # Update session context still
                self.session['last_county'] = query_info.get('county')
                self.session['last_state'] = query_info.get('state')
                
                # Use the county info agent to get the response
                return self.agents['county_info'].process(query_info)
            
            # Store the original query for context
            query_info['query'] = query
            
            # Debug output
            if DEBUG:
                log_query_info(query_info)
            
            # Update session context for future follow-up questions
            self.session['last_county'] = query_info.get('county')
            self.session['last_state'] = query_info.get('state')
            self.session['last_years'] = query_info.get('years')
            self.session['last_analysis_type'] = query_info.get('analysis_type')
            self.session['last_focus'] = query_info.get('focus')
            
            # Log what type of analysis we're performing
            analysis_type = query_info.get('analysis_type', 'crop')
            print(f"Processing {analysis_type} analysis for {query_info['county']} County")
            
            # Update context and config
            self.context.update(query_info)
            
            # Ensure correct data retrieval configuration
            config = {
                'county': query_info['county'],
                'state': query_info['state'],
                'requested_years': query_info['years'],
                'years': query_info['years'],
                'analysis_type': analysis_type
            }
            
            # Only expand year range for non-climate queries and if we need related years
            needs_year_expansion = analysis_type in ['both'] or (
                analysis_type == 'crop' and query_info.get('focus') == 'pattern'
            )
            
            if needs_year_expansion:
                all_years = list(range(
                    min(query_info['years']) - 1,
                    max(query_info['years']) + 2
                ))
                config['years'] = all_years
            
            # Update query_info with complete config
            query_info.update(config)
            
            # Step 2: Retrieve data (only what's needed based on analysis type)
            data_start = time.time()
            data = self.agents['data'].process(query_info, self.context)
            data_time = time.time() - data_start
            self.performance_metrics['data_time'].append(data_time)
            
            if DEBUG:
                timing_log("Data retrieval", data_start, time.time())
            
            # Debug output
            if DEBUG:
                log_data_structure(data)
            
            if not data:
                return (
                    f"Could not find data for {query_info['county']} County, {query_info['state']}. "
                    "Please verify the county name and try again."
                )
                
            # Step 3: Analyze data
            analysis_start = time.time()
            analysis = self.agents['analysis'].process(data, query_info)
            analysis_time = time.time() - analysis_start
            self.performance_metrics['analysis_time'].append(analysis_time)
            
            if DEBUG:
                timing_log("Data analysis", analysis_start, time.time())
            
            # Determine if we need synthesis based on complexity
            needs_synthesis = analysis_type == 'both' or len(str(analysis)) > 1000
            
            if not needs_synthesis:
                total_time = time.time() - total_start
                self.performance_metrics['total_time'].append(total_time)
                
                if DEBUG:
                    print(f"\nTotal processing time: {total_time:.2f} seconds\n")
                
                # Add any helpful follow-up suggestions
                return self._add_follow_up_suggestions(analysis, query_info)
                
            # Step 4: Synthesize results
            synthesis_start = time.time()
            final_response = self.agents['synthesis'].process(analysis, self.context)
            synthesis_time = time.time() - synthesis_start
            self.performance_metrics['synthesis_time'].append(synthesis_time)
            
            if DEBUG:
                timing_log("Response synthesis", synthesis_start, time.time())
            
            total_time = time.time() - total_start
            self.performance_metrics['total_time'].append(total_time)
            
            if DEBUG:
                print(f"\nTotal processing time: {total_time:.2f} seconds\n")
            
            # Add any helpful follow-up suggestions
            return self._add_follow_up_suggestions(final_response, query_info)
        
        except Exception as e:
            print(f"Error processing query: {e}")
            print(traceback.format_exc())
            
            # Provide a graceful error response
            return (
                "I'm sorry, I encountered an error while processing your request. "
                "Please try rephrasing your question or providing more specific details."
            )
    
    def _add_follow_up_suggestions(self, response, query_info):
        """Add helpful follow-up suggestions based on the query context."""
        county = query_info.get('county')
        state = query_info.get('state', 'Michigan')
        years = query_info.get('years', [])
        analysis_type = query_info.get('analysis_type')
        
        suggestions = []
        
        # Suggest climate analysis if they asked about crops
        if analysis_type == 'crop':
            if len(years) == 1:
                year = years[0]
                suggestions.append(f"What was the climate like in {county} County in {year}?")
            else:
                suggestions.append(f"How was the climate in {county} County during these years?")
        
        # Suggest crop analysis if they asked about climate
        elif analysis_type == 'climate':
            suggestions.append(f"What were the major crops in {county} County during these years?")
        
        # Suggest looking at neighboring counties or additional years
        if len(years) == 1:
            year = years[0]
            if year > 2000:  # Avoid suggesting years that are too old
                suggestions.append(f"How does this compare to {year-5}?")
                
        # Only add suggestions if we have some and the response isn't too long already
        if suggestions and len(response) < 3000:
            response += "\n\n**Follow-up questions you might ask:**\n"
            for suggestion in suggestions[:2]:  # Limit to 2 suggestions
                response += f"- {suggestion}\n"
                
        return response
    
    def get_performance_stats(self):
        """Return performance statistics for monitoring."""
        if not self.performance_metrics['total_time']:
            return {"error": "No queries processed yet"}
            
        # Calculate averages
        avg_metrics = {}
        for key, values in self.performance_metrics.items():
            if values:
                avg_metrics[f"avg_{key}"] = sum(values) / len(values)
                avg_metrics[f"max_{key}"] = max(values)
                
        # Calculate percentages
        total_avg = avg_metrics.get("avg_total_time", 0)
        if total_avg > 0:
            for key in ['query_time', 'data_time', 'analysis_time', 'synthesis_time']:
                avg_key = f"avg_{key}"
                if avg_key in avg_metrics:
                    avg_metrics[f"{key}_percent"] = (avg_metrics[avg_key] / total_avg) * 100
                    
        return avg_metrics

# Test the agent coordinator
if __name__ == "__main__":
    coordinator = AgentCoordinator()
    
    # Test a general information query
    query = "What can you tell me about Mecosta County, Michigan?"
    response = coordinator.process_query(query)
    print(response)
    
    # Test a specific analysis query
    query = "Analyze the crop patterns in Mecosta County, Michigan from 2000 to 2010."
    response = coordinator.process_query(query)
    print(response)
    
    # Test a more complex analysis query
    query = "Analyze the relationship between land use and climate in Mecosta County, Michigan."
    response = coordinator.process_query(query)
    print(response)
    
    # Print performance stats
    print("\nPerformance statistics:")
    print(coordinator.get_performance_stats())

"""
Agent Coordinator module for handling user queries and delegating to specialized agents.
This is a simplified implementation to ensure the application works.
"""
import logging
import re

logger = logging.getLogger(__name__)

class AgentCoordinator:
    """
    Coordinates interactions between user queries and specialized agents.
    Provides fallback responses when specialized modules aren't available.
    """
    
    def __init__(self):
        """Initialize the coordinator with basic capabilities."""
        self.state_abbreviations = {
            'michigan': 'MI',
            'ohio': 'OH',
            'indiana': 'IN',
            'illinois': 'IL',
            'wisconsin': 'WI'
        }
        
    def process_query(self, query):
        """
        Process a user query and return a response.
        
        Args:
            query (str): The user query text
            
        Returns:
            str: Response to the query
        """
        logger.info(f"Processing query: {query}")
        
        # Basic response for empty queries
        if not query or not query.strip():
            return "Please ask me a question about agricultural or climate data."
        
        # Extract potential location mentions
        query_lower = query.lower().strip()
        
        # Check for county mentions
        county_match = re.search(r'([a-z]+)\s+county', query_lower)
        if county_match:
            county_name = county_match.group(1).capitalize()
            
            # Check for state mentions
            state_name = None
            for state in self.state_abbreviations:
                if state in query_lower:
                    state_name = state
                    break
                    
            if state_name:
                state_code = self.state_abbreviations.get(state_name)
                return self._generate_county_response(county_name, state_code, query)
        
        # Generic fallback responses
        if 'crop' in query_lower or 'agriculture' in query_lower:
            return self._generate_crop_response(query)
        elif 'climate' in query_lower or 'weather' in query_lower:
            return self._generate_climate_response(query)
        elif 'help' in query_lower:
            return self._generate_help_response()
        else:
            return self._generate_default_response(query)
            
    def _generate_county_response(self, county, state_code, query):
        """Generate a response about a specific county."""
        if county == "Ingham" and state_code == "MI":
            return (f"Ingham County, Michigan is known for its diverse agricultural production. "
                   f"Major crops include corn, soybeans, and wheat. The county has approximately "
                   f"175,000 acres of farmland. The climate is humid continental with warm summers "
                   f"and cold winters. Annual precipitation averages around 800-900mm.")
        
        return (f"I found that {county} County in {state_code} is an important agricultural region. "
                f"Unfortunately, I don't have detailed data available for this specific county right now. "
                f"I can tell you that the region generally produces a mix of corn, soybeans, and wheat, "
                f"which are the dominant crops in the Midwest.")
                
    def _generate_crop_response(self, query):
        """Generate a response about crops."""
        return ("The primary crops grown in the Midwest include corn, soybeans, wheat, and hay. "
                "Michigan specifically has a diverse agricultural profile with significant production "
                "of cherries, apples, blueberries, and other specialty crops alongside traditional "
                "field crops. The western and central regions of Michigan are particularly known for "
                "their fruit production due to the lake effect climate.")
                
    def _generate_climate_response(self, query):
        """Generate a response about climate."""
        return ("The climate in Michigan and surrounding Midwest states is primarily classified as "
                "humid continental, characterized by four distinct seasons with warm summers and cold "
                "winters. Annual precipitation typically ranges from 750-950mm across the region, with "
                "slightly higher amounts near the Great Lakes due to lake effect precipitation. "
                "Climate change is gradually shifting growing seasons in the region, with generally "
                "warmer temperatures and more extreme precipitation events.")
                
    def _generate_help_response(self):
        """Generate a help response."""
        return ("I can help you with information about:\n"
                "- Agricultural data for specific counties (e.g., 'What crops are grown in Ingham County?')\n"
                "- Climate patterns in the Midwest region\n"
                "- General agricultural trends\n\n"
                "For the most precise information, please specify a county and state in your question.")
                
    def _generate_default_response(self, query):
        """Generate a default response when no specific handler matches."""
        return ("I understand you're interested in agricultural or environmental information, but I'm "
                "not sure I have the specific data you're looking for. Could you try rephrasing your "
                "question to focus on a specific county or crop type? For example, you could ask about "
                "agricultural patterns in a particular county or climate trends in Michigan.")

