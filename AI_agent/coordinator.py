import time
from specialized_agents import (
    QueryAnalysisAgent,
    DataRetrievalAgent,
    AnalysisAgent,
    SynthesisAgent
)
from model_selector import ModelSelector
try:
    from debug_utils import log_query_info, log_data_structure
    DEBUG = True
except ImportError:
    DEBUG = False
    def log_query_info(*args): pass
    def log_data_structure(*args): pass

class AgentCoordinator:
    def __init__(self):
        self.agents = {
            'query': QueryAnalysisAgent(),
            'data': DataRetrievalAgent(),
            'analysis': AnalysisAgent(),
            'synthesis': SynthesisAgent()
        }
        self.context = {}
        # Session context to remember previous queries
        self.session = {
            'last_county': None,
            'last_state': None,
            'last_analysis_type': None,
            'last_focus': None
        }
    
    def process_query(self, query):
        total_start = time.time()
        
        # Step 1: Understand the query - use simple model
        query_start = time.time()
        query_info = self.agents['query'].process(query, {'session': self.session})
        query_time = time.time() - query_start
        print(f"\nQuery analysis took {query_time:.2f} seconds")
        
        if not query_info:
            return "I couldn't understand your query. Please provide more specific information."
        
        if query_info.get('error'):
            return query_info['message']
        
        # Store the original query for context
        query_info['query'] = query
        
        # Debug output
        if DEBUG:
            log_query_info(query_info)
        
        # Update session context for future follow-up questions
        self.session['last_county'] = query_info.get('county')
        self.session['last_state'] = query_info.get('state')
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
            'requested_years': query_info['years'],  # Store the exact years requested
            'years': query_info['years'],  # For climate queries, use exact years
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
        print(f"Data retrieval took {data_time:.2f} seconds")
        
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
        print(f"Data analysis took {analysis_time:.2f} seconds")
        
        # Skip synthesis for specific queries
        needs_synthesis = analysis_type == 'both'
        if not needs_synthesis:
            return analysis
            
        # Step 4: Synthesize results
        synthesis_start = time.time()
        final_response = self.agents['synthesis'].process(analysis, self.context)
        synthesis_time = time.time() - synthesis_start
        print(f"Response synthesis took {synthesis_time:.2f} seconds")
        
        total_time = time.time() - total_start
        print(f"\nTotal processing time: {total_time:.2f} seconds\n")
        
        return final_response
