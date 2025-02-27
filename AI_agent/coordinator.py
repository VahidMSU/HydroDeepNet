import time
try:
    from specialized_agents import (
        QueryAnalysisAgent,
        DataRetrievalAgent,
        AnalysisAgent,
        SynthesisAgent
    )
    from county_info_agent import CountyInfoAgent
    from query_agent import BaseQueryAgent
    from model_selector import ModelSelector
except ImportError:
    from AI_agent.specialized_agents import (
        QueryAnalysisAgent,
        DataRetrievalAgent,
        AnalysisAgent,
        SynthesisAgent
    )
    from AI_agent.county_info_agent import CountyInfoAgent
    from AI_agent.query_agent import BaseQueryAgent
    from AI_agent.model_selector import ModelSelector
    
try:
    from debug_utils import log_query_info, log_data_structure, timing_log
    DEBUG = True
except ImportError:
    DEBUG = False
    def log_query_info(*args): pass
    def log_data_structure(*args): pass
    def timing_log(*args): pass

class AgentCoordinator:
    def __init__(self):
        self.agents = {
            'query': BaseQueryAgent(),
            'data': DataRetrievalAgent(),
            'analysis': AnalysisAgent(),
            'synthesis': SynthesisAgent(),
            'county_info': CountyInfoAgent()  # Add the new agent
        }
        self.context = {}
        # Session context to remember previous queries
        self.session = {
            'last_county': None,
            'last_state': None,
            'last_years': None,
            'last_analysis_type': None,
            'last_focus': None
        }
    
    def process_query(self, query):
        total_start = time.time()
        
        # Step 1: Understand the query using BaseQueryAgent
        query_start = time.time()
        query_info = self.agents['query'].process(query, {'session': self.session})
        query_time = time.time() - query_start
        if DEBUG:
            timing_log("Query analysis", query_start, time.time())
            print(f"Query analysis took {query_time:.2f} seconds")
        
        if not query_info:
            return "I couldn't understand your query. Please provide more specific information about which county you're interested in."
        
        # Check if this is a general information request
        if query_info.get('query_type') == 'general_info':
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
        if DEBUG:
            timing_log("Data analysis", analysis_start, time.time())
        
        # Skip synthesis for specific queries
        needs_synthesis = analysis_type == 'both'
        if not needs_synthesis:
            return analysis
            
        # Step 4: Synthesize results
        synthesis_start = time.time()
        final_response = self.agents['synthesis'].process(analysis, self.context)
        if DEBUG:
            timing_log("Response synthesis", synthesis_start, time.time())
        
        if DEBUG:
            total_time = time.time() - total_start
            print(f"\nTotal processing time: {total_time:.2f} seconds\n")
        
        return final_response
