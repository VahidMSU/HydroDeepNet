import time
from specialized_agents import (
    QueryAnalysisAgent,
    DataRetrievalAgent,
    AnalysisAgent,
    SynthesisAgent
)

class AgentCoordinator:
    def __init__(self):
        self.agents = {
            'query': QueryAnalysisAgent(),
            'data': DataRetrievalAgent(),
            'analysis': AnalysisAgent(),
            'synthesis': SynthesisAgent()
        }
        self.context = {}
    
    def process_query(self, query):
        total_start = time.time()
        
        # Step 1: Understand the query
        query_start = time.time()
        query_info = self.agents['query'].process(query, self.context)
        query_time = time.time() - query_start
        print(f"\nQuery analysis took {query_time:.2f} seconds")
        
        if not query_info:
            return "I couldn't understand the query. Please rephrase it."
            
        # Determine if synthesis is needed
        needs_synthesis = query_info.get('analysis_type') == 'both'
        
        # Update context
        self.context.update(query_info)
        
        # Step 2: Retrieve data
        data_start = time.time()
        data = self.agents['data'].process(query_info, self.context)
        data_time = time.time() - data_start
        print(f"Data retrieval took {data_time:.2f} seconds")
        
        if not data:
            return f"Couldn't retrieve data for {query_info['county']} County, {query_info['state']}"
            
        # Step 3: Analyze data
        analysis_start = time.time()
        analysis = self.agents['analysis'].process(data, query_info)
        analysis_time = time.time() - analysis_start
        print(f"Data analysis took {analysis_time:.2f} seconds")
        
        # Skip synthesis for specific queries
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
