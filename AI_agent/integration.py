import os
import sys
import logging
import argparse
from typing import Dict, Any, Optional
from pathlib import Path

# Import our components
from interactive_agent import InteractiveReportAgent
from ContextMemory import ContextMemory
from KnowledgeGraph import KnowledgeGraph
from utils.ai_logger import LoggerSetup

# Define log path
LOG_PATH = "/data/SWATGenXApp/codes/AI_agent/logs"

# Create directory if it doesn't exist
os.makedirs(LOG_PATH, exist_ok=True)

# Clear any existing root logging configuration
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Initialize central logger
logger_setup = LoggerSetup(report_path=LOG_PATH, verbose=True)
logger = logger_setup.setup_logger(name="report_analyzer")

class EnhancedReportAnalyzer:
    """
    Integrates our report agent with enhanced memory and knowledge graph capabilities.
    """
    
    def __init__(self, base_dir: str = "/data/SWATGenXApp/Users/admin/Reports/", logger=None):
        """
        Initialize the enhanced report analyzer.
        
        Args:
            base_dir: Base directory for reports
            logger: Logger instance to use
        """
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("Initializing EnhancedReportAnalyzer")
        
        # Initialize single ContextMemory for shared use
        self.context = ContextMemory(storage_path="report_analyzer_memory.db", logger=self.logger)
        
        # Initialize knowledge graph
        self.graph = KnowledgeGraph(storage_path="report_knowledge.json", logger=self.logger)
        
        # Start a new session
        self.session_id = self.context.start_session({
            "base_dir": base_dir,
            "start_time": self.context.session_id  # Use timestamp as ID
        })
        
        # Initialize the agent with our base directory, shared context, and logger
        self.agent = InteractiveReportAgent(base_dir=base_dir, context=self.context, logger=self.logger)
        
        # Initialize domain concepts in knowledge graph
        concepts_added = self.graph.initialize_domain_concepts()
        self.logger.info(f"Initialized knowledge graph with {concepts_added} initial concepts")
    
    def process_query(self, query: str) -> str:
        """
        Process a user query with enhanced capabilities.
        
        Args:
            query: User query
            
        Returns:
            Response for the user
        """
        self.logger.info(f"Processing query: {query}")
        
        # Save query in memory
        self.context.add_message("user", query)
        
        # Check if this is a knowledge graph specific query
        if query.lower().startswith("show knowledge") or "knowledge graph" in query.lower():
            # Generate and save a visualization
            self.graph.visualize("knowledge_graph.png")
            return "I've generated a visualization of my knowledge graph. You can view it in knowledge_graph.png."
        
        # Check if this is a memory-specific query
        if query.lower().startswith("show memory") or "what do you remember" in query.lower():
            insights = self.context.get_insights(limit=10)
            if not insights:
                return "I haven't recorded any insights yet. Let's analyze some data first!"
            
            response = "Here are the key insights I remember:\n\n"
            for i, insight in enumerate(insights):
                response += f"{i+1}. {insight['content']}\n   (from {insight['source']})\n\n"
            
            self.context.add_message("assistant", response)
            return response
        
        # Get recent history for context
        history = self.context.get_conversation_history(5)
        
        # Process query with the agent
        response = self.agent.process_query(query)
        
        # Save response in memory
        self.context.add_message("assistant", response)
        
        # Extract concepts from the query and response and add to knowledge graph
        self.graph.process_text(query, "user_query")
        self.graph.process_text(response, "agent_response")
        
        # Save the updated graph
        self.graph.save_graph()
        
        return response
    
    def interactive_session(self):
        """Start an interactive session with the user."""
        print("🌊 Welcome to the Enhanced Hydrology Report Analyzer 🌊")
        print("Type 'help' for available commands or 'exit' to quit")
        print("Additional commands:")
        print("- show knowledge: Visualize the knowledge graph")
        print("- show memory: Display insights I've remembered")
        
        while True:
            user_input = input("\n> ")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                # End session and save data
                self.context.end_session()
                self.graph.save_graph()
                print("\nThank you for using the Enhanced Hydrology Report Analyzer! Goodbye!")
                break
            
            if not user_input.strip():
                print("\nPlease enter a query or type 'help' for available commands.")
                continue
                
            print("\nProcessing your request...")
            
            # Process the query
            response = self.process_query(user_input)
            
            # Ensure we handle None values properly
            if response is None:
                response = "No response was generated. This may indicate an issue with processing your request."
                
            print("\n" + response)

def main():
    """Main entry point of the application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Enhanced Hydrology Report Analyzer")
    parser.add_argument("--base-dir", type=str, default="/data/SWATGenXApp/Users/admin/Reports/",
                      help="Base directory for reports")
    
    args = parser.parse_args()
    
    try:
        # Initialize the analyzer with the central logger
        analyzer = EnhancedReportAnalyzer(base_dir=args.base_dir, logger=logger)
        
        # Start interactive session
        analyzer.interactive_session()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        print(f"An error occurred: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 