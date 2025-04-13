#!/usr/bin/env python3

import sys
import os
from interactive_agent import InteractiveReportAgent
from ContextMemory import ContextMemory

def print_header(text):
    """Print a header with formatting."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)

def test_query_intent():
    """Test the query intent analyzer with different types of queries."""
    
    # Initialize the agent
    agent = InteractiveReportAgent()
    
    # Define test queries for different intents
    test_queries = {
        "Information Queries": [
            "what data are in cdl group?",
            "show me what files are available in groundwater",
            "list the information in climate_change group",
            "what files do you have about soil?",
            "tell me what data is in the prism group",
            "do you have any datasets for the southwestern region?",
            "what kind of reports are available for climate change?",
            "describe the available files in groundwater group",
            "what types of data do you have for crop yields?"
        ],
        "Analysis Queries": [
            "analyze the groundwater data",
            "compare precipitation and temperature trends",
            "evaluate crop rotation patterns",
            "show me the analysis of soil moisture",
            "generate a report on the climate change effects",
            "what can you find about temperature patterns in the climate_change group?",
            "identify trends in precipitation data",
            "calculate the statistics for crop yields over time",
            "what insights can you provide from the soil moisture measurements?",
            "find relationships between rainfall and crop production in cdl data"
        ],
        "Conversational Queries": [
            "tell me more about groundwater levels",
            "why is soil moisture important?",
            "can you explain how precipitation affects crop yields?",
            "what is the relationship between climate and water levels?",
            "I'm interested in learning about erosion",
            "could you explain what soil pH indicates about fertility?",
            "in your opinion, what's the most significant factor affecting crop yields?",
            "is there any correlation between rainfall patterns and groundwater recharge?",
            "what do the declining groundwater levels mean for agriculture?",
            "I'm curious about how climate change impacts local farming practices"
        ],
        "Command Queries": [
            "list reports",
            "list groups",
            "set report 20250324_222749",
            "set group climate_change",
            "help",
            "exit"
        ],
        "Mixed/Ambiguous Queries": [
            "groundwater",
            "I need data on climate change",
            "soil erosion in the southwestern region",
            "how does this work?",
            "crop rotation visualization",
            "climate_change group",
            "tell me about soil data and analyze the trends"
        ],
        "Edge Cases": [
            "just browsing",
            "hello",
            "I don't know what I'm looking for",
            "soil",
            "data",
            "visualization of prism group data and comparison with groundwater trends"
        ]
    }
    
    for category, queries in test_queries.items():
        print_header(category)
        
        for query in queries:
            # Get the intent analysis
            intent = agent.analyze_query_intent(query)
            
            # Print the result
            print(f"\nQuery: \"{query}\"")
            print(f"Intent Type: {intent.get('intent_type', 'unknown')}")
            print(f"Confidence: {intent.get('confidence', 0):.2f}")
            print(f"Action: {intent.get('action', 'unknown')}")
            
            # Print additional details based on intent type
            if intent.get('intent_type') == "information":
                print(f"Target Group: {intent.get('target_group', 'Not specified')}")
            elif intent.get('intent_type') == "analysis":
                print(f"Target Group: {intent.get('target_group', 'Not specified')}")
                if 'specific_file' in intent:
                    print(f"Specific File: {intent.get('specific_file')}")
                if 'target_file_type' in intent:
                    print(f"Target File Type: {intent.get('target_file_type')}")
            elif intent.get('intent_type') == "conversation":
                print(f"Topic: {intent.get('topic', 'Not specified')}")
                if 'target_group' in intent:
                    print(f"Target Group: {intent.get('target_group')}")
            elif intent.get('intent_type') == "command":
                print(f"Command: {intent.get('command', 'Not specified')}")
                
        print()  # Add a blank line after each category

def simulate_interaction():
    """Simulate an interaction with the system to demonstrate query intent handling."""
    print_header("INTERACTIVE DEMO")
    print("This shows how the agent responds to different query types.\n")
    
    # Initialize the agent
    agent = InteractiveReportAgent()
    
    # Define a sequence of queries that demonstrate the different behaviors
    demo_queries = [
        "list groups",
        "what data are in cdl group?",
        "tell me more about cdl data",
        "can you explain what the crop data means?",
        "analyze crop rotation patterns in cdl group",
        "what insights can you provide from this analysis?",
        "compare this with climate change data"
    ]
    
    # Process each query
    for i, query in enumerate(demo_queries, 1):
        print(f"\nUSER QUERY {i}: {query}")
        print("-" * 50)
        
        # Process the query
        response = agent.process_query(query)
        
        # Show response
        print(f"AGENT RESPONSE:\n{response}")
        
        # Add a separator between interactions
        if i < len(demo_queries):
            print("\n" + "~" * 80 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        simulate_interaction()
    else:
        test_query_intent()
        print("\nRun with --interactive flag to see a simulated interaction.") 