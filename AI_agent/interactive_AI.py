#!/usr/bin/env python3
"""
Interactive AI Chat with Agno and Gemini
This script provides a command-line interface for interacting with the HydroGeo Assistant
powered by Agno and Gemini. It also provides a function that can be called by the Flask 
application when Agno is not available.
"""

import os
import sys
import argparse
import traceback
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set API key
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', 'AIzaSyCNvHkGEo33oOYffvAkBql6kC9ty7ijklM')

# Fallback AI function for Flask application
def interactive_agent(message):
    """
    Legacy fallback function for processing messages when Agno is not available.
    This is called by the Flask application.
    
    Args:
        message (str): The message from the user
        
    Returns:
        str: A response to the message
    """
    logger.info(f"Legacy interactive_agent called with message: {message[:50]}...")
    
    # Simple rule-based fallback responses
    message_lower = message.lower()
    
    if "hello" in message_lower or "hi" in message_lower:
        return "Hello! I'm the legacy HydroGeo Assistant. I can help with basic questions about environmental data."
    
    if "help" in message_lower:
        return "I can answer questions about environmental and hydrological data, though my capabilities are limited in this legacy mode. For better responses, please ensure Agno is properly installed and configured."
    
    if any(term in message_lower for term in ["prism", "climate", "precipitation", "temperature"]):
        return "PRISM provides high-resolution spatial climate data for the United States. It includes variables like precipitation, temperature, and other climate indicators."
    
    if any(term in message_lower for term in ["loca", "projection", "climate change"]):
        return "LOCA (Localized Constructed Analogs) is a downscaling technique that provides climate projections at a finer resolution from global climate models."
    
    if any(term in message_lower for term in ["wellogic", "groundwater", "well"]):
        return "Wellogic is a database containing water well records, including information about well construction, geology, and groundwater levels."
    
    # Generic fallback response
    return "I understand you're asking about environmental or hydrological data. For more detailed responses, please ensure the Agno integration is properly set up."

# Direct interaction when run as a script
def print_header():
    """Print a nice header for the chat."""
    print("\n" + "=" * 80)
    print("                 HydroGeo Assistant - Interactive Mode")
    print("=" * 80)
    print("Type your questions about environmental and hydrological data.")
    print("Type 'exit', 'quit', or press Ctrl+C to end the conversation.")
    print("=" * 80 + "\n")

def interactive_chat(use_agno=True, model_id="gemini-1.5-flash"):
    """Run an interactive chat session with the AI."""
    print_header()
    
    # First try to use Agno if requested
    if use_agno:
        try:
            # Try to import Agno modules
            from agno_hydrogeo import get_agno_agent, agno_respond
            
            # Create an agent
            agent = get_agno_agent(model_id=model_id)
            print(f"Using Agno with model: {model_id}")
            
            # Introduction message
            try:
                welcome = agno_respond(agent, "Introduce yourself as an environmental data assistant. Keep it brief and friendly.")
                print(f"\nAssistant: {welcome}\n")
            except Exception as e:
                logger.error(f"Error getting introduction: {str(e)}")
                print("\nAssistant: Hello! I'm the HydroGeo Assistant. I can help with questions about environmental data.\n")
            
            # Main chat loop with Agno
            while True:
                # Get user input
                try:
                    user_input = input("You: ")
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
                    
                # Check if user wants to exit
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("\nExiting... Thanks for chatting!")
                    break
                    
                # Skip empty inputs
                if not user_input.strip():
                    continue
                    
                # Get response from agent
                try:
                    response = agno_respond(agent, user_input)
                    print(f"\nAssistant: {response}\n")
                except Exception as e:
                    logger.error(f"Error getting Agno response: {str(e)}")
                    logger.error(traceback.format_exc())
                    print(f"\nError: {str(e)}")
                    print("Please try again with a different question.")
            
            return  # Exit function if Agno was successfully used
            
        except (ImportError, Exception) as e:
            logger.error(f"Failed to use Agno: {str(e)}")
            logger.error(traceback.format_exc())
            print("Agno is not available. Falling back to rule-based responses.\n")
            # Fall through to the rule-based approach if Agno failed
    
    # Rule-based fallback (used if Agno fails or isn't requested)
    print(f"\nAssistant: Hello! I'm using rule-based fallback mode. I can help with basic questions about environmental data.\n")
    
    # Main chat loop with rule-based responses
    while True:
        # Get user input
        try:
            user_input = input("You: ")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
            
        # Check if user wants to exit
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nExiting... Thanks for chatting!")
            break
            
        # Skip empty inputs
        if not user_input.strip():
            continue
            
        # Get response using the interactive_agent function
        try:
            response = interactive_agent(user_input)
            print(f"\nAssistant: {response}\n")
        except Exception as e:
            logger.error(f"Error in rule-based response: {str(e)}")
            logger.error(traceback.format_exc())
            print(f"\nError: {str(e)}")
            print("Please try again with a different question.")

def main():
    """Main function to parse arguments and run the chat."""
    parser = argparse.ArgumentParser(description="Interactive HydroGeo Assistant")
    parser.add_argument("--model", "-m", type=str, default="gemini-1.5-flash",
                      choices=["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"],
                      help="Model to use for the chat (if Agno is available)")
    parser.add_argument("--no-agno", action="store_true", 
                      help="Disable Agno and use rule-based responses only")
    args = parser.parse_args()
    
    interactive_chat(use_agno=not args.no_agno, model_id=args.model)

if __name__ == "__main__":
    main()
