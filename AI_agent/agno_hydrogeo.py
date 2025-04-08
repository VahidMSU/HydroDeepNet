"""
Agno integration for HydroGeo Assistant.
This module uses the Agno library to create AI agents powered by Google's Gemini models.
"""

import os
import logging
import traceback
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get API key from environment with multiple fallbacks
def get_google_api_key():
    """Get Google API key from various possible sources"""
    # Try environment variable first
    api_key = os.environ.get('GOOGLE_API_KEY')
    
    if not api_key:
        # Try to import from Flask config
        try:
            from flask import current_app
            if current_app and current_app.config.get('GOOGLE_API_KEY'):
                api_key = current_app.config.get('GOOGLE_API_KEY')
                logger.info("Got API key from Flask config")
        except (ImportError, RuntimeError):
            logger.debug("Could not get API key from Flask config")
            
    if not api_key:
        # Try to import directly from config file
        try:
            sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if sys_path not in sys.path:
                sys.path.append(sys_path)
            from web_application.config import Config
            if hasattr(Config, 'GOOGLE_API_KEY'):
                api_key = Config.GOOGLE_API_KEY
                logger.info("Got API key from config.py")
        except (ImportError, Exception) as e:
            logger.debug(f"Could not import from config.py: {e}")
            
    # Last resort - hardcoded default (not recommended for production)
    if not api_key:
        api_key = 'AIzaSyCNvHkGEo33oOYffvAkBql6kC9ty7ijklM'
        logger.warning("Using default API key - not recommended for production")
        
    return api_key

# Get the API key
GOOGLE_API_KEY = get_google_api_key()

# Try to import Agno
try:
    from agno.agent import Agent
    from agno.models.google import Gemini
    AGNO_AVAILABLE = True
    logger.info("Successfully imported Agno modules")
except ImportError as e:
    AGNO_AVAILABLE = False
    logger.warning(f"Agno import failed: {str(e)}. AI functionality will be limited.")
except Exception as e:
    AGNO_AVAILABLE = False
    logger.error(f"Error importing Agno: {str(e)}")
    logger.error(traceback.format_exc())

# Cache for agents to maintain state during a session
agent_cache = {}

def get_agno_agent(model_id='gemini-1.5-flash', session_id=None):
    """
    Create or retrieve an Agno agent instance.
    
    Args:
        model_id (str): The ID of the model to use.
        session_id (str, optional): Session ID for agent caching.
        
    Returns:
        Agent: An Agno agent instance.
        
    Raises:
        ImportError: If Agno is not installed
        ValueError: If an invalid model is specified
        Exception: For other errors
    """
    # Check if Agno is available
    if not AGNO_AVAILABLE:
        raise ImportError("Agno is not installed or failed to import. Install with: pip install agno")
    
    # Return cached agent if available
    if session_id and session_id in agent_cache:
        logger.info(f"Using cached agent for session {session_id}")
        return agent_cache[session_id]
    
    # Set available models
    available_models = {
        'gemini-1.5-flash': 'gemini-1.5-flash',
        'gemini-1.5-pro': 'gemini-1.5-pro',
        'gemini-1.0-pro': 'gemini-1.0-pro',
    }
    
    # Validate and get the correct model ID
    if model_id not in available_models:
        logger.warning(f"Invalid model ID: {model_id}. Using default model: gemini-1.5-flash")
        model_id = 'gemini-1.5-flash'
    
    try:
        # Create a new Agno agent
        agent = Agent(
            model=Gemini(id=available_models[model_id], api_key=GOOGLE_API_KEY),
            description="""You are an expert environmental and hydrological data assistant called the HydroGeo Assistant.
            You help users understand data sources, interpret results, and navigate the HydroGeo dataset explorer.
            Provide informative, accurate, and helpful responses about environmental and climate data,
            especially regarding PRISM, LOCA2, and Wellogic records. Your responses should be well-formatted
            with markdown when appropriate, including proper headings, lists, and code formatting.""",
            markdown=True
        )
        
        # Cache the agent if session_id is provided
        if session_id:
            agent_cache[session_id] = agent
            logger.info(f"Created and cached new agent for session {session_id}")
        else:
            logger.info(f"Created new agent (not cached)")
        
        return agent
    except Exception as e:
        logger.error(f"Error creating Agno agent: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def agno_respond(agent, prompt):
    """
    Get a response from an Agno agent.
    
    Args:
        agent (Agent): The Agno agent to use.
        prompt (str): The user's prompt.
        
    Returns:
        str: The agent's response.
        
    Raises:
        Exception: If there's an error generating a response
    """
    if not AGNO_AVAILABLE:
        raise ImportError("Agno is not installed or failed to import")
    
    try:
        # Simple log of incoming prompt
        logger.info(f"Processing prompt: {prompt[:50]}...")
        
        # First try using run() which should return the response as a string
        try:
            response = agent.run(prompt)
            if response and isinstance(response, str):
                logger.info(f"Generated response using agent.run(): {response[:50]}...")
                return response
        except (AttributeError, TypeError) as e:
            logger.warning(f"agent.run() method failed: {e}")
        
        # Next try chat() which should return a response in most newer Agno versions
        try:
            response = agent.chat(prompt)
            if response and isinstance(response, str):
                logger.info(f"Generated response using agent.chat(): {response[:50]}...")
                return response
        except (AttributeError, TypeError) as e:
            logger.warning(f"agent.chat() method failed: {e}")
        
        # Fallback to print_response but capture stdout
        try:
            import io
            import sys
            original_stdout = sys.stdout
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            agent.print_response(prompt, stream=False)
            
            sys.stdout = original_stdout
            response = captured_output.getvalue()
            
            # Extracting just the assistant's response from ANSI-colored output
            import re
            # Try to extract the actual response text, skipping ANSI codes and headers
            clean_response = re.sub(r'\x1b\[[0-9;]*m', '', response)  # Remove ANSI color codes
            response_lines = [line for line in clean_response.split('\n') if line.strip() and not line.strip().startswith('┏') and not line.strip().startswith('┗') and not line.strip().startswith('┃') and 'Response' not in line]
            
            if response_lines:
                cleaned_response = ' '.join(response_lines).strip()
                if cleaned_response:
                    logger.info(f"Generated response by capturing print_response stdout: {cleaned_response[:50]}...")
                    return cleaned_response
            
            # If we got some response but couldn't clean it properly, return the raw captured output
            if response.strip():
                logger.info(f"Generated raw response: {response[:50]}...")
                return response.strip()
            
        except Exception as inner_e:
            logger.error(f"Output capture failed: {str(inner_e)}")
        
        # Last resort: try any available method that might return a response
        for method_name in ['generate', 'ask', 'respond']:
            if hasattr(agent, method_name):
                try:
                    method = getattr(agent, method_name)
                    response = method(prompt)
                    if response and isinstance(response, str):
                        logger.info(f"Generated response using agent.{method_name}(): {response[:50]}...")
                        return response
                except Exception as method_e:
                    logger.warning(f"agent.{method_name}() failed: {method_e}")
        
        raise ValueError("Could not generate response using any available method")
        
    except Exception as e:
        logger.error(f"Error generating Agno response: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# Example usage, for testing - only runs when file is executed directly
if __name__ == "__main__":
    if not AGNO_AVAILABLE:
        print("ERROR: Agno is not available. Please install it with: pip install agno")
        exit(1)
        
    # Create an agent
    test_agent = get_agno_agent('gemini-1.5-flash')
    
    # Test the agent with a simple prompt
    test_response = agno_respond(test_agent, "Tell me about PRISM climate data.")
    
    # Print the response
    print(test_response) 