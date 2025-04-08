#!/usr/bin/env python3
"""
Test script to check if the Google API key is properly configured.
This is used to diagnose issues with the Agno integration.

Run this script directly to test the API key:
    python test_api_key.py
"""

import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api-key-test")

def main():
    # Test environment variable
    api_key_env = os.environ.get('GOOGLE_API_KEY')
    if api_key_env:
        logger.info(f"✅ GOOGLE_API_KEY found in environment: {api_key_env[:10]}...")
    else:
        logger.warning("❌ GOOGLE_API_KEY not found in environment variables")
    
    # Try importing from agno_hydrogeo.py
    logger.info("Testing API key retrieval from agno_hydrogeo.py:")
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from agno_hydrogeo import GOOGLE_API_KEY, get_google_api_key
        
        logger.info(f"✅ GOOGLE_API_KEY from agno_hydrogeo.py: {GOOGLE_API_KEY[:10]}...")
        
        # Test the getter function directly
        fresh_key = get_google_api_key()
        logger.info(f"✅ Fresh API key from get_google_api_key(): {fresh_key[:10]}...")
        
    except ImportError as e:
        logger.error(f"❌ Could not import from agno_hydrogeo.py: {e}")
    except Exception as e:
        logger.error(f"❌ Error accessing API key from agno_hydrogeo.py: {e}")
    
    # Try creating an Agno agent
    logger.info("Testing Agno agent creation...")
    try:
        from agno_hydrogeo import get_agno_agent
        
        agent = get_agno_agent()
        logger.info("✅ Successfully created Agno agent")
        
        # Try a simple test message
        try:
            from agno_hydrogeo import agno_respond
            response = agno_respond(agent, "Say hello")
            logger.info(f"✅ Agent response: {response[:50]}...")
        except Exception as e:
            logger.error(f"❌ Error getting agent response: {e}")
        
    except ImportError as e:
        logger.error(f"❌ Could not import Agno functions: {e}")
    except Exception as e:
        logger.error(f"❌ Error creating Agno agent: {e}")
    
    logger.info("Test completed")

if __name__ == "__main__":
    main() 