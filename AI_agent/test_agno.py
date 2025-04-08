"""
Test file for Agno integration.
This script tests the integration with Agno and the Gemini model.
Run this file directly to test the integration.
"""

import os
import sys
from agno_hydrogeo import get_agno_agent, agno_respond

def test_agno_integration():
    """Test the Agno integration by creating an agent and getting a response."""
    print("Testing Agno integration...")
    
    # Check if the Google API key is set
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        print("Warning: GOOGLE_API_KEY environment variable is not set.")
        print("Using default key from the script, which may not work.")
    
    try:
        # Create an agent
        print("Creating agent...")
        agent = get_agno_agent('gemini-1.5-flash')
        print("Agent created successfully!")
        
        # Test a simple prompt
        print("\nTesting prompt: Tell me about yourself")
        response = agno_respond(agent, "Tell me about yourself")
        print("\nResponse:")
        print("=" * 80)
        print(response)
        print("=" * 80)
        
        # Test a domain-specific prompt
        print("\nTesting prompt: What is PRISM climate data?")
        response = agno_respond(agent, "What is PRISM climate data?")
        print("\nResponse:")
        print("=" * 80)
        print(response)
        print("=" * 80)
        
        print("\nAgno integration test completed successfully!")
        return True
    except Exception as e:
        print(f"Error testing Agno integration: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Run the test
    success = test_agno_integration()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 