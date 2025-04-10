#!/usr/bin/env python3
"""
Test script for the InteractiveDocumentReader class.
"""

import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestLogger")

# Add parent directory to path to make imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from report_analyser_example.document_reader import InteractiveDocumentReader
except ImportError as e:
    logger.error(f"Error importing InteractiveDocumentReader: {str(e)}")
    sys.exit(1)

def main():
    """Run a simple test of the InteractiveDocumentReader."""
    logger.info("Starting test of InteractiveDocumentReader")
    
    try:
        # Initialize the reader
        reader = InteractiveDocumentReader()
        logger.info("Created InteractiveDocumentReader instance")
        
        # Initialize with auto-discovery
        base_path = '/data/SWATGenXApp/Users/admin/Reports/20250324_222749/'
        if not os.path.exists(base_path):
            logger.warning(f"Path does not exist: {base_path}")
            base_path = input("Enter a valid base path for document discovery: ")
        
        logger.info(f"Initializing with base path: {base_path}")
        reader.initialize(auto_discover=True, base_path=base_path)
        
        # Test a simple chat interaction
        response = reader.chat("Hello, can you help me understand what documents are available?")
        logger.info(f"Received response: {response}")
        
        # Test a more complex query
        response = reader.chat("What markdown files are available?")
        logger.info(f"Received response: {response}")
        
        # Test document search functionality
        response = reader.chat("Find information about solar energy")
        logger.info(f"Received response: {response}")
        
        # Test the OpenAI fallback by querying twice with the same question
        test_query = "What is the impact of climate change on agriculture?"
        logger.info("Testing OpenAI fallback with repeated query...")
        
        # First attempt - should use document search
        logger.info("First attempt:")
        response1 = reader.chat(test_query)
        logger.info(f"First response: {response1}")
        
        # Second attempt - should use OpenAI fallback
        logger.info("Second attempt (should use OpenAI fallback):")
        response2 = reader.chat(test_query)
        logger.info(f"Second response: {response2}")
        
        # Check if the responses are different
        if response1 != response2:
            logger.info("SUCCESS: Received different responses for repeated query, indicating OpenAI fallback worked")
        else:
            logger.warning("WARNING: Received identical responses for repeated query, OpenAI fallback may not be working")
        
        logger.info("Test completed successfully")
        
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 