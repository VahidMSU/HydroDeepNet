#!/usr/bin/env python3
"""
Python replacement for run_ai.sh with better error handling and dependency management
"""

import os
import sys
import json
import argparse
import logging
import tempfile
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional, Union
import traceback

# Get the script directory
SCRIPT_DIR = Path(__file__).parent.absolute()
TOOLS_DIR = SCRIPT_DIR / "tools"
# Set the correct virtual environment path
VENV_PATH = Path("/data/SWATGenXApp/codes/.venv")
VENV_SITE_PACKAGES = VENV_PATH / "lib" / "python3.10" / "site-packages"

# Add directories to Python path
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(VENV_SITE_PACKAGES))

# Ensure logs directory exists
LOG_DIR = SCRIPT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Import the AI logger
from tools.ai_logger import LoggerSetup

# Set up logging using AI logger
logger_setup = LoggerSetup(report_path=str(LOG_DIR), verbose=True)
logger = logger_setup.setup_logger("run_ai")

class AIRunner:
    """Replacement for run_ai.sh functionality as a Python class"""
    
    def __init__(self):
        self.input_file = None
        self.output_file = None
        self.base_dir = "/data/SWATGenXApp/Users/admin/Reports/"
        self.model_id = "gpt-4o"
        self.interactive = True
        self.session_id = None
        self.database_path = None
        
    def parse_args(self) -> argparse.Namespace:
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(description="AI Agent Processing Script")
        parser.add_argument("--input", type=str, help="Path to input JSON file containing message")
        parser.add_argument("--output", type=str, help="Path to output JSON file for response")
        parser.add_argument("--dir", type=str, default=self.base_dir, help="Specify reports directory")
        parser.add_argument("--model", type=str, default=self.model_id, help="Specify model to use")
        parser.add_argument("--debug", action="store_true", help="Enable debug logging")
        
        args = parser.parse_args()
        
        if args.debug:
            logger.setLevel(logging.DEBUG)
            
        if args.input or args.output:
            if not (args.input and args.output):
                logger.error("Both --input and --output must be specified together")
                sys.exit(1)
            self.interactive = False
            
        self.input_file = args.input
        self.output_file = args.output
        self.base_dir = args.dir
        self.model_id = args.model
        
        return args
    
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are available"""
        logger.info("Checking module imports (dependencies should already be installed)")
        
        # Try to import required modules directly (no installation attempt)
        try:
            import ollama
            logger.info("Successfully imported ollama")
        except ImportError as e:
            logger.error(f"Error importing ollama: {e}")
            logger.error(f"PYTHONPATH: {sys.path}")
            return False
        
        try:
            import agno
            logger.info("Successfully imported agno")
        except ImportError as e:
            logger.error(f"Error importing agno: {e}")
            logger.error(f"PYTHONPATH: {sys.path}")
            return False
            
        # Try to load key modules from our tools directory
        try:
            from tools.integration import EnhancedReportAnalyzer
            logger.info("Successfully imported EnhancedReportAnalyzer")
            return True
        except ImportError as e:
            logger.error(f"Error importing EnhancedReportAnalyzer: {e}")
            # Try alternative import path
            try:
                from integration import EnhancedReportAnalyzer
                logger.info("Successfully imported EnhancedReportAnalyzer using alternative path")
                return True
            except ImportError as e2:
                logger.error(f"Error importing EnhancedReportAnalyzer from alternative path: {e2}")
                logger.error(f"PYTHONPATH: {sys.path}")
                return False
    
    def run_api_mode(self) -> bool:
        """Run in API mode - single query mode with output to file"""
        try:
            if not self.input_file or not os.path.exists(self.input_file):
                self._write_error_response("Input file does not exist")
                return False
            
            # Read the input file - first try as JSON then fall back to plain text
            try:
                with open(self.input_file, 'r') as f:
                    input_data = json.load(f)
                    
                # Extract data from JSON format if available
                if isinstance(input_data, dict):
                    query = input_data.get('message', '')
                    self.session_id = input_data.get('session_id', self.session_id)
                    self.model_id = input_data.get('model', self.model_id)
                else:
                    query = str(input_data)
            except json.JSONDecodeError:
                # Fall back to reading as plain text
                with open(self.input_file, 'r') as f:
                    query = f.read().strip()
            
            # Write a temporary processing message
            try:
                with open(self.output_file, 'w') as f:
                    json.dump({
                        "response": "Processing query...",
                        "status": "processing", 
                        "model": self.model_id,
                        "session_id": self.session_id
                    }, f)
            except Exception as e:
                logger.error(f"Error writing processing message: {str(e)}")
            
            # Handle list commands directly 
            if query.lower().startswith("list"):
                parts = query.lower().split()
                if len(parts) >= 2:
                    try:
                        # Use dir_discover directly instead of EnhancedReportAnalyzer
                        from tools.dir_discover import discover_reports
                        reports_dict = discover_reports(base_dir=self.base_dir, silent=True, recursive=True)
                        
                        if parts[1] == "reports":
                            # List all available reports
                            reports = sorted(reports_dict.keys()) if reports_dict else []
                            result = {"reports": reports}
                            response_text = f"Available reports: {', '.join(reports) if reports else 'None'}"
                        elif parts[1] == "groups" and reports_dict:
                            # List all available groups in current report
                            latest_report = sorted(reports_dict.keys())[-1] if reports_dict else None
                            groups = sorted(reports_dict[latest_report]["groups"].keys()) if latest_report else []
                            result = {"groups": groups}
                            response_text = f"Available groups in report '{latest_report}': {', '.join(groups) if groups else 'None'}"
                        else:
                            response_text = f"Unknown list command: {parts[1]}"
                            result = {"error": f"Unknown list command: {parts[1]}"}
                        
                        with open(self.output_file, 'w') as f:
                            json.dump({
                                "response": response_text,
                                "result": result,
                                "status": "success",
                                "model": self.model_id,
                                "session_id": self.session_id
                            }, f)
                        return True
                    except Exception as e:
                        logger.error(f"Error processing list command: {str(e)}", exc_info=True)
                        self._write_error_response(f"Error processing list command: {str(e)}")
                        return False
            
            # Process normal query
            try:
                response = self.process_query(query)
                with open(self.output_file, 'w') as f:
                    json.dump({
                        "response": response,
                        "status": "success",
                        "model": self.model_id,
                        "session_id": self.session_id
                    }, f)
                return True
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}", exc_info=True)
                self._write_error_response(f"Failed to process query: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"Unexpected error in API mode: {str(e)}", exc_info=True)
            self._write_error_response(f"Unexpected error occurred: {str(e)}")
            return False
    
    def run_interactive(self) -> bool:
        """Run in interactive mode"""
        logger.info("Starting Enhanced Hydrology Report Analyzer (interactive mode)")
        logger.info(f"Report directory: {self.base_dir}")
        
        try:
            # Import here to avoid dependency issues earlier
            from tools.integration import EnhancedReportAnalyzer
            
            # Initialize the report analyzer
            analyzer = EnhancedReportAnalyzer(
                base_dir=self.base_dir,
                model_id=self.model_id
            )
            
            # Start interactive session
            analyzer.interactive_session()
            return True
            
        except Exception as e:
            logger.error(f"Error in interactive mode: {str(e)}", exc_info=True)
            return False
    
    def _write_error_response(self, error_message: str) -> None:
        """Write a structured error response to the output file"""
        try:
            with open(self.output_file, 'w') as f:
                json.dump({
                    "response": error_message,
                    "status": "error",
                    "model": self.model_id,
                    "session_id": self.session_id
                }, f)
            logger.info(f"Wrote error response to {self.output_file}")
        except Exception as e:
            logger.error(f"Failed to write error response: {str(e)}", exc_info=True)
            # Last resort - try to create a valid JSON error response
            try:
                with open(self.output_file, 'w') as f:
                    f.write('{"response": "Critical error occurred", "status": "error"}')
            except:
                pass
    
    def process_query(self, query: str) -> str:
        """Process a user query by delegating to InteractiveReportAgent"""
        try:
            # Import the InteractiveReportAgent here to avoid circular imports
            from tools.interactive_agent import InteractiveReportAgent
            
            # Extract message content if it's a JSON object
            try:
                query_data = json.loads(query)
                if isinstance(query_data, dict) and 'message' in query_data:
                    query = query_data['message']
                    # Update session ID if provided
                    if 'session_id' in query_data:
                        self.session_id = query_data['session_id']
            except (json.JSONDecodeError, TypeError):
                # Not JSON or not a dict, use query as is
                pass
                
            logger.info(f"Processing query: {query[:100]}{'...' if len(query) > 100 else ''}")
            
            # Create the agent with the appropriate base directory
            agent = InteractiveReportAgent(
                base_dir=self.base_dir,
                logger=logger
            )
            
            # Process the query and return the response
            response = agent.process_query(query)
            return response
            
        except Exception as e:
            logger.error(f"Error in process_query: {str(e)}", exc_info=True)
            return f"Error processing your request: {str(e)}"
    
    def run(self) -> int:
        """Main entry point"""
        try:
            # Parse command line arguments
            self.parse_args()
            
            # Log Python environment information
            logger.info(f"Python version: {sys.version}")
            logger.info(f"Python executable: {sys.executable}")
            logger.info(f"PYTHONPATH: {sys.path}")
            
            # Check dependencies
            if not self.check_dependencies():
                if self.output_file:
                    self._write_error_response("Dependency check failed. See logs for details.")
                return 1
            
            # Run in appropriate mode
            if self.interactive:
                success = self.run_interactive()
            else:
                success = self.run_api_mode()
                
            return 0 if success else 1
            
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            return 130
        except Exception as e:
            logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
            if self.output_file:
                self._write_error_response(f"Unhandled exception: {str(e)}")
            return 1

def run_api_mode(input_file, output_file, model_name="gpt-4o", debug=False):
    """Run the AI assistant in API mode using specified input and output files."""
    # Setup logging
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('HydroGeo_AI')
    
    # Log some basic environment info for debugging
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    
    # Default response in case of errors
    response = {
        "response": "I'm sorry, I couldn't process your request. Please try again.",
        "model": model_name,
        "status": "error"
    }
    
    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            logger.error(f"Input file {input_file} does not exist")
            response["response"] = f"Error: Input file {input_file} not found"
            with open(output_file, 'w') as f:
                json.dump(response, f)
            return 1
            
        # Read input data from JSON file
        with open(input_file, 'r') as f:
            input_data = json.load(f)
        
        # Extract data
        query = input_data.get('message', '')
        session_id = input_data.get('session_id', 'default_session')
        model = input_data.get('model', model_name)
        
        logger.info(f"Processing query: '{query}' with model: {model} and session ID: {session_id}")
        
        # Handle list commands directly
        if query.lower().strip().startswith('list'):
            parts = query.lower().strip().split()
            
            # Write initial processing status
            with open(output_file, 'w') as f:
                json.dump({"status": "processing", "model": model}, f)
            
            if len(parts) == 1 or parts[1] == 'reports':
                # List all available reports
                try:
                    reports = []
                    reports_dir = './reports'
                    if os.path.exists(reports_dir):
                        reports = sorted([f for f in os.listdir(reports_dir) if os.path.isdir(os.path.join(reports_dir, f))])
                    
                    if reports:
                        response_msg = "Available reports:\n" + "\n".join(reports)
                    else:
                        response_msg = "No reports found. Please check the reports directory."
                    
                    response = {
                        "response": response_msg,
                        "model": model,
                        "session_id": session_id,
                        "status": "complete"
                    }
                except Exception as e:
                    logger.error(f"Error listing reports: {e}")
                    response["response"] = f"Error listing reports: {str(e)}"
            
            elif parts[1] == 'groups' or parts[1] == 'group':
                # List all available data groups for the most recent report
                try:
                    reports_dir = './reports'
                    reports = []
                    
                    if os.path.exists(reports_dir):
                        reports = sorted([f for f in os.listdir(reports_dir) if os.path.isdir(os.path.join(reports_dir, f))])
                    
                    if not reports:
                        response["response"] = "No reports found. Please check the reports directory."
                    else:
                        latest_report = reports[-1]  # Get the most recent report
                        report_path = os.path.join(reports_dir, latest_report)
                        
                        # Get all data groups in the report
                        groups = []
                        if os.path.exists(report_path):
                            groups = [f for f in os.listdir(report_path) 
                                     if os.path.isdir(os.path.join(report_path, f)) and not f.startswith('.')]
                        
                        if groups:
                            response["response"] = f"Available data groups in report '{latest_report}':\n" + "\n".join(groups)
                        else:
                            response["response"] = f"No data groups found in report '{latest_report}'."
                        
                        response["status"] = "complete"
                except Exception as e:
                    logger.error(f"Error listing groups: {e}")
                    response["response"] = f"Error listing groups: {str(e)}"
            
            else:
                # List specific group data
                group_name = parts[1]
                try:
                    reports_dir = './reports'
                    reports = []
                    
                    if os.path.exists(reports_dir):
                        reports = sorted([f for f in os.listdir(reports_dir) if os.path.isdir(os.path.join(reports_dir, f))])
                    
                    if not reports:
                        response["response"] = "No reports found. Please check the reports directory."
                    else:
                        latest_report = reports[-1]  # Get the most recent report
                        group_path = os.path.join(reports_dir, latest_report, group_name)
                        
                        if not os.path.exists(group_path):
                            # Check if group exists and provide suggestions
                            report_path = os.path.join(reports_dir, latest_report)
                            available_groups = []
                            if os.path.exists(report_path):
                                available_groups = [f for f in os.listdir(report_path) 
                                                  if os.path.isdir(os.path.join(report_path, f)) and not f.startswith('.')]
                            
                            if available_groups:
                                response["response"] = f"Group '{group_name}' not found in report '{latest_report}'.\n\nAvailable groups:\n" + "\n".join(available_groups)
                            else:
                                response["response"] = f"Group '{group_name}' not found and no available groups in report '{latest_report}'."
                        else:
                            # List files in the group
                            files = [f for f in os.listdir(group_path) if os.path.isfile(os.path.join(group_path, f))]
                            
                            if files:
                                response["response"] = f"Files in group '{group_name}' (report '{latest_report}'):\n" + "\n".join(files)
                            else:
                                response["response"] = f"No files found in group '{group_name}' (report '{latest_report}')."
                        
                        response["status"] = "complete"
                except Exception as e:
                    logger.error(f"Error listing group data: {e}")
                    response["response"] = f"Error accessing group '{group_name}': {str(e)}"
            
            # Write the response to the output file
            with open(output_file, 'w') as f:
                json.dump(response, f)
            
            return 0
        
        # Write initial processing status
        with open(output_file, 'w') as f:
            json.dump({"status": "processing", "model": model}, f)
        
        try:
            # Try to import the EnhancedReportAnalyzer
            try:
                from agnorafieiva.enhanced_report_analyzer import EnhancedReportAnalyzer
                logger.info("Successfully imported EnhancedReportAnalyzer")
            except ImportError:
                logger.warning("Failed to import from agnorafieiva package, trying alternative import path")
                sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
                from AI_agent.tools.enhanced_report_analyzer import EnhancedReportAnalyzer
                logger.info("Successfully imported EnhancedReportAnalyzer from alternative path")
            
            # Create the analyzer and process the query
            analyzer = EnhancedReportAnalyzer(model_name=model)
            
            try:
                # Try with session ID first
                logger.info(f"Processing query with session ID: {session_id}")
                result = analyzer.process_query(query, session_id=session_id)
            except AttributeError as e:
                if "'ContextMemory' object has no attribute 'restore_session'" in str(e):
                    # Handle the specific error when restore_session is not available
                    logger.warning(f"Session restoration not supported: {e}")
                    # Try processing the query without session_id
                    logger.info("Retrying query without session management")
                    result = analyzer.process_query(query)
                else:
                    # Re-raise other AttributeErrors
                    raise
            
            # Create the response
            response = {
                "response": result,
                "model": model,
                "session_id": session_id,
                "status": "complete"
            }
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error processing query: {error_message}")
            
            # Generate user-friendly error message based on the type of error
            if "restore_session" in error_message:
                friendly_error = "I'm having trouble with my session memory. My developers are working on it."
            elif "openai" in error_message.lower() or "ollama" in error_message.lower() or "model" in error_message.lower():
                friendly_error = "I'm having trouble connecting to my language model service. Please try again later."
            elif "import" in error_message.lower():
                friendly_error = "I'm having trouble loading some of my components. Please check your installation."
            else:
                friendly_error = "I encountered an error while processing your request. Please try again or rephrase your question."
            
            response = {
                "response": friendly_error,
                "model": model,
                "session_id": session_id,
                "status": "error",
                "error": error_message
            }
    except Exception as e:
        # Handle any other exceptions outside the main try/except
        error_message = str(e)
        logger.error(f"Unexpected error: {error_message}")
        logger.error(traceback.format_exc())
        
        response = {
            "response": "An unexpected error occurred. Please try again later.",
            "model": model_name,
            "status": "error",
            "error": error_message
        }
    
    # Write the final response to the output file
    with open(output_file, 'w') as f:
        json.dump(response, f)
    
    return 0  # Success

if __name__ == "__main__":
    runner = AIRunner()
    sys.exit(runner.run()) 