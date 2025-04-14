import os
import sys
import re
import logging
import json
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.media import Image
from tools.dir_discover import discover_reports
import text_reader
import json_reader
import image_reader
import csv_reader
import website_reader as website_reader
import combine_reader
import ollama # Added ollama import
import traceback  # Add traceback for detailed error logging

# Import ContextMemory as the unified context management solution
from ContextMemory import ContextMemory

# Import the AI logger
from ai_logger import LoggerSetup

# Configure logging using the AI logger
LOG_PATH = '/data/SWATGenXApp/codes/AI_agent/logs'
logger_setup = LoggerSetup(report_path=LOG_PATH, verbose=True)
logger = logger_setup.setup_logger("interactive_agent")

class InteractiveReportAgent:
    """
    An interactive agent that can understand user queries and route them to specialized readers.
    Includes enhanced reasoning, report selection, and response capabilities.
    """
    def __init__(self, context=None, logger=None, base_dir="/data/SWATGenXApp/Users/admin/Reports/"):
        """
        Initialize the Interactive Report Agent with a base directory for reports.
        
        Args:
            context: Optional conversation context object
            logger: Optional logger for debugging
            base_dir: Base directory where reports are stored
        """
        # Basic initialization of state variables
        self.reports_dict = {}
        self.current_report = None
        self.current_group = None
        self.base_dir = base_dir
        
        # Initialize conversation context
        if context is None:
            try:
                from ContextMemory import ContextMemory
                self.context = ContextMemory(logger=logger)
            except ImportError:
                # Fallback to simple context
                self.context = self._create_default_context()
        else:
            self.context = context
        
        # Initialize logger
        if logger is None:
            self.logger = logging.getLogger(__name__)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
        else:
            self.logger = logger
            
        # Initialize state tracking variables
        self._awaiting_clarification = False
        self._clarification_type = None
        self._clarification_group = None
        self._awaiting_file_selection = False
        self._selected_file_type = None
        
        # Load available reports
        self._load_reports()
        
        # Initialize the reasoning agent with more detailed instructions
        self.agent = Agent(
            model=OpenAIChat(id="gpt-4o"),
            markdown=True,
            instructions=[
                "You are HydroInsight, an assistant specialized in analyzing environmental and water resources reports.",
                "You help users navigate and analyze groundwater, climate, land use, and other environmental data.",
                "Respond conversationally while providing detailed technical analysis when appropriate.",
                "Your goal is to help the user understand their data by directing them to appropriate analysis tools.",
                "Always consider the user's previous questions and your past responses for context.",
                "When analyzing data, focus on trends, patterns, and relationships between different environmental factors.",
                "Remember key insights from previous analyses to build a comprehensive understanding of the data.",
                "Be proactive in suggesting related analyses that might interest the user based on their questions."
            ],
        )
        
        # Initialize the query analysis agent for more sophisticated intent recognition
        # REMOVED self.query_analyzer Agent
        # self.query_analyzer = Agent(...)
        
        self.logger.info(f"Initialized with {len(self.reports_dict)} reports")
        self.logger.info(f"Current report set to: {self.current_report}")
        
        # Enhanced keyword mapping with more specific subcategories
        # REMOVED self.keyword_to_group dictionary
        # self.keyword_to_group = { ... }
        
        # Map of file types to readers with additional metadata
        self.filetype_to_reader = {
            ".csv": {"reader": self._read_csv, "description": "Tabular data analysis"},
            ".md": {"reader": self._read_text, "description": "Text report analysis"},
            ".txt": {"reader": self._read_text, "description": "Plain text analysis"},
            ".png": {"reader": self._read_image, "description": "Image and visualization analysis"},
            ".jpg": {"reader": self._read_image, "description": "Image and visualization analysis"},
            ".jpeg": {"reader": self._read_image, "description": "Image and visualization analysis"},
            "config.json": {"reader": self._read_json, "description": "Configuration settings analysis"},
            "combined": {"reader": self._read_combined, "description": "Comprehensive group analysis"},
            "website": {"reader": self._read_website, "description": "Web content analysis"}
        }
    
    def _load_reports(self):
        """
        Load available reports from the base directory using dir_discover.
        """
        try:
            # Use discover_reports to get the full structure
            self.reports_dict = discover_reports(base_dir=self.base_dir, silent=True, recursive=True)
            
            # If no reports found, log warning and create a dummy entry
            if not self.reports_dict:
                self.logger.warning(f"No reports found in {self.base_dir} by discover_reports")
                # Ensure a minimal structure exists to prevent downstream errors
                self.reports_dict = {} 
                self.current_report = None
            else:
                # Set latest report as default
                self.current_report = sorted(self.reports_dict.keys())[-1]
                self.logger.info(f"Loaded {len(self.reports_dict)} reports. Current report set to: {self.current_report}")
                
        except ImportError as ie:
             self.logger.error(f"Error importing discover_reports: {str(ie)}. Cannot load report structure.")
             self.reports_dict = {}
             self.current_report = None
        except Exception as e:
            self.logger.error(f"Error loading reports using discover_reports: {str(e)}")
            # Fallback to an empty structure
            self.reports_dict = {}
            self.current_report = None
            
    def _create_default_context(self):
        """
        Create a minimal default context when ContextMemory is not available.
        
        Returns:
            A simple object with the basic methods needed for context tracking.
        """
        class SimpleContext:
            def __init__(self):
                self.messages = []
                self.session_data = {
                    "viewed_files": [],
                    "viewed_groups": [],
                    "viewed_reports": [],
                    "last_query_time": None,
                    "insights": []
                }
            
            def add_user_message(self, message):
                self.messages.append({"role": "user", "content": message})
            
            def add_assistant_message(self, message, metadata=None):
                self.messages.append({"role": "assistant", "content": message})
            
            def get_recent_history(self, num_entries=5):
                return self.messages[-num_entries:] if self.messages else []
            
            def record_insight(self, insight, source):
                self.session_data["insights"].append({
                    "content": insight,
                    "source": source,
                    "timestamp": datetime.now().isoformat()
                })
                
        return SimpleContext()
    
    def list_reports(self) -> List[str]:
        """Return list of available reports"""
        return sorted(self.reports_dict.keys())
    
    def list_groups(self, report: Optional[str] = None) -> List[str]:
        """Return list of available groups in the report"""
        report_id = report or self.current_report
        if report_id and report_id in self.reports_dict and "groups" in self.reports_dict[report_id]:
            return sorted(self.reports_dict[report_id]["groups"].keys())
        self.logger.warning(f"Could not list groups for report '{report_id}'. Report not found or has no groups.")
        return []
    
    def list_files_by_type(self, group: Optional[str] = None, report: Optional[str] = None) -> Dict[str, List[str]]:
        """Return dictionary of files organized by type for a given group and report."""
        report_id = report or self.current_report
        group_id = group or self.current_group
        
        if not report_id or report_id not in self.reports_dict:
            self.logger.warning(f"Report '{report_id}' not found.")
            return {}
            
        if not group_id or group_id not in self.reports_dict[report_id].get("groups", {}):
            self.logger.warning(f"Group '{group_id}' not found in report '{report_id}'.")
            return {}
            
        # Access the files dictionary for the specified group
        files_dict = self.reports_dict[report_id]["groups"].get(group_id, {}).get("files", {})
        
        # Format the output to be {file_extension: [list_of_filenames]}
        formatted_files = {}
        for ext, files_data in files_dict.items():
            formatted_files[ext] = sorted(list(files_data.keys()))
            
        return formatted_files
    
    def set_current_report(self, report: str) -> bool:
        """Set the current report for analysis."""
        if report in self.reports_dict:
            self.current_report = report
            self.context.set_conversation_state("current_report", report)
            return True
        return False
    
    def set_current_group(self, group: str) -> bool:
        """Set the current group for analysis."""
        if self.current_report and group in self.reports_dict[self.current_report]["groups"]:
            self.current_group = group
            self.context.set_conversation_state("current_group", group)
            return True
        return False
    
    def _read_text(self, file_path: str) -> str:
        """Process text files using the text reader"""
        response = text_reader.text_reader(file_path)
        if response:
            # Extract key insights
            self._extract_and_save_insights(response, f"text:{os.path.basename(file_path)}")
        return response
    
    def _read_json(self, file_path: str) -> str:
        """Process JSON files using the JSON reader"""
        return json_reader.json_reader(file_path)
    
    def _read_image(self, file_path: str) -> str:
        """Process image files using the image reader"""
        response = image_reader.image_reader(file_path)
        if response:
            # Extract key insights
            self._extract_and_save_insights(response, f"image:{os.path.basename(file_path)}")
        return response
    
    def _read_csv(self, file_path: str) -> str:
        """Process CSV files using the CSV reader"""
        response = csv_reader.csv_reader(file_path, recreate_db_str="false")
        if response:
            # Extract key insights
            self._extract_and_save_insights(response, f"csv:{os.path.basename(file_path)}")
        return response
    
    def _read_website(self, url: str) -> str:
        """Process websites using the website reader"""
        return website_reader.website_reader(url)
    
    def _read_combined(self, report: str, group: str) -> str:
        """Process entire group using the combined reader"""
        response = combine_reader.combined_reader(self.reports_dict, report, group)
        if response:
            # Extract key insights
            self._extract_and_save_insights(response, f"group:{group}")
        return response
    
    def _extract_and_save_insights(self, text: str, source: str):
        """Extract key insights from text and save to context."""
        # Ask the agent to extract insights
        prompt = f"""
        Extract 3-5 key insights from the following analysis:
        
        {text[:2000]}  # Use first 2000 chars for brevity
        
        Return just the insights as a bulleted list without any introduction or conclusion.
        """
        
        insights = self.agent.print_response(prompt)
        
        if insights:
            # Save each insight separately
            for line in insights.split('\n'):
                line = line.strip()
                if line and line.startswith(('•', '-', '*')):
                    insight = line[1:].strip()
                    self.context.record_insight(insight, source)
    
    def process_file(self, file_name: str, file_type: str, group: Optional[str] = None, report: Optional[str] = None) -> str:
        """Process a specific file using the appropriate reader"""
        report_id = report or self.current_report
        group_id = group or self.current_group
        
        if not group_id:
            return "Please select a group first."
            
        if (report_id in self.reports_dict and 
            group_id in self.reports_dict[report_id]["groups"] and
            file_type in self.reports_dict[report_id]["groups"][group_id]["files"] and
            file_name in self.reports_dict[report_id]["groups"][group_id]["files"][file_type]):
            
            file_path = self.reports_dict[report_id]["groups"][group_id]["files"][file_type][file_name]["path"]
            
            # Record this file view in context
            self.context.record_file_view(file_name, file_type, group_id, report_id)
            
            if file_type in self.filetype_to_reader:
                reader_func = self.filetype_to_reader[file_type]["reader"]
                return reader_func(file_path)
            else:
                return f"No reader available for file type {file_type}"
        return f"File {file_name} not found in {group_id} group of report {report_id}"
    
    def analyze_group(self, group: Optional[str] = None, report: Optional[str] = None) -> str:
        """Analyze an entire group using the combined reader"""
        report_id = report or self.current_report
        group_id = group or self.current_group
        
        if not group_id:
            return "Please select a group first."
            
        if (report_id in self.reports_dict and 
            group_id in self.reports_dict[report_id]["groups"]):
            
            # Record this group view in context
            self.context.record_file_view("all_files", "combined", group_id, report_id)
            
            try:
                # More robust error handling
                result = self._read_combined(report_id, group_id)
                
                # Ensure we return a string even if _read_combined returns None
                if result is None:
                    return f"Analysis completed for {group_id}, but no results were returned."
                    
                return result
            except Exception as e:
                self.logger.error(f"Error analyzing group {group_id}: {str(e)}", exc_info=True)
                
                # Check for common errors
                error_str = str(e).lower()
                if "database" in error_str or "connection" in error_str:
                    return f"I encountered a database connection issue when analyzing the {group_id} group. This might be due to configuration issues with the database. Please try again later or try a different group."
                elif "file not found" in error_str or "no such file" in error_str:
                    return f"Some files in the {group_id} group could not be found. This might be due to file permission issues or missing files."
                else:
                    # For cdl group specifically, provide a more helpful message
                    if group_id == "cdl":
                        return f"The Crop Data Layer (CDL) analysis encountered an issue. Would you like me to analyze just one of the csv files in the cdl group instead of the entire group? This is often more reliable."
                    else:
                        return f"I encountered an issue while analyzing the {group_id} group. Would you like to try looking at individual files within this group instead?"
        
        return f"Group {group_id} not found in report {report_id}"
    
    def process_query(self, query: str) -> str:
        """
        Process a user query conversationally, attempting autonomous analysis.

        Args:
            query: The user's query/question

        Returns:
            Response text to the user
        """
        # 1. Log and Save Query
        self.logger.info(f"Processing user query: {query}")
        if hasattr(self.context, 'add_user_message'):
            self.context.add_user_message(query)
        elif hasattr(self.context, 'add_message'):
            self.context.add_message("user", query)

        try:
            # --- ADD THIS CHECK --- 
            # Handle simple greetings directly before complex routing
            if query.strip().lower() in ["hi", "hello", "hey", "greetings"]:
                self.logger.info("Handling simple greeting directly.")
                final_response = "Hello! How can I help you analyze your hydrology reports today?"
                if hasattr(self.context, 'add_assistant_message'):
                    self.context.add_assistant_message(final_response)
                elif hasattr(self.context, 'add_message'):
                    self.context.add_message("assistant", final_response)
                return final_response
            # --- END ADDED CHECK --- 

            # 2. Handle History Request (Simple Keyword Check)
            history_keywords = ["what did we talk about", "conversation history", "show history", "history please"]
            if any(keyword in query.lower() for keyword in history_keywords):
                history = self.context.get_formatted_history(num_entries=10)
                if not history:
                    response = "We haven't talked about anything yet in this session."
                else:
                    response = "Here is the recent conversation history:\n\n" + history
                if hasattr(self.context, 'add_assistant_message'):
                    self.context.add_assistant_message(response)
                elif hasattr(self.context, 'add_message'):
                    self.context.add_message("assistant", response)
                return response

            # 3. Handle System Commands (Simple Keyword Check)
            system_commands = ["list reports", "list groups", "set report", "set group", "help", "exit", "quit"]
            if any(query.lower().startswith(cmd) for cmd in system_commands):
               response = self.handle_system_command(query)
               if hasattr(self.context, 'add_assistant_message'):
                     self.context.add_assistant_message(response)
               elif hasattr(self.context, 'add_message'):
                     self.context.add_message("assistant", response)
               return response

            # 4. Gather Context for LLM
            available_reports = self.list_reports()
            current_report_id = self.current_report or "None"
            available_groups = self.list_groups() # Defaults to current report if set
            current_group_id = self.current_group or "None"
            files_in_current_group = []
            if self.current_group:
                files_in_current_group = self.list_files_in_group(self.current_group)
            conversation_history = self.context.get_recent_history(num_entries=5)

            # 5. Construct *Modified* Ollama Prompt
            prompt_messages = [
                {
                    'role': 'system',
                    'content': f"""
You are HydroInsight Router, an AI assistant routing requests for a hydrology report analysis system.
Your goal is to understand the user's request and determine the MOST relevant data and the type of analysis needed. Avoid asking clarifying questions unless absolutely necessary.

Available reports (timestamps): {available_reports}
Available groups in the current report ('{current_report_id}'): {available_groups}
Available files in the current group ('{current_group_id}'): {files_in_current_group}

Current selected report: '{current_report_id}'
Current selected group: '{current_group_id}'

Based on the user query and conversation history, determine the action and target data.

Possible actions:
- 'perform_analysis': User wants analysis, insights, trends, summaries, or answers about the data.
- 'select_group': User explicitly wants to switch focus to a specific group.
- 'select_report': User explicitly wants to switch focus to a specific report.
- 'list_data_summary': User wants to know what kind of data is available (summary, not raw list).
- 'general_conversation': A general question or comment not specific to data actions.

Respond ONLY with a JSON object containing:
- 'action': (string) One of the possible actions listed above.
- 'target_report': (string or null) The most relevant report timestamp (use current if unsure or not specified).
- 'target_group': (string or null) The most relevant data group (use current if unsure or not specified).
- 'target_file': (string or null) A *specific relevant* filename if the query clearly points to one. Often null.
- 'analysis_focus': (string or null) Keywords indicating the analysis type (e.g., "trends", "summary", "comparison", "rainfall data", "well levels"). Only relevant for 'perform_analysis'.
- 'explanation': (string) Brief reasoning.

Prioritize the current report/group if not specified. If the query asks for analysis or information, lean towards 'perform_analysis'.
If the user asks "what data do you have about X", use 'list_data_summary'.
Only use 'select_group' or 'select_report' if the user explicitly uses 'set', 'switch to', 'focus on', etc.
If unsure about the target group/file for analysis, provide the best guess based on context and keywords.
"""
                }
            ]
            # Add conversation history...
            for msg in conversation_history:
                 prompt_messages.append({'role': msg['role'], 'content': msg['content']})
            # Add the current user query...
            prompt_messages.append({'role': 'user', 'content': query})

            # 6. Call Ollama
            try:
                self.logger.info(f"Sending request to Ollama. Model: mistral:latest") # Specify model
                response_ollama = ollama.chat(
                    model='mistral:latest', # Use your desired model
                    messages=prompt_messages,
                    format='json' # Request JSON output
                )
                llm_output_str = response_ollama['message']['content']
                self.logger.info(f"Ollama response content: {llm_output_str}")
                parsed_response = json.loads(llm_output_str)
                self.logger.info(f"Ollama parsed response: {parsed_response}")
            except Exception as e:
                self.logger.error(f"Error calling Ollama: {str(e)}")
                self.logger.error(traceback.format_exc())
                final_response = "I'm having trouble connecting to my language model. This might be a temporary issue. Could you try again?"
                if hasattr(self.context, 'add_assistant_message'):
                    self.context.add_assistant_message(final_response)
                elif hasattr(self.context, 'add_message'):
                    self.context.add_message("assistant", final_response)
                return final_response

            # 7. *Revised* Route Based on LLM Response
            action = parsed_response.get('action')
            target_report = parsed_response.get('target_report')
            target_group = parsed_response.get('target_group')
            target_file = parsed_response.get('target_file')
            analysis_focus = parsed_response.get('analysis_focus') # New field
            
            self.logger.info(f"Taking action: {action}")
            self.logger.info(f"Target report: {target_report}")
            self.logger.info(f"Target group: {target_group}")
            self.logger.info(f"Target file: {target_file}")
            self.logger.info(f"Analysis focus: {analysis_focus}")

            # --- Auto-select Report/Group based on LLM suggestion or context ---
            # If LLM suggested a report, try to set it.
            if target_report and target_report != "None" and target_report != self.current_report:
                if self.set_current_report(target_report):
                    self.logger.info(f"LLM suggested report change. Set current report to: {target_report}")
                    # Reset group if report changes, let LLM/heuristics find the new target group
                    self.current_group = None
                    target_group = parsed_response.get('target_group') # Re-evaluate suggested group
                else:
                    self.logger.warning(f"LLM suggested invalid report '{target_report}'. Keeping '{self.current_report}'.")
                    target_report = self.current_report # Use current if suggestion invalid

            # If LLM suggested a group, try to set it.
            if target_group and target_group != "None" and target_group != self.current_group:
                if self.set_current_group(target_group):
                     self.logger.info(f"LLM suggested group change. Set current group to: {target_group}")
                else:
                    # If LLM's group suggestion was invalid for the *current* report, keep the current group (or None)
                    self.logger.warning(f"LLM suggested invalid group '{target_group}' for report '{self.current_report}'. Keeping group as '{self.current_group}'.")
                    target_group = self.current_group # Fallback to current group

            # If no group is targeted/set by now, and an action needs one, try to infer
            if not target_group and action in ['perform_analysis', 'list_data_summary']:
                 target_group = self._infer_group(query) # Add a helper to infer group
                 if target_group:
                     self.set_current_group(target_group)
                     self.logger.info(f"Inferred and set current group to: {target_group}")
                 else:
                     # If we still can't determine a group, we HAVE to ask (last resort)
                     final_response = self.ask_for_group_clarification()
                     # Save and Return Response
                     if final_response is None:
                        final_response = "I received an empty response from the action handler. Please check the logs."
                     if hasattr(self.context, 'add_assistant_message'):
                         self.context.add_assistant_message(final_response)
                     elif hasattr(self.context, 'add_message'):
                         self.context.add_message("assistant", final_response)
                     return final_response

            # --- Execute Revised Actions ---
            final_response = ""
            try:
                if action == 'perform_analysis':
                    if not self.current_group:
                        # This case should be rare after inference, but handle it.
                        final_response = self.ask_for_group_clarification()
                    else:
                        self.logger.info(f"Performing analysis on group: {self.current_group}")
                        final_response = self._handle_analysis_request(
                            self.current_group,
                            target_file,      # Specific file hint from LLM
                            analysis_focus    # Analysis keywords hint from LLM
                        )
                        self.logger.info(f"Analysis complete. Response length: {len(final_response) if final_response else 0}")

                elif action == 'list_data_summary':
                     if not self.current_group:
                         final_response = self.ask_for_group_clarification()
                     else:
                         self.logger.info(f"Generating data summary for group: {self.current_group}")
                         final_response = self._summarize_group_content(self.current_group, query)

                elif action == 'select_group':
                     # Group should have been set above if valid and different from current
                     if self.current_group == target_group and target_group is not None:
                         final_response = f"Okay, focusing on group '{self.current_group}'."
                         # Optional: Proactively provide a summary after selection
                         final_response += "\n" + self._summarize_group_content(self.current_group, query)
                     elif target_group is None: # Invalid group suggested
                          final_response = self.ask_for_group_clarification() # Fallback if selection failed
                     # If group was set successfully but wasn't the target_group initially, no extra message needed.

                elif action == 'select_report':
                     # Report should have been set above if valid
                     if self.current_report == target_report and target_report is not None:
                         final_response = f"Okay, focusing on report '{self.current_report}'. Available groups: {', '.join(self.list_groups())}"
                     elif target_report is None:
                          final_response = self.ask_for_report_clarification() # Fallback if selection failed

                elif action == 'general_conversation':
                    # Use a more general approach for simple chat
                    if query.strip().lower() in ["hi", "hello", "hey", "greetings"]:
                        final_response = "Hello! How can I help you analyze your hydrology reports today?"
                    # Handle identity questions directly without using the main agent
                    elif any(identity_q in query.strip().lower() for identity_q in ["who are you", "who are u", "what are you", "what is your name", "introduce yourself"]):
                        final_response = "I'm HydroInsight, an AI assistant specialized in analyzing environmental and water resources data. I can help you explore and understand groundwater, climate, land use, and other environmental reports. How can I assist you today?"
                    else:
                        # Fallback to the main reasoning agent for more complex general queries
                        try:
                            self.logger.info("Handling as general query with the main HydroInsight agent.")
                            final_response = self.agent.print_response(query) # Pass query to the base agent
                        except Exception as e:
                            self.logger.error(f"Error using main agent for general conversation: {e}")
                            self.logger.error(traceback.format_exc())
                            # Fallback response if main agent fails
                            final_response = "I'm HydroInsight, an assistant specialized in environmental and water data analysis. I can help you explore reports, analyze data, and understand hydrology information. What specific aspect of your data would you like to explore?"

                else: # Unknown action
                     self.logger.warning(f"LLM returned unknown action: {action}. Treating as general conversation.")
                     # Also use the general approach here for unknown actions
                     if query.strip().lower() in ["hi", "hello", "hey", "greetings"]:
                        final_response = "Hello! How can I help you with your reports?"
                     # Handle identity questions directly without using the main agent
                     elif any(identity_q in query.strip().lower() for identity_q in ["who are you", "who are u", "what are you", "what is your name", "introduce yourself"]):
                        final_response = "I'm HydroInsight, an AI assistant specialized in analyzing environmental and water resources data. I can help you explore and understand groundwater, climate, land use, and other environmental reports. How can I assist you today?"
                     else:
                        self.logger.info("Handling unknown action as general query with the main HydroInsight agent.")
                        try:
                            final_response = self.agent.print_response(query) # Fallback
                        except Exception as e:
                            self.logger.error(f"Error using main agent for unknown action: {e}")
                            self.logger.error(traceback.format_exc())
                            # Fallback response if main agent fails
                            final_response = "I'm HydroInsight, an assistant specialized in environmental and water data analysis. I can help you explore reports, analyze data, and understand hydrology information. What specific aspect of your data would you like to explore?"

            except Exception as e:
                # Log the full stack trace for debugging
                self.logger.error(f"Error executing action '{action}': {str(e)}")
                self.logger.error(traceback.format_exc())
                final_response = f"Sorry, I encountered an error while trying to perform the action: {action}. Please try again."

            # 8. Save and Return Response
            if final_response is None:
                 self.logger.error("Received None response from action handler")
                 final_response = "I received an empty response from the action handler. Please check the logs."
                 
            if hasattr(self.context, 'add_assistant_message'):
                self.context.add_assistant_message(final_response)
            elif hasattr(self.context, 'add_message'):
                self.context.add_message("assistant", final_response)
                
            return final_response
            
        except Exception as e:
            # Catch-all exception handler to prevent crashes
            self.logger.error(f"Unexpected error in process_query: {str(e)}")
            self.logger.error(traceback.format_exc())
            error_response = "I encountered an unexpected error. This has been logged for investigation. Please try again or try a different question."
            
            # Try to save to context if possible
            try:
                if hasattr(self.context, 'add_assistant_message'):
                    self.context.add_assistant_message(error_response)
                elif hasattr(self.context, 'add_message'):
                    self.context.add_message("assistant", error_response)
            except:
                pass
                
            return error_response

    # --- New/Modified Helper Methods ---

    def _infer_group(self, query: str) -> Optional[str]:
        """Try to infer the target group based on query keywords and available groups."""
        if not self.current_report: return None # Cannot infer without a report

        available_groups = self.list_groups()
        if not available_groups: return None

        query_lower = query.lower()
        possible_matches = []
        for group in available_groups:
            # Simple keyword matching (can be improved with fuzzy matching or embeddings)
            if group.lower() in query_lower:
                possible_matches.append(group)

        if len(possible_matches) == 1:
            return possible_matches[0]
        # TODO: Add more sophisticated matching if needed
        return None # Cannot confidently infer

    def _summarize_group_content(self, group: str, query: str) -> str:
         """Provide a summary of the types of data available in the group."""
         files_by_type = self.list_files_by_type(group)
         if not files_by_type:
             return f"Group '{group}' exists but seems to be empty."

         summary_lines = [f"In the '{group}' group, I have the following types of data:"]
         file_type_descriptions = {
             ".csv": "CSV data files, likely containing tabular data (e.g., time series, statistics).",
             ".md": "Markdown reports, usually containing narrative descriptions, analysis, and summaries.",
             ".txt": "Plain text files, which could contain logs, notes, or simple data.",
             ".png": "PNG image files, typically visualizations like plots, maps, or charts.",
             ".jpg": "JPG image files, similar to PNGs, used for visualizations.",
             ".jpeg": "JPEG image files, similar to PNGs, used for visualizations.",
             "config.json": "JSON configuration files detailing settings used for generating the report.",
             # Add other common types if needed
         }

         for ftype, flist in files_by_type.items():
             desc = file_type_descriptions.get(ftype, f"{ftype} files.")
             summary_lines.append(f"- {len(flist)} {desc}")

         # Optional: Analyze the most relevant file based on the original query
         # This makes 'list_data_summary' more proactive
         focus = query # Use original user query to find relevant file
         relevant_file_summary = self._find_and_analyze_most_relevant(group, focus)
         if relevant_file_summary:
              summary_lines.append("\nBased on your query, here's an analysis of the most relevant item:")
              summary_lines.append(relevant_file_summary)
         else:
              summary_lines.append("\nWhat specific aspect of this group would you like me to analyze?")


         return "\n".join(summary_lines)

    def _find_most_relevant_file(self, group: str, analysis_focus: Optional[str]) -> Optional[Tuple[str, str]]:
        """Find the most relevant file in a group based on focus keywords."""
        if not analysis_focus: return None # Cannot find relevance without focus

        files = self.list_files_in_group(group)
        if not files: return None

        focus_terms = set(analysis_focus.lower().split())
        best_match = None
        max_score = -1

        for filename in files:
             # Score based on matching terms in filename
             score = sum(1 for term in focus_terms if term in filename.lower())
             # Optional: Boost score based on file type relevance (e.g., CSV for 'data', PNG for 'plot')
             ext = os.path.splitext(filename)[1].lower()
             if ext == ".csv" and any(t in focus_terms for t in ["data", "table", "stats", "statistics", "csv"]):
                 score += 2
             if ext in [".png", ".jpg", ".jpeg"] and any(t in focus_terms for t in ["plot", "chart", "map", "figure", "visualization", "image", "png"]):
                 score += 2
             if ext in [".md", ".txt"] and any(t in focus_terms for t in ["report", "summary", "text", "narrative", "md", "document"]):
                 score += 1


             if score > max_score:
                 max_score = score
                 best_match = (filename, ext)

        if max_score > 0:
            self.logger.info(f"Most relevant file for focus '{analysis_focus}' in group '{group}' determined to be: {best_match[0]}")
            return best_match
        else:
             self.logger.warning(f"Could not determine a relevant file for focus '{analysis_focus}' in group '{group}'.")
             return None # No relevant file found based on score

    def _find_and_analyze_most_relevant(self, group: str, analysis_focus: Optional[str]) -> Optional[str]:
         """Helper to find the most relevant file and analyze it."""
         relevant_file_info = self._find_most_relevant_file(group, analysis_focus)
         if relevant_file_info:
             file_name, file_type = relevant_file_info
             # Check it actually exists in our structure before processing
             file_path = self.get_file_path(group, file_name)
             if file_path:
                  self.logger.info(f"Analyzing most relevant file: {file_name}")
                  # Call process_file directly (which uses the specific readers)
                  return self.process_file(file_name, file_type, group)
         return None


    def _handle_analysis_request(self, group: str, target_file_hint: Optional[str], analysis_focus: Optional[str]) -> str:
         """Handle the 'perform_analysis' action."""
         # Special handling for cdl group which often has issues with combined analysis
         if group == "cdl" and not target_file_hint:
             self.logger.info("Using special handling for cdl group - analyzing individual files instead of whole group")
             # Try to find a CSV file in the cdl group to analyze
             files_by_type = self.list_files_by_type(group)
             if '.csv' in files_by_type and files_by_type['.csv']:
                 cdl_csv = files_by_type['.csv'][0]  # Get the first CSV file
                 self.logger.info(f"Automatically selecting {cdl_csv} from cdl group for analysis")
                 file_path = self.get_file_path(group, cdl_csv)
                 if file_path:
                     result = self.process_file(cdl_csv, '.csv', group)
                     return result + "\n\nNote: I analyzed this specific file because it's often more reliable than analyzing the entire CDL group at once."
         
         # Priority 1: Analyze specific file if hinted by LLM and it exists
         if target_file_hint:
             file_path = self.get_file_path(group, target_file_hint)
             if file_path:
                 self.logger.info(f"LLM suggested specific file: {target_file_hint}. Analyzing.")
                 file_ext = os.path.splitext(target_file_hint)[1].lower()
                 try:
                     return self.process_file(target_file_hint, file_ext, group)
                 except Exception as e:
                     self.logger.error(f"Error processing file {target_file_hint}: {str(e)}", exc_info=True)
                     return f"I encountered an issue analyzing {target_file_hint}. Error: {str(e)}"
             else:
                 self.logger.warning(f"LLM suggested file '{target_file_hint}' but it was not found in group '{group}'.")

         # Priority 2: Find and analyze the *most relevant* file based on analysis_focus
         try:
             relevant_analysis = self._find_and_analyze_most_relevant(group, analysis_focus)
             if relevant_analysis:
                 # For broad queries, add follow-up suggestions
                 if analysis_focus and len(analysis_focus.split()) < 5:
                     # This is likely a broad query
                     files_by_type = self.list_files_by_type(group)
                     sample_files = []
                     
                     # Get a sample of other relevant files (up to 3)
                     for ext, file_list in files_by_type.items():
                         if ext in ['.csv', '.md', '.txt', '.png', '.jpg', '.jpeg'] and file_list:
                             for file in file_list[:1]:  # Take just 1 from each type
                                 sample_files.append((file, ext))
                                 if len(sample_files) >= 3:
                                     break
                         if len(sample_files) >= 3:
                             break
                     
                     # Add suggestions for follow-up questions
                     if sample_files:
                         follow_up = "\n\nTo explore this topic further, you could ask about:"
                         for file, ext in sample_files:
                             if ext == '.csv':
                                 follow_up += f"\n- Trends or patterns in the data from {file}"
                             elif ext in ['.png', '.jpg', '.jpeg']:
                                 follow_up += f"\n- Details about the visualization in {file}"
                             elif ext in ['.md', '.txt']:
                                 follow_up += f"\n- Key findings from the report {file}"
                         
                         follow_up += "\n\nOr you could ask a more specific question about this topic."
                         return relevant_analysis + follow_up
                 
                 return relevant_analysis
         except Exception as e:
             self.logger.error(f"Error finding relevant file for analysis: {str(e)}", exc_info=True)
             # Continue to next approach if this fails

         # Priority 3: For broad queries, analyze a SAMPLE of files instead of the whole group
         try:
             if analysis_focus:
                 self.logger.info(f"Broad query detected. Analyzing a representative sample for '{analysis_focus}'.")
                 
                 # Get available file types in this group
                 files_by_type = self.list_files_by_type(group)
                 if not files_by_type:
                     return f"Group '{group}' exists but seems to be empty."
                 
                 # First, provide an overview of available data
                 summary = f"I found several files related to '{analysis_focus}' in the '{group}' group. "
                 summary += "Here's what I can tell you based on a sample of the data:\n\n"
                 
                 # Analyze up to 3 representative files of different types
                 analyzed_files = []
                 insights = []
                 
                 # Prioritize CSV and text files for initial analysis
                 for priority_ext in ['.csv', '.md', '.txt', '.png']:
                     if priority_ext in files_by_type and files_by_type[priority_ext]:
                         # Take the first file of this type
                         file = files_by_type[priority_ext][0]
                         file_path = self.get_file_path(group, file)
                         if file_path:
                             try:
                                 # Process the file but don't return yet
                                 result = self.process_file(file, priority_ext, group)
                                 if result:
                                     # Extract a shorter summary for the combined response
                                     short_summary = f"From {file} ({priority_ext}): " + result.split('\n\n')[0]
                                     insights.append(short_summary)
                                     analyzed_files.append(file)
                             except Exception as e:
                                 self.logger.error(f"Error processing sample file {file}: {str(e)}", exc_info=True)
                                 # Continue with other files even if one fails
                         
                         if len(analyzed_files) >= 2:  # Limit to 2 files for brevity
                             break
                 
                 # Add the insights to the summary
                 if insights:
                     summary += "\n\n".join(insights)
                     
                     # Add suggestions for more specific questions
                     summary += "\n\nTo get more specific insights, you could ask about:"
                     for file in analyzed_files:
                         file_name = os.path.splitext(file)[0]
                         summary += f"\n- More details about {file}"
                     
                     # Suggest analyzing the whole group if needed
                     summary += f"\n- A comprehensive analysis of all data in the {group} group"
                     summary += f"\n- Specific trends or patterns related to {analysis_focus}"
                     
                     return summary
         except Exception as e:
             self.logger.error(f"Error analyzing sample files: {str(e)}", exc_info=True)
             # Continue to next approach if this fails
         
         # Priority 4: Last resort - analyze the whole group
         self.logger.info(f"No specific file focus determined and partial analysis not applicable. Analyzing the whole group '{group}'.")
         try:
             return self.analyze_group(group) # analyze_group calls combine_reader
         except Exception as e:
             self.logger.error(f"Error in last resort whole group analysis: {str(e)}", exc_info=True)
             return f"I encountered an issue analyzing the {group} group. I recommend trying to look at specific files within this group instead of the entire group at once."

    # --- Helper methods for clarification (used as fallback) ---
    def ask_for_group_clarification(self) -> str:
        """Asks the user to clarify which group they want."""
        groups = self.list_groups()
        if not groups:
            return "No data groups are currently available in this report."
        group_list = ", ".join(groups)
        return f"Which data group are you interested in? Available groups: {group_list}."

    def ask_for_file_clarification(self, group: str) -> str:
        """Asks the user to clarify which file they want."""
        files = self.list_files_in_group(group)
        if not files:
            return f"Group '{group}' doesn't seem to contain any files."

        # Maybe list by type for clarity?
        files_by_type = self.list_files_by_type(group)
        response_lines = [f"Which file in group '{group}' would you like to analyze?"]
        for ftype, flist in files_by_type.items():
            response_lines.append(f"  {ftype} files:")
            for fname in flist[:5]: # Show max 5 per type
                response_lines.append(f"    - {fname}")
            if len(flist) > 5:
                 response_lines.append(f"    ... ({len(flist) - 5} more)")
        return "\n".join(response_lines)

    def ask_for_report_clarification(self) -> str:
        """Asks the user to clarify which report they want."""
        reports = self.list_reports()
        if not reports:
             return "No reports have been found."
        report_list = ", ".join(reports)
        return f"Which report would you like to work with? Available reports: {report_list}."

    def handle_system_command(self, query: str) -> str:
        """Handle system commands like list, set, help, exit."""
        query_lower = query.lower()
        parts = query.split()

        if query_lower.startswith("list reports"):
            return f"Available reports: {', '.join(self.list_reports())}"
        elif query_lower.startswith("list groups"):
            groups = self.list_groups()
            if not groups:
                 return f"No groups found in the current report ('{self.current_report}')."
            return f"Available groups in '{self.current_report}': {', '.join(groups)}"
        elif query_lower.startswith("set report") and len(parts) > 2:
            report_id = parts[2]
            if self.set_current_report(report_id):
                self.current_group = None # Reset group when report changes
                return f"Current report set to '{report_id}'. Available groups: {', '.join(self.list_groups())}"
            else:
                return f"Report '{report_id}' not found. Available reports: {', '.join(self.list_reports())}"
        elif query_lower.startswith("set group") and len(parts) > 2:
            group_id = parts[2]
            if not self.current_report:
                return "Please set a report first using 'set report [report_id]'."
            if self.set_current_group(group_id):
                return f"Current group set to '{group_id}' within report '{self.current_report}'."
            else:
                return f"Group '{group_id}' not found in report '{self.current_report}'. Available groups: {', '.join(self.list_groups())}"
        elif query_lower.startswith("help"):
             return self.explain_capabilities() # Use existing method for help
        elif query_lower.startswith(("exit", "quit")):
             return "Exiting... (Use the interactive shell's exit command)"
        else:
            return f"Unknown command: {query}"

    def get_file_types_in_group(self, group: str) -> List[str]:
        """
        Get a list of unique file types in a group.
        
        Args:
            group: The group to check
            
        Returns:
            List of file extensions (e.g., ['.csv', '.txt'])
        """
        files = self.list_files_in_group(group)
        file_types = set()
        
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext:  # Only add non-empty extensions
                file_types.add(ext)
        
        return sorted(list(file_types))

    def suggest_files_for_analysis(self, group: str, query: str, file_types: List[str]) -> List[str]:
        """Suggest specific files that might be relevant for the analysis."""
        files = self.list_files_in_group(group)
        suggested = []
        
        # Extract key terms from the query
        query_terms = set(query.lower().split())
        
        # Score files based on query term matches in filename
        scored_files = []
        for file in files:
            file_lower = file.lower()
            score = sum(1 for term in query_terms if term in file_lower)
            
            # Boost score for files matching mentioned file types
            file_ext = os.path.splitext(file)[1]
            for file_type in file_types:
                if file_type in query.lower() and file_ext == file_type:
                    score += 3
                    
            scored_files.append((file, score))
            
        # Sort by score descending and take top matches
        scored_files.sort(key=lambda x: x[1], reverse=True)
        suggested = [file for file, score in scored_files if score > 0][:5]
        
        # If no matches by score, include some representative files of each type
        if not suggested:
            for file_type in file_types:
                type_matches = [f for f in files if f.endswith(file_type)]
                if type_matches:
                    suggested.append(type_matches[0])
                    
            # Limit to 5 suggestions
            suggested = suggested[:5]
            
        return suggested

    def list_files_in_group(self, group: Optional[str] = None, report: Optional[str] = None) -> List[str]:
        """
        List all unique filenames within a specific group and report.
        
        Args:
            group: The group name
            report: The report name (optional, defaults to current)
            
        Returns:
            List of unique filenames in the group
        """
        report_id = report or self.current_report
        group_id = group or self.current_group
        
        files_by_type = self.list_files_by_type(group=group_id, report=report_id)
        
        all_files = set()
        for ext, file_list in files_by_type.items():
            for file_name in file_list:
                all_files.add(file_name)
                
        return sorted(list(all_files))
        
    def analyze_file(self, file_path: str) -> str:
        """
        Analyze a specific file based on its type.
        
        Args:
            file_path: The path to the file to analyze
            
        Returns:
            Analysis results
        """
        if not os.path.exists(file_path):
            return f"File not found: {file_path}"
            
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].lower()
        group = os.path.dirname(file_path)
        
        # Record file view in context
        self.context.record_file_view(file_name, file_ext, group, self.current_report)
        
        if file_ext == '.csv':
            # Note: In a real implementation, these would be actual module imports
            # For this example, we'll simulate the analysis
            return f"Analysis of CSV file {file_name}:\n\nThis CSV file contains 1024 rows and 15 columns.\nThe columns include: ID, Name, Date, Value, Category.\nThe data spans from 2022-01 to 2023-12.\nAverage value: 42.5\nMost common category: 'A'"
            
        elif file_ext in ['.txt', '.md']:
            # Simulate text file analysis
            return f"Analysis of {file_ext} file {file_name}:\n\nThis document contains approximately 2500 words.\nKey topics identified: data analysis, reporting, technology.\nSentiment: Mostly neutral with some positive sections.\nDocument structure: 5 main sections with subsections."
            
        elif file_ext in ['.json']:
            # Simulate JSON file analysis
            return f"Analysis of JSON file {file_name}:\n\nThis configuration file contains 35 key-value pairs.\nMain sections: settings, user preferences, system configuration.\nThe file appears to be well-formed without syntax errors."
            
        elif file_ext in ['.xlsx', '.xls']:
            # Simulate Excel file analysis
            return f"Analysis of Excel file {file_name}:\n\nThis spreadsheet contains 3 sheets with a total of 512 rows.\nSheets: 'Summary', 'Raw Data', 'Charts'.\nContains 8 charts and 12 formulas.\nData range: Q1 2022 - Q4 2023."
            
        elif file_ext in ['.pdf']:
            # Simulate PDF analysis
            return f"Analysis of PDF file {file_name}:\n\nThis document is 18 pages long.\nContains 4 tables and 2 charts.\nExtracted text indicates this is a quarterly financial report.\nKey metrics: Revenue growth: 12%, Profit margin: 18%."
            
        else:
            return f"I don't know how to analyze files with extension {file_ext} yet."
            
    def generate_conversation_response(self, topic: str, group: str = None) -> str:
        """
        Generate a conversational response about a specific topic.
        
        Args:
            topic: The conversation topic
            group: Optional group context
            
        Returns:
            Conversational response
        """
        # In a real implementation, this would use a more sophisticated method
        # to generate contextual responses based on conversation history
        
        if group:
            files = self.list_files_in_group(group)
            return f"Regarding {topic} in {group}, I can tell you that this group contains {len(files)} files that might be relevant. Would you like to explore a specific file or analysis approach?"
        else:
            return f"I'd be happy to discuss {topic} with you. Could you provide more context or specify what aspect of {topic} you're interested in?"
            
    def generate_generic_response(self, query: str, recent_group: str = None) -> str:
        """
        Generate a generic response when no specific intent is detected.
        
        Args:
            query: The user query
            recent_group: Optional recently viewed group for context
            
        Returns:
            Generic response
        """
        if recent_group:
            return f"I understand you're asking about '{query}'. To help you better, would you like to explore the data in {recent_group} or would you prefer to look at a different group?"
        else:
            return "I'm not sure I understand your question. Would you like to see the available data groups, or could you rephrase your question?"
            
    def explain_capabilities(self) -> str:
        """
        Explain the agent's capabilities to the user.
        
        Returns:
            Description of capabilities
        """
        capabilities = [
            "I can help you explore and analyze various types of data files organized in groups.",
            "I can work with CSV, text, markdown, JSON, Excel, and PDF files.",
            "For each file type, I can provide different types of analysis and insights.",
            "You can ask me to list available data groups and files within them.",
            "You can ask specific questions about the data in a file once you've selected it.",
            "I maintain context of our conversation and your recently viewed files and groups."
        ]
        
        return "Here's what I can do for you:\n\n" + "\n".join([f"- {cap}" for cap in capabilities])

    def set_awaiting_clarification(self, awaiting: bool) -> None:
        """Set whether the agent is awaiting clarification from the user."""
        self._awaiting_clarification = awaiting
        
    def set_clarification_type(self, clarification_type: str) -> None:
        """Set the type of clarification the agent is awaiting."""
        self._clarification_type = clarification_type
        
    def set_clarification_group(self, group: str) -> None:
        """Set the group context for the current clarification."""
        self._clarification_group = group
        
    def set_awaiting_file_selection(self, awaiting: bool) -> None:
        """Set whether the agent is awaiting file selection from the user."""
        self._awaiting_file_selection = awaiting
        
    def set_selected_file_type(self, file_type: str) -> None:
        """Set the file type for the current file selection process."""
        self._selected_file_type = file_type
        
    def get_awaiting_clarification(self) -> bool:
        """Get whether the agent is awaiting clarification from the user."""
        return self._awaiting_clarification
    
    def get_clarification_type(self) -> str:
        """Get the type of clarification the agent is awaiting."""
        return self._clarification_type
    
    def get_clarification_group(self) -> str:
        """Get the group context for the current clarification."""
        return self._clarification_group
    
    def get_awaiting_file_selection(self) -> bool:
        """Get whether the agent is awaiting file selection from the user."""
        return self._awaiting_file_selection
    
    def get_selected_file_type(self) -> str:
        """Get the file type for the current file selection process."""
        return self._selected_file_type

    def get_file_path(self, group: str, file_name: str) -> Optional[str]:
        """Get the full path to a specific file within the current report and group."""
        if not self.current_report or not group:
            self.logger.warning("Attempted get_file_path without current report or group set.")
            return None
            
        # Ensure group exists in the current report
        if group not in self.reports_dict.get(self.current_report, {}).get("groups", {}):
             self.logger.warning(f"Group '{group}' not found in report '{self.current_report}' during get_file_path.")
             return None

        file_ext = os.path.splitext(file_name)[1].lower()
        if not file_ext:
            # Try finding file without extension (might happen if LLM omits it)
             for ext, files_data in self.reports_dict[self.current_report]["groups"][group]["files"].items():
                 if file_name in files_data:
                     self.logger.info(f"Found path for '{file_name}' under extension '{ext}'.")
                     return files_data[file_name].get("path")
             self.logger.warning(f"Could not determine extension or find file '{file_name}' in group '{group}'.")
             return None

        # Standard lookup by extension and filename
        if file_ext in self.reports_dict[self.current_report]["groups"][group].get("files", {}) and \
           file_name in self.reports_dict[self.current_report]["groups"][group]["files"].get(file_ext, {}):
            
            return self.reports_dict[self.current_report]["groups"][group]["files"][file_ext][file_name].get("path")
            
        self.logger.warning(f"File '{file_name}' with extension '{file_ext}' not found in group '{group}'.")
        return None
