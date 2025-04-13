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

# Import ContextMemory as the unified context management solution
from ContextMemory import ContextMemory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        self.query_analyzer = Agent(
            model=OpenAIChat(id="gpt-4o"),
            markdown=True,
            instructions=[
                "You analyze user queries to determine their intent and information needs.",
                "You identify which report groups, file types, and specific files are most relevant to their query.",
                "You specialize in understanding hydrology, groundwater, climate, and environmental science terminology.",
                "Your goal is to accurately interpret what the user is asking for, even when their query is ambiguous."
            ],
        )
        
        self.logger.info(f"Initialized with {len(self.reports_dict)} reports")
        self.logger.info(f"Current report set to: {self.current_report}")
        
        # Enhanced keyword mapping with more specific subcategories
        self.keyword_to_group = {
            # Climate-related keywords
            "climate": "climate_change",
            "temperature": "climate_change", 
            "precipitation": "climate_change",
            "weather": "climate_change",
            "rainfall": "climate_change",
            "drought": "climate_change",
            "warming": "climate_change",
            "cooling": "climate_change",
            "humidity": "climate_change",
            "climate change": "climate_change",
            "climate data": "climate_change",
            
            # Groundwater-related keywords
            "groundwater": "groundwater",
            "aquifer": "groundwater",
            "well": "groundwater",
            "water table": "groundwater",
            "water level": "groundwater",
            "piezometric": "groundwater",
            "drawdown": "groundwater",
            "recharge": "groundwater",
            "pumping": "groundwater",
            "hydraulic conductivity": "groundwater",
            "transmissivity": "groundwater",
            "porosity": "groundwater",
            
            # Crop/land use related keywords
            "crop": "cdl",
            "land use": "cdl",
            "agriculture": "cdl",
            "farming": "cdl",
            "cdl": "cdl",
            "cropland": "cdl",
            "rotation": "cdl",
            "corn": "cdl",
            "soybeans": "cdl",
            "wheat": "cdl",
            "land cover": "cdl",
            
            # Soil-related keywords
            "soil": "soil",
            "erosion": "soil",
            "texture": "soil",
            "fertility": "soil",
            "drainage": "soil",
            "infiltration": "soil",
            "compaction": "soil",
            "organic matter": "soil",
            "soil moisture": "soil",
            "soil type": "soil",
            
            # PRISM-related keywords
            "prism": "prism",
            "elevation": "prism",
            "slope": "prism",
            "aspect": "prism",
            "topography": "prism",
            "terrain": "prism",
            "dem": "prism",
            "digital elevation": "prism",
            
            # MODIS-related keywords
            "modis": "modis",
            "satellite": "modis",
            "vegetation": "modis",
            "ndvi": "modis",
            "evi": "modis",
            "remote sensing": "modis",
            "leaf area": "modis",
            "phenology": "modis",
            "biomass": "modis",
            "land surface temperature": "modis",
        }
        
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
    
    def find_best_matching_report(self, query: str) -> str:
        """Find the best matching report based on user query."""
        reports = self.list_reports()
        
        # First check if query contains a date that matches a report timestamp
        date_patterns = [
            r'(\d{8}_\d{6})',  # 20250324_222749 format
            r'(\d{4}-\d{2}-\d{2})',  # 2025-03-24 format
            r'(\d{2}/\d{2}/\d{4})'   # 03/24/2025 format
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                for report in reports:
                    if match in report:
                        return report
        
        # If no direct date match, use the most recent report
        return sorted(reports)[-1]
    
    def find_best_matching_group(self, query: str) -> Optional[str]:
        """Find the best matching group based on user query using keyword matching."""
        # Check if query directly mentions a group name
        available_groups = self.list_groups()
        
        for group in available_groups:
            if group.lower() in query.lower():
                return group
        
        # Check for keywords that map to groups
        for keyword, group in self.keyword_to_group.items():
            if keyword.lower() in query.lower() and group in available_groups:
                return group
                
        # If no matching group is found, return None
        return None
    
    def find_best_matching_file(self, query: str, group: str, file_type: Optional[str] = None) -> Optional[Tuple[str, str]]:
        """Find the best matching file based on user query."""
        files_by_type = self.list_files_by_type(group=group)
        
        if not files_by_type:
            return None
            
        # If file_type is specified, only search that type
        if file_type and file_type in files_by_type:
            search_types = {file_type: files_by_type[file_type]}
        else:
            search_types = files_by_type
            
        # Look for file name mentions in the query
        for ext, files in search_types.items():
            for file in files:
                # Remove extension for matching
                file_base = os.path.splitext(file)[0].lower()
                # Check if file name is in query
                if file_base in query.lower():
                    return (file, ext)
        
        # If no direct match found, return the first file of the appropriate type
        # Prioritize markdown files, then csv, then images
        priority_types = ['.md', '.txt', '.csv', '.png', '.jpg', '.jpeg']
        
        for type_ext in priority_types:
            if type_ext in search_types and search_types[type_ext]:
                return (search_types[type_ext][0], type_ext)
                
        # If still no match, return first file of any type
        for ext, files in search_types.items():
            if files:
                return (files[0], ext)
                
        return None
    
    def _read_text(self, file_path: str) -> str:
        """Process text files using the text reader"""
        response = text_reader.analyze_report(file_path)
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
        response = csv_reader.csv_reader(file_path, recreate_db=True)
        if response:
            # Extract key insights
            self._extract_and_save_insights(response, f"csv:{os.path.basename(file_path)}")
        return response
    
    def _read_website(self, url: str) -> str:
        """Process websites using the website reader"""
        return website_reader.website_reader(url)
    
    def _read_combined(self, report: str, group: str) -> str:
        """Process entire group using the combined reader"""
        response = combine_reader.combined_reader(self.reports_dict, report, group, recreate_db=True)
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
            
            result = self._read_combined(report_id, group_id)
            # Ensure we return a string even if _read_combined returns None
            if result is None:
                return f"Analysis completed for {group_id}, but no results were returned."
            return result
        
        return f"Group {group_id} not found in report {report_id}"
    
    def process_query(self, query: str) -> str:
        """
        Process a user query and return the appropriate response.
        This is the main entry point for handling all user interactions.
        
        Args:
            query: The user's query/question
            
        Returns:
            Response text to the user
        """
        # First, log the user query and add to conversation history
        self.logger.info(f"Processing user query: {query}")
        
        # Use the appropriate method depending on what's available in the context object
        if hasattr(self.context, 'add_user_message'):
            self.context.add_user_message(query)
        elif hasattr(self.context, 'add_message'):
            self.context.add_message("user", query)
        
        # Check if we're in a clarification flow
        if self.get_awaiting_clarification():
            clarification_type = self.get_clarification_type()
            clarification_group = self.get_clarification_group()
            
            self.logger.info(f"Handling clarification. Type: {clarification_type}, Group: {clarification_group}")
            
            # Reset the clarification state
            self.set_awaiting_clarification(False)
            self.set_clarification_type(None)
            self.set_clarification_group(None)
            
            # Handle different types of clarifications
            if clarification_type == "group_selection":
                groups = self.list_groups()
                for group in groups:
                    if group.lower() in query.lower():
                        self.current_group = group
                        self.logger.info(f"Group selected from clarification: {group}")
                        return self.handle_group_selection(group)
                
                # If no valid group was found, offer suggestions again
                return "I couldn't identify a valid group from your response. Here are the available groups: " + \
                       ", ".join(groups) + ". Please select one."
                       
            elif clarification_type == "file_type_selection":
                # Handle file type selection clarification
                # Dynamically get available types for the relevant group
                available_types = list(self.list_files_by_type(clarification_group).keys())
                if not available_types:
                    return f"No files found in group '{clarification_group}'. Cannot select a file type."

                selected_type = None
                query_lower = query.lower()

                # Check against dynamically fetched types
                for file_type in available_types:
                    # Check for '.ext' or 'ext'
                    if file_type in query_lower or (file_type.startswith('.') and file_type[1:] in query_lower):
                        selected_type = file_type
                        break

                if selected_type:
                    self.set_selected_file_type(selected_type)
                    return self.filter_and_display_files(clarification_group, selected_type)
                else:
                    # Improved error message listing actual available types
                    type_list = ", ".join(available_types)
                    return f"I couldn't identify a valid file type from your response. Available types in '{clarification_group}' are: {type_list}. Please specify one."
            
            elif clarification_type == "analysis_target":
                # Handle clarifying what the user wants to analyze
                return self.determine_analysis_target_from_clarification(query, clarification_group)
        
        # Check if we're in a file selection flow
        if self.get_awaiting_file_selection():
            self.logger.info(f"Handling file selection. File type: {self.get_selected_file_type()}")
            
            # Reset the file selection state
            self.set_awaiting_file_selection(False)
            
            # Handle file selection
            if self.current_group:
                files_in_group = self.list_files_in_group(self.current_group)

                # Check if the query contains a file name
                matching_files = []
                for file in files_in_group:
                    # Use case-insensitive matching against the base filename (without extension)
                    file_base = os.path.splitext(file)[0].lower()
                    # Also check against the full filename
                    if file.lower() in query.lower() or file_base in query.lower():
                        matching_files.append(file)

                if len(matching_files) == 1:
                    selected_file_name = matching_files[0]
                    # Get the file type (extension)
                    file_ext = os.path.splitext(selected_file_name)[1].lower()
                    if not file_ext:
                        self.logger.warning(f"Could not determine file type for '{selected_file_name}'. Skipping analysis.")
                        return f"Could not determine file type for '{selected_file_name}'."

                    # Use process_file to call the appropriate reader
                    return self.process_file(selected_file_name, file_ext, self.current_group)

                elif len(matching_files) > 1:
                    file_list = "\n".join([f"- {file}" for file in matching_files])
                    return f"I found multiple matching files. Please specify which one you'd like to analyze:\n{file_list}"
                else:
                    # Improved error message if no file matches
                    return f"I couldn't find a file matching '{query}' in the group '{self.current_group}'. Please try again with one of these files: \n- " + "\n- ".join(files_in_group)
        
        # For normal query handling, first check if this is a system command
        if query.lower().startswith(("list reports", "list groups", "set report", "set group", "help", "exit", "quit")):
            return self.handle_system_command(query)
        
        # Analyze the query intent
        intent_analysis = self.analyze_query_intent(query)
        intent_type = intent_analysis.get("intent_type", "unknown")
        action = intent_analysis.get("action", "")
        confidence = intent_analysis.get("confidence", 0)
        
        self.logger.info(f"Query intent analysis: {intent_type} (confidence: {confidence})")
        
        # Handle query based on intent type
        if intent_type == "information":
            if action == "suggest_groups":
                groups = self.list_groups()
                if not groups:
                    return "No data groups are currently available."
                
                # Set clarification state for group selection
                self.set_awaiting_clarification(True)
                self.set_clarification_type("group_selection")
                
                if len(groups) == 1:
                    # Only one group available, automatically select it
                    self.current_group = groups[0]
                    self.logger.info(f"Auto-selected the only available group: {groups[0]}")
                    return self.handle_group_selection(groups[0])
                else:
                    group_list = ", ".join(groups)
                    return f"I have data available in the following groups: {group_list}. Which one would you like to explore?"
                    
            elif action == "list_files":
                target_group = intent_analysis.get("target_group")
                if not target_group and self.current_group:
                    target_group = self.current_group
                
                if target_group:
                    self.current_group = target_group
                    return self.display_group_files(target_group)
                else:
                    groups = self.list_groups()
                    # Set up clarification state
                    self.set_awaiting_clarification(True)
                    self.set_clarification_type("group_selection")
                    return f"Which data group would you like to see files for? Available groups: {', '.join(groups)}"
                    
            elif action == "explain_capabilities":
                return self.explain_capabilities()
        
        elif intent_type == "analysis":
            target_group = intent_analysis.get("target_group")
            target_file_type = intent_analysis.get("target_file_type")
            
            if action == "analyze_group" and target_group:
                self.current_group = target_group
                return self.handle_group_selection(target_group)
                
            elif action == "find_and_analyze_file" and target_file_type:
                if not self.current_group:
                    groups = self.list_groups()
                    if len(groups) == 1:
                        self.current_group = groups[0]
                    else:
                        # Set up clarification state
                        self.set_awaiting_clarification(True)
                        self.set_clarification_type("group_selection")
                        self.set_clarification_group(None)  # No specific group yet
                        return f"Which data group would you like to explore {target_file_type} files in? Available groups: {', '.join(groups)}"
                
                return self.filter_and_display_files(self.current_group, target_file_type)
                
            elif action == "determine_analysis_target":
                if not self.current_group:
                    groups = self.list_groups()
                    if len(groups) == 1:
                        self.current_group = groups[0]
                        # Continue to analysis with the only available group
                        return self.handle_analysis_request(query, groups[0])
                    else:
                        # Set up clarification state
                        self.set_awaiting_clarification(True)
                        self.set_clarification_type("group_selection")
                        self.set_clarification_group(None)
                        return f"Which data group would you like to analyze? Available groups: {', '.join(groups)}"
                else:
                    return self.handle_analysis_request(query, self.current_group)
        
        elif intent_type == "conversation":
            topic = intent_analysis.get("topic", "")
            target_group = intent_analysis.get("target_group")
            
            if target_group:
                self.current_group = target_group
            
            # Check if we're discussing a specific topic with enough context
            if topic and self.current_group:
                # Add to viewed groups if not already there
                if self.current_group not in self.context.session_data["viewed_groups"]:
                    self.context.session_data["viewed_groups"].append(self.current_group)
                
                # Generate a more conversational response about the topic in the current group
                return self.generate_conversation_response(topic, self.current_group)
            
            elif topic and not self.current_group:
                # Need to determine which group to discuss
                groups = self.list_groups()
                if len(groups) == 1:
                    self.current_group = groups[0]
                    return self.generate_conversation_response(topic, groups[0])
                else:
                    # Set up clarification state
                    self.set_awaiting_clarification(True)
                    self.set_clarification_type("group_selection")
                    self.set_clarification_group(None)
                    return f"Which data group would you like to discuss {topic} in? Available groups: {', '.join(groups)}"
            
            else:
                # Generic conversation without specific topic or group
                if self.context.session_data["viewed_groups"]:
                    recent_group = self.context.session_data["viewed_groups"][-1]
                    return self.generate_generic_response(query, recent_group)
                else:
                    return self.generate_generic_response(query)
                    
        elif intent_type == "command":
            command = intent_analysis.get("command", "")
            return self.handle_system_command(query)
        
        # Fall back to generic response if intent is unknown or low confidence
        return self.generate_generic_response(query)

    def generate_response(self, query: str, analysis: Dict[str, Any]) -> str:
        """
        Generate a response based on query analysis.
        Enhanced to include clarification steps.
        
        Args:
            query: User query
            analysis: Query analysis results
            
        Returns:
            Response text
        """
        # Handle clarification needs first
        if analysis.get("needs_clarification", False):
            clarification_type = analysis.get("clarification_type")
            options = analysis.get("clarification_options", [])
            
            if clarification_type == "intent":
                return (
                    "I'm not entirely sure what you're asking for. Would you like to:\n" +
                    "\n".join(f"- {option}" for option in options)
                )
                
            elif clarification_type == "group":
                group_options = "\n".join(f"- {group}" for group in options)
                return (
                    f"{analysis.get('clarification_message', 'Which data group are you interested in?')}\n" +
                    f"Available groups:\n{group_options}"
                )
                
            elif clarification_type == "file_type":
                type_options = "\n".join(f"- {ftype}" for ftype in options)
                return (
                    f"{analysis.get('clarification_message', 'Which type of files would you like to work with?')}\n" +
                    f"Available file types:\n{type_options}"
                )
                
            elif clarification_type == "specific_file":
                file_options = "\n".join(f"- {file}" for file in options[:5])
                additional = f" (showing 5 of {len(options)})" if len(options) > 5 else ""
                return (
                    f"{analysis.get('clarification_message', 'Which specific file would you like to analyze?')}\n" +
                    f"Available files{additional}:\n{file_options}"
                )
        
        # Command handling
        if analysis["intent_type"] == "command":
            # Handle command in the existing process_command method
            return None  # Let the command handler generate the response
            
        # Information intent - show what data is available
        elif analysis["intent_type"] == "information":
            if "target_group" in analysis:
                group = analysis["target_group"]
                files = analysis.get("available_files", self.list_files_in_group(group))
                file_types = analysis.get("file_types", self.get_file_types_in_group(group))
                
                type_summary = ""
                if file_types:
                    type_counts = {}
                    for file in files:
                        ext = os.path.splitext(file)[1]
                        type_counts[ext] = type_counts.get(ext, 0) + 1
                        
                    type_summary = "\n".join(f"- {count} {ftype} files" for ftype, count in type_counts.items())
                    
                files_preview = "\n".join(f"- {file}" for file in files[:5])
                additional = f" (showing 5 of {len(files)})" if len(files) > 5 else ""
                
                return (
                    f"Here's information about the {group} data group:\n\n"
                    f"Summary of available files:\n{type_summary}\n\n"
                    f"Sample files{additional}:\n{files_preview}\n\n"
                    f"Would you like to analyze any specific files or file types from this group?"
                )
            else:
                groups = analysis.get("available_groups", self.list_groups())
                groups_list = "\n".join(f"- {group}" for group in groups)
                
                return (
                    "Here are the available data groups:\n\n"
                    f"{groups_list}\n\n"
                    "Which group would you like to explore?"
                )
                
        # Analysis intent - analyze data
        elif analysis["intent_type"] == "analysis":
            # If we've passed all clarification steps, we can proceed with analysis
            # This part would be handled by process_query in the existing implementation
            return None
            
        # Conversation intent - discuss data
        elif analysis["intent_type"] == "conversation":
            # This is handled by process_query in existing implementation
            return None
            
        # Fallback for unexpected cases
        return "I'm not sure how to respond to that. Can you try rephrasing your question?"

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
        
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze the user query to determine intent, target data, and optimal processing approach.
        
        Args:
            query: The user's query text
            
        Returns:
            Dictionary containing analysis results
        """
        # First, check if this is a direct command
        command_patterns = {
            r"(?i)^\s*list\s+reports\s*$": {"command": "list_reports"},
            r"(?i)^\s*list\s+groups\s*$": {"command": "list_groups"},
            r"(?i)^\s*set\s+report\s+(\S+)\s*$": {"command": "set_report", "param": lambda m: m.group(1)},
            r"(?i)^\s*set\s+group\s+(\S+)\s*$": {"command": "set_group", "param": lambda m: m.group(1)},
            r"(?i)^\s*help\s*$": {"command": "help"},
            r"(?i)^\s*exit\s*$": {"command": "exit"}
        }
        
        import re
        for pattern, action in command_patterns.items():
            match = re.match(pattern, query)
            if match:
                result = {"intent_type": "command", "command": action["command"]}
                if "param" in action:
                    result["param"] = action["param"](match)
                return result
        
        # Otherwise, get a more detailed analysis using the intent classifier
        intent_analysis = self.analyze_query_intent(query)
        
        # If specific file types mentioned, add them
        file_type_patterns = {
            r"(?i)(csv|spreadsheet|table|excel)": ".csv",
            r"(?i)(image|png|jpg|jpeg|picture|graph|chart|plot|visualization)": ".png",
            r"(?i)(text|txt|markdown|md|report|document)": ".md"
        }
        
        mentioned_file_types = []
        for pattern, file_type in file_type_patterns.items():
            if re.search(pattern, query):
                mentioned_file_types.append(file_type)
        
        if mentioned_file_types:
            intent_analysis["file_types"] = mentioned_file_types
        
        # Check for specific report group mentions
        group_patterns = {}
        for group in self.list_groups():
            # Create a regex pattern for each group name
            pattern = f"(?i)\\b{re.escape(group)}\\b"
            group_patterns[pattern] = group
        
        # Find matching groups
        matching_groups = []
        for pattern, group in group_patterns.items():
            if re.search(pattern, query):
                matching_groups.append(group)
        
        # If we have matching groups, add the first one as the target
        if matching_groups:
            intent_analysis["target_group"] = matching_groups[0]
            intent_analysis["multiple_groups"] = len(matching_groups) > 1
            intent_analysis["all_matching_groups"] = matching_groups
            
        # Check if we need clarification
        needs_clarification = False
        clarification_type = None
        clarification_options = []
        
        # Case: Intent is information/analysis but no group specified
        if intent_analysis["intent_type"] in ["information", "analysis"] and "target_group" not in intent_analysis:
            needs_clarification = True
            clarification_type = "group"
            clarification_options = self.list_groups()
            clarification_message = "Which data group would you like to explore?"
            
        # Case: Intent is analysis but no specific file type
        elif intent_analysis["intent_type"] == "analysis" and "target_group" in intent_analysis and "file_types" not in intent_analysis:
            group = intent_analysis["target_group"]
            available_types = self.get_file_types_in_group(group)
            
            if len(available_types) > 1:
                needs_clarification = True
                clarification_type = "file_type"
                clarification_options = available_types
                clarification_message = f"What type of files from the {group} group would you like to analyze?"
            
        # Case: Intent ambiguous
        elif intent_analysis["intent_type"] == "unknown" or intent_analysis["confidence"] < 0.6:
            needs_clarification = True
            clarification_type = "intent"
            clarification_options = ["Show available data", "Analyze a specific file", "Explain a concept"]
            clarification_message = "I'm not sure what you're asking for. What would you like to do?"
        
        # Add clarification info if needed
        if needs_clarification:
            intent_analysis["needs_clarification"] = True
            intent_analysis["clarification_type"] = clarification_type
            intent_analysis["clarification_options"] = clarification_options
            intent_analysis["clarification_message"] = clarification_message
        
        return intent_analysis

    def get_available_groups(self):
        """Get the available data groups from current report"""
        if self.current_report and self.current_report in self.reports_dict:
            return sorted(self.reports_dict[self.current_report]["groups"].keys())
        return []

    def get_files_by_type(self, group, file_type):
        """Get files of a specific type in a group"""
        if self.current_report and group in self.reports_dict[self.current_report]["groups"]:
            if file_type in self.reports_dict[self.current_report]["groups"][group]["files"]:
                return sorted(list(self.reports_dict[self.current_report]["groups"][group]["files"][file_type].keys()))
        return []
        
    def get_file_path(self, group, file_name):
        """Get the full path to a specific file"""
        if not self.current_report or not group:
            return None
            
        file_ext = os.path.splitext(file_name)[1]
        
        if (self.current_report in self.reports_dict and 
            group in self.reports_dict[self.current_report]["groups"] and
            file_ext in self.reports_dict[self.current_report]["groups"][group]["files"] and
            file_name in self.reports_dict[self.current_report]["groups"][group]["files"][file_ext]):
            
            return self.reports_dict[self.current_report]["groups"][group]["files"][file_ext][file_name]["path"]
            
        return None
        
    def needs_clarification(self, analysis):
        """Check if query analysis indicates we need clarification"""
        return analysis.get("needs_clarification", False)

    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Analyze the user's query to determine its intent type and relevant details.
        
        Intent types:
        - information: User is asking what data exists
        - analysis: User is requesting analysis of data
        - conversation: User is having a discussion about existing data
        - command: User is issuing a system command
        
        Args:
            query: The user query string
            
        Returns:
            Dictionary with intent type and details
        """
        # First check if this is a system command
        if any(query.lower().startswith(cmd) for cmd in [
            "list reports", "list groups", "set report", "set group", "help", "exit", "quit"
        ]):
            return {
                "intent_type": "command",
                "command": query.lower().split()[0],
                "confidence": 0.95
            }
        
        # Enhanced information query patterns - user asking about what data exists
        info_patterns = [
            r"what (data|files|information|reports|datasets) (are|is|do you have) (in|about|on|for) (.+)",
            r"show me (what|the) (data|files|information|reports|datasets) (in|about|on|for) (.+)",
            r"(list|show) (all |the )?(data|files|information|reports|datasets) (in|about|on|for) (.+)",
            r"what (is|are) (available|there) (in|about|on|for) (.+)",
            r"tell me what (data|files|information|reports|datasets) (are|is) (in|about|on|for) (.+)",
            r"do you have (any |some )?(data|files|information|reports|datasets) (in|about|on|for) (.+)",
            r"what (kind of|types of) (data|files|information|reports|datasets) (are|is) (in|about|on|for) (.+)",
            r"(describe|summarize) (the |available )?(data|files|information|reports|datasets) (in|about|on|for) (.+)",
            r"(what|which) (groups|categories|types) (of data|of reports|of information) (are|do you have) (available|)",
            r"(can you|could you) (list|tell me|show me) (what|which) (files|reports|data) (you have|are available)",
            r"(show|list|get) (me )?available (data|files|reports)",
            r"what can I (analyze|look at) (here|)",
            r"(what|which) (reports|data sets|files) (can|should) I (look at|view|examine)"
        ]
        
        # Enhanced analysis request patterns - user asking for conclusions or insights
        analysis_patterns = [
            r"(analyze|analyse|study|examine|investigate) (.+)",
            r"(show|tell) me (about|the) (.+) (analysis|results|findings|trends|patterns|conclusions)",
            r"(what|how) (is|are|does|do) (.+) (affect|impact|influence|relate|correlate|mean)",
            r"(compare|contrast|evaluate|assess) (.+)",
            r"(generate|create|produce|provide) (a|an) (analysis|report|summary|overview) (of|on|about) (.+)",
            r"what (can you|do you) (tell|see|find|conclude) (about|from|in) (.+)",
            r"what (insights|conclusions|results) (can you show|do you have|are there) (for|from|about) (.+)",
            r"(calculate|compute|determine) (the|any) (statistics|metrics|measures|values) (for|of|from) (.+)",
            r"(find|identify|discover|detect) (patterns|trends|anomalies|outliers|relationships) (in|from|within) (.+)",
            r"(run|perform|conduct|do) (a|an) (full|complete|comprehensive|detailed) (analysis|assessment|examination) (of|on) (.+)",
            r"(deep dive|in-depth look) (into|at) (.+)",
            r"(what's|what is) (happening|going on) (in|with) (.+)",
            r"(visualize|plot|graph|chart) (.+)",
            r"(summarize|breakdown) the (data|information|metrics|statistics|findings) (in|for|about) (.+)",
            r"(process|mine|extract insights from) (.+)"
        ]
        
        # Enhanced conversational patterns - user wants to discuss or learn about data
        conversation_patterns = [
            r"(tell me more|elaborate|explain) (about|on) (.+)",
            r"(why|how) (is|are|does|do|can|would|should|might) (.+)",
            r"(can|could) you (explain|describe|clarify|help me understand) (.+)",
            r"(what|who|when|where) (is|are|was|were) (.+)",
            r"(i'm interested in|i want to know about|i'd like to learn|i'm curious about) (.+)",
            r"(what does|could you explain what) (.+) (mean|imply|suggest|indicate)",
            r"(can you|would you) (tell|explain to) me (why|how|what) (.+)",
            r"(is|are) there (any|some) (relationship|connection|correlation) between (.+)",
            r"(in your opinion|what do you think|do you believe) (.+)",
            r"(can|could) you (interpret|translate|decode) (.+)",
            r"(help me|assist me) (understand|interpret|grasp) (.+)",
            r"(I don't understand|I'm confused by|I'm not sure about) (.+)",
            r"(give me context|provide background|explain the significance) (about|of|for) (.+)",
            r"(how would you|how do you) (interpret|understand|approach) (.+)",
            r"let's (talk|discuss|chat) about (.+)",
            r"(I'd like your|give me your) (thoughts|perspective|take|opinion) on (.+)"
        ]
        
        # Check context for recently viewed data
        has_context = False
        if self.context.session_data["viewed_files"] or self.context.session_data["viewed_groups"]:
            has_context = True
        
        # Check for information intent with high specificity patterns first
        for pattern in info_patterns:
            match = re.search(pattern, query.lower())
            if match:
                # Try to extract the group or topic
                # In most patterns, the last capture group contains the target
                target = match.groups()[-1] if match.groups() else ""
                
                # Look for group mentions
                groups = self.list_groups()
                for group in groups:
                    if group.lower() in target.lower():
                        return {
                            "intent_type": "information",
                            "target_group": group,
                            "confidence": 0.9,  # Higher confidence for exact group match
                            "action": "list_files"
                        }
                
                # Information query without specific group
                return {
                    "intent_type": "information",
                    "confidence": 0.85,
                    "action": "suggest_groups"
                }
        
        # Check for analysis intent
        for pattern in analysis_patterns:
            match = re.search(pattern, query.lower())
            if match:
                # Try to extract what to analyze
                to_analyze = match.groups()[-1] if match.groups() else ""
                
                # Look for group mentions
                groups = self.list_groups()
                for group in groups:
                    if group.lower() in query.lower():
                        return {
                            "intent_type": "analysis",
                            "target_group": group,
                            "confidence": 0.9,
                            "action": "analyze_group"
                        }
                
                # Check for file type mentions
                file_types = [".csv", ".txt", ".json", ".xlsx", ".md", ".pdf"]
                for file_type in file_types:
                    if file_type in query.lower() or file_type[1:] in query.lower():
                        return {
                            "intent_type": "analysis",
                            "target_file_type": file_type,
                            "confidence": 0.85,
                            "action": "find_and_analyze_file"
                        }
                
                # If we have context (viewed files/groups), more likely to be analysis
                if has_context:
                    return {
                        "intent_type": "analysis",
                        "confidence": 0.82,
                        "action": "determine_analysis_target"
                    }
                
                # Generic analysis request
                return {
                    "intent_type": "analysis",
                    "confidence": 0.75,
                    "action": "determine_analysis_target"
                }
        
        # Check for conversation intent
        for pattern in conversation_patterns:
            match = re.search(pattern, query.lower())
            if match:
                topic = match.groups()[-1] if match.groups() else ""
                
                # Check if the conversation is about a specific group
                groups = self.list_groups()
                for group in groups:
                    if group.lower() in query.lower():
                        return {
                            "intent_type": "conversation",
                            "topic": topic,
                            "target_group": group,
                            "confidence": 0.88,
                            "action": "discuss_topic"
                        }
                
                # If we have context, more likely to be conversation about viewed data
                if has_context:
                    recent_groups = self.context.session_data["viewed_groups"][-1] if self.context.session_data["viewed_groups"] else None
                    
                    return {
                        "intent_type": "conversation",
                        "topic": topic,
                        "target_group": recent_groups,
                        "confidence": 0.86,
                        "action": "discuss_topic",
                        "context_based": True
                    }
                
                # Generic conversation
                return {
                    "intent_type": "conversation",
                    "topic": topic,
                    "confidence": 0.8,
                    "action": "discuss_topic"
                }
        
        # Simple group name detection - treat as analysis intent if it's just the group name
        groups = self.list_groups()
        for group in groups:
            if group.lower() == query.lower() or f"{group} group".lower() == query.lower():
                return {
                    "intent_type": "analysis",
                    "target_group": group,
                    "confidence": 0.95,  # Very high confidence for exact group name match
                    "action": "analyze_group"
                }
        
        # Detect simple information queries about the system
        system_info_keywords = ["what can you do", "capabilities", "what are you able to", "how do you work", 
                               "what can i ask", "how can you help", "what are your functions"]
        if any(keyword in query.lower() for keyword in system_info_keywords):
            return {
                "intent_type": "information",
                "confidence": 0.9,
                "action": "explain_capabilities",
                "system_query": True
            }
        
        # Check for follow-up questions without context
        followup_indicators = ["and", "also", "in addition", "further", "more", "what about", "how about"]
        if any(query.lower().startswith(indicator) for indicator in followup_indicators) and has_context:
            # Get the most recent query intent
            if len(self.context.conversation_history) >= 2:
                # This is likely a follow-up query, continue with previous intent type but lower confidence
                last_user_query = None
                for msg in reversed(self.context.conversation_history):
                    if msg["role"] == "user":
                        last_user_query = msg["content"]
                        break
                
                if last_user_query:
                    # Default to conversation intent for follow-ups with context
                    followup_result = {
                        "intent_type": "conversation",
                        "confidence": 0.7,
                        "action": "discuss_topic",
                        "is_followup": True
                    }
                    
                    # Try to determine a group from context
                    if self.context.session_data["viewed_groups"]:
                        recent_group = self.context.session_data["viewed_groups"][-1]
                        followup_result["target_group"] = recent_group
                    
                    return followup_result
        
        # If we reach here, use a combined approach: first check for specific group mentions
        for group in self.list_groups():
            if group.lower() in query.lower():
                # Group is mentioned but intent is unclear
                # Decide based on query length and complexity
                
                # Short queries mentioning a group are likely analysis requests
                if len(query.split()) < 5:
                    return {
                        "intent_type": "analysis", 
                        "target_group": group,
                        "confidence": 0.75,
                        "action": "analyze_group"
                    }
                
                # Queries with question words are more likely to be information or conversation
                question_words = ["what", "how", "why", "when", "where", "who", "which", "can", "do", "is", "are"]
                if any(query.lower().split()[0] == word for word in question_words):
                    # Distinguish between information and conversation based on context
                    if has_context and group in self.context.session_data["viewed_groups"]:
                        return {
                            "intent_type": "conversation",
                            "target_group": group,
                            "confidence": 0.72,
                            "action": "discuss_topic"
                        }
                    else:
                        return {
                            "intent_type": "information",
                            "target_group": group,
                            "confidence": 0.7,
                            "action": "list_files"
                        }
                
                # Default to analysis for unclear but group-specific queries
                return {
                    "intent_type": "analysis", 
                    "target_group": group,
                    "confidence": 0.68,
                    "action": "analyze_group"
                }
        
        # If we still haven't determined intent, use a more direct approach with the query_analyzer
        prompt = f"""
        Analyze the following user query and determine the most likely intent:
        
        Query: "{query}"
        
        Available groups: {", ".join(self.list_groups())}
        
        Respond with a JSON object with these fields:
        - intent_type: "information", "analysis", or "conversation"
        - confidence: a value between 0 and 1
        - target_group: any group the query seems focused on (if applicable)
        - explanation: brief reason for this classification
        """
        
        # Get direct analysis from the query analyzer
        try:
            # Use a direct call to the agent instead of going through analyze_query
            llm_analysis = self.query_analyzer.print_response(prompt)
            
            # Try to parse the response as JSON
            try:
                import json
                llm_result = json.loads(llm_analysis)
                
                # Add default action based on intent type
                if "intent_type" in llm_result:
                    if llm_result["intent_type"] == "information":
                        llm_result["action"] = "suggest_groups" if not llm_result.get("target_group") else "list_files"
                    elif llm_result["intent_type"] == "analysis":
                        llm_result["action"] = "determine_analysis_target"
                    elif llm_result["intent_type"] == "conversation":
                        llm_result["action"] = "discuss_topic"
                        
                    # Set a default confidence if not provided
                    if "confidence" not in llm_result:
                        llm_result["confidence"] = 0.6
                        
                    return llm_result
            except:
                # If parsing fails, fall back to default
                pass
        except:
            # If LLM call fails, continue to fallback
            pass
            
        # Simple fallback - look for key terms to classify intent
        conversation_indicators = ["explain", "discuss", "tell me about", "share", "what is", "what are", 
                                 "meaning of", "significance of", "importance of"]
        if any(indicator in query.lower() for indicator in conversation_indicators):
            return {
                "intent_type": "conversation",
                "confidence": 0.62,
                "action": "discuss_topic",
                "explanation": "Query appears conversational in nature"
            }
            
        # Check if it might be an information query
        info_indicators = ["show", "list", "available", "do you have", "what data", "what files"]
        if any(indicator in query.lower() for indicator in info_indicators):
            return {
                "intent_type": "information",
                "confidence": 0.61,
                "action": "suggest_groups",
                "explanation": "Query appears to be asking about available data"
            }
        
        # Default to analysis for remaining cases
        return {
            "intent_type": "analysis",
            "confidence": 0.6,
            "action": "determine_analysis_target",
            "explanation": "Default to analysis for ambiguous queries"
        }

    def handle_group_selection(self, group: str) -> str:
        """
        Handle when a user selects a specific data group.
        
        Args:
            group: The selected group name
            
        Returns:
            Response with information about the group
        """
        if not self.current_group:
            self.current_group = group
            
        files = self.list_files_in_group(group)
        if not files:
            return f"The group '{group}' exists but doesn't contain any files yet."
            
        # Record this group in viewed groups
        if group not in self.context.session_data["viewed_groups"]:
            self.context.session_data["viewed_groups"].append(group)
            
        # Group summary and file count by type
        file_types = {}
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in file_types:
                file_types[ext] = 0
            file_types[ext] += 1
            
        type_summary = ", ".join([f"{count} {ext} files" for ext, count in file_types.items()])
        
        return f"Group '{group}' contains {len(files)} files ({type_summary}). Would you like to see all files or a specific type?"
        
    def filter_and_display_files(self, group: str, file_type: str) -> str:
        """
        Filter files by type and prepare them for display/selection.
        
        Args:
            group: The group to filter files in
            file_type: The file type to filter by
            
        Returns:
            A formatted list of files of the specified type
        """
        self.logger.info(f"Filtering files in {group} by type: {file_type}")
        
        # Get all files of the specified type in the group
        files_by_type = self.list_files_by_type(group)
        
        # Check if we have files of this type
        if file_type not in files_by_type or not files_by_type[file_type]:
            all_types = ", ".join(files_by_type.keys())
            return f"No {file_type} files found in the {group} group. Available file types: {all_types}"
        
        # Store the file type for later
        self.set_selected_file_type(file_type)
        
        # Always set the state to awaiting selection and display files, even if there's only one match
        self.set_awaiting_file_selection(True)
        
        # Format the list of files for display
        file_list = "\n".join([f"- {f}" for f in files_by_type[file_type]])
        
        return f"Found {len(files_by_type[file_type])} {file_type} files in the {group} group. Which one would you like to analyze?\n\n{file_list}"
        
    def determine_analysis_target_from_clarification(self, query: str, group: str) -> str:
        """
        Process a clarification about what to analyze based on user's response.
        
        Args:
            query: The user's clarification response
            group: The group context for this clarification
            
        Returns:
            Response with analysis results or further questions
        """
        # Extract potential analysis targets from the query
        analysis_keywords = ["trends", "summary", "statistics", "compare", "analyze", "data", 
                             "overview", "insights", "patterns", "anomalies"]
                             
        file_types = [".csv", ".txt", ".json", ".xlsx", ".md", ".pdf"]
        
        # Check if user mentioned a specific file type
        mentioned_file_type = None
        for file_type in file_types:
            if file_type in query.lower() or file_type[1:] in query.lower():
                mentioned_file_type = file_type
                break
                
        if mentioned_file_type:
            return self.filter_and_display_files(group, mentioned_file_type)
            
        # Check if user mentioned a specific analysis goal
        for keyword in analysis_keywords:
            if keyword in query.lower():
                return self.handle_analysis_request(query, group)
                
        # If we can't determine a specific action, ask about file types
        self.set_awaiting_clarification(True)
        self.set_clarification_type("file_type_selection")
        self.set_clarification_group(group)
        
        return f"What type of files from '{group}' would you like to work with? (CSV, text, markdown, etc.)"
        
    def handle_analysis_request(self, query: str, group: str) -> str:
        """
        Handle a request to analyze data, checking if specific files or types are mentioned.
        
        Args:
            query: The user's query
            group: The current group to analyze
            
        Returns:
            Response text to the user
        """
        self.logger.info(f"Handling analysis request for group: {group}")
        
        # Get available file types in the group
        file_types = self.get_file_types_in_group(group)
        
        # Check if a specific file type is mentioned
        mentioned_file_type = None
        for file_type in file_types:
            if file_type in query.lower() or (file_type.startswith('.') and file_type[1:] in query.lower()):
                mentioned_file_type = file_type
                break
        
        if mentioned_file_type:
            # User mentioned a specific file type
            self.logger.info(f"User mentioned file type: {mentioned_file_type}")
            return self.filter_and_display_files(group, mentioned_file_type)
        else:
            # No specific file type mentioned, ask for clarification
            self.set_awaiting_clarification(True)
            self.set_clarification_type("file_type_selection")
            self.set_clarification_group(group)
            
            file_type_options = ", ".join([f"{ft}" for ft in file_types])
            return f"What type of data would you like to analyze in the {group} group? Available formats: {file_type_options}"
        
    def display_group_files(self, group: str) -> str:
        """
        Display all files in a group categorized by type.
        
        Args:
            group: The group to display files for
            
        Returns:
            Formatted string with file information
        """
        files = self.list_files_in_group(group)
        if not files:
            return f"The group '{group}' doesn't contain any files."
            
        # Record this group in viewed groups
        if group not in self.context.session_data["viewed_groups"]:
            self.context.session_data["viewed_groups"].append(group)
            
        # Organize files by type
        files_by_type = {}
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if not ext:
                ext = "(no extension)"
                
            if ext not in files_by_type:
                files_by_type[ext] = []
                
            files_by_type[ext].append(file)
            
        # Format the response
        response = [f"Files in group '{group}':"]
        
        for ext, file_list in files_by_type.items():
            response.append(f"\n{ext} files ({len(file_list)}):")
            for file in file_list:
                response.append(f"- {file}")
                
        return "\n".join(response)
        
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
