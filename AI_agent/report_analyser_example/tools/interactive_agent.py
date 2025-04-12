import os
import sys
import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from dir_discover import discover_reports
import text_reader
import json_reader
import image_reader
import csv_reader
import website_reader
import combine_reader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InteractiveReportAgent:
    """
    An interactive agent that can understand user queries and route them to specialized readers.
    """
    def __init__(self, base_dir="/data/SWATGenXApp/Users/admin/Reports/"):
        """Initialize the agent with report data and LLM backend"""
        self.base_dir = base_dir
        self.reports_dict = discover_reports(silent=True, recursive=True)
        
        # If no reports found, raise ValueError
        if not self.reports_dict:
            raise ValueError(f"No reports found in {base_dir}")
        
        # Set latest report as default
        self.current_report = sorted(self.reports_dict.keys())[-1]
        self.current_group = None
        
        # Initialize the conversation history
        self.conversation_history = []
        
        # Initialize the LLM agent
        self.agent = Agent(
            model=OpenAIChat(id="gpt-4o"),
            markdown=True,
            instructions=[
                "You are HydroInsight, an assistant specialized in analyzing environmental and water resources reports.",
                "You help users navigate and analyze groundwater, climate, land use, and other environmental data.",
                "Respond conversationally while understanding technical details about water resources.",
                "Your goal is to help the user understand their data by directing them to appropriate analysis tools."
            ],
        )
        
        logger.info(f"Initialized with {len(self.reports_dict)} reports")
        logger.info(f"Current report set to: {self.current_report}")
        
        # Map of keywords to report groups
        self.keyword_to_group = {
            # Climate-related keywords
            "climate": "climate_change",
            "temperature": "climate_change", 
            "precipitation": "climate_change",
            "weather": "climate_change",
            "rainfall": "climate_change",
            "drought": "climate_change",
            
            # Groundwater-related keywords
            "groundwater": "groundwater",
            "aquifer": "groundwater",
            "well": "groundwater",
            "water table": "groundwater",
            "water level": "groundwater",
            
            # Crop/land use related keywords
            "crop": "cdl",
            "land use": "cdl",
            "agriculture": "cdl",
            "farming": "cdl",
            "cdl": "cdl",
            "cropland": "cdl",
            
            # Soil-related keywords
            "soil": "soil",
            "erosion": "soil",
            "texture": "soil",
            "fertility": "soil",
            
            # PRISM-related keywords
            "prism": "prism",
            "precipitation": "prism",
            "elevation": "prism",
            
            # MODIS-related keywords
            "modis": "modis",
            "satellite": "modis",
            "vegetation": "modis",
            "ndvi": "modis",
            "evi": "modis",
            "remote sensing": "modis",
        }
        
        # Map of file types to readers
        self.filetype_to_reader = {
            ".csv": self._read_csv,
            ".md": self._read_text,
            ".txt": self._read_text,
            ".png": self._read_image,
            ".jpg": self._read_image,
            ".jpeg": self._read_image,
            "config.json": self._read_json,
            "combined": self._read_combined,
            "website": self._read_website
        }
    
    def list_reports(self) -> List[str]:
        """Return list of available reports"""
        return sorted(self.reports_dict.keys())
    
    def list_groups(self, report: Optional[str] = None) -> List[str]:
        """Return list of available groups in the report"""
        report_id = report or self.current_report
        if report_id in self.reports_dict:
            return sorted(self.reports_dict[report_id]["groups"].keys())
        return []
    
    def list_files_by_type(self, group: Optional[str] = None, report: Optional[str] = None, file_type: Optional[str] = None) -> Dict[str, List[str]]:
        """Return dictionary of files organized by type"""
        report_id = report or self.current_report
        group_id = group or self.current_group
        
        if not group_id:
            logger.warning("No group selected")
            return {}
            
        if report_id in self.reports_dict and group_id in self.reports_dict[report_id]["groups"]:
            files_dict = self.reports_dict[report_id]["groups"][group_id]["files"]
            
            if file_type:
                return {file_type: list(files_dict.get(file_type, {}).keys())}
            return {ext: list(files.keys()) for ext, files in files_dict.items()}
            
        return {}
    
    def set_current_report(self, report: str) -> bool:
        """Set the current report to work with"""
        if report in self.reports_dict:
            self.current_report = report
            logger.info(f"Current report set to: {report}")
            return True
        return False
    
    def set_current_group(self, group: str) -> bool:
        """Set the current group to work with"""
        if self.current_report and group in self.reports_dict[self.current_report]["groups"]:
            self.current_group = group
            logger.info(f"Current group set to: {group}")
            return True
        return False
    
    def _read_text(self, file_path: str) -> str:
        """Process text files using the text reader"""
        return text_reader.analyze_report(file_path)
    
    def _read_json(self, file_path: str) -> str:
        """Process JSON files using the JSON reader"""
        return json_reader.json_reader(file_path)
    
    def _read_image(self, file_path: str) -> str:
        """Process image files using the image reader"""
        return image_reader.image_reader(file_path)
    
    def _read_csv(self, file_path: str) -> str:
        """Process CSV files using the CSV reader"""
        return csv_reader.csv_reader(file_path, recreate_db=True)
    
    def _read_website(self, url: str) -> str:
        """Process websites using the website reader"""
        return website_reader.website_reader(url)
    
    def _read_combined(self, report: str, group: str) -> str:
        """Process entire group using the combined reader"""
        return combine_reader.combined_reader(self.reports_dict, report, group, recreate_db=True)
    
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
            
            if file_type in self.filetype_to_reader:
                reader_func = self.filetype_to_reader[file_type]
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
            
            result = self._read_combined(report_id, group_id)
            # Ensure we return a string even if _read_combined returns None
            if result is None:
                return f"Analysis completed for {group_id}, but no results were returned."
            return result
        
        return f"Group {group_id} not found in report {report_id}"
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze user query to determine intention and relevant files"""
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Create a context for the agent
        report_list = self.list_reports()
        current_report_str = f"Current report: {self.current_report}"
        
        group_list = self.list_groups()
        current_group_str = f"Current group: {self.current_group}" if self.current_group else "No group selected"
        
        files_str = ""
        if self.current_group:
            files_dict = self.list_files_by_type()
            for ext, files in files_dict.items():
                files_str += f"\n- {ext} files: {', '.join(files)}"
        
        # Build the context prompt for analysis
        prompt = f"""
The user has provided the following query about their environmental data report: 
"{query}"

Based on this query, I need to:
1. Identify what type of data or information they're looking for
2. Determine which report group (climate_change, groundwater, cdl, soil, prims, modis, etc.) is most relevant
3. Determine what file type or analysis method would best answer their question

SYSTEM CONTEXT:
{current_report_str}
{current_group_str}
Available reports: {', '.join(report_list)}
Available groups: {', '.join(group_list)}
{files_str}

Please provide a structured analysis in the following JSON-formatted manner (don't include backticks):
{{
  "interpreted_query": "Brief rephrasing of what the user is asking for",
  "file_types": ["csv", "md", "png", etc...],
  "suggested_group": "most relevant group name",
  "suggested_action": "read_specific_file or analyze_entire_group",
  "specific_file": "filename if a specific file is relevant, otherwise null",
  "explanation": "Brief explanation of your reasoning"
}}
"""
        
        # Get the agent's analysis - use print_response instead of generate
        response = self.agent.print_response(prompt)
        
        # Check if response is None and provide a default
        if response is None:
            logger.warning("Agent returned None response")
            return {
                "interpreted_query": query,
                "file_types": [],
                "suggested_group": self.current_group,
                "suggested_action": "analyze_entire_group",
                "specific_file": None,
                "explanation": "Failed to generate agent response"
            }
        
        # Parse JSON from the response
        # Find JSON content between curly braces
        match = re.search(r'({.*})', response.replace('\n', ' '), re.DOTALL)
        if match:
            import json
            try:
                result = json.loads(match.group(1))
                logger.info(f"Successfully parsed query intent: {result}")
                return result
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON from response")
        
        # If we can't parse JSON, return default
        logger.warning("Could not extract JSON from response")
        return {
            "interpreted_query": query,
            "file_types": [],
            "suggested_group": self.current_group,
            "suggested_action": "analyze_entire_group",
            "specific_file": None,
            "explanation": "Failed to parse agent response"
        }
    
    def process_query(self, query: str) -> str:
        """Process a user query and return appropriate response"""
        # Handle empty queries
        if not query or query.strip() == "":
            return "Please enter a query or type 'help' for available commands."
            
        # Handle common greetings
        greetings = ["hi", "hello", "hey", "greetings", "howdy"]
        if query.lower() in greetings:
            return f"Hello! I'm your hydrology data assistant. Currently analyzing report {self.current_report}, group {self.current_group or 'none selected'}. How can I help you analyze this data today?"
        
        # Handle system commands first
        if query.lower().startswith("list reports"):
            reports = self.list_reports()
            return f"Available reports:\n" + "\n".join([f"- {r}" for r in reports])
        
        if query.lower().startswith("list groups"):
            groups = self.list_groups()
            return f"Available groups in report {self.current_report}:\n" + "\n".join([f"- {g}" for g in groups])
        
        if query.lower().startswith("set report "):
            report_id = query[11:].strip()
            if self.set_current_report(report_id):
                return f"Current report set to: {report_id}"
            return f"Report {report_id} not found"
        
        if query.lower().startswith("set group "):
            group_id = query[10:].strip()
            if self.set_current_group(group_id):
                return f"Current group set to: {group_id}"
            return f"Group {group_id} not found in report {self.current_report}"
        
        if query.lower() == "help":
            return """
Available commands:
- list reports: Show all available reports
- list groups: Show all groups in the current report
- set report [report_id]: Set the current report
- set group [group_id]: Set the current group
- help: Show this help message

You can also ask natural language questions about the reports, such as:
- Show me the groundwater data
- Analyze the climate change trends
- What do the crop rotation images tell us?
- Give me statistics from the soil data
            """
        
        # Check for direct group name input
        if query in self.list_groups():
            self.set_current_group(query)
            result = self.analyze_group(query)
            return result if result is not None else f"Analysis completed for {query}, but no results were returned."
        
        # For natural language queries, analyze and route to appropriate reader
        analysis = self.analyze_query(query)
        
        # Check if we need to switch groups based on the query
        suggested_group = analysis.get("suggested_group")
        
        # If query suggests a different group, check if it exists and switch
        if (suggested_group and 
            (not self.current_group or suggested_group != self.current_group) and 
            suggested_group in self.list_groups()):
            
            self.set_current_group(suggested_group)
            group_switch_msg = f"Switched to group: {suggested_group}\n\n"
        else:
            group_switch_msg = ""
            
            # Try to detect keywords in query if no group is selected yet
            if not self.current_group:
                for keyword, group in self.keyword_to_group.items():
                    if keyword.lower() in query.lower() and group in self.list_groups():
                        self.set_current_group(group)
                        group_switch_msg = f"Switched to group: {group}\n\n"
                        break
        
        # Still no group? Ask user to select one
        if not self.current_group:
            groups = self.list_groups()
            return f"Please select a group to analyze:\n" + "\n".join([f"- {g}" for g in groups]) + "\n\nUse 'set group [group_name]' to select."
        
        # Determine action based on analysis
        action = analysis.get("suggested_action", "analyze_entire_group")
        
        if action == "read_specific_file" and analysis.get("specific_file"):
            # Find the file in available files
            file_name = analysis.get("specific_file")
            files_by_type = self.list_files_by_type()
            
            for file_type, files in files_by_type.items():
                matching_files = [f for f in files if file_name.lower() in f.lower()]
                if matching_files:
                    # Use the first matching file
                    result = self.process_file(matching_files[0], file_type)
                    if result is None:
                        return f"Processed file {matching_files[0]}, but no results were returned."
                    return result
            
            # If we get here, no matching file was found
            return f"No matching file found for '{file_name}'. Available files: {', '.join([f for files in files_by_type.values() for f in files])}"
        
        # Default to analyzing the entire group
        result = self.analyze_group()
        if result is None:
            return f"{group_switch_msg}Analysis completed for {self.current_group}, but no results were returned."
        return f"{group_switch_msg}{result}"
    
    def chat(self):
        """Start an interactive chat session with the user"""
        print("🌊 Welcome to HydroInsight - Your Water Resources Analysis Assistant 🌊")
        print("Type 'help' for available commands or 'exit' to quit")
        print(f"Current report: {self.current_report}")
        print(f"Available groups: {', '.join(self.list_groups())}")
        
        while True:
            user_input = input("\n> ")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Goodbye!")
                break
            
            if not user_input.strip():
                print("\nPlease enter a query or type 'help' for available commands.")
                continue
                
            print("\nProcessing your request...")
            
            # Ensure we handle None values properly
            response = self.process_query(user_input)
            if response is None:
                response = "No response was generated. This may indicate an issue with processing your request."
                
            print("\n" + response)

if __name__ == "__main__":
    agent = InteractiveReportAgent()
    agent.chat()
