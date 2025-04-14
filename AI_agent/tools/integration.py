import os
import sys
import logging
import argparse
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import re

# Import our components
from interactive_agent import InteractiveReportAgent
from ContextMemory import ContextMemory
from KnowledgeGraph import KnowledgeGraph
from tools.ai_logger import LoggerSetup

# Define log path
LOG_PATH = "/data/SWATGenXApp/codes/AI_agent/logs"

# Create directory if it doesn't exist
os.makedirs(LOG_PATH, exist_ok=True)

# Clear any existing root logging configuration
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Initialize central logger - set verbose to False for cleaner console output
logger_setup = LoggerSetup(report_path=LOG_PATH, verbose=False)
logger = logger_setup.setup_logger(name="report_analyzer")

class EnhancedReportAnalyzer:
    """
    Integrates our report agent with enhanced memory and knowledge graph capabilities.
    """
    
    def __init__(self, base_dir: str = "/data/SWATGenXApp/Users/admin/Reports/", logger=None, model_id: str = "gpt-4o"):
        """
        Initialize the enhanced report analyzer.
        
        Args:
            base_dir: Base directory for reports
            logger: Logger instance to use
            model_id: ID of the model to use for generation
        """
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info(f"Initializing EnhancedReportAnalyzer with model: {model_id}")
        
        # Store model ID
        self.model_id = model_id
        
        # Initialize single ContextMemory for shared use
        self.context = ContextMemory(storage_path="report_analyzer_memory.db", logger=self.logger)
        
        # Initialize knowledge graph
        self.graph = KnowledgeGraph(storage_path="report_knowledge.json", logger=self.logger)
        
        # Start a new session
        self.session_id = self.context.start_session({
            "base_dir": base_dir,
            "model_id": model_id,
            "start_time": self.context.session_id  # Use timestamp as ID
        })
        
        # Initialize the agent with our base directory, shared context, and logger
        self.agent = InteractiveReportAgent(base_dir=base_dir, context=self.context, logger=self.logger)
        
        # Initialize domain concepts in knowledge graph
        concepts_added = self.graph.initialize_domain_concepts()
        self.logger.info(f"Initialized knowledge graph with {concepts_added} initial concepts")
    
    def process_query(self, query: str, session_id: Optional[str] = None) -> str:
        """
        Process a user query with enhanced capabilities and autonomous data access.
        
        Args:
            query: User query
            session_id: Optional session ID for persistence across calls
            
        Returns:
            Response for the user
        """
        self.logger.info(f"Processing query with model {self.model_id}: {query}")
        
        # If a session_id is provided, log it but don't try to restore
        # The ContextMemory class doesn't have a restore_session method
        if session_id and session_id != "None" and session_id != self.session_id:
            self.logger.info(f"Received request with session_id: {session_id}")
            # Just store the session_id for continuity
            self.session_id = session_id
        
        # Save query in memory
        self.context.add_message("user", query)
        
        # Check if user is asking about general available data
        query_lower = query.lower()
        if any(phrase == query_lower.strip() for phrase in ["what data", "available data", "data you have", "data u have", "what information", "what reports"]):
            self.logger.info("User is asking about general available data. Providing complete data overview.")
            response = self._provide_data_overview()
            self.context.add_message("assistant", response)
            return response
            
        # Check if user is asking about specific type of data
        data_topic_patterns = [
            r"what data (?:do )?(?:we|you) have (?:on|about|for) ([\w\s]+)",
            r"what ([\w\s]+) data (?:do )?(?:we|you) have",
            r"(?:do )?(?:we|you) have data (?:on|about|for) ([\w\s]+)",
            r"tell me about (?:the )?([\w\s]+) data"
        ]
        
        for pattern in data_topic_patterns:
            match = re.search(pattern, query_lower)
            if match:
                topic = match.group(1).strip()
                self.logger.info(f"User is asking about specific data topic: '{topic}'")
                response = self._provide_topic_data_overview(topic)
                self.context.add_message("assistant", response)
                return response
        
        # Check if this is a knowledge graph specific query
        if query.lower().startswith("show knowledge") or "knowledge graph" in query.lower():
            # Generate and save a visualization
            self.graph.visualize("knowledge_graph.png")
            return "I've generated a visualization of my knowledge graph. You can view it in knowledge_graph.png."
        
        # Check if this is a memory-specific query
        if query.lower().startswith("show memory") or "what do you remember" in query.lower():
            insights = self.context.get_insights(limit=10)
            if not insights:
                return "I haven't recorded any insights yet. Let's analyze some data first!"
            
            response = "Here are the key insights I remember:\n\n"
            for i, insight in enumerate(insights):
                response += f"{i+1}. {insight['content']}\n   (from {insight['source']})\n\n"
            
            self.context.add_message("assistant", response)
            return response

        # ENHANCED AUTONOMOUS PROCESSING
        # 1. Directly analyze the query to determine what data would be relevant
        data_topics = self._extract_data_topics(query)
        self.logger.info(f"Extracted data topics: {data_topics}")
        
        # 2. Find most relevant report and group based on the topics
        relevant_data = self._find_relevant_data(data_topics)
        
        # 3. Analyze the data directly without asking user for choices
        if relevant_data:
            report_id, group_id, file_hints = relevant_data
            response = self._autonomous_analysis(query, report_id, group_id, file_hints)
        else:
            # If we couldn't determine relevant data, fall back to the agent
            self.logger.info("No specific relevant data found, passing query to agent")
            response = self.agent.process_query(query)

        # Save response in memory
        self.context.add_message("assistant", response)
        
        # Extract concepts from the query and response and add to knowledge graph
        self.graph.process_text(query, "user_query")
        self.graph.process_text(response, "agent_response")
        
        # Save the updated graph
        self.graph.save_graph()
        
        return response
        
    def _extract_data_topics(self, query: str) -> List[str]:
        """Extract relevant data topics from user query for autonomous mode."""
        # Topic mapping to normalized terms (expand as needed)
        topic_mapping = {
            "climat": ["climate", "temperature", "precipitation", "rainfall", "weather", "warming"],
            "groundwater": ["groundwater", "aquifer", "well", "water level", "water table"],
            "soil": ["soil", "land", "erosion", "fertility"],
            "agriculture": ["crop", "farm", "agri", "irrigation", "yield"],
            "hydrology": ["water", "river", "stream", "flow", "flood", "drought"]
        }
        
        # Check for mention of topics in the query
        found_topics = []
        query_lower = query.lower()
        
        for topic, keywords in topic_mapping.items():
            if any(keyword in query_lower for keyword in keywords):
                found_topics.append(topic)
                
        # If no specific topics found but query seems analysis-oriented, return general analysis hint
        if not found_topics and any(term in query_lower for term in ["analyze", "report", "data", "show", "tell me about"]):
            # Try to match any existing group name
            groups = []
            for report_id in self.agent.reports_dict:
                for group_id in self.agent.reports_dict[report_id].get("groups", {}):
                    if group_id.lower() in query_lower:
                        found_topics.append(group_id)
                        return found_topics
            
            # If nothing specific found but analysis requested, do general analysis
            found_topics.append("general_analysis")
        
        return found_topics
    
    def _find_relevant_data(self, topics: List[str]) -> Optional[Tuple[str, str, List[str]]]:
        """Find the most relevant report, group, and potential files based on topics."""
        if not topics:
            return None
            
        # Get all available reports sorted by timestamp (newest first)
        available_reports = sorted(self.agent.reports_dict.keys(), reverse=True)
        if not available_reports:
            return None
            
        # Take the most recent report
        report_id = available_reports[0]
        
        # Topic to group mapping - expand as needed
        topic_group_mapping = {
            "climat": "climate_change",
            "groundwater": "groundwater",
            "soil": "gssurgo",
            "agriculture": "cdl",
            "hydrology": "prism", # Or other relevant group
            "general_analysis": None # Will select based on other criteria
        }
        
        # Try to find a matching group for the requested topics
        group_id = None
        for topic in topics:
            if topic in topic_group_mapping:
                mapped_group = topic_group_mapping[topic]
                if mapped_group and mapped_group in self.agent.reports_dict[report_id].get("groups", {}):
                    group_id = mapped_group
                    break
            # Direct match if topic is an actual group name
            elif topic in self.agent.reports_dict[report_id].get("groups", {}):
                group_id = topic
                break
        
        # If no specific group matches, choose the first available group
        if not group_id:
            groups = list(self.agent.reports_dict[report_id].get("groups", {}).keys())
            if groups:
                group_id = groups[0]
            else:
                return None  # No groups available
        
        # Provide hints about file types to analyze based on topics
        file_hints = self._get_file_hints(topics, report_id, group_id)
        
        return (report_id, group_id, file_hints)
    
    def _get_file_hints(self, topics: List[str], report_id: str, group_id: str) -> List[str]:
        """Generate hints about what file types or specific files might be relevant based on topics."""
        file_hints = []
        
        # Check what files are available in this group
        if report_id in self.agent.reports_dict and group_id in self.agent.reports_dict[report_id].get("groups", {}):
            files_dict = self.agent.reports_dict[report_id]["groups"][group_id].get("files", {})
            
            # Prioritize certain file types based on topic
            if "climat" in topics:
                # For climate topics, prioritize markdown reports and CSV data
                if '.md' in files_dict:
                    for filename in files_dict['.md']:
                        if 'climate' in filename.lower():
                            file_hints.append(filename)
                if '.csv' in files_dict:
                    for filename in files_dict['.csv']:
                        if 'temp' in filename.lower() or 'precip' in filename.lower():
                            file_hints.append(filename)
                # Also look for climate-related visualizations
                if '.png' in files_dict:
                    for filename in files_dict['.png']:
                        if 'climate' in filename.lower() or 'temp' in filename.lower():
                            file_hints.append(filename)
            
            # Add more topic-specific file hint logic for other topics here
            
            # If no specific files found by keyword, include one of each main type if available
            if not file_hints:
                for ext in ['.md', '.csv', '.png']:
                    if ext in files_dict and files_dict[ext]:
                        # Add the first file of each type as a fallback
                        file_hints.append(next(iter(files_dict[ext].keys())))
        
        return file_hints
    
    def _autonomous_analysis(self, query: str, report_id: str, group_id: str, file_hints: List[str]) -> str:
        """Autonomously analyze relevant data based on the query and return insights."""
        self.logger.info(f"Performing autonomous analysis: report={report_id}, group={group_id}, file_hints={file_hints}")
        
        # Set these as the current report and group in the agent
        self.agent.set_current_report(report_id)
        self.agent.set_current_group(group_id)
        
        collected_insights = []
        analysed_something = False
        
        # First, try to analyze specifically hinted files if they exist
        if file_hints:
            for filename in file_hints:
                file_ext = os.path.splitext(filename)[1].lower()
                try:
                    file_path = self.agent.get_file_path(group_id, filename)
                    if file_path:
                        self.logger.info(f"Analyzing specific file: {filename}")
                        result = self.agent.process_file(filename, file_ext, group_id)
                        if result:
                            collected_insights.append(f"Analysis of {filename}:\n{result}")
                            analysed_something = True
                            # Don't analyze too many files to avoid verbose output
                            if len(collected_insights) >= 2:
                                break
                except Exception as e:
                    self.logger.error(f"Error analyzing file {filename}: {e}")
        
        # If no specific files were successfully analyzed, try group analysis
        if not analysed_something:
            try:
                self.logger.info(f"Analyzing entire group: {group_id}")
                result = self.agent.analyze_group(group_id)
                if result:
                    collected_insights.append(result)
                    analysed_something = True
            except Exception as e:
                self.logger.error(f"Error analyzing group {group_id}: {e}")
        
        # If we have insights, synthesize them into a cohesive response
        if collected_insights:
            # Use agent's own reasoning to synthesize the insights
            from agno.agent import Agent
            from agno.models.openai import OpenAIChat
            from agno.exceptions import ModelProviderError
            
            # Try using the specified model_id, with fallbacks if not available
            try:
                synthesis_agent = Agent(
                    model=OpenAIChat(id=self.model_id),
                    instructions=[
                        "You are HydroInsight, an AI that provides insights on environmental and water resources data.",
                        "Synthesize the following analysis results into a clear, conversational response.",
                        "Focus on the most important findings and their implications.",
                        "Directly address the user's query without stating that you analyzed files or performed analysis.",
                        "Make the response sound natural, as if from a knowledgeable expert, not a data retrieval system.",
                        "Do not mention file names, data sources, or the analysis process in your response."
                    ]
                )
            except ModelProviderError:
                # First fallback: Try gpt-4 model
                try:
                    self.logger.warning(f"Model {self.model_id} not available. Falling back to gpt-4.")
                    synthesis_agent = Agent(
                        model=OpenAIChat(id="gpt-4"),
                        instructions=[
                            "You are HydroInsight, an AI that provides insights on environmental and water resources data.",
                            "Synthesize the following analysis results into a clear, conversational response.",
                            "Focus on the most important findings and their implications.",
                            "Directly address the user's query without stating that you analyzed files or performed analysis.",
                            "Make the response sound natural, as if from a knowledgeable expert, not a data retrieval system.",
                            "Do not mention file names, data sources, or the analysis process in your response."
                        ]
                    )
                except ModelProviderError:
                    # Second fallback: Try gpt-3.5-turbo as a last resort
                    self.logger.warning("GPT-4 not available. Falling back to gpt-3.5-turbo.")
                    synthesis_agent = Agent(
                        model=OpenAIChat(id="gpt-3.5-turbo"),
                        instructions=[
                            "You are HydroInsight, an AI that provides insights on environmental and water resources data.",
                            "Synthesize the following analysis results into a clear, conversational response.",
                            "Focus on the most important findings and their implications.",
                            "Directly address the user's query without stating that you analyzed files or performed analysis.",
                            "Make the response sound natural, as if from a knowledgeable expert, not a data retrieval system.",
                            "Do not mention file names, data sources, or the analysis process in your response."
                        ]
                    )
            
            # Define newline separator outside of f-string
            nl_separator = "\n\n"
            prompt = f"""
            User query: {query}
            
            Analysis results:
            {nl_separator.join(collected_insights)}
            
            Synthesize these results into a clear, conversational response that directly addresses the user's query.
            Your response should integrate the key findings without mentioning the files or analysis process.
            Make it sound natural, as if from a knowledgeable expert, not a data retrieval system.
            """
            
            # Try to get synthesized response with error handling
            try:
                synthesized_response = synthesis_agent.print_response(prompt)
                if synthesized_response:
                    return synthesized_response
            except Exception as e:
                self.logger.error(f"Error during synthesis: {str(e)}")
                self.logger.info("Falling back to returning collected insights directly")
                
            # Fallback to just returning the collected insights if synthesis fails
            return "\n\n".join(collected_insights)
        
        # If no insights were collected, fall back to the agent's normal processing
        self.logger.info("No insights collected, falling back to agent processing")
        return self.agent.process_query(query)
    
    def _provide_data_overview(self) -> str:
        """Provides an overview of all available data in the system"""
        self.logger.info("Generating data overview")
        
        # Get all available reports
        available_reports = sorted(self.agent.reports_dict.keys(), reverse=True)
        
        if not available_reports:
            return "I don't currently have access to any reports. Please check if the data directories are properly configured."
        
        # Structure to store our overview
        overview = ["I have access to the following data:"]
        
        # Count total groups and files
        total_groups = 0
        total_files = 0
        file_type_counts = {}
        
        # Go through each report
        for report_id in available_reports:
            report_info = self.agent.reports_dict[report_id]
            groups = report_info.get("groups", {})
            
            # Count groups in this report
            report_groups = len(groups)
            total_groups += report_groups
            
            # Initialize files counter for this report
            report_files = 0
            
            # Add report to overview
            report_date = report_id.split('_')[0]  # Extract date part from ID
            overview.append(f"\n## Report {report_id} (Date: {report_date})")
            
            # Add groups in this report
            if groups:
                group_summaries = []
                for group_id, group_info in groups.items():
                    files = group_info.get("files", {})
                    group_file_count = sum(len(files.get(ext, {})) for ext in files)
                    report_files += group_file_count
                    
                    # Get file type distribution in this group
                    file_types = []
                    for ext, files_of_type in files.items():
                        file_count = len(files_of_type)
                        file_types.append(f"{file_count} {ext} files")
                        
                        # Update global file type counter
                        file_type_counts[ext] = file_type_counts.get(ext, 0) + file_count
                    
                    # Create summary for this group
                    file_types_str = ", ".join(file_types)
                    group_summaries.append(f"- {group_id}: {group_file_count} files ({file_types_str})")
                
                overview.append("### Groups:")
                overview.extend(group_summaries)
            else:
                overview.append("### Groups: None found")
            
            total_files += report_files
        
        # Add summary statistics
        overview.insert(1, f"\n# Summary: {len(available_reports)} reports with {total_groups} groups containing {total_files} files")
        
        # Add file type distribution
        file_type_summary = []
        for ext, count in sorted(file_type_counts.items(), key=lambda x: x[1], reverse=True):
            file_type_summary.append(f"- {ext}: {count} files")
        
        if file_type_summary:
            overview.append("\n## File Types:")
            overview.extend(file_type_summary)
        
        # Add examples of the types of analysis that can be performed
        overview.append("\n## Available Analyses:")
        overview.append("- Climate change analysis")
        overview.append("- Crop data layer (CDL) analysis")
        overview.append("- Groundwater conditions assessment")
        overview.append("- Land use and land cover trends")
        overview.append("- Statistical summaries of temporal data")
        
        # Add instructions for the user
        overview.append("\nYou can ask me to analyze specific aspects of this data, compare trends over time, or provide summaries of particular groups or reports.")
        
        return "\n".join(overview)
    
    def _provide_topic_data_overview(self, topic: str) -> str:
        """Provides an overview of data related to a specific topic."""
        self.logger.info(f"Generating focused data overview for topic: {topic}")
        
        # Topic to keywords mapping to improve matching
        topic_keywords = {
            "land": ["land", "landcover", "land use", "land cover", "cdl", "modis", "gssurgo", "soil", "crop"],
            "climate": ["climate", "temperature", "precipitation", "rainfall", "weather", "climate_change", "prism"],
            "groundwater": ["groundwater", "aquifer", "water level", "water table", "well"],
            "water": ["water", "hydrology", "river", "stream", "flow", "snodas", "prism", "precipitation"],
            "solar": ["solar", "radiation", "nsrdb", "sun"],
            "crop": ["crop", "agriculture", "cdl", "harvest", "yield"],
            "soil": ["soil", "gssurgo", "land", "erosion", "fertility"],
            "snow": ["snow", "snodas", "water equivalent", "snowpack"],
            "vegetation": ["vegetation", "modis", "landcover", "ndvi", "evi"]
        }
        
        # Find relevant keywords for the topic
        relevant_keywords = []
        for key, words in topic_keywords.items():
            if topic in words or any(word in topic for word in words):
                relevant_keywords.extend(words)
                
        # If no matching keywords found, just use the topic itself
        if not relevant_keywords:
            relevant_keywords = [topic]
            
        self.logger.info(f"Keywords for topic '{topic}': {relevant_keywords}")
        
        # Get all available reports
        available_reports = sorted(self.agent.reports_dict.keys(), reverse=True)
        
        if not available_reports:
            return f"I don't currently have access to any reports that might contain data about {topic}. Please check if the data directories are properly configured."
        
        # Structure to store our overview
        overview = [f"Here's what I found related to {topic}:"]
        
        # Mapping between keywords and relevant group IDs
        group_relevance = {
            "cdl": ["land use", "agriculture", "crop", "landcover"],
            "modis": ["vegetation", "landcover", "land use"],
            "gssurgo": ["soil", "land", "agriculture"],
            "groundwater": ["groundwater", "water", "aquifer"],
            "climate_change": ["climate", "temperature", "precipitation"],
            "prism": ["climate", "precipitation", "rainfall", "weather"],
            "snodas": ["snow", "water", "precipitation"],
            "nsrdb": ["solar", "radiation", "climate"]
        }
        
        # Track relevant groups and files found
        relevant_groups_found = False
        relevant_files_found = False
        total_relevant_files = 0
        matched_groups_by_report = {}
        
        # Find the most recent report with relevant data
        most_recent_report = None
        
        # Go through each report to find relevant groups and files
        for report_id in available_reports:
            report_info = self.agent.reports_dict[report_id]
            groups = report_info.get("groups", {})
            
            # Find groups that might be relevant to the topic
            matched_groups = []
            
            for group_id, group_info in groups.items():
                # Check if group matches topic directly
                group_relevant = False
                
                # Check if group ID directly matches any keyword
                if any(keyword in group_id.lower() for keyword in relevant_keywords):
                    group_relevant = True
                
                # Check if group is relevant to our topic based on the mapping
                if group_id in group_relevance:
                    relevant_topics = group_relevance[group_id]
                    if any(kw in topic.lower() for kw in relevant_topics) or \
                       any(kw in relevant_topics for kw in relevant_keywords):
                        group_relevant = True
                
                if group_relevant:
                    matched_groups.append(group_id)
                    relevant_groups_found = True
                    
                    # Check files in this group to find particularly relevant ones
                    files = group_info.get("files", {})
                    relevant_files = []
                    
                    for ext, files_of_type in files.items():
                        for filename, file_info in files_of_type.items():
                            # Check if filename contains any of our keywords
                            if any(keyword in filename.lower() for keyword in relevant_keywords):
                                relevant_files.append((filename, ext))
                                relevant_files_found = True
                                total_relevant_files += 1
                    
                    # Store the relevant files for this group
                    if relevant_files:
                        if report_id not in matched_groups_by_report:
                            matched_groups_by_report[report_id] = {}
                        
                        matched_groups_by_report[report_id][group_id] = relevant_files
            
            # Store the most recent report that has relevant groups
            if matched_groups and most_recent_report is None:
                most_recent_report = report_id
        
        if not relevant_groups_found:
            return f"I couldn't find any data specifically about {topic} in my available reports. Would you like to see the general data overview instead?"
        
        # Add summary of what was found
        if most_recent_report:
            report_date = most_recent_report.split('_')[0]
            overview.append(f"\n## Most Recent Data (Report {most_recent_report}, Date: {report_date})")
            
            # List relevant groups and files from the most recent report
            if most_recent_report in matched_groups_by_report:
                for group_id, relevant_files in matched_groups_by_report[most_recent_report].items():
                    overview.append(f"\n### Group: {group_id}")
                    
                    if relevant_files:
                        overview.append("Particularly relevant files:")
                        for filename, ext in relevant_files:
                            overview.append(f"- {filename}")
                    else:
                        # If no specifically relevant files, mention the group is relevant
                        overview.append(f"This group contains data related to {topic}.")
        
        # Add a summary of all reports with relevant data
        overview.append(f"\n## All Reports with {topic.title()} Data")
        
        for report_id in matched_groups_by_report.keys():
            report_date = report_id.split('_')[0]
            group_names = list(matched_groups_by_report[report_id].keys())
            overview.append(f"- Report {report_id} (Date: {report_date}): Groups {', '.join(group_names)}")
        
        # Add relevant analysis suggestions
        overview.append(f"\n## Suggested Analyses for {topic.title()} Data:")
        
        # Customize suggestions based on the topic
        if any(kw in topic.lower() for kw in ["land", "landcover", "cdl"]):
            overview.append("- Land use and land cover changes over time")
            overview.append("- Crop distribution analysis")
            overview.append("- Agricultural land use patterns")
        elif any(kw in topic.lower() for kw in ["climate", "temperature", "precipitation"]):
            overview.append("- Climate trends and anomalies")
            overview.append("- Precipitation patterns and changes")
            overview.append("- Temperature variation analysis")
        elif any(kw in topic.lower() for kw in ["groundwater", "water table", "aquifer"]):
            overview.append("- Groundwater level trends")
            overview.append("- Aquifer condition assessment")
            overview.append("- Well data time series analysis")
        elif any(kw in topic.lower() for kw in ["soil", "gssurgo"]):
            overview.append("- Soil type distribution")
            overview.append("- Soil characteristics analysis")
            overview.append("- Land capability classification")
            
        # Generic suggestions if none of the above
        if len(overview) < 3:
            overview.append(f"- {topic.title()} trend analysis")
            overview.append(f"- Statistical summary of {topic} data")
            overview.append(f"- {topic.title()} spatial distribution")
        
        # Add instructions for next steps
        overview.append(f"\nTo analyze this data, you could try asking questions like:")
        overview.append(f"- \"Analyze {topic} trends in the most recent report\"")
        overview.append(f"- \"Show me the {topic} data from the {most_recent_report} report\"")
        overview.append(f"- \"Compare {topic} patterns between different years\"")
        
        return "\n".join(overview)
    
    def interactive_session(self):
        """Start an interactive session with the user."""
        print("🌊 Welcome to the Enhanced Hydrology Report Analyzer 🌊")
        print("Type 'help' for available commands or 'exit' to quit")
        print("Additional commands:")
        print("- show knowledge: Visualize the knowledge graph")
        print("- show memory: Display insights I've remembered")
        
        while True:
            user_input = input("\n> ")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                # End session and save data
                self.context.end_session()
                self.graph.save_graph()
                print("\nThank you for using the Enhanced Hydrology Report Analyzer! Goodbye!")
                break
            
            if not user_input.strip():
                print("\nPlease enter a query or type 'help' for available commands.")
                continue
                
            print("\nProcessing your request...")
            
            # Process the query
            response = self.process_query(user_input)
            
            # Ensure we handle None values properly
            if response is None:
                response = "No response was generated. This may indicate an issue with processing your request."
                
            print("\n" + response)

def main():
    """Main entry point of the application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Enhanced Hydrology Report Analyzer")
    parser.add_argument("--base-dir", type=str, default="/data/SWATGenXApp/Users/admin/Reports/",
                      help="Base directory for reports")
    parser.add_argument("--model-id", type=str, default="gpt-4",
                      help="Model ID to use for generation")
    
    args = parser.parse_args()
    
    try:
        # Initialize the analyzer with the central logger
        analyzer = EnhancedReportAnalyzer(
            base_dir=args.base_dir, 
            logger=logger,
            model_id=args.model_id
        )
        
        # Start interactive session
        analyzer.interactive_session()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        print(f"An error occurred: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 