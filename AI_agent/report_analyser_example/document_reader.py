from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.knowledge.text import TextKnowledgeBase
from agno.knowledge.csv import CSVKnowledgeBase
from agno.knowledge.json import JSONKnowledgeBase
from agno.knowledge.docx import DocxKnowledgeBase
from agno.knowledge.website import WebsiteKnowledgeBase
from agno.knowledge.combined import CombinedKnowledgeBase
from agno.vectordb.pgvector import PgVector
from redis_tools import RedisTools
from data_visualizer import DataVisualizer
from knowledge_graph import KnowledgeGraph
from context_manager import context
from agno.agent import Agent
from agno.embedder.openai import OpenAIEmbedder
from agno.media import Image
from agno.models.openai import OpenAIChat
import os
import json
import pandas as pd
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import uuid

import re
import traceback
from PIL import Image as PILImage
import logging
from AI_agent.report_analyser_example.Logger import LoggerSetup
import os 
from utils import (
    extract_image_name,
    create_image_summary,
    extract_main_topic
)
from prompt_handler import (
                deduplicate_content,
                  calculate_similarity, 
                  is_response_complete,
                  clean_response_output,
                  extract_keywords,
                  should_visualize,
                  validate_response,
                  complete_response,
                  enhance_relevance,
)



logger = LoggerSetup(verbose=False, rewrite=True)   
logger = logger.setup_logger("AI_AgentLogger")


class QueryProcessor:
    def process_query(self, query):
        # 1. Check conversation history
        relevant_history = self.get_relevant_history(query)
        
        # 2. Search knowledge base
        kb_results = self.knowledge_base.search(query)
        
        # 3. Route to appropriate agent
        agent_response = self.coordinator.route_query(query)
        
        # 4. Combine and enhance response
        final_response = self.enhance_response(
            query, 
            relevant_history, 
            kb_results, 
            agent_response
        )
        
        return final_response


class InteractiveDocumentReader:
    """An interactive AI system for understanding and interpreting various document types."""
    
    def __init__(self, config=None, redis_url="redis://localhost:6379/0"):
        """Initialize the document reader with the given configuration."""
        self.logger = None
        self._setup_logger()
        
        # Initialize Redis connection
        self.redis_url = redis_url
        self.redis_client = None
        try:
            import redis
            self.redis_client = redis.from_url(redis_url)
            self.logger.info("Connected to Redis")
        except Exception as e:
            self.logger.warning(f"Could not connect to Redis: {str(e)}")
            
        # Store basic configuration
        self.config = config or {}
        self.session_id = os.environ.get('SESSION_ID', str(uuid.uuid4()))
        self.logger.info(f"Session ID: {self.session_id}")
        
        # Multi-agent system components
        self.interactive_agent = None
        self.knowledge_base = {}
        self.image_reader = None
        
        # Document discovery and processing
        self.discovered_files = {}
        self.base_path = None
        
        # Update the global context
        context.session_id = self.session_id
        context.discovered_files = self.discovered_files
        context.base_path = self.base_path
        
        # Set default conversation history
        self._sync_conversation_history()
        
        self.db_url = self.config.get('db_url', "postgresql+psycopg://ai:ai@localhost:5432/ai")
        self.knowledge_bases = {}
        self.combined_knowledge_base = None
        self.agent = None
        self.embedder = OpenAIEmbedder()
        self.image_agent = None
        self.images = []
        
        # New fields for interactive capabilities
        self.conversation_history = []
        self.data_summaries = {}
        
        # Multi-agent system
        self.agents = {
            "coordinator": None,
            "document_navigator": None,
            "visual_analyst": None,
            "data_scientist": None,
            "domain_expert": None
        }
        
        # State machine for conversation management
        self.conversation_state = {
            "current_topic": None,
            "depth_level": 0,
            "pending_requests": [],
            "answered_questions": set(),
            "follow_up_suggestions": []
        }
        
        # Performance metrics
        self.metrics = {
            "response_times": [],
            "query_success_rate": 0,
            "useful_responses": 0,
            "total_responses": 0
        }
        
        # Initialize Redis tools
        self.redis_tools = RedisTools(redis_url, self.session_id)
        
        # Initialize utility classes
        self.data_visualizer = None
        self.knowledge_graph_manager = None
        
        # Set up context with session info
        context.session_id = self.session_id
        if config:
            context.config = config
    
    def _setup_logger(self):
        """Setup the logger for the document reader."""
        self.logger = LoggerSetup(verbose=False, rewrite=False) 
        self.logger = self.logger.setup_logger("AI_AgentLogger")
        
    def _sync_conversation_history(self):
        """Sync conversation history with Redis."""
        from context_manager import context
        
        if not self.redis_client:
            return False
            
        try:
            # Attempt to retrieve existing context from Redis
            redis_context = self.redis_client.get(f"context:{self.session_id}")
            if redis_context:
                # Load and update our local context
                updated_data = json.loads(redis_context)
                context.conversation_history = updated_data.get('conversation_history', [])
                self.logger.info(f"Loaded conversation history from Redis ({len(context.conversation_history)} messages)")
                return True
                
            return False
        except Exception as e:
            self.logger.error(f"Error syncing conversation history: {str(e)}")
            return False
    
    def initialize(self, auto_discover=False, base_path=None):
        """Initialize the interactive document reader with optional file discovery."""
        # Configure logging first
        self.logger = LoggerSetup(verbose=False, rewrite=True)
        self.logger = self.logger.setup_logger("AI_AgentLogger")
        
        logger.info("Initializing InteractiveDocumentReader")
        
        # Initialize metrics
        self.metrics = {
            "response_times": [],
            "total_responses": 0,
            "conversations": 0
        }
        
        # Initialize document paths
        self.discovered_files = {
            'csv': [],
            'txt': [],
            'md': [],
            'png': [],
            'jpg': [],
            'pdf': [],
            'json': [],
        }
        
        # Initialize knowledge base and agents
        self.knowledge_bases = {}
        self.agent = None
        self.interactive_agent = None
        self.image_agent = None
        self.images = []
        self.agents = {}  # For multi-agent system
        
        # Initialize conversation history
        self.conversation_history = []
        
        # Initialize utility classes
        self.data_visualizer = DataVisualizer(logger=logger)


        
        # Set base path in context
        context.base_path = base_path
        
        saved_history = self.redis_tools.load_from_redis('conversation_history')
        if saved_history:
            self.conversation_history = saved_history
            context.add_to_conversation("assistant", "Hello! I'm your interactive document assistant. How can I help you analyze your data today?")
            logger.info(f"Loaded conversation history from Redis: {len(saved_history)} messages")
        
        saved_context_data = self.redis_tools.load_from_redis('context')
        if saved_context_data:
            # Transfer data to our AppContext singleton
            if 'current_topic' in saved_context_data:
                context.current_topic = saved_context_data['current_topic']
            if 'current_files' in saved_context_data:
                context.current_files = saved_context_data['current_files']
            logger.info(f"Loaded context from Redis")
        
        # Auto-discover files if requested
        if auto_discover and base_path:
            self._discover_files(base_path)
        
        # Initialize multi-agent system if we have a config
        if self.config:
            self._initialize_multi_agent_system()
            self._initialize_interactive_agent()
            
            # Create knowledge graph for deeper semantic understanding
            self.create_knowledge_graph()
            
        # Add default welcome message if history is empty
        if not self.conversation_history:
            self.conversation_history.append({
                "role": "assistant", 
                "content": "Hello! I'm your interactive document assistant. How can I help you analyze your data today?"
            })
            context.add_to_conversation("assistant", "Hello! I'm your interactive document assistant. How can I help you analyze your data today?")
            
        logger.info("InteractiveDocumentReader initialized successfully")
        return True

    
    
    def _initialize_multi_agent_system(self):
        """Initialize the multi-agent system with specialized agents."""
        try:
            # 1. Document Navigator - handles file discovery and organization
            self.agents["document_navigator"] = Agent(
                model=OpenAIChat(id="gpt-4o"),
                agent_id="document-navigator",
                name="Document Navigator",
                markdown=True,
                debug_mode=True,
                instructions=[
                    "You are a Document Navigator agent responsible for file discovery and organization.",
                    "Your primary role is to help users find and navigate through available files.",
                    "You provide information about file structure, content summaries, and relationships between files.",
                    "When users ask about available files or specific file types, you provide clear listings and descriptions.",
                    "You understand file formats and can explain their contents at a high level.",
                    "You help categorize and organize information for easier access.",
                ]
            )
            
            # 2. Visual Analyst - specialized in image analysis
            self.agents["visual_analyst"] = Agent(
                model=OpenAIChat(id="gpt-4o"),
                agent_id="visual-analyst",
                name="Visual Analyst",
                markdown=True,
                debug_mode=True,
                instructions=[
                    "You are a Visual Analyst agent specialized in analyzing images, charts, and visualizations.",
                    "Your expertise is in extracting insights from visual data representations.",
                    "You can identify trends, patterns, and anomalies in charts and graphs.",
                    "You understand different visualization types (bar charts, scatter plots, maps, etc.) and their purposes.",
                    "You can describe spatial patterns and distributions shown in maps and geographic visualizations.",
                    "You can correlate visual information with underlying data when available.",
                    "You provide detailed, structured analyses of visual content focusing on the most important insights."
                ]
            )
            
            # 3. Data Scientist - handles data analysis
            self.agents["data_scientist"] = Agent(
                model=OpenAIChat(id="gpt-4o"),
                agent_id="data-scientist",
                name="Data Scientist",
                markdown=True,
                debug_mode=True,
                instructions=[
                    "You are a Data Scientist agent specialized in analyzing structured data such as CSV files.",
                    "You can interpret statistical information and explain patterns in datasets.",
                    "You identify correlations, trends, and outliers in numerical data.",
                    "You can suggest appropriate visualizations for different data types.",
                    "You understand common data formats and their structures.",
                    "You can explain statistical concepts in clear, accessible language.",
                    "You provide insights on data quality, completeness, and limitations."
                ]
            )
            
            # 4. Domain Expert - provides subject matter expertise
            self.agents["domain_expert"] = Agent(
                model=OpenAIChat(id="gpt-4o"),
                agent_id="domain-expert",
                name="Domain Expert",
                markdown=True,
                debug_mode=True,
                instructions=[
                    "You are a Domain Expert specialized in hydrology, climate, agriculture, and environmental sciences.",
                    "You provide subject matter expertise on topics related to water resources, groundwater, aquifers,",
                    "precipitation patterns, temperature trends, crop systems, vegetation indices, and land use.",
                    "You understand terminology and concepts specific to these domains.",
                    "You can explain complex environmental processes and their relationships.",
                    "You help interpret domain-specific data and visualizations in their proper context.",
                    "You can relate local observations to broader scientific understanding.",
                    "You maintain scientific accuracy while making concepts accessible."
                ]
            )
            
            # 5. Coordinator - orchestrates the other agents
            self.agents["coordinator"] = Agent(
                model=OpenAIChat(id="gpt-4o"),
                agent_id="coordinator",
                name="Coordinator",
                markdown=True,
                debug_mode=True,
                instructions=[
                    "You are the Coordinator agent that orchestrates the multi-agent system.",
                    "Your job is to analyze user requests and determine which specialized agent should handle them.",
                    "You maintain conversation coherence by tracking context and previous interactions.",
                    "You can combine insights from multiple agents to provide comprehensive responses.",
                    "You ensure responses are complete, relevant, and free of contradictions or repetition.",
                    "You manage conversation flow and can suggest follow-up questions or topics.",
                    "You prioritize responding to the user's immediate needs while suggesting additional valuable insights."
                ]
            )
            
            logger.info("Multi-agent system initialized")
            return True
            
        except Exception as e:
            logger.error(f"_initialize_multi_agent_system Error initializing multi-agent system: {str(e)}")
            return False
            
    def _discover_files(self, base_path):
        try:
            from utils import discover_files
            self.discovered_files = discover_files(base_path, self.config, logger)
            # Generate data summaries for all discovered files
            self._generate_data_summaries()
            # Ensure the discovered files are properly reflected in the agent's instructions
            self._update_interactive_agent_instructions()

        except Exception as e:
            logger.error(f"_discover_files Error discovering files: {str(e)}")
            return False
    
    def _update_interactive_agent_instructions(self):
        """Update the instructions for the interactive agent with current file information."""
        if not hasattr(self, 'interactive_agent') or self.interactive_agent is None:
            return
            
        instructions = [
            "You are an AI data analysis assistant that can interpret and explain various data sources.",
            "You have access to multiple document types including PDFs, CSVs, images, and more.",
            "Your goal is to help users understand their data and answer questions about it.",
            "When appropriate, suggest visualizations or analyses that might be insightful.",
            "If you're unsure about something, be honest about your limitations."
        ]
        
        # Add data summaries to the instructions
        if self.data_summaries:
            instructions.append("\nAvailable data sources:")
            for file_path, summary in self.data_summaries.items():
                file_name = summary.get('file_name', os.path.basename(file_path))
                
                # Different display for skipped files
                if summary.get('status') == 'skipped':
                    instructions.append(f"- {file_name}: {summary.get('reason', 'Skipped due to size limits')}")
                elif 'rows' in summary:
                    instructions.append(f"- {file_name}: {summary['rows']} rows, {summary['columns']} columns")
                    instructions.append(f"  Columns: {', '.join(summary['column_names'])}")
                else:
                    instructions.append(f"- {file_name}: {summary.get('data_type', 'unknown type')}")
        
        # Add CSV files
        if self.discovered_files.get('csv'):
            csv_files = self.discovered_files.get('csv', [])
            if csv_files:
                instructions.append("\nAvailable CSV files:")
                for csv_path in csv_files:
                    instructions.append(f"- {os.path.basename(csv_path)}")
        
        # Add information about image files
        if self.discovered_files.get('png') or self.discovered_files.get('jpg'):
            instructions.append("\nAvailable image files:")
            for img_type in ['png', 'jpg']:
                for img_path in self.discovered_files.get(img_type, []):
                    instructions.append(f"- {os.path.basename(img_path)}: {img_type.upper()} image file")
        
        # Add information about markdown and text files
        for doc_type in ['md', 'txt']:
            if self.discovered_files.get(doc_type):
                instructions.append(f"\nAvailable {doc_type.upper()} files:")
                for path in self.discovered_files.get(doc_type, []):
                    instructions.append(f"- {os.path.basename(path)}")
        
        # Update the agent's instructions
        self.interactive_agent.instructions = instructions
    
    def _generate_data_summaries(self):
        """Generate summaries for discovered data files."""
        try:
            # Generate summaries for CSV files - they contain structured data
            for file_path in self.discovered_files.get('csv', []):
                try:
                    # Check file size first - read just enough to count rows
                    row_count = 0
                    with open(file_path, 'r') as f:
                        for i, _ in enumerate(f):
                            row_count = i + 1
                            if row_count > 1000:
                                logger.warning(f"CSV file {file_path} has more than 1000 rows ({row_count}). Skipping detailed analysis.")
                                self.data_summaries[file_path] = {
                                    'file_name': os.path.basename(file_path),
                                    'path': file_path,
                                    'rows': f">{row_count} (too large to process fully)",
                                    'status': 'skipped',
                                    'reason': 'CSV file too large (>1000 rows)'
                                }
                                break
                    
                    # If file is not too large, process it normally
                    if row_count <= 1000:
                        df = pd.read_csv(file_path)
                        file_name = os.path.basename(file_path)
                        
                        # Create a basic summary
                        summary = {
                            'file_name': file_name,
                            'path': file_path,
                            'rows': len(df),
                            'columns': len(df.columns),
                            'column_names': df.columns.tolist(),
                            'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
                            'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
                            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
                            'date_columns': df.select_dtypes(include=['datetime']).columns.tolist(),
                            'has_missing_values': df.isnull().any().any(),
                            'sample_data': df.head(5).to_dict() if len(df) > 0 else {},
                            'status': 'processed'
                        }
                        
                        self.data_summaries[file_path] = summary
                    
                except Exception as e:
                    logger.error(f"_generate_data_summaries Error generating summary for {file_path}: {str(e)}")
            
            # Generate summaries for JSON files
            for file_path in self.discovered_files.get('json', []):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    file_name = os.path.basename(file_path)
                    
                    # Create a basic summary
                    summary = {
                        'file_name': file_name,
                        'path': file_path,
                        'data_type': type(data).__name__,
                    }
                    
                    if isinstance(data, dict):
                        summary['top_level_keys'] = list(data.keys())
                        summary['num_keys'] = len(data)
                    elif isinstance(data, list):
                        summary['num_items'] = len(data)
                        if data and isinstance(data[0], dict):
                            summary['sample_keys'] = list(data[0].keys())
                    
                    self.data_summaries[file_path] = summary
                    
                except Exception as e:
                    logger.error(f"_generate_data_summaries Error generating summary for {file_path}: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"_generate_data_summaries Error generating data summaries: {str(e)}")
            return False
    
    def _initialize_interactive_agent(self):
        """Initialize the main interactive agent."""
        try:
            # Create system instructions for the interactive agent
            instructions = [
                "Your Name is HydroDeepNet"
                "You are an AI data analysis assistant that can interpret and explain various data sources.",
                "You have access to multiple document types including images (png) statistics (CSVs), description (md), and more.",
                "These files are the reports/summaries of the extracted data from a geological boundary",
                "Your goal is to help users understand their data and answer questions about it.",
                "When appropriate, suggest visualizations or analyses that might be insightful.",
                "If you're unsure about something, be honest about your limitations.",
                "When asked to list files of a particular type, make sure to check self.discovered_files for that type."
            ]
            
            # Add information about specific capabilities
            instructions.append("\nCapabilities:")
            instructions.append("- You can analyze CSV data and generate statistics")
            instructions.append("- You can interpret and describe images, charts and visualizations")
            instructions.append("- You can summarize content from markdown and text files")
            instructions.append("- You can create visualizations from numeric data")
            
            # Add data summaries to the instructions
            if self.data_summaries:
                instructions.append("\nAvailable data sources:")
                for file_path, summary in self.data_summaries.items():
                    file_name = summary.get('file_name', os.path.basename(file_path))
                    
                    # Different display for skipped files
                    if summary.get('status') == 'skipped':
                        instructions.append(f"- {file_name}: {summary.get('reason', 'Skipped due to size limits')}")
                    elif 'rows' in summary:
                        instructions.append(f"- {file_name}: {summary['rows']} rows, {summary['columns']} columns")
                        instructions.append(f"  Columns: {', '.join(summary['column_names'])}")
                    else:
                        instructions.append(f"- {file_name}: {summary.get('data_type', 'unknown type')}")
            
            # Add CSV files
            if self.discovered_files.get('csv'):
                csv_files = self.discovered_files.get('csv', [])
                if csv_files:
                    instructions.append("\nAvailable CSV files:")
                    for csv_path in csv_files:
                        instructions.append(f"- {os.path.basename(csv_path)}")
            
            # Add information about image files
            if self.discovered_files.get('png') or self.discovered_files.get('jpg'):
                instructions.append("\nAvailable image files:")
                for img_type in ['png', 'jpg']:
                    for img_path in self.discovered_files.get(img_type, []):
                        instructions.append(f"- {os.path.basename(img_path)}: {img_type.upper()} image file")
            
            # Add information about markdown and text files
            for doc_type in ['md', 'txt']:
                if self.discovered_files.get(doc_type):
                    instructions.append(f"\nAvailable {doc_type.upper()} files:")
                    for path in self.discovered_files.get(doc_type, []):
                        instructions.append(f"- {os.path.basename(path)}")
            
            # Initialize the agent with the OpenAI chat model
            self.interactive_agent = Agent(
                model=OpenAIChat(id="gpt-4o"),
                agent_id="interactive-data-analyst",
                name="Interactive Data Analyst",
                markdown=True,
                debug_mode=True,
                show_tool_calls=True,
                instructions=instructions,
            )
            
            logger.info("Interactive agent initialized")
            return True
            
        except Exception as e:
            logger.error(f"_initialize_interactive_agent Error initializing interactive agent: {str(e)}")
            return False
    
    def _initialize_knowledge_base(self, doc_type, config):
        """Initialize a specific knowledge base based on document type."""
        try:
            # Handle both single config and list of configs
            configs = config if isinstance(config, list) else [config]
            
            for cfg in configs:
                # Special handling for CSV files
                if doc_type == 'csv':
                    path = cfg['path']
                    
                    # Check if file was already analyzed and is too large
                    if path in self.data_summaries and self.data_summaries[path].get('status') == 'skipped':
                        logger.warning(f"Skipping knowledge base creation for large CSV file: {path}")
                        continue
                    
                    # If not analyzed yet, check the file size
                    if path not in self.data_summaries:
                        row_count = 0
                        with open(path, 'r') as f:
                            for i, _ in enumerate(f):
                                row_count = i + 1
                                if row_count > 1000:
                                    logger.warning(f"CSV file {path} has more than 1000 rows ({row_count}). Skipping knowledge base creation.")
                                    continue
                
                table_name = cfg.get('table_name', f"{doc_type}_documents")
                vector_db = PgVector(
                    table_name=table_name,
                    db_url=self.db_url,
                )
                
                if doc_type == 'pdf':
                    kb = PDFKnowledgeBase(
                        path=cfg['path'],
                        vector_db=vector_db,
                        reader=PDFReader(chunk=True),
                        embedder=self.embedder
                    )
                elif doc_type == 'text':
                    kb = TextKnowledgeBase(
                        path=cfg['path'],
                        vector_db=vector_db,
                        embedder=self.embedder
                    )
                elif doc_type == 'csv':
                    kb = CSVKnowledgeBase(
                        path=cfg['path'],
                        vector_db=vector_db,
                        embedder=self.embedder
                    )
                elif doc_type == 'json':
                    kb = JSONKnowledgeBase(
                        path=cfg['path'],
                        vector_db=vector_db,
                        embedder=self.embedder
                    )
                elif doc_type == 'docx':
                    kb = DocxKnowledgeBase(
                        path=cfg['path'],
                        vector_db=vector_db,
                        embedder=self.embedder
                    )
                elif doc_type == 'website':
                    kb = WebsiteKnowledgeBase(
                        urls=[cfg['url']],
                        max_links=cfg.get('max_links', 10),
                        vector_db=vector_db,
                        embedder=self.embedder
                    )
                    
                kb.load(recreate=False)
                
                # Store the knowledge base with a unique key
                key = f"{doc_type}_{len(self.knowledge_bases)}"
                self.knowledge_bases[key] = kb
                logger.info(f"{doc_type.upper()} knowledge base initialized and loaded for {cfg['path']}")
            
        except Exception as e:
            logger.error(f"_initialize_knowledge_base Error initializing {doc_type} knowledge base: {str(e)}")
            raise

    def _initialize_image_reader(self, doc_type, config):
        """Initialize image reader agent."""
        try:
            # Handle both single config and list of configs
            configs = config if isinstance(config, list) else [config]
            
            for cfg in configs:
                image_path = cfg['path']
                logger.info(f"Initializing image reader for {image_path}")
                
                # Create an Image object
                image_obj = Image(filepath=image_path)
                self.images.append(image_obj)
            
            # Initialize the image agent (only once)
            if not self.image_agent and self.images:
                self.image_agent = Agent(
                    model=OpenAIChat(id="gpt-4o"),
                    agent_id="image-to-text",
                    name="Image to Text Agent",
                    markdown=True,
                    debug_mode=True,
                    show_tool_calls=True,
                    instructions=[
                        "You are an AI agent that can generate text descriptions based on an image.",
                        "You have to return a text response describing the image.",
                        "Focus on analytical aspects of charts, graphs, and visualizations if present.",
                        "Identify trends, patterns, and key insights from data visualizations.",
                    ],
                )
            
                logger.info(f"Image reader initialized for {len(self.images)} images")
            
        except Exception as e:
            logger.error(f"_initialize_image_reader Error initializing image reader: {str(e)}")
            raise

    def _create_combined_knowledge_base(self):
        """Create a combined knowledge base from all initialized knowledge bases."""
        try:
            combined_table = self.config.get('combined_table_name', 'combined_documents')
            vector_db = PgVector(
                table_name=combined_table,
                db_url=self.db_url,
            )
            
            self.combined_knowledge_base = CombinedKnowledgeBase(
                sources=list(self.knowledge_bases.values()),
                vector_db=vector_db,
                embedder=self.embedder
            )
            
            self.agent = Agent(
                knowledge=self.combined_knowledge_base,
                search_knowledge=True
            )
            
            self.combined_knowledge_base.load(recreate=False)
            logger.info("Combined knowledge base created and loaded")
            
        except Exception as e:
            logger.error(f"_create_combined_knowledge_base Error creating combined knowledge base: {str(e)}")
            raise
    
    def visualize_data(self, file_path, x_column=None, y_column=None, chart_type='auto'):
        """Generate a visualization for numerical data in CSV files."""
        return self.data_visualizer.visualize_data(file_path, x_column, y_column, chart_type)
    
    def analyze_data(self, file_path):
        """Generate a basic statistical analysis for data files."""
        return self.data_visualizer.analyze_csv_data(file_path)
    

    def _create_contextual_visualization(self, query, csv_path):
        """Create a visualization customized to answer the specific query."""
        return self.data_visualizer.create_contextual_visualization(query, csv_path)
    

    
    def chat(self, message):
        """Process user's message and return a response through the appropriate agent."""
        start_time = datetime.now()
        
        try:
            # Add the message to conversation history
            self.conversation_history.append({"role": "user", "content": message})
            context.add_to_conversation("user", message)
            logger.debug(f"Processing user message: '{message[:50] if message else ''}...'")
            
            # Check if there are pending actions to confirm
            pending_handled = False
            pending_actions = context.config.get('pending_actions', [])
            if pending_actions:
                for i, action in enumerate(pending_actions):
                    if action['type'] == 'file_check' and "yes" in message.lower():
                        # User confirmed file check
                        logger.debug(f"Handling file check confirmation for {len(action['file_paths'])} files")
                        response = self._handle_file_check(action['file_paths'])
                        pending_actions.pop(i)
                        context.config['pending_actions'] = pending_actions
                        pending_handled = True
                        break
                    elif action['type'] == 'image_analysis' and "analyze" in message.lower():
                        # Extract image name from user message
                        image_name = extract_image_name(message)
                        if image_name:
                            logger.debug(f"Handling image analysis for '{image_name}'")
                            response = self._analyze_specific_image(image_name)
                            pending_actions.pop(i)
                            context.config['pending_actions'] = pending_actions
                            pending_handled = True
                            break
            
            # Handle special commands if the message wasn't a response to a pending action
            if not pending_handled:
                # Check for special commands
                if message.startswith("/"):
                    logger.debug(f"Handling command: {message}")
                    response = self._handle_command(message)
                else:
                    # Check if it's specifically asking to analyze a file
                    file_match = re.search(r'(analyze|analyse|view|read|show|open)\s+(\S+\.\w+)', message.lower())
                    if file_match:
                        file_name = file_match.group(2)
                        file_ext = os.path.splitext(file_name)[1].lower()
                        
                        if file_ext in ['.png', '.jpg']:
                            # Handle image analysis
                            image_name = os.path.basename(file_name)
                            logger.debug(f"Handling explicit image analysis request for '{image_name}'")
                            response = self._analyze_specific_image(image_name)
                        elif file_ext in ['.md', '.markdown']:
                            # Handle markdown viewing
                            logger.debug(f"Handling explicit markdown viewing request for '{file_name}'")
                            response = self._handle_markdown_command(os.path.basename(file_name))
                        elif file_ext == '.csv':
                            # Handle CSV analysis
                            logger.debug(f"Handling explicit CSV analysis request for '{file_name}'")
                            response = self._handle_csv_command(os.path.basename(file_name))
                        else:
                            # Fallback to multi-agent routing
                            logger.debug(f"No specific handler for file type {file_ext}, routing through multi-agent system")
                            multi_agent_response = self._route_to_multi_agent_system(message)
                            
                            if multi_agent_response:
                                logger.debug("Got response from multi-agent system")
                                raw_response = multi_agent_response
                            else:
                                # Fallback to traditional handling
                                logger.debug("Falling back to traditional question handling")
                                raw_response = self._handle_question(message)
                            
                            # Apply response validation framework
                            if raw_response:
                                logger.debug(f"Validating raw response of length: {len(raw_response) if raw_response else 0}")
                                response = validate_response(raw_response, message, self.logger)
                            else:
                                logger.warning("Received empty response from agent")
                                response = "I apologize, but I couldn't generate a response. Please try rephrasing your question."
                    else:
                        # First try to route through the multi-agent system
                        logger.debug("Routing through multi-agent system")
                        multi_agent_response = self._route_to_multi_agent_system(message)
                        
                        if multi_agent_response:
                            logger.debug("Got response from multi-agent system")
                            raw_response = multi_agent_response
                        else:
                            # Fallback to traditional handling
                            logger.debug("Falling back to traditional question handling")
                            raw_response = self._handle_question(message)
                        
                        # Apply response validation framework
                        if raw_response:
                            logger.debug(f"Validating raw response of length: {len(raw_response) if raw_response else 0}")
                            response = validate_response(raw_response, message, self.logger)
                        else:
                            logger.warning("Received empty response from agent")
                            response = "I apologize, but I couldn't generate a response. Please try rephrasing your question."
            
            # Ensure response is never None before adding to history
            if response is None:
                logger.warning("Response is None, using generic fallback")
                response = "I processed your request, but couldn't generate a specific response."
            
            # Clean the response of any debug information or formatting
            logger.debug(f"Cleaning response output of length: {len(response) if response else 0}")
            cleaned_response = clean_response_output(response, logger)
            logger.debug(f"Cleaned response of length: {len(cleaned_response) if cleaned_response else 0}")
            
            # Don't use apology messages if we have a real response
            if (cleaned_response and "apologize" in cleaned_response.lower() and len(cleaned_response) < 100 and 
                response and response != cleaned_response and len(response) > 100):
                # We had a real response before cleaning, use that instead
                logger.debug("Using original response instead of apology-only cleaned response")
                cleaned_response = response
            
            # Set final response
            response = cleaned_response
                
            # Add the response to conversation history - ensure we don't add streaming responses twice
            # Check if the last message is already from the assistant and similar to avoid duplication
            should_add_to_history = True
            if self.conversation_history and self.conversation_history[-1]["role"] == "assistant":
                # If the last message is very similar to the current response, don't add it again
                last_response = self.conversation_history[-1]["content"]
                similarity = calculate_similarity(last_response, response)
                if similarity > 0.8:  # 80% similarity threshold
                    # Update the existing message rather than adding a new one
                    logger.debug(f"Updating existing message in history (similarity: {similarity:.2f})")
                    self.conversation_history[-1]["content"] = response
                    should_add_to_history = False
                    
            if should_add_to_history:
                logger.debug("Adding response to conversation history")
                self.conversation_history.append({"role": "assistant", "content": response})
                context.add_to_conversation("assistant", response)
            
            # Save updated conversation history and context to Redis
            self.redis_tools.save_to_redis('conversation_history', self.conversation_history)
            self.redis_tools.save_to_redis('context', {
                'current_topic': context.current_topic,
                'current_files': context.current_files,
                'analysis_results': context.analysis_results,
                'config': context.config
            })
                
            # Update metrics
            response_time = (datetime.now() - start_time).total_seconds()
            self.metrics["response_times"].append(response_time)
            self.metrics["total_responses"] += 1
            logger.debug(f"Response completed in {response_time:.2f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Error processing your request: {str(e)}"

    def _route_to_multi_agent_system(self, message):
        """Route a user message through the multi-agent system for processing."""
        try:
            # Check if we have any specialized agents
            if not self.agents or not self.agents.get("coordinator"):
                logger.debug("Multi-agent system not available or coordinator not initialized")
                return None
            
            # First, let the coordinator determine which specialist agent to use
            coordinator_prompt = f"""
            The user asked: "{message}"
            

            Based on this message, determine which specialist agent would be most appropriate
            to handle this query. The available specialist agents are:
            
            - visual_analyst: Expert in analyzing images, charts, and visualizations
            - document_navigator: Expert in retrieving and summarizing document content
            - data_scientist: Expert in analyzing numerical data and statistics
            
            If the query involves multiple domains, identify the primary domain and relevant agents.
            If the query doesn't require a specialist, or if it's a general question about the project,
            you can handle it directly without delegating.
            
            Provide your assessment in this format:
            {{
                "primary_agent": "[agent_name or 'coordinator' if you'll handle directly]",
                "supporting_agents": ["list of any supporting agents needed"],
                "reasoning": "[brief explanation of your decision]"
            }}
            """
            
            logger.debug(f"Requesting coordinator to route message: '{message[:50] if message and isinstance(message, str) else ''}'...")
            
            try:
                # Use print_response with stream=False to get the full response
                coordinator_response = self.agents["coordinator"].print_response(coordinator_prompt, stream=False)
            except Exception as e:
                logger.error(f"Error getting response from coordinator: {str(e)}")
                logger.error(traceback.format_exc())
                return None
            
            # Check if coordinator_response is None before trying to access it
            if coordinator_response is None:
                logger.warning("Received None response from coordinator agent")
                coordinator_response = ""
            
            try:
                # Only try to log if we have a valid string response
                if isinstance(coordinator_response, str) and coordinator_response:
                    logger.debug(f"Got coordinator response (first 100 chars): {coordinator_response[:100]}")
                else:
                    logger.warning(f"Unexpected coordinator response type: {type(coordinator_response)}")
            except Exception as log_error:
                logger.error(f"Error logging coordinator response: {str(log_error)}")
            
            # Try to extract JSON from the response
            try:
                # Find JSON pattern in the response
                import re
                if not coordinator_response or not isinstance(coordinator_response, str):
                    logger.warning("Coordinator response is empty, None, or not a string")
                    primary_agent = "coordinator" 
                    supporting_agents = []
                else:
                    # Find JSON pattern in the response - search for anything that looks like JSON with primary_agent in it
                    json_pattern = r'\{[\s\S]*?"primary_agent"[\s\S]*?\}'
                    json_match = re.search(json_pattern, coordinator_response)
                    
                    if json_match:
                        json_str = json_match.group(0)
                        logger.debug(f"Found JSON pattern: {json_str}")
                        try:
                            routing_info = json.loads(json_str)
                            primary_agent = routing_info.get("primary_agent")
                            supporting_agents = routing_info.get("supporting_agents", [])
                            logger.info(f"Routing to primary agent: {primary_agent}, supporting: {supporting_agents}")
                        except json.JSONDecodeError as je:
                            logger.warning(f"JSON decode error: {str(je)} in pattern: {json_str}")
                            primary_agent = "coordinator"
                            supporting_agents = []
                    else:
                        logger.warning(f"No JSON pattern found in coordinator response")
                        # Fallback: assume coordinator handles it
                        primary_agent = "coordinator"
                        supporting_agents = []
            except Exception as json_e:
                logger.warning(f"Error parsing coordinator response: {str(json_e)}")
                # Fallback to coordinator directly
                primary_agent = "coordinator"
                supporting_agents = []
            
            # If the coordinator will handle it directly, return its response
            if primary_agent == "coordinator":
                # The coordinator already provided a response, use it
                logger.debug("Using coordinator's direct response")
                return coordinator_response
            
            # Get responses from all relevant agents
            agent_responses = {}
            
            # Add the primary agent
            if primary_agent in self.agents and primary_agent != "coordinator":
                logger.debug(f"Getting response from primary agent: {primary_agent}")
                # Create a prompt for the primary agent, using the original query
                prompt = f"""
                The user asked: "{message}"
                
                Please provide a detailed response with your expertise as the {primary_agent}.
                """
                
                # Get response from this agent
                try:
                    agent_responses[primary_agent] = self.agents[primary_agent].print_response(prompt, stream=False)
                    if agent_responses[primary_agent]:
                        logger.debug(f"Got response from {primary_agent} (length: {len(agent_responses[primary_agent])})")
                    else:
                        logger.warning(f"Empty response from {primary_agent}")
                except Exception as e:
                    logger.error(f"Error getting response from {primary_agent}: {str(e)}")
            
            # Add supporting agents if needed
            for agent_name in supporting_agents:
                if agent_name not in self.agents or agent_name == "coordinator":
                    continue
                
                logger.debug(f"Getting response from supporting agent: {agent_name}")
                # Create a prompt for the supporting agent
                prompt = f"""
                The user asked: "{message}"
                
                Please provide a supporting response with your expertise as the {agent_name}.
                Focus on aspects of the question related to your specialization.
                """
                
                # Get response from this agent
                try:
                    agent_responses[agent_name] = self.agents[agent_name].print_response(prompt, stream=False)
                    if agent_responses[agent_name]:
                        logger.debug(f"Got response from {agent_name} (length: {len(agent_responses[agent_name])})")
                    else:
                        logger.warning(f"Empty response from {agent_name}")
                except Exception as e:
                    logger.error(f"Error getting response from {agent_name}: {str(e)}")
            
            # Filter out None values for JSON serialization
            valid_responses = {k: v for k, v in agent_responses.items() if v is not None}
            
            # Have the coordinator synthesize the responses
            if valid_responses:
                logger.debug(f"Synthesizing {len(valid_responses)} agent responses")
                
                # Create a JSON-safe representation for synthesis
                json_safe_responses = {}
                for agent_name, response in valid_responses.items():
                    if isinstance(response, str):
                        # Limit response length to avoid overloading the context
                        json_safe_responses[agent_name] = response[:5000] if len(response) > 5000 else response
                    else:
                        # Handle non-string responses
                        json_safe_responses[agent_name] = str(response)
                
                synthesis_prompt = f"""
                The user asked: "{message}"
                
                Responses from specialist agents:
                
                {json.dumps(json_safe_responses, indent=2)}
                
                Please synthesize these responses into a single coherent answer that addresses the user's question.
                Focus on the most relevant information while ensuring all important insights are included.
                Avoid repetition and ensure a logical flow of information.
                """
                
                try:
                    final_response = self.agents["coordinator"].print_response(synthesis_prompt, stream=False)
                    if final_response:
                        logger.debug(f"Got synthesized response (length: {len(final_response)})")
                    else:
                        logger.warning("Empty synthesized response")
                    return final_response
                except Exception as e:
                    logger.error(f"Error synthesizing responses: {str(e)}")
                    # Return the primary agent's response as fallback
                    if agent_responses.get(primary_agent):
                        return agent_responses[primary_agent]
                    elif valid_responses:
                        # Return any agent response we have
                        return next(iter(valid_responses.values()))
                    else:
                        return coordinator_response or "I'm sorry, I couldn't process your request properly. Could you try rephrasing your question?"
            else:
                # If no specialist agent was invoked, use the coordinator's response directly
                logger.debug("No specialist agent responses received, using coordinator response directly")
                return coordinator_response or "I'm sorry, I couldn't process your request properly. Could you try rephrasing your question?"
                
        except Exception as e:
            logger.error(f"Error routing through multi-agent system: {str(e)}")
            logger.error(traceback.format_exc())
            return "I'm sorry, I encountered an error while processing your request. Could you try again with a different question?"

    def _check_cached_analysis(self, query, keywords):
        """Check if the analysis is cached in Redis or in the global context."""
        from context_manager import context
        
        # Create a cache key from the query
        cache_key = f"query:{query}"
        
        # First check Redis
        if self.redis_tools and self.redis_tools.has_redis:
            cached_analysis = self.redis_tools.get_cached_analysis(cache_key)
            if cached_analysis and cached_analysis != query and len(cached_analysis) > 30:
                self.logger.info(f"Found cached analysis in Redis for query: {query[:50]}...")
                return cached_analysis
        
        # Then check global context
        if keywords:
            # Try to find matching analysis results in context
            for key, result in context.analysis_results.items():
                # Avoid returning the user's question as the answer
                if result == query or not isinstance(result, str) or len(result) < 30:
                    continue
                    
                # Check if the key or result contains any of our keywords
                key_matches = any(kw.lower() in key.lower() for kw in keywords if len(kw) > 3)
                result_matches = isinstance(result, str) and any(kw.lower() in result.lower() for kw in keywords if len(kw) > 3)
                
                if (key_matches or result_matches) and key != 'last_question':
                    self.logger.info(f"Found relevant cached analysis in context for: {key}")
                    return result
                    
        return None

    def _analyze_query_type(self, message):
        """Analyze a user query to determine its type and extract keywords."""
        message_lower = message.lower()
        
        # Extract keywords from the message
        keywords = []
        # Remove common stop words and tokenize
        stop_words = ['a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 'at', 'from', 'by', 'for', 'with', 'about', 'to', 'in', 'on', 'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing']
        
        # Split message and filter out stop words
        words = message_lower.split()
        keywords = [word.strip('.,?!()[]{}:;"\'') for word in words 
                   if word.strip('.,?!()[]{}:;"\'') and word.strip('.,?!()[]{}:;"\'') not in stop_words 
                   and len(word.strip('.,?!()[]{}:;"\'')) > 2]
        
        # Check for file listing queries
        if any(pattern in message_lower for pattern in [
            "list", "show", "what files", "available files", "what documents",
            "display files", "show me the files", "what images"
        ]):
            return "file_listing", keywords
            
        # Check for image analysis queries
        if any(pattern in message_lower for pattern in [
            "analyze image", "analyze the image", "analyze this image",
            "describe image", "explain image", "what does the image show",
            "interpret the chart", "explain the graph", "analyze the chart"
        ]):
            return "image_analysis", keywords
            
        # Check for data analysis queries
        if any(pattern in message_lower for pattern in [
            "analyze data", "statistics", "correlation", "trend", "pattern",
            "what does the data show", "data analysis", "statistical", "mean",
            "average", "distribution", "csv analysis", "analyze csv"
        ]):
            return "data_analysis", keywords
            
        # Check for domain knowledge queries
        if any(pattern in message_lower for pattern in [
            "explain", "why does", "how does", "what causes", "relationship between",
            "impact of", "effect of", "mechanism", "process", "scientific explanation"
        ]):
            return "domain_knowledge", keywords
            
        # Default to general query
        return "general", keywords
    
    def _handle_command(self, message):
        """Handle special commands starting with /."""
        parts = message.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if command == "/visualize":
            return self._handle_visualize_command(args)
        elif command == "/analyze":
            return self._handle_analyze_command(args)
        elif command == "/discover":
            return self._handle_discover_command(args)
        elif command == "/help":
            return self._handle_help_command()
        elif command == "/files":
            return self._handle_files_command()
        elif command == "/markdown" or command == "/md":
            return self._handle_markdown_command(args)
        elif command == "/csv":
            return self._handle_csv_command(args)
        elif command == "/clear":
            # Clear conversation history
            self.conversation_history = []
            self.redis_tools.clear_redis_session()
            return "Conversation history and context cleared."
        elif command == "/context":
            # Debug command to show current context
            return f"Current context:\n{json.dumps(self.context, indent=2)}"
        else:
            return f"Unknown command: {command}. Type /help for available commands."
    
    def _handle_visualize_command(self, args):
        """Handle /visualize command to visualize CSV data."""
        from context_manager import context
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib import style
        
        # Parse arguments - format is "/visualize file_name [x_column] [y_column] [chart_type]"
        args_list = args.strip().split()
        if not args_list:
            return "Please specify a CSV file to visualize. Example: `/visualize data.csv column1 column2`"
        
        # Extract file_name and optional column names
        file_path = args_list[0]
        x_column = args_list[1] if len(args_list) > 1 else None
        y_column = args_list[2] if len(args_list) > 2 else None
        chart_type = args_list[3] if len(args_list) > 3 else 'auto'
        
        # Find matching files if path is partial
        matching_files = []
        for path in self.discovered_files.get('csv', []):
            if file_path.lower() in path.lower():
                matching_files.append(path)
        
        if not matching_files:
            return f"No CSV files found matching '{file_path}'. Use `/files` to see available files."
        
        if len(matching_files) > 1:
            file_list = "\n".join([f"- {os.path.basename(path)}" for path in matching_files])
            return f"Multiple CSV files found matching '{file_path}'. Please be more specific:\n\n{file_list}"
        
        # We found exactly one matching file
        csv_path = matching_files[0]
        
        # Update context
        context.set_current_topic('visualization', [csv_path])
        
        try:
            # Create visualization
            result = self.visualize_data(csv_path, x_column, y_column, chart_type)
            context.save_visualization(f"visualize_{os.path.basename(csv_path)}", csv_path, result)
            return result
        except Exception as e:
            self.logger.error(f"Error visualizing CSV data: {str(e)}")
            return f"Error visualizing CSV file '{os.path.basename(csv_path)}': {str(e)}"
    
    def _handle_analyze_command(self, args):
        """Handle /analyze command to analyze a file."""
        from context_manager import context
        
        if not args.strip():
            return "Please specify a file to analyze. Example: `/analyze data.csv`"
        
        file_path = args.strip()
        
        # Find matching files if path is partial
        matching_files = []
        for file_type, files in self.discovered_files.items():
            for path in files:
                if file_path.lower() in path.lower():
                    matching_files.append(path)
        
        if not matching_files:
            return f"No files found matching '{file_path}'. Use `/files` to see available files."
        
        if len(matching_files) > 1:
            file_list = "\n".join([f"- {os.path.basename(path)}" for path in matching_files])
            return f"Multiple files found matching '{file_path}'. Please be more specific:\n\n{file_list}"
        
        # We found exactly one matching file
        target_file = matching_files[0]
        target_file_ext = os.path.splitext(target_file)[1].lower()
        
        # Update context
        context.set_current_topic('analysis', [target_file])
        
        # Call appropriate analyzer based on file type
        try:
            if target_file_ext == '.csv':
                from csv_utils import handle_csv_command
                return handle_csv_command(os.path.basename(target_file), self.discovered_files, self.logger)
            elif target_file_ext == '.json':
                from json_queries import analyze_json_file
                return analyze_json_file(target_file, self.logger)
            elif target_file_ext in ['.png', '.jpg', '.jpeg']:
                return self._analyze_specific_image(os.path.basename(target_file))
            else:
                return f"Analysis of {target_file_ext} files is not yet supported."
        except Exception as e:
            self.logger.error(f"Error in _handle_analyze_command: {str(e)}")
            return f"Error analyzing file '{os.path.basename(target_file)}': {str(e)}"
    
    def _handle_discover_command(self, args):
        """Handle /discover command to find files in a directory."""
        from context_manager import context
        import os
        from utils import discover_files
        
        # Get the directory path from args, or use the current base_path
        path = args.strip()
        if not path:
            path = self.base_path
            if not path:
                return "Please specify a directory path to discover files."
        
        # If path is relative, make it absolute based on the base_path
        if not os.path.isabs(path) and self.base_path:
            path = os.path.join(self.base_path, path)
        
        # Check if path exists
        if not os.path.exists(path):
            return f"Directory path '{path}' does not exist."
        
        # Update context
        context.set_current_topic('file_discovery', [path])
        context.base_path = path
        
        # Discover files
        self.discovered_files = discover_files(path, self.config, self.logger)
        context.discovered_files = self.discovered_files
        
        # Summarize discovered files
        file_counts = {k: len(v) for k, v in self.discovered_files.items() if v}
        
        if not file_counts:
            return f"No files found in '{path}'."
        
        # Format the response
        response = f"## Discovered Files in {path}\n\n"
        for file_type, count in file_counts.items():
            if count > 0:
                response += f"- {file_type.upper()}: {count} files\n"
        
        response += "\nUse `/files` to list all files or `/files [type]` to list files of a specific type."
        return response
    
    def _handle_help_command(self):
        """Handle /help command."""
        help_text = """
        # Available Commands

        ## File Discovery and Management
        - `/discover path` - Discover files in the specified path
        - `/files` - List all discovered files
        
        ## File Type Commands
        - `/csv [filename]` - List all CSV files or analyze a specific CSV file
        - `/markdown` or `/md [filename]` - List all markdown files or view a specific markdown file
        
        ## Analysis and Visualization
        - `/visualize file_path [x_column] [y_column] [chart_type]` - Create a visualization
        - `/analyze file_path` - Perform statistical analysis on a file
        
        ## System Commands
        - `/help` - Display this help message
        - `/clear` - Clear conversation history and context
        
        ## Tips
        - You can ask questions about specific files by mentioning their names
        - For image analysis, say "analyze [image name]"
        - For any other queries, just ask questions about your data
        """
        return help_text
    
    def _handle_files_command(self):
        """Handle /files command."""
        # List all discovered files
        if not self.discovered_files or all(len(files) == 0 for files in self.discovered_files.values()):
            # Check if we haven't properly initialized file discovery
            if hasattr(self, 'config') and self.config.get('csv'):
                # We have CSV files in the config but they haven't been added to discovered_files
                csv_files = []
                for csv_config in self.config.get('csv', []):
                    if 'path' in csv_config:
                        csv_files.append(csv_config['path'])
                
                if csv_files:
                    if 'csv' not in self.discovered_files:
                        self.discovered_files['csv'] = []
                    self.discovered_files['csv'].extend(csv_files)
                    
            # After checking, if we still have no files
            if not self.discovered_files or all(len(files) == 0 for files in self.discovered_files.values()):
                return "No files have been discovered yet. Use /discover path to find files."
        
        summary = "Discovered files:\n"
        
        # First, create sections by file type
        for file_type, file_paths in self.discovered_files.items():
            if file_paths:
                summary += f"\n## {file_type.upper()} files ({len(file_paths)}):\n"
                for path in file_paths:
                    summary += f"- {os.path.basename(path)}\n"
        
        return summary
    
    def _handle_file_check(self, file_paths):
        """Handle a confirmed file check action."""
        results = []
        for path in file_paths:
            if path.endswith('.json'):
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                    
                    # Create a formatted summary
                    if isinstance(data, dict):
                        summary = f"Contents of {os.path.basename(path)}:\n"
                        summary += json.dumps(data, indent=2)
                        results.append(summary)
                    else:
                        results.append(f"File {os.path.basename(path)} contains a JSON array with {len(data)} items.")
                except Exception as e:
                    results.append(f"Error reading {os.path.basename(path)}: {str(e)}")
            elif path.endswith('.csv'):
                try:
                    # Check file size first
                    row_count = 0
                    with open(path, 'r') as f:
                        for i, _ in enumerate(f):
                            row_count = i + 1
                            if row_count > 1000:
                                results.append(f"CSV file {os.path.basename(path)} has more than 1000 rows ({row_count}). Showing header and first 5 rows only.")
                                break
                    
                    # Read the file with limit if needed
                    if row_count > 1000:
                        df = pd.read_csv(path, nrows=5)
                    else:
                        df = pd.read_csv(path)
                    
                    results.append(f"CSV file {os.path.basename(path)} has {row_count} rows and {len(df.columns)} columns.")
                    results.append(f"Columns: {', '.join(df.columns.tolist())}")
                    results.append("First few rows:")
                    results.append(df.head(5).to_string())
                except Exception as e:
                    results.append(f"Error reading {os.path.basename(path)}: {str(e)}")
        
        return "\n\n".join(results)
    
    def _handle_question(self, message):
        """Handle a question message."""
        try:
            # Analyze the query type
            query_type, keywords = self._analyze_query_type(message)
            
            # Store the question in context
            from context_manager import context
            context.save_analysis_result('last_question', message)
            
            # Check for cached analysis results that might be relevant
            cached_results = self._check_cached_analysis(message, keywords)
            if cached_results:
                self.logger.info(f"Found cached analysis results for query: {message[:50]}...")
                return cached_results
            
            # Check if it's specifically an image analysis request
            analyze_match = re.search(r'(analyze|analyse|analysis)\s+([^\s\.]+(?:\.\w+)?)', message.lower())
            if analyze_match:
                requested_item = analyze_match.group(2)
                # Check if it has an extension
                if '.' in requested_item:
                    # It has an extension, handle by extension type
                    file_ext = os.path.splitext(requested_item)[1].lower()
                    if file_ext in ['.png', '.jpg']:
                        # It's an image file
                        image_name = os.path.basename(requested_item)
                        return self._analyze_specific_image(image_name)
                    elif file_ext in ['.md', '.markdown']:
                        # It's a markdown file
                        md_name = os.path.basename(requested_item)
                        return self._handle_markdown_command(md_name)
                    elif file_ext == '.csv':
                        # It's a CSV file
                        csv_name = os.path.basename(requested_item)
                        return self._handle_csv_command(csv_name)
                else:
                    # No extension, check against various file types starting with images
                    # Check if it's a valid image name
                    for img_type in ['png', 'jpg']:
                        for img_path in self.discovered_files.get(img_type, []):
                            if requested_item.lower() in os.path.basename(img_path).lower():
                                # It's a valid image, so analyze it directly
                                return self._analyze_specific_image(requested_item)
                
                    # Check if it's a markdown file
                    for md_path in self.discovered_files.get('md', []):
                        if requested_item.lower() in os.path.basename(md_path).lower():
                            # It's a markdown file
                            return self._handle_markdown_command(requested_item)
                
                    # Check if it's a CSV file
                    for csv_path in self.discovered_files.get('csv', []):
                        if requested_item.lower() in os.path.basename(csv_path).lower():
                            # It's a CSV file
                            return self._handle_csv_command(requested_item)
        
            # Check if we can answer using the knowledge graph - for non-image-analysis questions
            kg_answer = self._answer_from_knowledge_graph(message)
            if kg_answer:
                return kg_answer
            
            # Rest of the existing _handle_question method...
            # Check if the question is about listing files
            list_file_patterns = [
                r'list\s+(the\s+)?(all\s+)?(\w+)(\s+files)?',
                r'show\s+(the\s+)?(all\s+)?(\w+)(\s+files)?',
                r'what\s+(\w+)(\s+files)?\s+(do\s+you\s+have|are\s+available)',
                r'display\s+(the\s+)?(\w+)(\s+files)?'
            ]
            
            for pattern in list_file_patterns:
                match = re.search(pattern, message.lower())
                if match:
                    file_type = match.group(3)
                    
                    # For direct "list csv files" or "list images" commands
                    if message.lower().strip() == f"list {file_type} files" or message.lower().strip() == f"list {file_type}":
                        # Map simple requests to the /files command or specific file type commands
                        if file_type == "csv":
                            return self._handle_csv_command("")
                        elif file_type in ["image", "images"]:
                            return self._handle_files_command()
                    
                    # Map common file type references to our internal categories
                    file_type_mapping = {
                        'image': ['png', 'jpg'],
                        'images': ['png', 'jpg'],
                        'picture': ['png', 'jpg'],
                        'pictures': ['png', 'jpg'],
                        'photo': ['png', 'jpg'],
                        'photos': ['png', 'jpg'],
                        'png': ['png'],
                        'jpg': ['jpg'],
                        'text': ['txt'],
                        'markdown': ['md'],
                        'md': ['md'],
                        'csv': ['csv'],
                        'spreadsheet': ['csv'],
                        'spreadsheets': ['csv'],
                        'excel': ['csv'],
                        'doc': ['docx'],
                        'document': ['docx', 'pdf', 'txt', 'md'],
                        'documents': ['docx', 'pdf', 'txt', 'md'],
                        'pdf': ['pdf']
                    }
                    
                    if file_type in file_type_mapping:
                        file_types = file_type_mapping[file_type]
                        
                        # Gather all files of the requested types
                        all_files = []
                        for ft in file_types:
                            if ft in self.discovered_files:
                                all_files.extend([(ft, f) for f in self.discovered_files[ft]])
                        
                        if all_files:
                            # Format the response
                            response = f"## Available {file_type.title()} Files:\n\n"
                            
                            for i, (ft, file_path) in enumerate(all_files, 1):
                                file_name = os.path.basename(file_path)
                                response += f"{i}. {file_name} ({ft.upper()} file)\n"
                            
                            if file_type in ['image', 'images', 'picture', 'pictures']:
                                response += "\nTo analyze an image, use `analyze [image name]`"
                            elif file_type in ['csv', 'spreadsheet', 'spreadsheets']:
                                response += "\nTo analyze a CSV file, use `/csv [filename]` or `/analyze [filename]`"
                            elif file_type in ['markdown', 'md']:
                                response += "\nTo view a markdown file, use `/markdown [filename]` or `/md [filename]`"
                            
                            return response
                        else:
                            return f"No {file_type} files were found. Use /discover to scan for files."
        
            # Check for introduction or overview requests
            if any(phrase in message.lower() for phrase in ["introduce yourself", "who are you", "what are you"]):
                intro = """
                Hello! I'm HydroDeepNet, an AI data analysis assistant designed to help you analyze and understand environmental data.
                
                I can help with:
                - Analyzing CSV data files and providing statistical insights
                - Interpreting images, charts, and visualizations
                - Summarizing markdown reports and documentation
                - Creating visualizations from data
                
                You can get started by using commands like:
                - `/files` to see available files
                - `/csv [filename]` to analyze CSV files
                - `analyze [image name]` to analyze images
                - `/help` for a list of all commands
                """
                return intro.strip()
                
            # Check for general data overview requests
            if any(phrase in message.lower() for phrase in ["what data", "available data", "data available", "data you have", "data types"]):
                data_overview = f"""
                # Available Data Sources
                
                ## CSV Files ({len(self.discovered_files.get('csv', []))})
                Statistical data about groundwater, climate, vegetation, and more
                
                ## Images ({len(self.discovered_files.get('png', []) + self.discovered_files.get('jpg', []))})
                Visualizations including maps, charts, time series, and spatial analyses
                
                ## Documentation ({len(self.discovered_files.get('md', []))})
                Detailed reports about the different data domains
                
                ## JSON Files ({len(self.discovered_files.get('json', []))})
                Configuration and metadata information
                
                Use `/files` to see a complete list of files, or specify a type like `/csv` or `/markdown`
                """
                return data_overview.strip()
            
            # Original code continues from here...
            
            # Process the message to extract key information
            keywords = extract_keywords(message)
            
            # If the query is specifically about analyzing an image, handle it directly
            if message.lower().startswith(('analyze ', 'analysis ', 'analyse ')):
                image_name = message.lower().replace('analyze ', '').replace('analysis ', '').replace('analyse ', '').strip()
                return self._analyze_specific_image(image_name)
            
            # Try to find the most relevant data sources for this question
            relevant_sources = self._find_relevant_data_sources(message, keywords)
            
            # If we found highly relevant sources, automatically use them to answer the question
            if relevant_sources and relevant_sources['high_relevance']:
                return self._answer_with_relevant_sources(message, relevant_sources)
            
            # If there are potentially relevant sources but we're not sure, suggest them
            if relevant_sources and relevant_sources['medium_relevance']:
                from context_manager import context
                suggested_files = relevant_sources['medium_relevance']
                
                # Ensure pending_actions exists in context.config
                if 'pending_actions' not in context.config:
                    context.config['pending_actions'] = []
                    
                context.config['pending_actions'].append({
                    'type': 'file_check',
                    'file_paths': suggested_files
                })
                file_names = [os.path.basename(f) for f in suggested_files]
                return f"I think these files might help answer your question about {keywords[:3]}: {', '.join(file_names)}. Would you like me to analyze them for you?"
            
            # Update context with the current question
            from context_manager import context
            context.save_analysis_result('last_question', message)
            
            # Check if question is about images generally
            is_image_question = any(keyword in message.lower() for keyword in 
                                ['image', 'picture', 'graph', 'chart', 'plot', 'visualization', 'png', 'jpg'])
            
            # If asking about images specifically, provide a list of available images
            if is_image_question and (self.discovered_files.get('png') or self.discovered_files.get('jpg')):
                relevant_images = self._find_relevant_images(message, keywords)
                if relevant_images:
                    # If we found specifically relevant images, offer those
                    image_list = "\n".join([f"- {os.path.basename(img)}" for img in relevant_images])
                    return f"I found these images that might relate to your question about {keywords[:3]}:\n\n{image_list}\n\nWould you like me to analyze any of these images? You can say 'analyze [image name]'."
                else:
                    # Otherwise show all images
                    image_files = []
                    for img_type in ['png', 'jpg']:
                        image_files.extend(self.discovered_files.get(img_type, []))
                    
                    if image_files:
                        image_list = "\n".join([f"- {os.path.basename(img)}" for img in image_files])
                        return f"I found the following image files:\n\n{image_list}\n\nWould you like me to analyze any of these images? You can say 'analyze [image name]'."
        
            # Forward the regular questions to the appropriate agent
            if self.interactive_agent:
                # Use the interactive agent with carefully filtered conversation history
                filtered_history = self._prepare_conversation_history()
                
                try:
                    # Set stream=False to avoid repetition issues with streamed responses
                    response = self.interactive_agent.print_response(
                        message,
                        context=filtered_history,
                        stream=False
                    )
                    return response
                except Exception as e:
                    logger.error(f"_handle_question Error with interactive agent: {str(e)}")
                    # Fallback to direct print_response if the first attempt fails
                    return self.interactive_agent.print_response(
                        message,
                        context=filtered_history,
                        stream=False
                    )
            elif self.image_agent and self.images and is_image_question:
                # If the question seems to be about images
                try:
                    return self.image_agent.print_response(
                        message,
                        images=self.images,
                        stream=False
                    )
                except Exception as e:
                    logger.error(f"_handle_question Error with image agent: {str(e)}")
                    return self.image_agent.print_response(
                        message,
                        images=self.images,
                        stream=False
                    )
            elif self.agent:
                # Use the regular agent for text-based knowledge
                return self.agent.print_response(message, stream=False)
            else:
                return "No agent or knowledge base has been initialized. Please initialize first or use /discover to find files."
                
        except Exception as e:
            logger.error(f"_handle_question Error: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Error processing your question: {str(e)}"
    
    def _prepare_conversation_history(self):
        """Prepare a filtered conversation history to avoid repetition."""
        if not self.conversation_history:
            return []
            
        # Filter out duplicate consecutive messages
        filtered_history = []
        prev_role = None
        prev_content = None
        
        for msg in self.conversation_history:
            role = msg.get('role')
            content = msg.get('content', '')
            
            # Skip if this is a duplicate of the previous message
            if role == prev_role and content == prev_content:
                continue
                
            # Skip empty messages
            if not content.strip():
                continue
                
            # Add to filtered history
            filtered_history.append(msg)
            prev_role = role
            prev_content = content
            
        # Limit history size to avoid context bloat
        if len(filtered_history) > 10:
            # Keep the first 2 messages (system context) and the last 8 for recency
            filtered_history = filtered_history[:2] + filtered_history[-8:]
            
        return filtered_history
    
    def _find_relevant_data_sources(self, query, keywords):
        """Intelligently find relevant data sources based on the user's query."""
        from context_manager import context
        
        result = {
            'high_relevance': [],  # Will answer directly with these
            'medium_relevance': [] # Will suggest these to the user
        }
        
        # Topic categories and related terms
        topic_keywords = {
            'solar': ['solar', 'pv', 'photovoltaic', 'sun', 'energy', 'power', 'renewable', 'electricity', 'panel'],
            'groundwater': ['water', 'groundwater', 'aquifer', 'well', 'hydrology', 'level', 'flow', 'quality'],
            'precipitation': ['rain', 'precipitation', 'rainfall', 'weather', 'storm', 'climate', 'wet', 'dry'],
            'temperature': ['temperature', 'heat', 'warm', 'cold', 'climate', 'degree', 'thermal'],
            'statistics': ['statistics', 'stats', 'average', 'mean', 'median', 'trend', 'analysis', 'correlation']
        }
        
        # Check if the query strongly matches any topics
        query_topics = {}
        for topic, terms in topic_keywords.items():
            score = 0
            for term in terms:
                if term in query.lower():
                    score += 3  # Exact match in query
                if any(term in kw for kw in keywords):
                    score += 1  # Partial match in keywords
            
            if score > 0:
                query_topics[topic] = score
        
        # Check each file for relevance to the query topics
        for file_type in self.discovered_files:
            for file_path in self.discovered_files[file_type]:
                file_name = os.path.basename(file_path).lower()
                
                # Calculate file relevance to identified topics
                file_relevance = 0
                for topic, score in query_topics.items():
                    if topic in file_name:
                        file_relevance += score * 2  # Topic directly in filename
                        
                    # Check for topic keywords in filename
                    for term in topic_keywords[topic]:
                        if term in file_name:
                            file_relevance += score
                
                # Direct keyword matches in filename
                for kw in keywords:
                    if kw in file_name:
                        file_relevance += 5
                
                # Check if file has been mentioned in conversation history
                for message in self.conversation_history[-5:]:  # Check last 5 messages
                    if message.get('role') == 'user' and file_name in message.get('content', '').lower():
                        file_relevance += 10  # Recently mentioned by user
                
                # Add to appropriate relevance category
                if file_relevance > 10:  # High confidence threshold
                    result['high_relevance'].append(file_path)
                elif file_relevance > 5:  # Medium confidence threshold
                    result['medium_relevance'].append(file_path)
        
        # Sort by filename to ensure stable order
        result['high_relevance'].sort()
        result['medium_relevance'].sort()
        
        # Limit the number of suggestions to prevent overwhelming the user
        result['high_relevance'] = result['high_relevance'][:3]  # Top 3 most relevant
        result['medium_relevance'] = result['medium_relevance'][:3]  # Top 3 medium relevance
        
        return result
    
    def _find_relevant_images(self, query, keywords):
        """Find images that are relevant to the user's query."""
        relevant_images = []
        
        # Extract topic keywords from the query
        topics = ['solar', 'groundwater', 'precipitation', 'temperature', 'seasonal', 
                'correlation', 'trend', 'map', 'spatial', 'time series', 'energy']
        
        query_topics = [topic for topic in topics if topic in query.lower()]
        if not query_topics:
            query_topics = [kw for kw in keywords if any(kw in topic for topic in topics)]
        
        # Search for images related to the detected topics
        for img_type in ['png', 'jpg']:
            for img_path in self.discovered_files.get(img_type, []):
                img_name = os.path.basename(img_path).lower()
                
                # Check if any topic is in the image name
                for topic in query_topics:
                    if topic in img_name:
                        relevant_images.append(img_path)
                        break
                
                # Also check for specific keywords
                for kw in keywords:
                    if len(kw) > 3 and kw in img_name:  # Only consider substantial keywords
                        relevant_images.append(img_path)
                        break
        
        return sorted(set(relevant_images))  # Remove duplicates and sort
    
    def _answer_with_relevant_sources(self, query, relevant_sources):
        """Proactively analyze and answer using the identified relevant sources."""
        from context_manager import context
        results = []
        
        # First, analyze each highly relevant source
        for source_path in relevant_sources['high_relevance']:
            source_name = os.path.basename(source_path)
            results.append(f"I found that '{source_name}' is relevant to your question. Here's what it shows:")
            
            # Handle different file types
            if source_path.endswith('.csv'):
                # Analyze CSV data
                analysis = self._quick_csv_analysis(source_path)
                if analysis is not None:
                    results.append(analysis)
                
                # Create a contextual visualization if appropriate
                if should_visualize(query, source_path):
                    viz_path = self._create_contextual_visualization(query, source_path)
                    if viz_path:
                        results.append(f"I've created a visualization to help answer your question: {os.path.basename(viz_path)}")
                        # Add the visualization to our images for reference
                        image_obj = Image(filepath=viz_path)
                        self.images.append(image_obj)
                        
                        # Analyze the visualization
                        viz_analysis = self._analyze_specific_image(os.path.basename(viz_path))
                        if viz_analysis is not None:
                            results.append(viz_analysis)
            
            elif source_path.endswith(('.png', '.jpg')):
                # Directly analyze the image
                image_analysis = self._analyze_specific_image(os.path.basename(source_path))
                if image_analysis is not None:
                    results.append(image_analysis)
                
            elif source_path.endswith('.json'):
                # Extract relevant info from JSON
                json_analysis = self._analyze_json_for_query(query, source_path)
                if json_analysis is not None:
                    results.append(json_analysis)
        
        # Update context to reflect what we've analyzed
        context.set_current_topic(extract_main_topic(query), relevant_sources['high_relevance'])
        
        # Combine all results with synthesis - ensure we have actual results
        if results:
            if len(results) > 1:
                synthesis = f"Based on the {len(relevant_sources['high_relevance'])} relevant files I analyzed, "
                synthesis += f"here's what I can tell you about {context.current_topic}:\n\n"
                # Filter out any None values before joining
                filtered_results = [r for r in results if r is not None]
                synthesis += "\n\n".join(filtered_results)
                return synthesis
            else:
                # Ensure we don't have None
                return results[0] if results[0] is not None else "I analyzed the file but couldn't extract meaningful information."
        else:
            return "I couldn't find any relevant information in the selected files."
    
    def _quick_csv_analysis(self, csv_path):
        """Quick analysis of CSV data focused on answering the user's query."""
        try:
            from csv_utils import quick_csv_analysis
            return quick_csv_analysis(csv_path)
        except Exception as e:
            logger.error(f"_quick_csv_analysis Error in quick CSV analysis: {str(e)}")
            return f"I tried to analyze {os.path.basename(csv_path)} but encountered an error: {str(e)}"
    
    def _analyze_json_for_query(self, query, json_path):
        """Extract relevant information from JSON based on the query."""
        try:
            from json_queries import analyze_json_for_query
            return analyze_json_for_query(query, json_path)
        except Exception as e:
            logger.error(f"_analyze_json_for_query Error analyzing JSON: {str(e)}")
            return f"I tried to analyze the JSON file but encountered an error: {str(e)}"
    
    
    def _analyze_specific_image(self, image_name):
        """Analyze a specific image file based on the name."""
        from context_manager import context
        import os
        
        # If image has an extension, strip it for better matching
        base_name = os.path.splitext(image_name)[0]
        
        # Find all image files in discovered files
        all_images = []
        for img_type in ['png', 'jpg']:
            all_images.extend(self.discovered_files.get(img_type, []))
        
        # Find matching images
        target_image_path = None
        for path in all_images:
            path_basename = os.path.basename(path)
            path_basename_noext = os.path.splitext(path_basename)[0]
            
            # Match with or without extension
            if base_name.lower() in path_basename_noext.lower():
                target_image_path = path
                break
        
        if not target_image_path:
            return f"No image found matching '{image_name}'. Use `/files` to see available files."
        
        # Update context
        context.set_current_topic('image_analysis', [target_image_path])
        
        try:
            # Check if we have a cached analysis for this image
            cached_analysis = context.get_file_analysis(target_image_path)
            if cached_analysis:
                return cached_analysis
            
            # Get context information about the image
            img_context = self._get_image_context(target_image_path)
            
            # Generate a summary of the image
            from utils import create_image_summary
            summary = create_image_summary(target_image_path)
            
            # Combine information
            response = f"## Analysis of {os.path.basename(target_image_path)}\n\n"
            response += summary + "\n\n"
            
            if img_context:
                response += "### Context\n"
                response += img_context
            
            # Save analysis in context
            context.save_analysis_result(target_image_path, response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error analyzing image: {str(e)}")
            return f"Error analyzing image '{os.path.basename(target_image_path)}': {str(e)}"

    
    def _get_image_context(self, image_path):
        """Get context information for an image by looking for related data files."""
        image_name = os.path.basename(image_path).lower()
        context = []
        
        # Extract potential keywords from the filename
        parts = image_name.replace('.png', '').replace('.jpg', '').split('_')
        
        # Look for related CSV files
        for csv_path in self.discovered_files.get('csv', []):
            csv_name = os.path.basename(csv_path).lower()
            
            # Check if any part of the image name matches the CSV name
            if any(part in csv_name for part in parts if len(part) > 2):
                context.append(f"Related data file: {os.path.basename(csv_path)}")
                
                # Try to get column names from the CSV
                try:
                    df = pd.read_csv(csv_path, nrows=1)
                    columns = df.columns.tolist()
                    context.append(f"Data columns: {', '.join(columns)}")
                except Exception:
                    pass
        
        # Look for related markdown files that might have descriptions
        for md_path in self.discovered_files.get('md', []):
            md_name = os.path.basename(md_path).lower()
            
            # Check if any part of the image name matches the markdown name
            if any(part in md_name for part in parts if len(part) > 2):
                context.append(f"Related documentation: {os.path.basename(md_path)}")
                
                # Try to extract relevant sections from the markdown
                try:
                    with open(md_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Look for sections that mention the image name
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if image_name in line.lower():
                            # Extract the surrounding context (5 lines before and after)
                            start = max(0, i-5)
                            end = min(len(lines), i+5)
                            relevant_lines = lines[start:end]
                            context.append("Related documentation excerpt:")
                            context.append("\n".join(relevant_lines))
                            break
                except Exception:
                    pass
        
        # If we have context, format it nicely
        if context:
            return "Context information:\n" + "\n".join(context)
        else:
            return "No additional context information found for this image."

    def enhance_image_analysis_capabilities(self):
        # Add specialized vision model capabilities
        self.vision_model = Agent(
            model=OpenAIChat(id="gpt-4o"),
            agent_id="detailed-vision-analyzer",
            name="Detailed Image Analyzer",
            instructions=[
                "Analyze images in extreme detail with focus on scientific data visualization",
                "Extract quantitative data from charts and graphs when possible",
                "Identify trends, patterns, outliers and statistical significance in visualizations",
                "Connect image content to relevant CSV data sources available in the system"
            ]
        )
        
        # Add image categorization by domain
        self.image_categories = {
            "climate": ["climate", "temperature", "precipitation", "seasonal", "spatial_change"],
            "vegetation": ["NDVI", "EVI", "LAI", "ET", "vegetation"],
            "soil": ["soil", "texture", "distribution", "map"],
            "groundwater": ["groundwater", "aquifer", "water", "H_COND"],
            "land_use": ["cdl", "crop", "land", "diversity"]
        }

    def implement_contextual_analysis(self):
        # Create a system to connect related data assets
        self.data_context = {
            "images": {},  # Organized by category 
            "csv_files": {},  # Organized by data domain
            "md_files": {},  # Documentation by topic
            "relationships": {}  # Tracks connections between assets
        }
        
        # Map relationships between files
        def map_data_relationships():
            for image_path in self.discovered_files.get('png', []):
                image_name = os.path.basename(image_path)
                # Find related CSV files with similar name patterns
                for csv_path in self.discovered_files.get('csv', []):
                    csv_name = os.path.basename(csv_path)
                    if calculate_similarity(image_name.split('.')[0], csv_name.split('.')[0]) > 0.7:
                        self.data_context["relationships"][image_name] = csv_name



    def create_knowledge_graph(self):
        """Create and populate a knowledge graph to represent domain concepts and their relationships."""
        logger.info("Creating knowledge graph")
        
        # Initialize knowledge graph with discovered files
        self.knowledge_graph_manager = KnowledgeGraph(
            discovered_files=self.discovered_files,
            logger=logger
        )
        
        # Create the knowledge graph
        return self.knowledge_graph_manager.create_knowledge_graph()
            
    def _answer_from_knowledge_graph(self, query):
        """Try to answer a query using the knowledge graph."""
        if not self.knowledge_graph_manager:
            logger.debug("Knowledge graph manager not initialized")
            return None
        
        logger.info(f"Attempting to answer query using knowledge graph: {query[:50]}...")
        try:
            answer = self.knowledge_graph_manager.answer_query(query)
            if answer:
                logger.info("Found answer in knowledge graph")
            else:
                logger.info("No answer found in knowledge graph")
            return answer
        except Exception as e:
            logger.error(f"Error querying knowledge graph: {str(e)}")
            return None

    def _handle_csv_command(self, args):
        """Handle /csv command to analyze a CSV file or list available CSV files."""
        try:
            from csv_utils import handle_csv_command
            return handle_csv_command(args, self.discovered_files, logger)
        except Exception as e:
            logger.error(f"_handle_csv_command Error in CSV command: {str(e)}")
            return f"Error processing CSV command: {str(e)}"

    def _handle_markdown_command(self, args):
        """Handle /markdown command to view markdown files."""
        try:
            from prompt_handler import handle_markdown_command
            return handle_markdown_command(args, self.discovered_files, logger)
        except Exception as e:
            logger.error(f"_handle_markdown_command Error in markdown command: {str(e)}")
            return f"Error processing markdown command: {str(e)}"
    
    def _handle_context_command(self, args=None):
        """Display the current context."""
        from context_manager import context
        
        # Get the current context as a dictionary
        ctx = {
            'session_id': context.session_id,
            'current_topic': context.current_topic,
            'current_files': context.current_files,
            'discovered_file_count': {k: len(v) for k, v in context.discovered_files.items() if v},
            'analysis_results_count': len(context.analysis_results),
            'conversation_history_count': len(context.conversation_history)
        }
        
        return f"Current context:\n{json.dumps(ctx, indent=2)}"
        
        
def interactive_document_reader(base_path):
    """Example usage of the InteractiveDocumentReader."""
    reader = InteractiveDocumentReader()
    
    # Auto-discover files in the base path
    reader.initialize(auto_discover=True, base_path=base_path)
    
    print("Interactive Document Reader initialized.")
    print("Type '/help' for a list of available commands.")
    print("Type 'exit' to quit.")
    
    # Interactive chat loop
    while True:
        message = input("\nYou: ")
        
        if message.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye!")
            break
        
        response = reader.chat(message)
        print(f"\nAI: {response}")

if __name__ == "__main__":
    username = "admin"
    report_num = "20250324_222749"
    generated_report_path = f"/data/SWATGenXApp/Users/{username}/Reports/{report_num}/"
    
    # Use the interactive document reader
    interactive_document_reader(generated_report_path)