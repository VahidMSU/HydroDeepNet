from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.knowledge.text import TextKnowledgeBase
from agno.knowledge.csv import CSVKnowledgeBase
from agno.knowledge.json import JSONKnowledgeBase
from agno.knowledge.docx import DocxKnowledgeBase
from agno.knowledge.website import WebsiteKnowledgeBase
from agno.vectordb.pgvector import PgVector
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
from agno.knowledge.combined import CombinedKnowledgeBase
import re
import traceback
from PIL import Image as PILImage
import logging
from AI_agent.report_analyser_example.Logger import LoggerSetup
import os 
from utils import (
    extract_image_name,
    create_image_summary,
    is_response_complete,
    deduplicate_content,
    extract_main_topic,
    calculate_similarity
)
from redis_tools import RedisTools
from prompt_handler import extract_keywords, clean_response_output, should_visualize
from data_visualizer import DataVisualizer
from knowledge_graph import KnowledgeGraph

logger = LoggerSetup(verbose=False, rewrite=True)   
logger = logger.setup_logger("AI_AgentLogger")

class InteractiveDocumentReader:
    """An interactive AI system for understanding and interpreting various document types."""
    
    def __init__(self, config=None, redis_url="redis://localhost:6379/0"):
        """
        Initialize the interactive document reader.
        
        Args:
            config: Dictionary containing configuration for different document types and database
                If None, will attempt auto-discovery
            redis_url: URL for Redis connection (used for persistent conversation context)
        """
        self.config = config or {}
        self.db_url = self.config.get('db_url', "postgresql+psycopg://ai:ai@localhost:5432/ai")
        self.knowledge_bases = {}
        self.combined_knowledge_base = None
        self.agent = None
        self.embedder = OpenAIEmbedder()
        self.image_agent = None
        self.images = []
        
        # New fields for interactive capabilities
        self.conversation_history = []
        self.discovered_files = {}
        self.data_summaries = {}
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        self.interactive_agent = None
        
        # Multi-agent system
        self.agents = {
            "coordinator": None,
            "document_navigator": None,
            "visual_analyst": None,
            "data_scientist": None,
            "domain_expert": None
        }
        
        # Context tracking
        self.context = {
            'current_topic': None,
            'current_files': [],
            'pending_questions': [],
            'pending_actions': [],
            'last_question': None,
            'agent_states': {}  # Track state of each agent
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
        self.has_redis = self.redis_tools.has_redis
        
        # Initialize utility classes
        self.data_visualizer = None
        self.knowledge_graph_manager = None
    
    def _sync_conversation_history(self):
        """Sync conversation history with Redis."""
        if self.has_redis:
            # Sync conversation history and context with Redis
            updated_data = self.redis_tools.sync_data(
                {
                    'conversation_history': self.conversation_history,
                    'context': self.context
                },
                ['conversation_history', 'context']
            )
            
            self.conversation_history = updated_data['conversation_history']
            self.context = updated_data['context']
    
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
        
        # Initialize context
        self.context = {
            'current_topic': None,
            'current_files': [],
            'pending_questions': [],
            'pending_actions': [],
            'last_question': None
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
        
        # Check for Redis
        if self.has_redis:
            # Try to load conversation history and context from Redis
            saved_history = self.redis_tools.load_from_redis('conversation_history')
            if saved_history:
                self.conversation_history = saved_history
                logger.info(f"Loaded conversation history from Redis: {len(saved_history)} messages")
            
            saved_context = self.redis_tools.load_from_redis('context')
            if saved_context:
                self.context = saved_context
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
        """Auto-discover and categorize files in the given directory."""
        logger.info(f"Auto-discovering files in {base_path}")
        
        # Initialize discovered files by category
        self.discovered_files = {
            'pdf': [],
            'csv': [],
            'json': [],
            'docx': [],
            'md': [],
            'txt': [],
            'png': [],
            'jpg': [],
            'html': [],
            'other': []
        }
        
        # Walk through the directory structure
        for root, dirs, files in os.walk(base_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()[1:]
                
                # Categorize file by extension
                if file_ext in self.discovered_files:
                    self.discovered_files[file_ext].append(file_path)
                else:
                    self.discovered_files['other'].append(file_path)
        
        # Log information about discovered files
        discovered_count = {k: len(v) for k, v in self.discovered_files.items() if v}
        logger.info(f"Discovered file counts: {discovered_count}")
        
        # Update config with discovered files
        for file_type, file_paths in self.discovered_files.items():
            if file_paths:
                if file_type == 'csv':
                    for i, path in enumerate(file_paths):
                        table_name = f"{file_type}_{i}_docs"
                        if 'csv' not in self.config:
                            self.config['csv'] = []
                        
                        # Check if path is already in config
                        if not any(cfg.get('path') == path for cfg in self.config['csv']):
                            self.config['csv'].append({
                                'path': path,
                                'table_name': table_name
                            })
                elif file_type == 'json':
                    for i, path in enumerate(file_paths):
                        table_name = f"{file_type}_{i}_docs"
                        if 'json' not in self.config:
                            self.config['json'] = []
                        
                        # Check if path is already in config
                        if not any(cfg.get('path') == path for cfg in self.config['json']):
                            self.config['json'].append({
                                'path': path,
                                'table_name': table_name
                            })
                elif file_type in ['png', 'jpg']:
                    for i, path in enumerate(file_paths):
                        table_name = f"image_{i}_analysis"
                        if 'image' not in self.config:
                            self.config['image'] = []
                        
                        # Check if path is already in config
                        if not any(cfg.get('path') == path for cfg in self.config['image']):
                            self.config['image'].append({
                                'path': path,
                                'table_name': table_name
                            })
                        # Create a basic image summary
                        create_image_summary(path)
                elif file_type in ['pdf', 'docx', 'md', 'txt']:
                    for i, path in enumerate(file_paths):
                        table_name = f"{file_type}_{i}_docs"
                        if file_type not in self.config:
                            self.config[file_type] = []
                        
                        # Check if path is already in config
                        if not any(cfg.get('path') == path for cfg in self.config[file_type]):
                            self.config[file_type].append({
                                'path': path,
                                'table_name': table_name
                            })
        
        # Generate data summaries for all discovered files
        self._generate_data_summaries()
        
        # Ensure the discovered files are properly reflected in the agent's instructions
        self._update_interactive_agent_instructions()
        
        logger.info(f"Discovered files: {json.dumps({k: len(v) for k, v in self.discovered_files.items()})}")
        return True
        

    
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
    
    def _should_visualize(self, query, csv_path):
        """Determine if we should create a visualization based on the query and data."""
        return self.data_visualizer.should_visualize(query, csv_path)
    
    def _create_contextual_visualization(self, query, csv_path):
        """Create a visualization customized to answer the specific query."""
        return self.data_visualizer.create_contextual_visualization(query, csv_path)
    
    def chat(self, message):
        """Process user's message and return a response through the appropriate agent."""
        start_time = datetime.now()
        
        try:
            # Add the message to conversation history
            self.conversation_history.append({"role": "user", "content": message})
            logger.debug(f"Processing user message: '{message[:50] if message else ''}...'")
            
            # Check if there are pending actions to confirm
            pending_handled = False
            if self.context.get('pending_actions'):
                for i, action in enumerate(self.context.get('pending_actions', [])):
                    if action['type'] == 'file_check' and "yes" in message.lower():
                        # User confirmed file check
                        logger.debug(f"Handling file check confirmation for {len(action['file_paths'])} files")
                        response = self._handle_file_check(action['file_paths'])
                        self.context['pending_actions'].pop(i)
                        pending_handled = True
                        break
                    elif action['type'] == 'image_analysis' and "analyze" in message.lower():
                        # Extract image name from user message
                        image_name = extract_image_name(message)
                        if image_name:
                            logger.debug(f"Handling image analysis for '{image_name}'")
                            response = self._analyze_specific_image(image_name)
                            self.context['pending_actions'].pop(i)
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
                                response = self.validate_response(raw_response, message)
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
                            response = self.validate_response(raw_response, message)
                        else:
                            logger.warning("Received empty response from agent")
                            response = "I apologize, but I couldn't generate a response. Please try rephrasing your question."
            
            # Ensure response is never None before adding to history
            if response is None:
                logger.warning("Response is None, using generic fallback")
                response = "I processed your request, but couldn't generate a specific response."
            
            # Clean the response of any debug information or formatting
            logger.debug(f"Cleaning response output of length: {len(response) if response else 0}")
            cleaned_response = self._clean_response_output(response)
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
            
            # Save updated conversation history and context to Redis
            if self.has_redis:
                self.redis_tools.save_to_redis('conversation_history', self.conversation_history)
                self.redis_tools.save_to_redis('context', self.context)
                
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
    
    def _clean_response_output(self, text):
        """
        Clean output text removing debug information and formatting.
        """
        return clean_response_output(text, logger)
    
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
    
    def _analyze_query_type(self, message):
        """Analyze a user query to determine its type."""
        message_lower = message.lower()
        
        # Check for file listing queries
        if any(pattern in message_lower for pattern in [
            "list", "show", "what files", "available files", "what documents",
            "display files", "show me the files", "what images"
        ]):
            return "file_listing"
            
        # Check for image analysis queries
        if any(pattern in message_lower for pattern in [
            "analyze image", "analyze the image", "analyze this image",
            "describe image", "explain image", "what does the image show",
            "interpret the chart", "explain the graph", "analyze the chart"
        ]):
            return "image_analysis"
            
        # Check for data analysis queries
        if any(pattern in message_lower for pattern in [
            "analyze data", "statistics", "correlation", "trend", "pattern",
            "what does the data show", "data analysis", "statistical", "mean",
            "average", "distribution", "csv analysis", "analyze csv"
        ]):
            return "data_analysis"
            
        # Check for domain knowledge queries
        if any(pattern in message_lower for pattern in [
            "explain", "why does", "how does", "what causes", "relationship between",
            "impact of", "effect of", "mechanism", "process", "scientific explanation"
        ]):
            return "domain_knowledge"
            
        # Default to general query
        return "general"
    
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
            self.context = {
                'current_topic': None,
                'current_files': [],
                'pending_questions': [],
                'pending_actions': [],
                'last_question': None
            }
            if self.has_redis:
                self.redis_tools.clear_redis_session()
            return "Conversation history and context cleared."
        elif command == "/context":
            # Debug command to show current context
            return f"Current context:\n{json.dumps(self.context, indent=2)}"
        else:
            return f"Unknown command: {command}. Type /help for available commands."
    
    def _handle_visualize_command(self, args):
        """Handle /visualize command."""
        # Command format: /visualize file_path [x_column] [y_column] [chart_type]
        parts = args.split(maxsplit=3)
        file_path = parts[0] if len(parts) > 0 else None
        x_column = parts[1] if len(parts) > 1 else None
        y_column = parts[2] if len(parts) > 2 else None
        chart_type = parts[3] if len(parts) > 3 else 'auto'
        
        if not file_path:
            return "Please specify a file path to visualize."
        
        # Find matching files if path is partial
        matching_files = []
        for file_type in self.discovered_files:
            for path in self.discovered_files[file_type]:
                if file_path in path:
                    matching_files.append(path)
        
        if not matching_files:
            return f"No files found matching '{file_path}'."
        
        if len(matching_files) > 1:
            return f"Multiple files found matching '{file_path}'. Please be more specific:\n" + \
                    "\n".join(matching_files)
        
        # Visualize the file
        result = self.visualize_data(matching_files[0], x_column, y_column, chart_type)
        
        # Add the generated visualization to image agent
        if result and os.path.exists(result) and result.endswith('.png'):
            image_obj = Image(filepath=result)
            self.images.append(image_obj)
            
            if not self.image_agent:
                self._initialize_image_reader('image', {'path': result})
            
            # Update context with the current visualization
            self.context['current_topic'] = 'visualization'
            self.context['current_files'].append(matching_files[0])
            self.context['last_visualization'] = result
            
            return f"Visualization created: {result}"
        
        return result
    
    def _handle_analyze_command(self, args):
        """Handle /analyze command."""
        # Command format: /analyze file_path
        file_path = args.strip()
        
        if not file_path:
            return "Please specify a file path to analyze."
        
        # Find matching files if path is partial
        matching_files = []
        for file_type in self.discovered_files:
            for path in self.discovered_files[file_type]:
                if file_path in path:
                    matching_files.append(path)
        
        if not matching_files:
            return f"No files found matching '{file_path}'."
        
        if len(matching_files) > 1:
            return f"Multiple files found matching '{file_path}'. Please be more specific:\n" + \
                    "\n".join(matching_files)
        
        # Update context
        self.context['current_topic'] = 'analysis'
        self.context['current_files'] = [matching_files[0]]
        
        # Analyze the file
        return self.analyze_data(matching_files[0])
    
    def _handle_discover_command(self, args):
        """Handle /discover command."""
        # Command format: /discover base_path
        base_path = args.strip()
        
        if not base_path:
            return "Please specify a base path to discover files."
        
        success = self._discover_files(base_path)
        
        if success:
            # Summarize discovered files
            summary = "Discovered files:\n"
            for file_type, file_paths in self.discovered_files.items():
                if file_paths:
                    summary += f"- {file_type}: {len(file_paths)} files\n"
            
            # Update context
            self.context['current_topic'] = 'file_discovery'
            
            return summary
        else:
            return "Error discovering files. Check the logs for details."
    
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
        """Handle regular user questions with intelligent data source selection."""
        # First, check if it's specifically an image analysis request
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
        
        # Original code continues from here...
        
        # Process the message to extract key information
        keywords = self._extract_keywords(message)
        
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
            suggested_files = relevant_sources['medium_relevance']
            self.context['pending_actions'].append({
                'type': 'file_check',
                'file_paths': suggested_files
            })
            file_names = [os.path.basename(f) for f in suggested_files]
            return f"I think these files might help answer your question about {keywords[:3]}: {', '.join(file_names)}. Would you like me to analyze them for you?"
        
        # Update context with the current question
        self.context['last_question'] = message
        
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
                if self._should_visualize(query, source_path):
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
        self.context['current_topic'] = extract_main_topic(query)
        self.context['current_files'] = relevant_sources['high_relevance']
        
        # Combine all results with synthesis - ensure we have actual results
        if results:
            if len(results) > 1:
                synthesis = f"Based on the {len(relevant_sources['high_relevance'])} relevant files I analyzed, "
                synthesis += f"here's what I can tell you about {self.context['current_topic']}:\n\n"
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
            # Check file size first
            row_count = 0
            with open(csv_path, 'r') as f:
                for i, _ in enumerate(f):
                    row_count = i + 1
                    if row_count > 1000:
                        return f"CSV file {os.path.basename(csv_path)} has {row_count} rows (showing summary only due to size)."
            
            # Read the CSV data
            df = pd.read_csv(csv_path)
            
            # Basic statistics
            summary = f"The file contains {len(df)} rows and {len(df.columns)} columns.\n"
            summary += f"Columns: {', '.join(df.columns.tolist())}\n\n"
            
            # Find numeric columns for statistical analysis
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                summary += "Key statistics:\n"
                for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                    summary += f"- {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, avg={df[col].mean():.2f}\n"
            
            # Check for interesting patterns
            if len(numeric_cols) >= 2:
                # Find highest correlation
                corr_matrix = df[numeric_cols].corr()
                highest_corr = 0
                pair = (None, None)
                
                for i, col1 in enumerate(numeric_cols):
                    for col2 in numeric_cols[i+1:]:
                        if abs(corr_matrix.loc[col1, col2]) > abs(highest_corr):
                            highest_corr = corr_matrix.loc[col1, col2]
                            pair = (col1, col2)
                
                if pair[0] and abs(highest_corr) > 0.5:
                    corr_type = "positive" if highest_corr > 0 else "negative"
                    summary += f"\nI noticed a strong {corr_type} correlation ({highest_corr:.2f}) between {pair[0]} and {pair[1]}."
            
            return summary
            
        except Exception as e:
            logger.error(f"_quick_csv_analysis Error in quick CSV analysis: {str(e)}")
            return f"I tried to analyze {os.path.basename(csv_path)} but encountered an error: {str(e)}"
    
    def _analyze_json_for_query(self, query, json_path):
        """Extract relevant information from JSON based on the query."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Extract query keywords
            keywords = self._extract_keywords(query)
            
            # Simple summary
            summary = f"The JSON file '{os.path.basename(json_path)}' "
            
            if isinstance(data, dict):
                summary += f"contains {len(data)} top-level keys.\n"
                
                # Find keys that match keywords
                matching_keys = []
                for key in data.keys():
                    if any(kw in key.lower() for kw in keywords):
                        matching_keys.append(key)
                
                if matching_keys:
                    summary += "Here are the sections that seem relevant to your question:\n\n"
                    for key in matching_keys[:3]:  # Limit to first 3 matches
                        value = data[key]
                        if isinstance(value, (dict, list)) and len(str(value)) > 500:
                            # Summarize large objects/arrays
                            if isinstance(value, dict):
                                summary += f"- {key}: A dictionary with {len(value)} keys\n"
                            else:
                                summary += f"- {key}: An array with {len(value)} items\n"
                        else:
                            # Show the actual value for smaller data
                            summary += f"- {key}: {json.dumps(value)}\n"
                else:
                    # No direct matches, show top-level structure
                    summary += "Top-level keys: " + ", ".join(list(data.keys())[:10])
                    if len(data) > 10:
                        summary += f" and {len(data) - 10} more."
            elif isinstance(data, list):
                summary += f"contains a list with {len(data)} items.\n"
                if data and isinstance(data[0], dict):
                    # Show structure of list items
                    keys = data[0].keys()
                    summary += f"Each item has these fields: {', '.join(keys)}\n"
                    
                    # Try to find items matching the query
                    matching_items = []
                    for item in data[:20]:  # Only check first 20 items
                        if any(kw in str(item).lower() for kw in keywords):
                            matching_items.append(item)
                    
                    if matching_items:
                        summary += f"\nFound {len(matching_items)} items relevant to your query. First match:\n"
                        summary += json.dumps(matching_items[0], indent=2)
            
            return summary
            
        except Exception as e:
            logger.error(f"_analyze_json_for_query Error analyzing JSON: {str(e)}")
            return f"I tried to analyze the JSON file but encountered an error: {str(e)}"
    
    def _extract_keywords(self, text):
        """Extract important keywords from a message."""
        # Use the standalone function from prompt_handler.py
        return extract_keywords(text)
    
    def _analyze_specific_image(self, image_name):
        """Analyze a specific image using the visual analyst agent and specialized functions."""
        # Remove extension if present to improve matching
        base_image_name = os.path.splitext(image_name)[0]
        
        # Find matching image paths
        matching_images = []
        for img_type in ['png', 'jpg']:
            for path in self.discovered_files.get(img_type, []):
                path_basename = os.path.basename(path)
                path_basename_noext = os.path.splitext(path_basename)[0]
                
                # Match with or without extension
                if (base_image_name.lower() in path_basename_noext.lower() or 
                    image_name.lower() in path_basename.lower()):
                    matching_images.append(path)
        
        if not matching_images:
            return f"I couldn't find any image matching '{image_name}'. Please check the name and try again."
        
        if len(matching_images) > 1:
            image_list = "\n".join([f"- {os.path.basename(img)}" for img in matching_images])
            return f"I found multiple images matching '{image_name}'. Please be more specific:\n\n{image_list}"
        
        # We found exactly one matching image
        target_image_path = matching_images[0]
        target_image_name = os.path.basename(target_image_path)
        
        # Update context
        self.context['current_topic'] = 'image_analysis'
        self.context['current_files'] = [target_image_path]
                # Create a direct Image object
        image_obj = Image(filepath=target_image_path)
        
        # First try using the visual analyst from the multi-agent system
        if self.agents.get("visual_analyst") is not None:
            logger.info(f"Using visual analyst agent to analyze {target_image_name}")
            
            # Get analysis context by looking for related CSV files
            context_info = self._get_image_context(target_image_path)
            
            prompt = f"""
            Please analyze this image: {target_image_name} in detail.
            
            Focus on these aspects:
            1. What type of visualization or chart is this?
            2. What are the main features, trends, or patterns visible?
            3. What variables or data are being represented?
            4. What scientific insights can be drawn from this visualization?
            5. How does this relate to environmental or geological data?
            
            {context_info}
            
            Provide a comprehensive analysis with key observations and insights.
            """
            
            try:
                # Use print_response with stream=False instead of ask
                analysis = self.agents["visual_analyst"].print_response(prompt, images=[image_obj], stream=False)
                return analysis
            except Exception as e:
                logger.error(f"Error using visual analyst agent: {str(e)}")
                # Fall back to the regular image agent
                pass
        
        # Fallback to the simple image agent
        logger.info(f"Falling back to simple image agent for {target_image_name}")
        if self.image_agent is not None:
            try:
                prompt = f"""
                Please analyze this image: {target_image_name}
                
                What does this image show? What type of visualization is it?
                What are the main patterns or trends visible?
                What scientific insights can we gain from this image?
                """
                
                # Use print_response for the image agent as well, not ask
                analysis = self.image_agent.print_response(prompt, images=[image_obj], stream=False)
                return analysis
            except Exception as e:
                logger.error(f"Error using image agent fallback: {str(e)}")
        
        # Last resort: basic image description if both agents fail
        return f"I found the image {target_image_name} but couldn't analyze it with my AI vision capabilities. The image agents encountered errors."
        

    
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

    def validate_response(self, response, query):
        """
        Validate and improve the agent's response.
        
        Args:
            response: The raw response from the agent
            query: The user's query that prompted this response
            
        Returns:
            The validated and potentially enhanced response
        """
        logger.debug(f"Validating response: length={len(response) if response else 0}")
        
        # Skip validation if response is None or empty
        if not response or not response.strip():
            logger.warning("Empty response received during validation")
            return "I apologize, but I couldn't generate a complete response."
            
        # If the response is very short, check if it's an error or apology
        if len(response.strip()) < 20:
            logger.warning(f"Very short response: '{response}'")
            if "error" in response.lower() or "apologize" in response.lower() or "sorry" in response.lower():
                return response
            else:
                # For very short non-error responses, we assume they're valid
                return response
            
        # Check for incomplete or truncated responses
        if not is_response_complete(response):
            logger.debug("Response appears incomplete, completing it")
            response = self._complete_response(response)
            
        # Check for repeated content
        if self._has_repeated_sections(response):
            logger.debug("Response has repeated sections, deduplicating")
            response = deduplicate_content(response)
            
        # Only add this context clarification for long responses that seem to be
        # generic or not directly addressing the query
        relevance_score = self._calculate_query_relevance(response, query)
        logger.debug(f"Response relevance score: {relevance_score:.2f}")
        
        if len(response) > 500 and relevance_score < 0.3:
            query_terms = ", ".join(self._extract_keywords(query)[:3])
            if query_terms:
                logger.debug(f"Adding clarification for low-relevance response about: {query_terms}")
                # Only add clarification text if the response seems generic and unrelated
                if not response.endswith("\n"):
                    response += "\n\n"
                else:
                    response += "\n"
                response += f"Regarding your specific question about {query_terms}: I've provided the available information above. If you need more specific details, please let me know."
        
        # Remove any empty list items or bullet points
        response = re.sub(r'\n\s*[-*]\s*\n', '\n\n', response)
        
        # Remove any remaining debug information
        cleaned_response = self._clean_response_output(response)
        
        return cleaned_response

    def _has_repeated_sections(self, text):
        """Check if a response has repeated sections."""
        if not text:
            return False
            
        # Split text into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) <= 1:
            return False
            
        # Check for duplicate paragraphs
        seen_paragraphs = set()
        for paragraph in paragraphs:
            if paragraph in seen_paragraphs:
                return True
            seen_paragraphs.add(paragraph)
            
        # Check for similar consecutive sections
        for i in range(len(paragraphs) - 1):
            similarity = calculate_similarity(paragraphs[i], paragraphs[i+1])
            if similarity > 0.7:  # 70% similarity threshold
                return True
                
        return False
  
    def _complete_response(self, text):
        """Try to make an incomplete response complete."""
        if not text:
            return "I apologize, but I couldn't generate a complete response."
            
        # If cut off with ellipsis, add a proper ending
        if text.strip().endswith(('...', '…')):
            return text.strip() + "\n\nI apologize, but the response was truncated. Please let me know if you'd like more information on this topic."
            
        # Balance unmatched parentheses, brackets, braces
        for open_char, close_char in [('(', ')'), ('[', ']'), ('{', '}')]:
            open_count = text.count(open_char)
            close_count = text.count(close_char)
            if open_count > close_count:
                # Add missing closing characters
                text = text + (close_char * (open_count - close_count))
                
        # If ends with a colon, add a concluding sentence
        if text.strip().endswith(':'):
            text = text + " I'll provide more details if you'd like additional information."
            
        return text
        
    def _calculate_query_relevance(self, response, query):
        """Calculate how relevant a response is to the original query."""
        if not response or not query:
            return 0.0
            
        # Extract key terms from the query
        query_terms = set(self._extract_keywords(query))
        if not query_terms:
            return 1.0  # No meaningful terms to match
            
        # Check how many query terms appear in the response
        response_lower = response.lower()
        matched_terms = sum(1 for term in query_terms if term.lower() in response_lower)
        
        # Calculate relevance score
        relevance_score = matched_terms / len(query_terms) if query_terms else 0.0
        
        return relevance_score
        
    def _enhance_relevance(self, response, query):
        """Try to enhance the relevance of a response to the query."""
        # If response already seems relevant, leave it as is
        if self._calculate_query_relevance(response, query) >= 0.7:
            return response
            
        # Extract key information from the query
        query_terms = self._extract_keywords(query)
        
        # Append a note addressing the query more directly
        enhanced = response.strip()
        enhanced += "\n\nRegarding your specific question about " + ", ".join(query_terms[:3]) + ": "
        enhanced += f"I've provided the available information above. If you need more specific details about {' or '.join(query_terms[:2])}, please let me know."
        
        return enhanced

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
            return None
            
        return self.knowledge_graph_manager.answer_query(query)

    def _handle_csv_command(self, args):
        """Handle /csv command to analyze a CSV file or list available CSV files."""
        # If no args are provided, list all CSV files
        if not args.strip():
            csv_files = self.discovered_files.get('csv', [])
            if not csv_files:
                return "No CSV files have been discovered yet. Use /discover path to find files."
            
            csv_list = "## Available CSV Files:\n\n"
            for i, path in enumerate(csv_files, 1):
                csv_list += f"{i}. {os.path.basename(path)}\n"
            
            csv_list += "\nTo analyze a CSV file, use `/csv [filename]` or `/analyze [filename]`"
            return csv_list
        
        # If a filename is provided, find and analyze the file
        file_path = args.strip()
        
        # Remove extension if present to improve matching
        base_file_name = os.path.splitext(file_path)[0]
        
        # Find matching files if path is partial
        matching_files = []
        for path in self.discovered_files.get('csv', []):
            path_basename = os.path.basename(path)
            path_basename_noext = os.path.splitext(path_basename)[0]
            
            # Match with or without extension
            if (base_file_name.lower() in path_basename_noext.lower() or
                file_path.lower() in path_basename.lower()):
                matching_files.append(path)
        
        if not matching_files:
            return f"No CSV files found matching '{file_path}'. Use `/files` to see available files."
        
        if len(matching_files) > 1:
            file_list = "\n".join([f"- {os.path.basename(path)}" for path in matching_files])
            return f"Multiple CSV files found matching '{file_path}'. Please be more specific:\n\n{file_list}"
        
        # We found exactly one matching file
        target_csv = matching_files[0]
        
        # Update context
        self.context['current_topic'] = 'csv_analysis'
        self.context['current_files'] = [target_csv]
        
        try:
            # Read the CSV data
            df = pd.read_csv(target_csv)
            
            # Prepare response
            filename = os.path.basename(target_csv)
            response = f"## Analysis of {filename}\n\n"
            response += f"This CSV file contains {len(df)} rows and {len(df.columns)} columns.\n\n"
            
            # List all columns
            response += "### Columns:\n"
            for col in df.columns:
                response += f"- {col}\n"
            
            # Show a preview
            response += "\n### Preview (first 5 rows):\n"
            response += df.head(5).to_string()
            
            # Add basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                response += "\n\n### Basic Statistics:\n"
                # Calculate statistics for numeric columns
                stats_df = df[numeric_cols].describe().round(2)
                response += stats_df.to_string()
                
                # Suggest visualization if there are numeric columns
                response += "\n\nYou can create visualizations with `/visualize " + filename + " [x_column] [y_column]`"
            
            return response
            
        except Exception as e:
            logger.error(f"Error analyzing CSV file: {str(e)}")
            return f"Error analyzing CSV file '{os.path.basename(target_csv)}': {str(e)}"
    
    def _handle_markdown_command(self, args):
        """Handle /markdown command to view markdown files."""
        # If no args are provided, list all markdown files
        if not args.strip():
            md_files = self.discovered_files.get('md', [])
            if not md_files:
                return "No markdown files have been discovered yet. Use `/discover path` to find files."
            
            md_list = "## Available Markdown Files:\n\n"
            for i, path in enumerate(md_files, 1):
                md_list += f"{i}. {os.path.basename(path)}\n"
            
            md_list += "\nTo view a markdown file, use `/markdown [filename]` or `/md [filename]`"
            return md_list
        
        # If a filename is provided, find and read the file
        file_path = args.strip()
        
        # Remove extension if present to improve matching
        base_file_name = os.path.splitext(file_path)[0]
        
        # Find matching files if path is partial
        matching_files = []
        for path in self.discovered_files.get('md', []):
            path_basename = os.path.basename(path)
            path_basename_noext = os.path.splitext(path_basename)[0]
            
            # Match with or without extension
            if (base_file_name.lower() in path_basename_noext.lower() or
                file_path.lower() in path_basename.lower()):
                matching_files.append(path)
        
        if not matching_files:
            return f"No markdown files found matching '{file_path}'. Use `/files` to see available files."
        
        if len(matching_files) > 1:
            file_list = "\n".join([f"- {os.path.basename(path)}" for path in matching_files])
            return f"Multiple markdown files found matching '{file_path}'. Please be more specific:\n\n{file_list}"
        
        # We found exactly one matching file
        target_md = matching_files[0]
        
        # Update context
        self.context['current_topic'] = 'markdown_view'
        self.context['current_files'] = [target_md]
        
        try:
            # Read the markdown file
            with open(target_md, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Prepare response
            response = f"## Markdown File: {os.path.basename(target_md)}\n\n"
            response += content
            
            return response
            
        except Exception as e:
            logger.error(f"Error reading markdown file: {str(e)}")
            return f"Error reading markdown file '{os.path.basename(target_md)}': {str(e)}"

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