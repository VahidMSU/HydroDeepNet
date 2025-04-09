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
import logging
import os
import json
import pandas as pd
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import redis
import uuid
import pickle
from agno.knowledge.combined import CombinedKnowledgeBase


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        
        # Context tracking
        self.context = {
            'current_topic': None,
            'current_files': [],
            'pending_questions': [],
            'pending_actions': [],
            'last_question': None
        }
        
        # Redis integration for persistent conversation history
        try:
            self.redis = redis.Redis.from_url(redis_url)
            logger.info(f"Connected to Redis at {redis_url}")
            self.has_redis = True
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {str(e)}. Using in-memory storage instead.")
            self.has_redis = False

    def _save_to_redis(self, key, value, expire_seconds=86400):
        """Save a value to Redis with expiration time (default 24 hours)."""
        if not self.has_redis:
            return False
        
        try:
            full_key = f"doc_reader:{self.session_id}:{key}"
            serialized = pickle.dumps(value)
            self.redis.set(full_key, serialized, ex=expire_seconds)
            return True
        except Exception as e:
            logger.error(f"Error saving to Redis: {str(e)}")
            return False
    
    def _load_from_redis(self, key):
        """Load a value from Redis."""
        if not self.has_redis:
            return None
        
        try:
            full_key = f"doc_reader:{self.session_id}:{key}"
            serialized = self.redis.get(full_key)
            if serialized:
                return pickle.loads(serialized)
            return None
        except Exception as e:
            logger.error(f"Error loading from Redis: {str(e)}")
            return None
    
    def _clear_redis_session(self):
        """Clear all keys for the current session."""
        if not self.has_redis:
            return
        
        try:
            pattern = f"doc_reader:{self.session_id}:*"
            for key in self.redis.scan_iter(match=pattern):
                self.redis.delete(key)
        except Exception as e:
            logger.error(f"Error clearing Redis session: {str(e)}")
    
    def _sync_conversation_history(self):
        """Sync conversation history with Redis."""
        if self.has_redis:
            # Try to load from Redis first
            redis_history = self._load_from_redis('conversation_history')
            if redis_history:
                self.conversation_history = redis_history
            else:
                # Save current history
                self._save_to_redis('conversation_history', self.conversation_history)
            
            # Also sync context
            redis_context = self._load_from_redis('context')
            if redis_context:
                self.context = redis_context
            else:
                self._save_to_redis('context', self.context)
    
    def initialize(self, auto_discover=False, base_path=None):
        """Initialize individual knowledge bases and create combined knowledge base."""
        try:
            # Auto-discover files if requested
            if auto_discover and base_path:
                self._discover_files(base_path)
                
            # Initialize individual knowledge bases
            if 'pdf' in self.config:
                self._initialize_knowledge_base('pdf', self.config['pdf'])
                
            if 'text' in self.config:
                self._initialize_knowledge_base('text', self.config['text'])
                
            if 'csv' in self.config:
                self._initialize_knowledge_base('csv', self.config['csv'])
                
            if 'json' in self.config:
                self._initialize_knowledge_base('json', self.config['json'])
                
            if 'docx' in self.config:
                self._initialize_knowledge_base('docx', self.config['docx'])
                
            if 'website' in self.config:
                self._initialize_knowledge_base('website', self.config['website'])
                
            if 'image' in self.config:
                self._initialize_image_reader('image', self.config['image'])
            
            # Create combined knowledge base if multiple knowledge bases exist
            if len(self.knowledge_bases) > 1:
                self._create_combined_knowledge_base()
            elif len(self.knowledge_bases) == 1:
                # Use the single knowledge base if only one was initialized
                doc_type = list(self.knowledge_bases.keys())[0]
                self.agent = Agent(
                    knowledge=self.knowledge_bases[doc_type],
                    search_knowledge=True
                )
            else:
                logger.warning("No knowledge bases were initialized")
                
            # Initialize interactive agent
            self._initialize_interactive_agent()
                
            return True
            
        except Exception as e:
            logger.error(f"Error initializing knowledge bases: {str(e)}")
            raise
    
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
        
        try:
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
            
            # Update config with discovered files
            for file_type, file_paths in self.discovered_files.items():
                if file_paths:
                    if file_type == 'csv':
                        for i, path in enumerate(file_paths):
                            table_name = f"{file_type}_{i}_docs"
                            if 'csv' not in self.config:
                                self.config['csv'] = []
                            self.config['csv'].append({
                                'path': path,
                                'table_name': table_name
                            })
                    elif file_type == 'json':
                        for i, path in enumerate(file_paths):
                            table_name = f"{file_type}_{i}_docs"
                            if 'json' not in self.config:
                                self.config['json'] = []
                            self.config['json'].append({
                                'path': path,
                                'table_name': table_name
                            })
                    elif file_type in ['png', 'jpg']:
                        for i, path in enumerate(file_paths):
                            table_name = f"image_{i}_analysis"
                            if 'image' not in self.config:
                                self.config['image'] = []
                            self.config['image'].append({
                                'path': path,
                                'table_name': table_name
                            })
                            # Create a basic image summary
                            self._create_image_summary(path)
                    elif file_type in ['pdf', 'docx', 'md', 'txt']:
                        for i, path in enumerate(file_paths):
                            table_name = f"{file_type}_{i}_docs"
                            if file_type not in self.config:
                                self.config[file_type] = []
                            self.config[file_type].append({
                                'path': path,
                                'table_name': table_name
                            })
            
            # Generate data summaries for all discovered files
            self._generate_data_summaries()
            
            logger.info(f"Discovered files: {json.dumps({k: len(v) for k, v in self.discovered_files.items()})}")
            return True
            
        except Exception as e:
            logger.error(f"Error discovering files: {str(e)}")
            return False
    
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
                    logger.error(f"Error generating summary for {file_path}: {str(e)}")
            
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
                    logger.error(f"Error generating summary for {file_path}: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error generating data summaries: {str(e)}")
            return False
    
    def _initialize_interactive_agent(self):
        """Initialize the main interactive agent."""
        try:
            # Create system instructions for the interactive agent
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
            
            # Add information about image files
            if self.discovered_files.get('png') or self.discovered_files.get('jpg'):
                instructions.append("\nAvailable image files:")
                for img_type in ['png', 'jpg']:
                    for img_path in self.discovered_files.get(img_type, []):
                        instructions.append(f"- {os.path.basename(img_path)}: {img_type.upper()} image file")
            
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
            logger.error(f"Error initializing interactive agent: {str(e)}")
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
            logger.error(f"Error initializing {doc_type} knowledge base: {str(e)}")
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
            logger.error(f"Error initializing image reader: {str(e)}")
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
            logger.error(f"Error creating combined knowledge base: {str(e)}")
            raise
    
    def visualize_data(self, file_path, x_column=None, y_column=None, chart_type='auto'):
        """Generate a visualization for numerical data in CSV files."""
        try:
            if not file_path.endswith('.csv'):
                return f"Visualization is currently only supported for CSV files. The file {file_path} is not a CSV."
            
            # Check if file is too large
            row_count = 0
            with open(file_path, 'r') as f:
                for i, _ in enumerate(f):
                    row_count = i + 1
                    if row_count > 1000:
                        return f"Warning: CSV file {os.path.basename(file_path)} has more than 1000 rows ({row_count}). Visualization skipped to prevent memory issues. Consider visualizing a subset of the data instead."
            
            df = pd.read_csv(file_path)
            
            # If columns aren't specified, try to guess appropriate ones
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if not numeric_cols:
                return "No numeric columns found for visualization."
            
            if not x_column:
                # Try to find a good x-axis - prefer date-like or first column
                if any(col.lower().find('date') >= 0 for col in df.columns):
                    date_cols = [col for col in df.columns if col.lower().find('date') >= 0]
                    x_column = date_cols[0]
                else:
                    x_column = df.columns[0]
            
            if not y_column:
                # Use first numeric column that's not the x column
                for col in numeric_cols:
                    if col != x_column:
                        y_column = col
                        break
                else:
                    y_column = numeric_cols[0]
            
            # Determine chart type if auto
            if chart_type == 'auto':
                if len(df) > 50:
                    chart_type = 'line'
                else:
                    chart_type = 'bar'
            
            # Create the plot
            plt.figure(figsize=(10, 6))
            
            if chart_type == 'line':
                plt.plot(df[x_column], df[y_column])
            elif chart_type == 'bar':
                plt.bar(df[x_column], df[y_column])
            elif chart_type == 'scatter':
                plt.scatter(df[x_column], df[y_column])
            
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.title(f"{y_column} vs {x_column} from {os.path.basename(file_path)}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save the plot to a file
            output_dir = os.path.join(os.path.dirname(file_path), "visualizations")
            os.makedirs(output_dir, exist_ok=True)
            
            filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_{x_column}_{y_column}_{chart_type}.png"
            output_path = os.path.join(output_dir, filename)
            plt.savefig(output_path)
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error visualizing data: {str(e)}")
            return f"Error creating visualization: {str(e)}"
    
    def analyze_data(self, file_path):
        """Generate a basic statistical analysis for data files."""
        try:
            if not file_path.endswith('.csv'):
                return f"Statistical analysis is currently only supported for CSV files. The file {file_path} is not a CSV."
            
            # Check if file is too large
            row_count = 0
            with open(file_path, 'r') as f:
                for i, _ in enumerate(f):
                    row_count = i + 1
                    if row_count > 1000:
                        return f"Warning: CSV file {os.path.basename(file_path)} has more than 1000 rows ({row_count}). Analysis skipped to prevent memory issues. Consider analyzing a subset of the data instead."
            
            df = pd.read_csv(file_path)
            
            # Get numeric columns for analysis
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if not numeric_cols:
                return "No numeric columns found for analysis."
            
            # Generate basic statistics
            stats = df[numeric_cols].describe().to_dict()
            
            # Calculate correlations if there are multiple numeric columns
            correlations = None
            if len(numeric_cols) > 1:
                correlations = df[numeric_cols].corr().to_dict()
            
            # Prepare the analysis results
            analysis = {
                'file_name': os.path.basename(file_path),
                'path': file_path,
                'rows': len(df),
                'columns': len(df.columns),
                'statistics': stats,
                'correlations': correlations,
                'missing_values_count': df.isnull().sum().to_dict()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing data: {str(e)}")
            return f"Error analyzing data: {str(e)}"
    
    def chat(self, message):
        """
        Interactive chat interface for document analysis with improved context handling.
        
        Args:
            message: User's question or command
            
        Returns:
            Agent's response
        """
        try:
            # Sync with Redis to ensure we have the latest conversation history
            self._sync_conversation_history()
            
            # Add user message to conversation history
            self.conversation_history.append({"role": "user", "content": message})
            
            # Check for pending actions from previous interactions
            pending_handled = False
            if self.context['pending_actions'] and len(message.strip()) < 20:
                # Short messages like "yes", "no", "ok" likely respond to a pending action
                for action in self.context['pending_actions']:
                    if action['type'] == 'file_check' and message.lower() in ['yes', 'y', 'sure', 'ok']:
                        # User confirmed they want to check a file
                        response = self._handle_file_check(action['file_paths'])
                        self.context['pending_actions'].remove(action)
                        pending_handled = True
                        break
            
            # Handle special commands if the message wasn't a response to a pending action
            if not pending_handled:
                # Check for special commands
                if message.startswith("/"):
                    response = self._handle_command(message)
                else:
                    # Not a command, interpret as a regular question
                    response = self._handle_question(message)
                
            # Add the response to conversation history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Save updated conversation history and context to Redis
            if self.has_redis:
                self._save_to_redis('conversation_history', self.conversation_history)
                self._save_to_redis('context', self.context)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return f"Error processing your request: {str(e)}"
    
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
                self._clear_redis_session()
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
        Available commands:
        /help - Display this help message
        /discover path - Discover files in the specified path
        /visualize file_path [x_column] [y_column] [chart_type] - Create a visualization
        /analyze file_path - Perform statistical analysis on a file
        /files - List all discovered files
        /clear - Clear conversation history and context
        /context - Show current conversation context (debug)
        
        For any other queries, just ask questions about your data.
        """
        return help_text
    
    def _handle_files_command(self):
        """Handle /files command."""
        # List all discovered files
        if not self.discovered_files:
            return "No files have been discovered yet. Use /discover path to find files."
        
        summary = "Discovered files:\n"
        for file_type, file_paths in self.discovered_files.items():
            if file_paths:
                summary += f"\n{file_type.upper()} files:\n"
                for path in file_paths:
                    summary += f"- {path}\n"
        
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
            # Use the interactive agent with the full conversation history
            response = self.interactive_agent.print_response(
                message,
                context=self.conversation_history,
                stream=True
            )
        elif self.image_agent and self.images and is_image_question:
            # If the question seems to be about images
            response = self.image_agent.print_response(
                message,
                images=self.images,
                stream=True
            )
        elif self.agent:
            # Use the regular agent for text-based knowledge
            response = self.agent.print_response(message)
        else:
            response = "No agent or knowledge base has been initialized. Please initialize first or use /discover to find files."
        
        return response
    
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
                        results.append(viz_analysis)
            
            elif source_path.endswith(('.png', '.jpg')):
                # Directly analyze the image
                image_analysis = self._analyze_specific_image(os.path.basename(source_path))
                results.append(image_analysis)
                
            elif source_path.endswith('.json'):
                # Extract relevant info from JSON
                json_analysis = self._analyze_json_for_query(query, source_path)
                results.append(json_analysis)
        
        # Update context to reflect what we've analyzed
        self.context['current_topic'] = self._extract_main_topic(query)
        self.context['current_files'] = relevant_sources['high_relevance']
        
        # Combine all results with synthesis
        if len(results) > 1:
            synthesis = f"Based on the {len(relevant_sources['high_relevance'])} relevant files I analyzed, "
            synthesis += f"here's what I can tell you about {self.context['current_topic']}:\n\n"
            synthesis += "\n\n".join(results)
            return synthesis
        else:
            return "\n\n".join(results)
    
    def _extract_main_topic(self, query):
        """Extract the main topic from a query."""
        topic_indicators = {
            'solar': ['solar', 'pv', 'photovoltaic', 'sun', 'energy', 'power', 'renewable'],
            'groundwater': ['water', 'groundwater', 'aquifer', 'well', 'hydrology'],
            'precipitation': ['rain', 'precipitation', 'rainfall', 'weather', 'storm', 'climate'],
            'temperature': ['temperature', 'heat', 'warm', 'cold', 'climate'],
            'statistics': ['statistics', 'stats', 'average', 'mean', 'median', 'trend', 'analysis']
        }
        
        # Count indicators for each topic
        topic_scores = {topic: 0 for topic in topic_indicators}
        for topic, indicators in topic_indicators.items():
            for indicator in indicators:
                if indicator in query.lower():
                    topic_scores[topic] += 1
        
        # Find topic with highest score
        if any(score > 0 for score in topic_scores.values()):
            main_topic = max(topic_scores.items(), key=lambda x: x[1])[0]
            return main_topic
        else:
            return "general_inquiry"
    
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
            logger.error(f"Error in quick CSV analysis: {str(e)}")
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
            logger.error(f"Error analyzing JSON: {str(e)}")
            return f"I tried to analyze the JSON file but encountered an error: {str(e)}"
    
    def _should_visualize(self, query, csv_path):
        """Determine if we should create a visualization based on the query and data."""
        visualization_indicators = ['show', 'plot', 'graph', 'chart', 'visualize', 'trend', 
                                   'compare', 'relationship', 'correlation', 'pattern']
        
        # Check if query explicitly asks for visualization
        if any(indicator in query.lower() for indicator in visualization_indicators):
            return True
        
        try:
            # Check if data is suitable for visualization
            df = pd.read_csv(csv_path)
            
            # Need at least one numeric column
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if not numeric_cols:
                return False
            
            # Check for time-series data (date columns or sequential values)
            has_date_col = any('date' in col.lower() or 'time' in col.lower() or 'year' in col.lower() 
                              for col in df.columns)
            
            # If data has multiple numeric columns or time dimension, it's good for visualization
            return has_date_col or len(numeric_cols) >= 2
            
        except Exception:
            return False
    
    def _create_contextual_visualization(self, query, csv_path):
        """Create a visualization customized to answer the specific query."""
        try:
            df = pd.read_csv(csv_path)
            
            # Determine which columns to visualize based on the query
            keywords = self._extract_keywords(query)
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if not numeric_cols:
                return None
            
            # Try to find columns mentioned in the query
            mentioned_cols = []
            for col in df.columns:
                if any(kw in col.lower() for kw in keywords):
                    mentioned_cols.append(col)
            
            # Determine x and y columns
            x_col = None
            y_col = None
            
            # Look for date/time columns first for x-axis
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower() or 'year' in col.lower():
                    x_col = col
                    break
            
            # If no date column, use the first column
            if not x_col:
                x_col = df.columns[0]
            
            # For y-axis, prioritize columns mentioned in the query
            for col in mentioned_cols:
                if col in numeric_cols:
                    y_col = col
                    break
            
            # If no mentioned column is numeric, use the first numeric column
            if not y_col and numeric_cols:
                y_col = numeric_cols[0]
            
            # Determine appropriate chart type
            chart_type = 'line'  # Default
            if 'distribution' in query.lower() or 'histogram' in query.lower():
                chart_type = 'histogram'
            elif 'scatter' in query.lower() or 'correlation' in query.lower() or 'relationship' in query.lower():
                chart_type = 'scatter'
            elif len(df) < 30:  # Small datasets look better as bar charts
                chart_type = 'bar'
            
            # Create the visualization
            plt.figure(figsize=(10, 6))
            
            if chart_type == 'line':
                plt.plot(df[x_col], df[y_col])
                plt.title(f"{y_col} over {x_col}")
            elif chart_type == 'bar':
                plt.bar(df[x_col], df[y_col])
                plt.title(f"{y_col} by {x_col}")
            elif chart_type == 'scatter':
                plt.scatter(df[x_col], df[y_col])
                plt.title(f"Relationship between {x_col} and {y_col}")
            elif chart_type == 'histogram':
                plt.hist(df[y_col], bins=15)
                plt.title(f"Distribution of {y_col}")
            
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save the plot
            output_dir = os.path.join(os.path.dirname(csv_path), "visualizations")
            os.makedirs(output_dir, exist_ok=True)
            
            # Use query keywords in the filename
            query_slug = "_".join(keywords[:2]).replace(' ', '_')
            filename = f"auto_viz_{query_slug}_{y_col}_{chart_type}.png"
            output_path = os.path.join(output_dir, filename)
            plt.savefig(output_path)
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return None
    
    def _extract_keywords(self, text):
        """Extract important keywords from a message."""
        # This is a simple implementation that could be enhanced with NLP techniques
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'about'}
        words = text.lower().split()
        return [word for word in words if word not in common_words and len(word) > 2]
    
    def _analyze_specific_image(self, image_name):
        """Analyze a specific image using the simple approach from image_reader.py."""
        # Find matching image paths
        matching_images = []
        for img_type in ['png', 'jpg']:
            for path in self.discovered_files.get(img_type, []):
                if image_name.lower() in os.path.basename(path).lower():
                    matching_images.append(path)
        
        if not matching_images:
            return f"I couldn't find any image matching '{image_name}'. Please check the name and try again."
        
        if len(matching_images) > 1:
            image_list = "\n".join([f"- {os.path.basename(img)}" for img in matching_images])
            return f"I found multiple images matching '{image_name}'. Please be more specific:\n\n{image_list}"
        
        # We found exactly one matching image
        target_image_path = matching_images[0]
        
        # Update context
        self.context['current_topic'] = 'image_analysis'
        self.context['current_files'] = [target_image_path]
        
        # Create a direct Image object
        image_obj = Image(filepath=target_image_path)
        
        # Create an image agent using the simple approach from image_reader.py
        image_agent = Agent(
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
                "For seasonal data, explain the seasonal patterns and any anomalies.",
                "For maps, describe spatial patterns and regional variations.",
                "For time series, describe trends, cycles, and notable events.",
            ],
        )
        
        # Analyze the image directly using the agent
        logger.info(f"Analyzing image: {os.path.basename(target_image_path)}")
        
        # Use the exact same simple approach as the image_reader.py example
        response = image_agent.print_response(
            f"What do you see in this image? Please analyze it in detail.",
            images=[image_obj],
            stream=True,
        )
        
        return response

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
