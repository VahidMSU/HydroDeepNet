import os
import re
import json
import pickle
import redis
import logging
import traceback
from datetime import datetime

# Import related modules
from document_reader_agents import AgentHandler
from document_reader_files import FileManager
from document_reader_commands import CommandHandler
from document_reader_analysis import AnalysisHandler
from document_reader_knowledge import KnowledgeHandler
from document_reader_response import ResponseHandler

# Get logger
from Logger import LoggerSetup
# Initialize logger using setup_logger method
logger_setup = LoggerSetup()
logger = logger_setup.setup_logger()

class InteractiveDocumentReader:
    """An interactive AI system for understanding and interpreting various document types."""
    
    def __init__(self, config=None, redis_url="redis://localhost:6379/0"):
        """Initialize the document reader with optional configuration."""
        # Store configuration
        self.config = config or {}
        
        # Set up Redis connection if available
        self.redis_url = redis_url
        self.has_redis = True
        try:
            self.redis = redis.from_url(redis_url)
            logger.info(f"Connected to Redis at {redis_url}")
        except Exception as e:
            self.has_redis = False
            logger.warning(f"Failed to connect to Redis: {str(e)}")
        
        # Prepare other attributes that will be initialized later
        self.knowledge_bases = {}
        self.agent = None
        self.interactive_agent = None
        self.image_agent = None
        self.images = []
        self.agents = {}  # For multi-agent system
        
        # Initialize conversation history
        self.conversation_history = []
        
        # Initialize document paths
        self.discovered_files = {
            'csv': [],
            'txt': [],
            'md': [],
            'png': [],
            'jpg': [],
            'pdf': [],
            'json': [],
            'html': [],
            'docx': [],
        }
        
        # Initialize context
        self.context = {
            'current_topic': None,
            'current_files': [],
            'pending_questions': [],
            'pending_actions': [],
            'last_question': None
        }
        
        # Initialize metrics
        self.metrics = {
            "response_times": [],
            "total_responses": 0,
            "conversations": 0
        }
        
        # Create handlers
        self.agent_handler = AgentHandler(self)
        self.file_handler = FileManager(self)
        self.command_handler = CommandHandler(self)
        self.analysis_handler = AnalysisHandler(self)
        self.knowledge_handler = KnowledgeHandler(self)
        self.response_handler = ResponseHandler(self)
    
    def initialize(self, auto_discover=False, base_path=None):
        """Initialize the interactive document reader with optional file discovery."""
        # Configure logging first
        self._configure_logging()
        
        logger.info("Initializing InteractiveDocumentReader")
        
        # Check for Redis
        if self.has_redis:
            # Try to load conversation history from Redis
            saved_history = self._load_from_redis('conversation_history')
            if saved_history:
                self.conversation_history = saved_history
                logger.info(f"Loaded conversation history from Redis: {len(saved_history)} messages")
            
            # Try to load context from Redis
            saved_context = self._load_from_redis('context')
            if saved_context:
                self.context = saved_context
                logger.info(f"Loaded context from Redis")
        
        # Auto-discover files if requested
        if auto_discover and base_path:
            self.file_handler.discover_files(base_path)
        
        # Initialize multi-agent system if we have a config
        if self.config:
            self.agent_handler.initialize_multi_agent_system()
            self.agent_handler.initialize_interactive_agent()
            
            # Create knowledge graph for deeper semantic understanding
            self.knowledge_handler.create_knowledge_graph()
            
        # Add default welcome message if history is empty
        if not self.conversation_history:
            self.conversation_history.append({
                "role": "assistant", 
                "content": "Hello! I'm your interactive document assistant. How can I help you analyze your data today?"
            })
            
        logger.info("InteractiveDocumentReader initialized successfully")
        return True
    
    def _configure_logging(self):
        """Configure logging for both our code and Agno's logging."""
        try:
            # Redirect Agno's logging to our logger
            import logging
            import os
            
            # Get the log file path from Logger's setup - using the current log path
            log_file = None
            try:
                from Logger import LoggerSetup
                # Initialize logger correctly
                logger_setup = LoggerSetup()
                logger_instance = logger_setup.setup_logger()
                # Try to find the log file from handlers
                for handler in logger_instance.handlers:
                    if isinstance(handler, logging.FileHandler):
                        log_file = handler.baseFilename
                        break
                
                if not log_file:
                    # Fallback to a fixed path
                    log_file = "/data/SWATGenXApp/codes/AI_agent/logs/AI_AgentLogger.log"
            except:
                # Fallback to a fixed path
                log_file = "/data/SWATGenXApp/codes/AI_agent/logs/AI_AgentLogger.log"
            
            # Create a file handler that writes directly to our log file
            # rather than going through our logger (which would cause recursion)
            if log_file and os.path.exists(os.path.dirname(log_file)):
                file_handler = logging.FileHandler(log_file)
                formatter = logging.Formatter('%(asctime)s - AGNO - %(levelname)s - %(message)s')
                file_handler.setFormatter(formatter)
                file_handler.setLevel(logging.WARNING)  # Only capture warnings and errors
                
                # Capture all existing loggers
                root_logger = logging.getLogger()
                
                # Remove any existing handlers to prevent double logging
                for handler in list(root_logger.handlers):
                    if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                        # Keep file handlers but remove console handlers
                        root_logger.removeHandler(handler)
                
                # Add our direct file handler to the root logger
                root_logger.addHandler(file_handler)
                
                # Set level on the root logger
                root_logger.setLevel(logging.WARNING)
                
                # Silence specific noisy loggers
                for logger_name in ['openai', 'httpx', 'urllib3', 'asyncio', 'agno']:
                    logging.getLogger(logger_name).setLevel(logging.WARNING)
                
                logger.info(f"Configured Agno logging to write directly to log file: {log_file}")
            else:
                # Fallback: just silence everything to avoid console output
                logging.getLogger().setLevel(logging.ERROR)
                for logger_name in ['openai', 'httpx', 'urllib3', 'asyncio', 'agno']:
                    logging.getLogger(logger_name).setLevel(logging.ERROR)
                logger.info("Configured Agno logging with minimal output (no log file found)")
            
        except Exception as e:
            logger.error(f"Error configuring logging: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _save_to_redis(self, key, value, expire_seconds=86400):
        """Save a value to Redis with expiration."""
        if not self.has_redis:
            return False
        
        try:
            # Create a unique key for this session
            session_key = f"doc_reader:{key}"
            
            # Serialize value to bytes
            serialized_value = pickle.dumps(value)
            
            # Save to Redis with expiration
            self.redis.set(session_key, serialized_value, ex=expire_seconds)
            return True
        except Exception as e:
            logger.error(f"Error saving to Redis: {str(e)}")
            return False
    
    def _load_from_redis(self, key):
        """Load a value from Redis."""
        if not self.has_redis:
            return None
        
        try:
            # Get the unique key for this session
            session_key = f"doc_reader:{key}"
            
            # Get from Redis
            value = self.redis.get(session_key)
            
            if value:
                # Deserialize bytes to original value
                return pickle.loads(value)
            else:
                return None
        except Exception as e:
            logger.error(f"Error loading from Redis: {str(e)}")
            return None
    
    def _clear_redis_session(self):
        """Clear all Redis keys for this session."""
        if not self.has_redis:
            return False
        
        try:
            # Get all keys for this session
            session_keys = self.redis.keys("doc_reader:*")
            
            if session_keys:
                # Delete all keys
                self.redis.delete(*session_keys)
            
            return True
        except Exception as e:
            logger.error(f"Error clearing Redis session: {str(e)}")
            return False
    
    def _sync_conversation_history(self):
        """Sync conversation history with Redis."""
        if not self.has_redis:
            return
        
        try:
            # Load conversation history from Redis
            saved_history = self._load_from_redis('conversation_history')
            
            if saved_history and len(saved_history) > len(self.conversation_history):
                # Redis has more messages, use that
                self.conversation_history = saved_history
            elif self.conversation_history:
                # We have messages, save to Redis
                self._save_to_redis('conversation_history', self.conversation_history)
        except Exception as e:
            logger.error(f"Error syncing conversation history: {str(e)}")
    
    def chat(self, message):
        """Process user's message and return a response through the appropriate agent."""
        # Delegate to the response handler
        return self.response_handler.chat(message) 
    

