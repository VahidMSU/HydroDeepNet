from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, ServiceContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import TextNode
from pathlib import Path
import json
import base64
import os
import pandas as pd
import ollama
from Logger import LoggerSetup
from memory_system import MemorySystem
from query_understanding import QueryUnderstanding
from response_generator import ResponseGenerator
import re
import numpy as np
import concurrent.futures
from functools import partial
import time
import multiprocessing
import uuid

# ==== IMAGE ANALYSIS TOOL ====
def describe_image(path, DEFAULT_MODEL, logger, memory=None, prompt="Analyze the image and describe it in detail"):
    logger.info(f"Describing image... {path}")
    
    # Check if we already have this analysis in memory
    if memory:
        # Get file from memory system if it exists
        file_records = memory.get_related_files(f"image {os.path.basename(path)}", ["image"])
        for file in file_records:
            if file.get("original_path") == path:
                logger.info(f"Using cached image description for {path}")
                return file.get("content", '')
    
    with open(path, "rb") as img_file:
        img_data = base64.b64encode(img_file.read()).decode("utf-8")
    response = ollama.chat(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "user", "content": prompt, "images": [img_data]}
        ]
    )
    result = response["message"]["content"]
    
    # Store in memory if available
    if memory:
        memory.add_file(path, result, "image", {'prompt': prompt})
        
    return result

# ==== FILE INDEXING WITH PARALLEL PROCESSING ====
def load_all_documents(report_dir, logger, only_md_files=True, max_workers=4):

    logger.info(f"Loading documents with parallel processing...")
    
    # If only loading markdown files, use a file filter
    if only_md_files:
        # Create a custom file filter for SimpleDirectoryReader
        def md_file_filter(file_path):
            return file_path.endswith('.md')
        
        logger.info("Loading only markdown files for faster indexing")
        file_filter = md_file_filter
    else:
        logger.info("Loading all document types")
        file_filter = None
    
    # Get all files that match our filter
    all_files = []
    for root, _, files in os.walk(report_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file_filter is None or file_filter(file_path):
                all_files.append(file_path)
    
    logger.info(f"Found {len(all_files)} files matching filter criteria")
    if not all_files:
        return []
    
    # Function to load a single file
    def load_single_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        file_name = os.path.basename(file_path)
        # Create a TextNode with the file content
        return TextNode(text=content, metadata={"file_name": file_name, "file_path": file_path})

    
    # Use parallel processing to load files
    documents = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for doc in executor.map(load_single_file, all_files):
            if doc is not None:
                documents.append(doc)
    
    logger.info(f"Loaded {len(documents)} documents using parallel processing")
    return documents


def index_documents(documents, logger, max_workers=4):
    if not documents:
        logger.warning("No documents to index")
        return None
        
    logger.info(f"Indexing {len(documents)} documents with parallel processing...")
    
    # Print document structure for debugging
    logger.info(f"Document sample: {str(documents[0].__dict__)[:200]}..." if documents else "No documents")
    
    # Configure Settings instead of using deprecated ServiceContext
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 20
    
    # Check if embeddings model is configured correctly
    assert hasattr(Settings, 'embed_model') and Settings.embed_model is not None, "Embeddings model is not configured. Check Settings.embed_model"
        
    logger.info(f"Using embedding model: {type(Settings.embed_model).__name__}")
    
    # Import the Node base class
    from llama_index.core.schema import TextNode
    
    # Create properly configured TextNodes from our documents
    processed_nodes = []
    
    for doc in documents:
        # Get the content
        content = doc.get_content() if hasattr(doc, 'get_content') else str(doc)
        
        # Get the document ID or create one
        doc_id = getattr(doc, 'id_', None) or str(uuid.uuid4())
        
        # Get the metadata or create empty dict
        metadata = getattr(doc, 'metadata', {}).copy()
        
        # Add hash to metadata
        metadata['hash'] = str(hash(content + str(metadata)))
        metadata['node_id'] = doc_id
        
        # Create a new TextNode with all the required attributes
        node = TextNode(
            text=content,
            id_=doc_id,
            metadata=metadata
        )
        
        processed_nodes.append(node)
    
    # Log the processed nodes for debugging
    logger.info(f"Created {len(processed_nodes)} TextNode objects")
    
    # Create the index using the constructor approach
    logger.info("Creating VectorStoreIndex with TextNode objects")
    index = VectorStoreIndex(processed_nodes, show_progress=True)
    
    logger.info(f"Successfully indexed {len(documents)} documents")
    return index



# ==== CSV ANALYSIS TOOL ====
def summarize_csv(path, DEFAULT_MODEL, logger, memory=None):
    logger.info(f"Summarizing CSV... {path}")
    
    # Check if file exists
    assert os.path.exists(path), f"CSV file does not exist: {path}"
    
    # Check if we already have this analysis in memory
    if memory:
        # Get file from memory system if it exists
        file_records = memory.get_related_files(f"csv {os.path.basename(path)}", ["csv"])
        for file in file_records:
            if file.get("original_path") == path:
                logger.info(f"Using cached CSV summary for {path}")
                return file.get("content", '')
    
    df = pd.read_csv(path)
    summary = df.describe(include='all').to_string()
    
    # Extract key statistics for better memory retention
    stats = {}
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            stats[column] = {
                'mean': float(df[column].mean()),
                'min': float(df[column].min()),
                'max': float(df[column].max()),
                'std': float(df[column].std())
            }
        
    # Convert stats to string for consistent storage
    stats_text = "\n\nKey Statistics:\n" + json.dumps(stats, indent=2)
    
    response = ollama.generate(
        model=DEFAULT_MODEL,
        prompt=f"Summarize the following CSV file: {path}\n\n{summary}{stats_text}",
    )
    result = response.response
    
    # Store in memory if available
    if memory:
        metadata = {'statistics': stats, 'columns': list(df.columns)}
        memory.add_file(path, result, "csv", metadata)
        
    return result

# ==== QUERY FUNCTION ====
def ask_query(engine, question, conversation_history, memory=None, DEFAULT_MODEL=None):
    # Get context from memory if available
    memory_context = ""
    if memory:
        # Get related interactions from memory
        related_interactions = memory.get_related_interactions(question, [])
        if related_interactions:
            memory_context = "Previous relevant interactions:\n"
            for interaction in related_interactions[:3]:
                if "query" in interaction and "response" in interaction:
                    memory_context += f"Q: {interaction['query']}\nA: {interaction['response']}\n\n"
    
    # Add conversation history context (last 3 exchanges)
    conversation_context = "\n".join([f"Q: {q}\nA: {a}" for q, a in conversation_history[-3:]])
    
    # Combine contexts with the question
    if memory_context:
        prompt = f"{conversation_context}\n\nRelevant information:\n{memory_context}\n\nUser: {question}\nAI:"
    else:
        prompt = f"{conversation_context}\n\nUser: {question}\nAI:"
    
    # Check if engine is available
    if engine is None:
        print("Warning: Query engine is not available. Using fallback response.")
        return f"I don't have access to the document index right now. I can still help with basic questions or analyzing specific files.", conversation_history
    
    # Query the engine directly - let any errors bubble up
    response = engine.query(prompt)
    result = str(response)
    
    # Store in memory if available
    if memory:
        # Basic query info for storing the interaction
        query_info = {
            "keywords": question.lower().split(),
            "intent": "general"
        }
        memory.store_interaction(question, result, query_info, [])
        
    conversation_history.append((question, result))
    return result, conversation_history

def handle_basic_query(query):
    """Handle basic queries without needing the index"""
    query_lower = query.lower()
    
    # Basic greeting patterns
    greetings = {
        "hi": "Hello! How can I help you with the report data?",
        "hello": "Hi there! I can help you analyze reports and data files. What would you like to know?",
        "hey": "Hey! I'm ready to help you with your data analysis needs.",
        "greetings": "Greetings! I'm here to help with your report analysis."
    }
    
    # Check for exact matches first
    for greeting, response in greetings.items():
        if query_lower == greeting or query_lower == f"{greeting}!":
            return response
    
    # Check for greeting patterns
    if any(greeting in query_lower.split() for greeting in greetings.keys()):
        return "Hello! I can help you analyze report data, show visualizations, or explain files. What would you like to explore?"
    
    # Help request patterns
    if "help" in query_lower or "what can you do" in query_lower or "capabilities" in query_lower:
        return (
            "I can help you with the following:\n"
            "1. Finding and analyzing CSV data files\n"
            "2. Describing images and visualizations\n"
            "3. Explaining report content from markdown files\n"
            "4. Answering questions about the report data\n"
            "5. Listing available files by type (try 'list images' or 'list csv files')\n\n"
            "What would you like to know about?"
        )
    
    # About the system
    if "who are you" in query_lower or "what are you" in query_lower:
        return "I'm an AI assistant designed to help you analyze and understand reports. I can process CSV data, describe images, and explain report content."
    
    return None

def describe_markdown(path, DEFAULT_MODEL, logger, memory=None):
    logger.info(f"Describing markdown... {path}")
    
    # Check if we already have this analysis in memory
    if memory:
        # Get file from memory system if it exists
        file_records = memory.get_related_files(f"markdown {os.path.basename(path)}", ["markdown"])
        for file in file_records:
            if file.get("original_path") == path:
                logger.info(f"Using cached markdown summary for {path}")
                return file.get("content", '')
    
    with open(path, "r") as f:
        content = f.read()
    response = ollama.generate(
        model=DEFAULT_MODEL,
        prompt=f"Summarize the following markdown file: {path}\n\n{content}",
    )
    result = response.response
    
    # Store in memory if available
    if memory:
        memory.add_file(path, result, "markdown", {'raw_content': content[:500]})
        
    return result

# ==== PRELOAD MEMORY WITH IMPORTANT FILES ====
def preload_memory(memory, report_structure, logger, DEFAULT_MODEL):
    """
    Preload important files into memory to enhance the system's knowledge
    before user interaction begins.
    
    Args:
        memory: MemorySystem instance
        report_structure: Dictionary containing report file structure
        logger: Logger instance
        DEFAULT_MODEL: Model to use for analysis
    """
    logger.info("Preloading memory with important files...")
    
    # Track what's been preloaded
    preloaded_files = {
        'csv': [],
        'image': [],
        'markdown': [],
        'html': []
    }
    
    # Skip CSV files preloading
    logger.info("Skipping CSV files preloading...")
    
    # Skip image files preloading
    logger.info("Skipping image files preloading...")
    
    # 3. Preload markdown files (reports, documentation)
    logger.info("Preloading markdown files...")
    md_files = []
    
    # Find all markdown files in the report structure
    for group_name, group_data in report_structure.items():
        # With the simplified structure, files are in the 'files' dictionary
        for file_name, file_data in group_data.get('files', {}).items():
            if file_name.lower().endswith('.md'):
                file_path = file_data.get('path')
                if file_path:
                    md_files.append((file_name, file_path, group_name))
    
    # Process markdown files
    for file_name, file_path, group_name in md_files:
        logger.info(f"Preloading markdown: {file_name} from {group_name}")
        result = describe_markdown(file_path, DEFAULT_MODEL, logger, memory)
        preloaded_files['markdown'].append(file_name)
        
        # Add a general note about this document to memory
        query_info = {
            "keywords": ["document", "markdown", file_name, group_name],
            "intent": "search"
        }
        memory.store_interaction(
            f"What information is in the {file_name} document?",
            f"The document {file_name} in {group_name} contains: {result[:200]}...",
            query_info,
            [file_path]
        )

    
    # Log summary of what was preloaded
    preload_summary = []
    for file_type, files in preloaded_files.items():
        if files:
            preload_summary.append(f"Preloaded {len(files)} {file_type} files")
    
    logger.info("Memory preloading complete: " + ", ".join(preload_summary))
    
    # Return summary of what was preloaded
    return preloaded_files

# ==== FILE LISTING FUNCTION ====
def list_files(file_type, report_structure, memory, logger):
    """
    List files of a specific type or all files in the report structure.
    
    Args:
        file_type: Type of files to list ('csv', 'image', 'markdown', 'all')
        report_structure: Dictionary containing report file structure
        memory: MemorySystem instance
        logger: Logger instance
    """
    logger.info(f"list_files function called with file_type: '{file_type}'")
    
    # Clean up and normalize the file type
    file_type = file_type.strip().lower()
    
    # Handle common variations
    if file_type in ['img', 'imgs', 'pictures', 'picture', 'photos', 'photo']:
        file_type = 'images'
    elif file_type in ['document', 'documents', 'docs', 'doc']:
        file_type = 'markdown'
    elif file_type in ['everything', 'any', 'any file', 'any files']:
        file_type = 'all'
    
    # Map common file type names to their extensions
    type_to_extensions = {
        'csv': ['.csv'],
        'image': ['.png', '.jpg', '.jpeg', '.gif'],
        'images': ['.png', '.jpg', '.jpeg', '.gif'],
        'markdown': ['.md'],
        'md': ['.md'],
        'html': ['.html', '.htm'],
        'text': ['.txt'],
        'climate': ['.csv', '.png', '.md'],  # Multiple extensions for climate data
        'geographic': ['.png'],  # Usually maps are geographic
        'spatial': ['.png'],  # Spatial visualizations
        'all': None  # All extensions
    }
    
    # Get the extensions to look for
    extensions = type_to_extensions.get(file_type)
    if extensions is None and file_type.lower() != 'all':
        logger.warning(f"Unknown file type: {file_type}. Falling back to 'all'")
        print(f"I'm not familiar with '{file_type}' files. Showing all files instead.")
        extensions = None
        file_type = 'all'
    
    # Collect files matching the criteria
    matching_files = []
    
    for group_name, group_data in report_structure.items():
        for file_name, file_data in group_data.get('files', {}).items():
            # Get file extension
            file_ext = os.path.splitext(file_name)[1].lower()
            
            # Check if this extension matches our criteria
            if extensions is None or file_ext in extensions:
                # For special types like 'climate' or 'geographic', filter by keywords in filename
                if file_type.lower() in ['climate', 'geographic', 'spatial']:
                    relevant_keywords = {
                        'climate': ['climate', 'temp', 'precipitation', 'weather', 'rainfall'],
                        'geographic': ['map', 'spatial', 'location', 'region', 'area'],
                        'spatial': ['spatial', 'map', 'distribution']
                    }
                    keywords = relevant_keywords.get(file_type.lower(), [])
                    if not any(kw in file_name.lower() for kw in keywords) and not any(kw in group_name.lower() for kw in keywords):
                        continue
                
                # Get file path from the data
                file_path = file_data.get('path')
                if file_path:
                    matching_files.append({
                        'name': file_name,
                        'path': file_path,
                        'group': group_name,
                        'extension': file_ext
                    })
    
    # If no files found
    if not matching_files:
        message = f"No {file_type} files found in the report."
        print(message)
        
        # Store in memory
        query_info = {
            "keywords": ["list", file_type, "files"],
            "intent": "search"
        }
        memory.store_interaction(f"list {file_type} files", message, query_info, [])
        return
    
    # Group files by their group (folder)
    files_by_group = {}
    for file in matching_files:
        if file['group'] not in files_by_group:
            files_by_group[file['group']] = []
        files_by_group[file['group']].append(file)
    
    # Format output message
    output_lines = [f"Found {len(matching_files)} {file_type} files:"]
    
    for group, files in files_by_group.items():
        output_lines.append(f"\n{group} ({len(files)} files):")
        
        # Sort files for consistent output
        sorted_files = sorted(files, key=lambda x: x['name'])
        
        for i, file in enumerate(sorted_files):
            # Show all files or truncate if too many
            if i < 10 or len(sorted_files) <= 15:
                output_lines.append(f"  {i+1}. {file['name']}")
            elif i == 10 and len(sorted_files) > 15:
                output_lines.append(f"  ... and {len(sorted_files) - 10} more files")
                break
    
    # Add notes about already analyzed files
    analyzed_files = []
    for file in matching_files:
        # Check if we have this file in memory by searching for it
        file_records = memory.get_related_files(os.path.basename(file['path']), [os.path.basename(file['path'])]) 
        if file_records:
            analyzed_files.append(file['name'])
    
    if analyzed_files:
        output_lines.append(f"\nI've already analyzed {len(analyzed_files)} of these files:")
        for i, name in enumerate(analyzed_files[:5]):
            output_lines.append(f"  • {name}")
        if len(analyzed_files) > 5:
            output_lines.append(f"  • ...and {len(analyzed_files) - 5} more")
        output_lines.append("\nYou can ask me about these files directly.")
    
    # Get document status
    output_message = "\n".join(output_lines)
    print(output_message)
    
    # Save to memory
    query_info = {
        "keywords": ["list", file_type, "files"],
        "intent": "search"
    }
    memory.store_interaction(
        f"list {file_type} files", 
        f"Listed {len(matching_files)} {file_type} files from the report.",
        query_info,
        []
    )

# Fix for semantic search errors
def safe_cosine_similarity(vec1, vec2):
    """
    Safely calculate cosine similarity between two vectors, handling empty vectors
    and dimensionality mismatches.
    """
    # Check for empty vectors
    if vec1 is None or vec2 is None:
        return 0
        
    # Check for dimensionality mismatch
    if len(vec1) != len(vec2):
        return 0
        
    # Calculate similarity
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    
    if norm_a == 0 or norm_b == 0:
        return 0
        
    return dot_product / (norm_a * norm_b)


# Add new completion_agent function at the appropriate place after the imports
def completion_agent(raw_response, query, DEFAULT_MODEL, logger):
    """
    Takes a raw response and generates a polished, complete response.
    This ensures consistent style and format for all system responses.
    
    Args:
        raw_response (str): The raw response from various analysis components
        query (str): The original user query
        DEFAULT_MODEL: The language model to use
        logger: Logger instance
        
    Returns:
        str: A polished final response
    """
    # Skip completion for very short responses or error messages
    if len(raw_response) < 25 or "error" in raw_response.lower():
        return raw_response
        
    logger.info("Using completion agent to polish response")
    
    # Create a prompt that instructs the model to polish the response
    prompt = f"""You are a helpful AI assistant specializing in data analysis and explanations.
Given the user query and a draft response, improve the response to be clear, direct, and helpful.
Maintain all factual information but improve clarity and readability.

User query: {query}

Draft response: {raw_response}

Improved response:"""

    response = ollama.generate(
        model=DEFAULT_MODEL,
        prompt=prompt,
    )
    
    improved_response = response.response.strip()
    
    # If the improved response is significantly shorter, stick with the original
    if len(improved_response) < len(raw_response) * 0.7 and len(raw_response) > 100:
        logger.warning("Completion agent produced significantly shorter response, using original")
        return raw_response
        
    return improved_response

# Add after the imports but before the existing functions

class UserQueryAgent:
    """
    Agent that handles user input before it reaches the query understanding pipeline.
    Responsible for classifying queries, processing commands, and maintaining conversation context.
    """
    
    def __init__(self, memory, query_understanding, response_generator, report_structure, logger, DEFAULT_MODEL):
        """Initialize the user query agent"""
        self.memory = memory
        self.query_understanding = query_understanding
        self.response_generator = response_generator
        self.report_structure = report_structure
        self.logger = logger
        self.DEFAULT_MODEL = DEFAULT_MODEL
        self.conversation_history = []
        self.last_mentioned_files = []
        self.last_mentioned_dataset = None
        self.command_patterns = {
            "list": r"(?i)list\s+([\w\s]+)(?:\s+files)?$",
            "analyze": r"(?i)anal[yz]e\s+(.*)",
            "show": r"(?i)show\s+(.*)",
            "what_is": r"(?i)what\s+is\s+(.*)",
            "tell_about": r"(?i)tell\s+(?:me\s+)?about\s+(.*)",
            "help": r"(?i)help|help me|how to|what can you do"
        }
    
    def process_query(self, user_input, query_engine=None):
        """
        Process a user query and determine the appropriate handling strategy
        
        Args:
            user_input: The raw user input
            query_engine: Optional query engine for RAG queries
            
        Returns:
            response: The response to the user
        """
        self.logger.info(f"UserQueryAgent processing: {user_input}")
        
        # Check if this is a simple query that can be handled directly
        simple_response = handle_basic_query(user_input)
        if simple_response:
            # Apply completion agent to simple responses for consistency
            polished_response = completion_agent(simple_response, user_input, self.DEFAULT_MODEL, self.logger)
            self.memory.store_interaction(user_input, polished_response, {"intent": "general"}, [])
            return polished_response
            
        # Check for specific commands
        command_match = self._match_command(user_input)
        if command_match:
            command_type, command_args = command_match
            return self._handle_command(command_type, command_args, user_input)
            
        # Check if this might be a file analysis request with just the filename
        # This handles cases like "seasonal_tmax.png" without any command prefix
        for group_data in self.report_structure.values():
            for file_name in group_data.get('files', {}):
                if file_name.lower() in user_input.lower():
                    self.logger.info(f"Detected possible file analysis for: {file_name}")
                    # Create analysis intent and handle as file analysis
                    query_info = {
                        "intent": "analyze",
                        "file_references": [file_name],
                        "keywords": user_input.split()
                    }
                    enhanced_query = self.query_understanding.enhance_query(user_input, query_info, self.memory)
                    return self._handle_file_analysis(enhanced_query, user_input)
            
        # Process through regular query understanding pipeline
        query_info = self.query_understanding.analyze_query(user_input)
        self.logger.info(f"Query intent: {query_info.get('intent')}")
        
        # Check for references to previously mentioned entities using pronouns
        if any(word in user_input.lower() for word in ['it', 'this', 'that', 'these', 'those', 'them']):
            self._resolve_references(user_input, query_info)
            
        # Store dataset if mentioned for future reference
        if "target_dataset" in query_info:
            self.last_mentioned_dataset = query_info["target_dataset"]
            
        # For dataset inquiry, process via enhanced query
        if query_info.get("intent") == "dataset_inquiry":
            enhanced_query = self.query_understanding.enhance_query(user_input, query_info, self.memory)
            return self._handle_dataset_inquiry(enhanced_query, user_input)
            
        # For analyze intent, look for file references
        if query_info.get("intent") == "analyze" and "file_references" in query_info:
            # Store these files as last mentioned
            self.last_mentioned_files = query_info["file_references"]
            enhanced_query = self.query_understanding.enhance_query(user_input, query_info, self.memory)
            return self._handle_file_analysis(enhanced_query, user_input)
            
        # For general queries, use standard pipeline with the query engine
        enhanced_query = self.query_understanding.enhance_query(user_input, query_info, self.memory)
        
        # Get related files
        related_files = self.memory.get_related_files(user_input, query_info.get("keywords", []))
        related_file_paths = [file.get("original_path") for file in related_files if "original_path" in file]
        
        # Update last mentioned files if we found any
        if related_file_paths:
            self.last_mentioned_files = related_file_paths
            
        if query_engine:
            # Get RAG response for general questions
            rag_response, history = ask_query(
                query_engine, 
                user_input, 
                self.conversation_history, 
                self.memory, 
                self.DEFAULT_MODEL
            )
            self.conversation_history = history
            
            # Generate a better response using our response generator
            response_data = self.response_generator.generate_response(
                user_input,
                enhanced_query,
                related_file_paths,
                self.memory
            )
            
            # Get the final answer
            answer = response_data.get("answer", rag_response)
            
            # Polish with completion agent
            polished_answer = completion_agent(answer, user_input, self.DEFAULT_MODEL, self.logger)
            
            # Store in memory
            self.memory.store_interaction(user_input, polished_answer, query_info, related_file_paths)
            
            return polished_answer
        else:
            # No query engine available, use general response from response generator
            response_data = self.response_generator.generate_response(
                user_input,
                enhanced_query,
                related_file_paths,
                self.memory
            )
            
            answer = response_data.get("answer", "I'm sorry, I don't have enough information to answer that question.")
            polished_answer = completion_agent(answer, user_input, self.DEFAULT_MODEL, self.logger)
            self.memory.store_interaction(user_input, polished_answer, query_info, related_file_paths)
            
            return polished_answer
            

    
    def _match_command(self, query):
        """Match user input against command patterns"""
        self.logger.info(f"Attempting to match command patterns for: '{query}'")
        
        for cmd_type, pattern in self.command_patterns.items():
            match = re.search(pattern, query)
            if match:
                # Extract command arguments
                args = match.groups()[0] if match.groups() else ""
                self.logger.info(f"Command matched: {cmd_type} with args: '{args}'")
                return (cmd_type, args)
                
        self.logger.info("No command pattern matched, treating as general query")
        return None
        
    def _handle_command(self, command_type, command_args, original_query):
        """Handle a matched command"""
        self.logger.info(f"Handling command: {command_type} with args: {command_args}")
        
        if command_type == "list":
            # Handle list files command
            file_type = command_args.strip()
            
            # Handle multi-word args like "all images"
            if file_type.startswith("all "):
                file_type = file_type.replace("all ", "")
                self.logger.info(f"Processing 'list all' command for file type: {file_type}")
            
            self.logger.info(f"Listing files of type: {file_type}")
            list_files(file_type, self.report_structure, self.memory, self.logger)
            
            # Store command in memory with appropriate intent
            query_info = {
                "keywords": ["list", file_type, "files"],
                "intent": "search"
            }
            self.memory.store_interaction(original_query, f"Listed {file_type} files", query_info, [])
            
            # The list_files function already prints output, so we return None
            return None
            
        elif command_type in ["analyze", "show"]:
            # For analysis commands, process with analyze intent
            query_info = {
                "intent": "analyze",
                "file_references": [command_args],
                "keywords": command_args.split()
            }
            
            enhanced_query = self.query_understanding.enhance_query(original_query, query_info, self.memory)
            return self._handle_file_analysis(enhanced_query, original_query)
            
        elif command_type in ["what_is", "tell_about"]:
            # Check if this is about a dataset
            lower_args = command_args.lower()
            
            # Check against dataset names first
            for dataset in self.query_understanding.known_datasets:
                if dataset in lower_args or any(term in lower_args for term in self.query_understanding.known_datasets[dataset][:3]):
                    query_info = {
                        "intent": "dataset_inquiry",
                        "target_dataset": dataset,
                        "keywords": lower_args.split()
                    }
                    enhanced_query = self.query_understanding.enhance_query(original_query, query_info, self.memory)
                    return self._handle_dataset_inquiry(enhanced_query, original_query)
            
            # Not a dataset, treat as general query
            return None
            
        elif command_type == "help":
            help_response = (
                "I can help you with the following:\n"
                "1. Finding and analyzing CSV data files\n"
                "2. Describing images and visualizations\n"
                "3. Explaining report content from markdown files\n"
                "4. Answering questions about the report data\n"
                "5. Listing available files by type (try 'list images' or 'list csv files')\n\n"
                "You can also ask about specific datasets like MODIS, NSRDB, or PRISM data."
            )
            
            query_info = {"intent": "help", "keywords": ["help"]}
            self.memory.store_interaction(original_query, help_response, query_info, [])
            
            return help_response
        
        # If no command matched or not handled, return None to fall back to standard processing
        return None
    
    def _handle_dataset_inquiry(self, enhanced_query, original_query):
        """Handle dataset-specific inquiry"""
        if "target_dataset" not in enhanced_query:
            return None
            
        target_dataset = enhanced_query["target_dataset"]
        self.logger.info(f"Processing dataset inquiry for: {target_dataset}")
        
        # Get files specifically related to this dataset using the dataset name
        dataset_files = self.memory.get_related_files(target_dataset, 
                                                enhanced_query.get("dataset_terms", [target_dataset]), 
                                                limit=5)
        
        # Verify these files actually belong to the target dataset
        verified_dataset_files = []
        for file in dataset_files:
            file_path = file.get("original_path", "").lower()
            # Check if file is in the correct dataset directory
            if f"/{target_dataset}/" in file_path or f"\\{target_dataset}\\" in file_path:
                verified_dataset_files.append(file)
                self.logger.info(f"Verified {target_dataset} file: {file.get('file_name')}")
            else:
                self.logger.warning(f"Rejecting file not in {target_dataset} directory: {file_path}")
        
        if verified_dataset_files:
            # Update last mentioned files
            self.last_mentioned_files = [file.get("original_path") for file in verified_dataset_files 
                                       if "original_path" in file]
            
            # Format a response about the dataset
            dataset_paths = self.last_mentioned_files
            self.logger.info(f"Found {len(verified_dataset_files)} verified files for {target_dataset} dataset")
            
            # Generate response for this dataset inquiry
            dataset_response = f"**Summary of {target_dataset.upper()} Data**\n\n"
            dataset_response += f"I've located {len(verified_dataset_files)} relevant files related to {target_dataset} data. "
            
            if len(verified_dataset_files) > 0:
                dataset_response += "Here's a brief overview of each file:\n\n"
                
                for i, file in enumerate(verified_dataset_files, 1):
                    file_name = file.get("file_name", "Unknown file")
                    file_type = file.get("file_type", "unknown")
                    content_summary = file.get("content", "")[:200] + "..." if file.get("content") else "No content available"
                    
                    dataset_response += f"{i}. **{file_name}**: {content_summary}\n\n"
            
            # Polish with completion agent
            polished_response = completion_agent(dataset_response, original_query, self.DEFAULT_MODEL, self.logger)
            
            # Store in memory
            query_info = {
                "intent": "dataset_inquiry", 
                "target_dataset": target_dataset,
                "keywords": enhanced_query.get("keywords", [])
            }
            self.memory.store_interaction(original_query, polished_response, query_info, dataset_paths)
            
            return polished_response
            
        else:
            # No valid files found for this dataset
            self.logger.warning(f"No valid files found for dataset: {target_dataset}")
            
            # Try to determine if the dataset exists in the report structure
            dataset_exists = False
            for group_name in self.report_structure.keys():
                if target_dataset.lower() == group_name.lower():
                    dataset_exists = True
                    break
            
            if dataset_exists:
                response = f"I can see that {target_dataset} data exists in the reports, but I haven't been able to properly index or access those files yet. You might want to try listing all files to see what's available."
            else:
                response = f"I couldn't find any files related to {target_dataset} in the current reports. This dataset might not be available or might be named differently. You can try 'list all files' to see what's available."
            
            # Polish with completion agent
            polished_response = completion_agent(response, original_query, self.DEFAULT_MODEL, self.logger)
            
            # Store in memory
            query_info = {"intent": "dataset_inquiry", "target_dataset": target_dataset}
            self.memory.store_interaction(original_query, polished_response, query_info, [])
            
            return polished_response
    
    def _handle_file_analysis(self, enhanced_query, original_query):
        """Handle file analysis requests"""
        file_paths = []
        has_explicit_file_reference = False
        
        # Extract file paths from enhanced query if available
        if "file_paths" in enhanced_query:
            file_paths = enhanced_query["file_paths"]
            has_explicit_file_reference = True
            self.logger.info(f"Found explicit file paths: {file_paths}")
        
        # If no explicit file paths but we have file references, try to find them
        elif "file_references" in enhanced_query:
            file_references = enhanced_query["file_references"]
            self.logger.info(f"Found file references: {file_references}")
            
            # Try to find these files in the report structure
            for reference in file_references:
                # Clean up the file reference (could be a path or just a filename)
                if os.path.sep in reference:
                    # Extract just the filename if it's a path
                    reference_file = os.path.basename(reference)
                else:
                    reference_file = reference
                    
                self.logger.info(f"Looking for file with name: {reference_file}")
                
                # First try exact filename match
                file_found = False
                for group_name, group_data in self.report_structure.items():
                    for file_name, file_data in group_data.get('files', {}).items():
                        # Check for exact file name match first
                        if file_name.lower() == reference_file.lower():
                            file_path = file_data.get('path')
                            if file_path:
                                self.logger.info(f"Found exact match: {file_path}")
                                file_paths.append(file_path)
                                has_explicit_file_reference = True
                                file_found = True
                
                # If no exact match, try partial match
                if not file_found:
                    for group_name, group_data in self.report_structure.items():
                        for file_name, file_data in group_data.get('files', {}).items():
                            # Check if filename contains reference
                            if reference_file.lower() in file_name.lower():
                                file_path = file_data.get('path')
                                if file_path:
                                    self.logger.info(f"Found partial match: {file_path}")
                                    file_paths.append(file_path)
                                    has_explicit_file_reference = True
                                    file_found = True
                
                # If still no match, log it
                if not file_found:
                    self.logger.warning(f"Could not find file matching: {reference_file}")
        
        # No explicit references, check if we should use previously mentioned files
        elif not file_paths and self.last_mentioned_files and any(word in original_query.lower() for word in ['it', 'this', 'that', 'the file']):
            file_paths = self.last_mentioned_files
            has_explicit_file_reference = True
            self.logger.info(f"Using previously mentioned files: {file_paths}")
        
        # If we have explicit file references, analyze those files directly
        if has_explicit_file_reference and file_paths:
            self.logger.info(f"Analyzing explicitly referenced files: {file_paths}")
            analysis_results = []
            
            for file_path in file_paths:
                # Determine file type from extension
                ext = os.path.splitext(file_path)[1].lower()
                file_name = os.path.basename(file_path)
                
                # Process file based on type
                if ext in ['.csv']:
                    result = summarize_csv(file_path, self.DEFAULT_MODEL, self.logger, self.memory)
                    analysis_results.append(result)
                    # Polish with completion agent
                    polished_result = completion_agent(result, f"Analyze the CSV file {file_name}", self.DEFAULT_MODEL, self.logger)
                    
                elif ext in ['.png', '.jpg', '.jpeg', '.gif']:
                    result = describe_image(file_path, self.DEFAULT_MODEL, self.logger, self.memory)
                    analysis_results.append(result)
                    # Polish with completion agent
                    polished_result = completion_agent(result, f"Describe the image {file_name}", self.DEFAULT_MODEL, self.logger)
                    
                elif ext in ['.md', '.txt']:
                    result = describe_markdown(file_path, self.DEFAULT_MODEL, self.logger, self.memory)
                    analysis_results.append(result)
                    # Polish with completion agent
                    polished_result = completion_agent(result, f"Summarize the file {file_name}", self.DEFAULT_MODEL, self.logger)
                    
                else:
                    self.logger.warning(f"Unsupported file type for direct analysis: {ext}")
                    continue
                
                # Store in memory - use the polished result
                query_info = {
                    "intent": "analyze",
                    "file_references": [file_name]
                }
                self.memory.store_interaction(
                    f"Analyze {file_name}",
                    polished_result if 'polished_result' in locals() else result,
                    query_info,
                    [file_path]
                )
                
                return polished_result

        
        # No files were found, return None to fall back to general query handling
        return None
    
    def _resolve_references(self, query, query_info):
        """Resolve references to previously mentioned entities"""
        if not self.last_mentioned_files and not self.last_mentioned_dataset:
            return
            
        # References to dataset
        if self.last_mentioned_dataset and any(word in query.lower() for word in ['it', 'this dataset', 'that dataset', 'these data']):
            query_info["target_dataset"] = self.last_mentioned_dataset
            query_info["dataset_terms"] = self.query_understanding.known_datasets.get(self.last_mentioned_dataset, [])
            self.logger.info(f"Resolved reference to dataset: {self.last_mentioned_dataset}")
            
        # References to files
        if self.last_mentioned_files and any(word in query.lower() for word in ['it', 'this file', 'that file', 'the file', 'those files']):
            query_info["file_references"] = [os.path.basename(f) for f in self.last_mentioned_files]
            self.logger.info(f"Resolved references to files: {query_info['file_references']}")


# Modify the run_chat function to use the UserQueryAgent
def run_chat(logger, REPORT_DIR, DEFAULT_MODEL, DEFAULT_EMBED_MODEL, report_structure=None, max_workers=100):
    # Initialize the memory system
    base_path = "/data/SWATGenXApp/codes/assistant"
    memory = MemorySystem(f"{base_path}/memory")
    query_understanding = QueryUnderstanding()
    response_generator = ResponseGenerator(llm_service=None)
    conversation_history = []
    
    logger.info("Indexing documents, please wait...")
    start_time = time.time()
    # Load only markdown files for faster indexing using parallel processing
    docs = load_all_documents(REPORT_DIR, logger, only_md_files=True, max_workers=max_workers)
    loading_time = time.time() - start_time
    logger.info(f"Document loading completed in {loading_time:.2f} seconds")
    
    logger.info(f"Loaded {len(docs)} documents, proceeding to indexing...")
    # Inspect document types to help diagnose issues
    doc_types = set(type(doc).__name__ for doc in docs)
    logger.info(f"Document types: {doc_types}")
    
    start_time = time.time()
    index = index_documents(docs, logger, max_workers=max_workers)
    indexing_time = time.time() - start_time
    logger.info(f"Document indexing completed in {indexing_time:.2f} seconds")
    
    # Check if indexing was successful
    if index is None:
        logger.error("Document indexing failed, query engine will not be available")
        print("Warning: Document indexing failed. The search functionality will be limited.")
        query_engine = None
    else:
        # Create the query engine
        query_engine = index.as_query_engine()
        logger.info("Document index created successfully")


    # If report_structure is not provided, try to load it from file
    if report_structure is None:
        report_structure = find_report_paths(REPORT_DIR, logger)
    else:
        # Use the provided report_structure, but pass it through find_report_paths for consistency
        report_structure = find_report_paths(REPORT_DIR, logger, report_structure=report_structure)

    ### print number of individual files that has been indexed
    logger.info(f"Number of individual files that has been indexed: {len(docs) if docs else 0}")
    
    # Preload memory with important files
    preloaded_files = preload_memory(memory, report_structure, logger, DEFAULT_MODEL)

    # Initialize the user query agent
    user_agent = UserQueryAgent(
        memory=memory,
        query_understanding=query_understanding,
        response_generator=response_generator,
        report_structure=report_structure,
        logger=logger,
        DEFAULT_MODEL=DEFAULT_MODEL
    )

    # Print instructions for the user
    logger.info("\nInteractive Report Assistant (type 'exit' to quit):")
    print("\nInteractive Report Assistant with Memory System (type 'exit' to quit):")
    print("You can ask about:")
    print("1. CSV files - e.g., 'Tell me about the CSV files'")
    print("2. Images - e.g., 'Show me the images in this report'")
    print("3. Markdown docs - e.g., 'What do the markdown files contain?'")
    print("4. General questions about the report data")
    print("5. Memory management - e.g., 'What's in your memory?', 'Clear your memory'")
    print("6. List files - e.g., 'list images', 'list csv files', 'list all files'")
    
    # Print summary of preloaded files
    print("\nI've preloaded information about:")
    for file_type, files in preloaded_files.items():
        if files:
            print(f"- {len(files)} {file_type} files")
            for i, file in enumerate(files[:3]):  # Show up to 3 examples
                print(f"  • {file}")
            if len(files) > 3:
                print(f"  • ...and {len(files)-3} more")
    print()
    
    while True:
        user_input = input("\n>> ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        # Process query through the UserQueryAgent
        response = user_agent.process_query(user_input, query_engine)
        
        # If the response is None, it means the command output was already printed
        # or the command wasn't handled and should be processed by the default logic
        if response is not None:
            print(response)
            


def find_report_paths(report_dir, logger, report_structure=None):
    """
    Get the report structure, either from the provided structure or from disk
    
    Args:
        report_dir: Directory containing the reports
        logger: Logger instance
        report_structure: Optional pre-loaded report structure
        
    Returns:
        dict: Report structure
    """
    # If a report structure was provided, use it
    if report_structure is not None:
        return report_structure
        
    # Otherwise read from disk
    logger.info(f"Reading report structure... {Path(report_dir) / 'report_structure.json'}")
    with open(Path(report_dir) / "report_structure.json") as f:
        return json.load(f)


if __name__ == "__main__":
    import os
    import multiprocessing
    
    logger = LoggerSetup(rewrite=True, verbose=True)

    # ==== CONFIGURATION ====
    REPORT_DIR = "/data/SWATGenXApp/Users/admin/Reports/20250412_172208"
    DEFAULT_MODEL = "Llama3.2-Vision:latest"
    DEFAULT_EMBED_MODEL = "nomic-embed-text"
    
    # Determine optimal number of workers for parallel processing
    num_cores = multiprocessing.cpu_count()
    max_workers = min(num_cores, 8)  # Use at most 8 workers to avoid overloading
    logger.info(f"Using {max_workers} workers for parallel processing (system has {num_cores} cores)")
    
    # ==== SETUP ====
    Settings.embed_model = OllamaEmbedding(model_name=DEFAULT_EMBED_MODEL)
    Settings.llm = Ollama(model=DEFAULT_MODEL)
    base_path = "/data/SWATGenXApp/codes/assistant"
    # Create memory directory if it doesn't exist
    os.makedirs(f"{base_path}/memory", exist_ok=True)
    os.makedirs(f"{base_path}/memory/interactions", exist_ok=True)
    os.makedirs(f"{base_path}/memory/files", exist_ok=True)
    os.makedirs(f"{base_path}/memory/geographic", exist_ok=True)

    # ==== Run the chat interface ====
    from discover_reports import discover_reports

    report_structure = discover_reports(base_dir=REPORT_DIR)

    # Debug output - show how many markdown files were found in each group
    logger.info("--- Report Structure Debug ---")
    md_file_count = 0
    for group_name, group_data in report_structure.items():
        md_files_in_group = [f for f in group_data.get('files', {}).keys() if f.lower().endswith('.md')]
        if md_files_in_group:
            logger.info(f"Group: {group_name}, MD Files: {len(md_files_in_group)}")
            for md_file in md_files_in_group:
                logger.info(f"  - {md_file}")
            md_file_count += len(md_files_in_group)
    logger.info(f"Total MD Files found: {md_file_count}")
    logger.info("-----------------------------")
    
    # Use the pre-loaded report structure instead of reading from disk again
    run_chat(logger, REPORT_DIR, DEFAULT_MODEL, DEFAULT_EMBED_MODEL, 
             report_structure=report_structure, max_workers=max_workers)
    
    # ==== END OF SCRIPT ====   
    