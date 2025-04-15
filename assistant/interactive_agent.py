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
from discover_reports import discover_reports
from Logger import LoggerSetup
from memory_system import MemorySystem
from query_understanding import QueryUnderstanding
from response_generator import ResponseGenerator
import re
import numpy as np

# ==== IMAGE ANALYSIS TOOL ====
def describe_image(path, DEFAULT_MODEL, logger, memory=None, prompt="Analyze the image and describe it in detail"):
    try:
        logger.info(f"Describing image... {path}")
        
        # Check if we already have this analysis in memory
        if memory:
            # Get file from memory system if it exists
            file_records = memory.get_related_files(f"image {os.path.basename(path)}", ["image"], file_type="image")
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
    except Exception as e:
        logger.error(f"Error analyzing image {path}: {str(e)}")
        return f"Error analyzing image {path}: {str(e)}"

# ==== FILE INDEXING ====
def load_all_documents(report_dir, logger):
    try:
        logger.info(f"Loading documents...")
        reader = SimpleDirectoryReader(report_dir, recursive=True)
        return reader.load_data()
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        return []

def index_documents(documents, logger):
    try:
        logger.info(f"Indexing documents...")
        return VectorStoreIndex.from_documents(documents)
    except Exception as e:
        logger.error(f"Error indexing documents: {str(e)}")
        return None

def find_report_paths(report_dir, logger):
    try:
        logger.info(f"Reading report structure... {Path(report_dir) / 'report_structure.json'}")
        with open(Path(report_dir) / "report_structure.json") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading report structure: {str(e)}")
        return {}

# ==== CSV ANALYSIS TOOL ====
def summarize_csv(path, DEFAULT_MODEL, logger, memory=None):
    try:
        logger.info(f"Summarizing CSV... {path}")
        
        # Check if we already have this analysis in memory
        if memory:
            # Get file from memory system if it exists
            file_records = memory.get_related_files(f"csv {os.path.basename(path)}", ["csv"], file_type="csv")
            for file in file_records:
                if file.get("original_path") == path:
                    logger.info(f"Using cached CSV summary for {path}")
                    return file.get("content", '')
        
        df = pd.read_csv(path)
        summary = df.describe(include='all').to_string()
        
        # Extract key statistics for better memory retention
        stats = {}
        try:
            for column in df.columns:
                if pd.api.types.is_numeric_dtype(df[column]):
                    stats[column] = {
                        'mean': float(df[column].mean()),
                        'min': float(df[column].min()),
                        'max': float(df[column].max()),
                        'std': float(df[column].std())
                    }
        except Exception as e:
            logger.error(f"Error extracting statistics: {str(e)}")
            stats = {"error": str(e)}
            
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
    except Exception as e:
        logger.error(f"Error reading CSV {path}: {str(e)}")
        return f"Error reading CSV {path}: {str(e)}"

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
    
    try:
        response = engine.query(prompt)
        result = str(response)
    except Exception as e:
        print(f"Error querying engine: {str(e)}")
        # Provide a fallback response
        simple_response = handle_basic_query(question)
        if simple_response:
            result = simple_response
        else:
            result = "I'm having trouble accessing my knowledge base at the moment. Could you try a different question or ask about a specific file?"
    
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
    try:
        logger.info(f"Describing markdown... {path}")
        
        # Check if we already have this analysis in memory
        if memory:
            # Get file from memory system if it exists
            file_records = memory.get_related_files(f"markdown {os.path.basename(path)}", ["markdown"], file_type="markdown")
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
    except Exception as e:
        logger.error(f"Error reading markdown {path}: {str(e)}")
        return f"Error reading markdown {path}: {str(e)}"

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
    
    # 1. Preload CSV files (statistics and data files)
    logger.info("Preloading CSV files...")
    csv_files = []
    
    # Find all CSV files in the report structure
    for report_name, report_data in report_structure.items():
        for group_name, group_data in report_data["groups"].items():
            if ".csv" in group_data["files"]:
                for file_name, file_data in group_data["files"][".csv"].items():
                    csv_files.append((file_name, file_data["path"], group_name))
    
    # Prioritize stats files as they contain the most useful summary information
    priority_csvs = [f for f in csv_files if "stats" in f[0].lower()]
    other_csvs = [f for f in csv_files if "stats" not in f[0].lower()]
    
    # Process priority CSVs first, then others if we have space
    for file_name, file_path, group_name in priority_csvs:
        try:
            logger.info(f"Preloading CSV: {file_name} from {group_name}")
            result = summarize_csv(file_path, DEFAULT_MODEL, logger, memory)
            preloaded_files['csv'].append(file_name)
            
            # Add a general note about this file to memory
            query_info = {
                "keywords": ["file", "csv", file_name, group_name],
                "intent": "search"
            }
            memory.store_interaction(
                f"What data is in {file_name}?",
                f"The file {file_name} in {group_name} contains statistical data: {result[:200]}...",
                query_info,
                [file_path]
            )
            
        except Exception as e:
            logger.error(f"Error preloading CSV {file_name}: {str(e)}")
    
    # 2. Preload important images (maps, visualizations)
    logger.info("Preloading important images...")
    image_files = []
    
    # Find all image files in the report structure
    for report_name, report_data in report_structure.items():
        for group_name, group_data in report_data["groups"].items():
            for ext in [".png", ".jpg", ".jpeg"]:
                if ext in group_data["files"]:
                    for file_name, file_data in group_data["files"][ext].items():
                        image_files.append((file_name, file_data["path"], group_name))
    
    # Prioritize maps and key visualizations
    priority_images = [
        f for f in image_files 
        if any(key in f[0].lower() for key in ["map", "spatial", "visualization", "comparison"])
    ]
    
    # Process priority images (limit to ~5 to avoid overloading)
    for file_name, file_path, group_name in priority_images[:5]:
        try:
            logger.info(f"Preloading image: {file_name} from {group_name}")
            result = describe_image(file_path, DEFAULT_MODEL, logger, memory)
            preloaded_files['image'].append(file_name)
            
            # Add a general note about this image to memory
            query_info = {
                "keywords": ["image", file_name, group_name],
                "intent": "search"
            }
            memory.store_interaction(
                f"What does the image {file_name} show?",
                f"The image {file_name} in {group_name} shows: {result[:200]}...",
                query_info,
                [file_path]
            )
            
        except Exception as e:
            logger.error(f"Error preloading image {file_name}: {str(e)}")
    
    # 3. Preload markdown files (reports, documentation)
    logger.info("Preloading markdown files...")
    md_files = []
    
    # Find all markdown files in the report structure
    for report_name, report_data in report_structure.items():
        for group_name, group_data in report_data["groups"].items():
            if ".md" in group_data["files"]:
                for file_name, file_data in group_data["files"][".md"].items():
                    md_files.append((file_name, file_data["path"], group_name))
    
    # Process markdown files
    for file_name, file_path, group_name in md_files:
        try:
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
            
        except Exception as e:
            logger.error(f"Error preloading markdown {file_name}: {str(e)}")
    
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
    extensions = type_to_extensions.get(file_type.lower())
    if extensions is None and file_type.lower() != 'all':
        print(f"Unknown file type: {file_type}. Available types: csv, image, markdown, html, text, climate, geographic, spatial, all")
        return
    
    # Collect files matching the criteria
    matching_files = []
    
    for report_name, report_data in report_structure.items():
        for group_name, group_data in report_data["groups"].items():
            for ext, files in group_data["files"].items():
                # Check if this extension matches our criteria
                if extensions is None or ext in extensions:
                    for file_name, file_data in files.items():
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
                                
                        matching_files.append({
                            'name': file_name,
                            'path': file_data['path'],
                            'group': group_name,
                            'extension': ext
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
    try:
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
    except Exception as e:
        # Log error and return 0 (no similarity)
        print(f"Error in cosine similarity calculation: {str(e)}")
        return 0

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
    try:
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
    except Exception as e:
        logger.error(f"Error in completion agent: {str(e)}")
        # Fall back to the original response
        return raw_response

# ==== MAIN INTERACTIVE AGENT ====
def run_chat(logger, REPORT_DIR, DEFAULT_MODEL, DEFAULT_EMBED_MODEL):
    # Initialize the memory system
    base_path = "/data/SWATGenXApp/codes/assistant"
    memory = MemorySystem(f"{base_path}/memory", logger=logger)
    query_understanding = QueryUnderstanding(logger=logger)
    response_generator = ResponseGenerator(logger=logger)
    conversation_history = []
    
    logger.info("Indexing documents, please wait...")
    docs = load_all_documents(REPORT_DIR, logger)
    index = index_documents(docs, logger)
    
    if index:
        query_engine = index.as_query_engine()
        logger.info("Document index created successfully")
    else:
        logger.warning("Failed to create document index. Basic functionality will still work.")
        query_engine = None

    report_structure = find_report_paths(REPORT_DIR, logger)

    ### print number of individual files that has been indexed
    logger.info(f"Number of individual files that has been indexed: {len(docs)}")
    
    # Preload memory with important files
    preloaded_files = preload_memory(memory, report_structure, logger, DEFAULT_MODEL)

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

        try:
            # Handle listing commands
            list_match = re.search(r"^list\s+(\w+)(?:\s+files)?$", user_input.lower())
            if list_match or user_input.lower() == "list all files":
                file_type = list_match.group(1) if list_match else "all"
                list_files(file_type, report_structure, memory, logger)
                continue
                
            # Special commands for memory management
            if user_input.lower() in ["what's in your memory?", "what is in your memory", "memory status", "show memory"]:
                # Get session info
                session_info = memory.get_session_info()
                interactions_count = len(session_info.get("interactions", []))
                active_files = session_info.get("active_files", [])
                
                memory_summary = (
                    f"Memory Status:\n"
                    f"- Session ID: {session_info.get('session_id', 'unknown')}\n"
                    f"- Stored interactions: {interactions_count}\n"
                    f"- Active files: {len(active_files)}\n"
                )
                
                if active_files:
                    memory_summary += "\nActive files:\n"
                    for i, file in enumerate(active_files[:10]):
                        memory_summary += f"  {i+1}. {os.path.basename(file)}\n"
                    if len(active_files) > 10:
                        memory_summary += f"  ...and {len(active_files) - 10} more files\n"
                
                print(memory_summary)
                
                # Store in memory
                query_info = {
                    "keywords": ["memory", "status"],
                    "intent": "help"
                }
                memory.store_interaction(user_input, memory_summary, query_info, [])
                continue
                
            if user_input.lower() in ["clear memory", "reset memory", "forget everything"]:
                memory.clear_memory(confirm=True)
                response = "Memory has been cleared. Starting fresh."
                print(response)
                
                # Start new memory session
                memory = MemorySystem(f"{base_path}/memory", logger)
                
                # Store in new memory
                query_info = {
                    "keywords": ["clear", "memory"],
                    "intent": "help"
                }
                memory.store_interaction(user_input, response, query_info, [])
                continue
                
            # For normal queries, first analyze the query
            query_info = query_understanding.analyze_query(user_input)
            logger.info(f"Query analysis: {query_info}")
                
            # Tool Routing based on user input and intent
            if query_info["intent"] == "search" and "csv" in user_input.lower():
                csv_analysis_results = []
                csv_files = []
                
                # First gather all relevant CSV files
                for r in report_structure.values():
                    for g in r["groups"].values():
                        if ".csv" in g["files"]:
                            for file_name, file_data in g["files"][".csv"].items():
                                csv_files.append((file_name, file_data["path"]))
                
                # If we have too many files, select most relevant or first few
                if len(csv_files) > 3:
                    # Try to find most relevant based on query terms
                    query_terms = user_input.lower().split()
                    relevant_files = []
                    
                    for file_name, file_path in csv_files:
                        if any(term in file_name.lower() for term in query_terms if len(term) > 2):
                            relevant_files.append((file_name, file_path))
                    
                    if relevant_files:
                        csv_files = relevant_files[:3]  # Limit to 3 most relevant
                    else:
                        csv_files = csv_files[:3]  # Just take first 3
                
                # Now analyze the selected files
                for file_name, file_path in csv_files:
                    result = summarize_csv(file_path, DEFAULT_MODEL, logger, memory)
                    csv_analysis_results.append(result)
                    # Apply completion agent
                    polished_result = completion_agent(result, f"Analyze the CSV file {file_name}", DEFAULT_MODEL, logger)
                    print(polished_result)  # Print to console for user
                
                # Store conversation in memory
                file_names = [name for name, _ in csv_files]
                file_paths = [path for _, path in csv_files]
                response = f"Analyzed the following CSV files: {', '.join(file_names)}"
                memory.store_interaction(user_input, response, query_info, file_paths)
                        
            elif query_info["intent"] == "search" and any(k in user_input.lower() for k in ["image", "plot", "map"]):
                image_analysis_results = []
                image_files = []
                
                # Check if user is asking about a specific image file by name
                specific_file_patterns = [
                    r'(.+\.(png|jpg|jpeg|gif))', 
                    r'image (.+)', 
                    r'(.+) image', 
                    r'show (.+)'
                ]
                specific_file = None
                
                for pattern in specific_file_patterns:
                    matches = re.findall(pattern, user_input.lower())
                    if matches:
                        specific_file = matches[0]
                        if isinstance(specific_file, tuple):  # For the case where the pattern includes file extension
                            specific_file = specific_file[0]
                        specific_file = specific_file.strip()
                        logger.info(f"Detected specific file request: {specific_file}")
                        break
                        
                # Gather relevant image files
                for r in report_structure.values():
                    for g in r["groups"].values():
                        for ext in [".png", ".jpg", ".jpeg", ".gif"]:
                            if ext in g["files"]:
                                for file_name, file_data in g["files"][ext].items():
                                    # If specific file was requested, only include exact matches
                                    if specific_file and specific_file not in file_name.lower():
                                        continue
                                    image_files.append((file_name, file_data["path"]))
                
                # If no images found and we were looking for a specific file, try memory
                if not image_files and specific_file:
                    # Try getting the file from memory
                    file_records = memory.get_related_files(user_input, query_info.get("keywords", []), "image")
                    
                    for file_record in file_records:
                        file_path = file_record.get("original_path", "")
                        file_name = file_record.get("file_name", "")
                        
                        if file_path and file_name:
                            # Check if the specific file name is in the file record
                            if specific_file in file_name.lower():
                                image_files.append((file_name, file_path))
                                logger.info(f"Found specific file in memory: {file_name}")
                                
                                # If we have content in memory, use it directly
                                if "content" in file_record:
                                    print(file_record["content"])
                                    
                                    # Store in memory
                                    response = f"Analyzed image file: {file_name}"
                                    memory.store_interaction(user_input, file_record["content"], query_info, [file_path])
                                    continue
                
                # Select relevant images if we didn't find a specific match
                if len(image_files) > 2 and not specific_file:
                    # Try to find most relevant based on query terms
                    query_terms = user_input.lower().split()
                    relevant_files = []
                    
                    for file_name, file_path in image_files:
                        if any(term in file_name.lower() for term in query_terms if len(term) > 2):
                            relevant_files.append((file_name, file_path))
                    
                    if relevant_files:
                        image_files = relevant_files[:2]  # Limit to 2 most relevant
                    else:
                        image_files = image_files[:2]  # Just take first 2
                
                # Analyze selected images
                for file_name, file_path in image_files:
                    result = describe_image(file_path, DEFAULT_MODEL, logger, memory)
                    image_analysis_results.append(result)
                    # Apply completion agent
                    polished_result = completion_agent(result, f"Describe the image {file_name}", DEFAULT_MODEL, logger)
                    print(polished_result)  # Print to console for user
                
                # Store conversation in memory
                file_names = [name for name, _ in image_files]
                file_paths = [path for _, path in image_files]
                
                if image_files:
                    response = f"Analyzed the following image files: {', '.join(file_names)}"
                    memory.store_interaction(user_input, response, query_info, file_paths)
                else:
                    response = f"I couldn't find any relevant image files matching your query."
                    print(response)
                    memory.store_interaction(user_input, response, query_info, [])
                    
            elif query_info["intent"] == "search" and ".md" in user_input.lower():
                md_files = []
                
                # Gather markdown files
                for r in report_structure.values():
                    for g in r["groups"].values():
                        if ".md" in g["files"]:
                            for file_name, file_data in g["files"][".md"].items():
                                md_files.append((file_name, file_data["path"]))
                
                # Analyze first markdown file or most relevant
                if md_files:
                    file_name, file_path = md_files[0]
                    result = describe_markdown(file_path, DEFAULT_MODEL, logger, memory)
                    # Apply completion agent
                    polished_result = completion_agent(result, f"Summarize the markdown file {file_name}", DEFAULT_MODEL, logger)
                    print(polished_result)  # Print to console for user
                    
                    # Store conversation in memory
                    response = f"Analyzed markdown file: {file_name}"
                    memory.store_interaction(user_input, response, query_info, [file_path])
                else:
                    response = "No markdown files found"
                    print(response)
                    memory.store_interaction(user_input, response, query_info, [])
                    
            else:
                # For general queries, use enhanced query to get better results
                enhanced_query = query_understanding.enhance_query(user_input, query_info, memory)
                
                # Check if this is a simple query that can be handled directly
                simple_response = handle_basic_query(user_input)
                if simple_response:
                    # Apply completion agent to simple responses for consistency
                    polished_response = completion_agent(simple_response, user_input, DEFAULT_MODEL, logger)
                    print(polished_response)
                    memory.store_interaction(user_input, polished_response, query_info, [])
                    continue
                
                # Check if this is a dataset inquiry (asking about a specific dataset)
                if query_info["intent"] == "dataset_inquiry" and "target_dataset" in enhanced_query:
                    target_dataset = enhanced_query["target_dataset"]
                    logger.info(f"Processing dataset inquiry for: {target_dataset}")
                    
                    # Get files specifically related to this dataset using the dataset name
                    dataset_files = memory.get_related_files(target_dataset, 
                                                           enhanced_query.get("dataset_terms", [target_dataset]), 
                                                           limit=5)
                    
                    # Verify these files actually belong to the target dataset
                    verified_dataset_files = []
                    for file in dataset_files:
                        file_path = file.get("original_path", "").lower()
                        # Check if file is in the correct dataset directory
                        if f"/{target_dataset}/" in file_path or f"\\{target_dataset}\\" in file_path:
                            verified_dataset_files.append(file)
                            logger.info(f"Verified {target_dataset} file: {file.get('file_name')}")
                        else:
                            logger.warning(f"Rejecting file not in {target_dataset} directory: {file_path}")
                    
                    if verified_dataset_files:
                        # Format a response about the dataset
                        dataset_paths = [file.get("original_path") for file in verified_dataset_files if "original_path" in file]
                        logger.info(f"Found {len(verified_dataset_files)} verified files for {target_dataset} dataset")
                        
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
                        polished_response = completion_agent(dataset_response, user_input, DEFAULT_MODEL, logger)
                        print(polished_response)
                        
                        # Store in memory
                        memory.store_interaction(user_input, polished_response, query_info, dataset_paths)
                        continue
                    else:
                        # No valid files found for this dataset
                        logger.warning(f"No valid files found for dataset: {target_dataset}")
                        
                        # Try to determine if the dataset exists in the report structure
                        dataset_exists = False
                        for r in report_structure.values():
                            for group_name in r["groups"].keys():
                                if target_dataset.lower() == group_name.lower():
                                    dataset_exists = True
                                    break
                        
                        if dataset_exists:
                            response = f"I can see that {target_dataset} data exists in the reports, but I haven't been able to properly index or access those files yet. You might want to try listing all files to see what's available."
                        else:
                            response = f"I couldn't find any files related to {target_dataset} in the current reports. This dataset might not be available or might be named differently. You can try 'list all files' to see what's available."
                        
                        # Polish with completion agent
                        polished_response = completion_agent(response, user_input, DEFAULT_MODEL, logger)
                        print(polished_response)
                        
                        # Store in memory
                        memory.store_interaction(user_input, polished_response, query_info, [])
                        continue
                
                # Check for explicit file references in the enhanced query
                has_explicit_file_reference = False
                file_paths = []
                
                # Extract file paths from enhanced query if available
                if "file_paths" in enhanced_query:
                    file_paths = enhanced_query["file_paths"]
                    has_explicit_file_reference = True
                    logger.info(f"Found explicit file references: {file_paths}")
                
                # If no explicit file paths but we have file references, try to find them
                elif "file_references" in enhanced_query:
                    file_references = enhanced_query["file_references"]
                    logger.info(f"Found file references: {file_references}")
                    
                    # Try to find these files in the report structure
                    for reference in file_references:
                        for r in report_structure.values():
                            for g in r["groups"].values():
                                for ext, files in g["files"].items():
                                    for file_name, file_data in files.items():
                                        # Check if filename matches reference
                                        if reference.lower() in file_name.lower():
                                            file_paths.append(file_data["path"])
                                            has_explicit_file_reference = True
                
                # If we have explicit file references, analyze those files directly
                if has_explicit_file_reference and file_paths:
                    logger.info(f"Analyzing explicitly referenced files: {file_paths}")
                    analysis_results = []
                    
                    for file_path in file_paths:
                        try:
                            # Determine file type from extension
                            ext = os.path.splitext(file_path)[1].lower()
                            file_name = os.path.basename(file_path)
                            
                            # Process file based on type
                            if ext in ['.csv']:
                                result = summarize_csv(file_path, DEFAULT_MODEL, logger, memory)
                                analysis_results.append(result)
                                # Polish with completion agent
                                polished_result = completion_agent(result, f"Analyze the CSV file {file_name}", DEFAULT_MODEL, logger)
                                print(polished_result)
                            elif ext in ['.png', '.jpg', '.jpeg', '.gif']:
                                result = describe_image(file_path, DEFAULT_MODEL, logger, memory)
                                analysis_results.append(result)
                                # Polish with completion agent
                                polished_result = completion_agent(result, f"Describe the image {file_name}", DEFAULT_MODEL, logger)
                                print(polished_result)
                            elif ext in ['.md', '.txt']:
                                result = describe_markdown(file_path, DEFAULT_MODEL, logger, memory)
                                analysis_results.append(result)
                                # Polish with completion agent
                                polished_result = completion_agent(result, f"Summarize the file {file_name}", DEFAULT_MODEL, logger)
                                print(polished_result)
                            else:
                                logger.warning(f"Unsupported file type for direct analysis: {ext}")
                                continue
                            
                            # Store in memory - use the polished result
                            memory.store_interaction(
                                f"Analyze {file_name}",
                                polished_result if 'polished_result' in locals() else result,
                                query_info,
                                [file_path]
                            )
                        except Exception as e:
                            logger.error(f"Error analyzing file {file_path}: {str(e)}")
                            print(f"I encountered an error while analyzing {os.path.basename(file_path)}: {str(e)}")
                    
                    # If we successfully analyzed files, we're done
                    if analysis_results:
                        continue
                
                # Get related files from memory (as backup if explicit references failed)
                related_files = memory.get_related_files(user_input, query_info.get("keywords", []))
                related_file_paths = [file.get("original_path") for file in related_files if "original_path" in file]
                
                # Get RAG response
                try:
                    rag_response, conversation_history = ask_query(
                        query_engine, 
                        user_input, 
                        conversation_history, 
                        memory, 
                        DEFAULT_MODEL
                    )
                    
                    # Generate a better response using our response generator
                    response_data = response_generator.generate_response(
                        user_input,
                        enhanced_query,
                        related_file_paths,
                        memory
                    )
                    
                    # Get the final answer
                    answer = response_data.get("answer", rag_response)
                    # Apply completion agent for final polishing
                    polished_answer = completion_agent(answer, user_input, DEFAULT_MODEL, logger)
                    print(polished_answer)
                    
                    # Store in memory - store the polished answer
                    memory.store_interaction(user_input, polished_answer, query_info, related_file_paths)
                    
                except Exception as e:
                    error_msg = f"Error processing query: {str(e)}"
                    logger.error(error_msg)
                    print("I encountered an error while processing your query. Please try a different question.")
                    
        except Exception as e:
            error_msg = f"Error in run_chat: {str(e)}"
            logger.error(error_msg)
            print("Sorry, I encountered an error. Please try a different question or approach.")
            
            # Try to still save the conversation even if there was an error
            try:
                query_info = {"keywords": user_input.lower().split(), "intent": "general"}
                memory.store_interaction(user_input, "Error: " + str(e), query_info, [])
            except:
                pass

if __name__ == "__main__":
    logger = LoggerSetup(rewrite=True, verbose=True, report_path="/data/SWATGenXApp/codes/assistant/logs")
    logger = logger.setup_logger("AI_AgentLogger")

    # ==== CONFIGURATION ====
    REPORT_DIR = "/data/SWATGenXApp/Users/admin/Reports/20250324_222749"
    DEFAULT_MODEL = "Llama3.2-Vision:latest"
    DEFAULT_EMBED_MODEL = "nomic-embed-text"
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
    discover_reports(base_dir=REPORT_DIR, logger=logger)
    run_chat(logger, REPORT_DIR, DEFAULT_MODEL, DEFAULT_EMBED_MODEL)