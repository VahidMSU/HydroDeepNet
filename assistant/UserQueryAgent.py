import re
import os
try:
    from .utils import summarize_csv, describe_image, describe_markdown
except ImportError:
    from utils import summarize_csv, describe_image, describe_markdown
import ollama



# Add new completion_agent function at the appropriate place after the imports
def completion_agent(raw_response, query, DEFAULT_MODEL, logger, memory=None, use_context=True):
    """
    Takes a raw response and generates a polished, complete response with full context access.
    This ensures consistent style and format for all system responses.
    
    Args:
        raw_response (str): The raw response from various analysis components
        query (str): The original user query
        DEFAULT_MODEL: Default model to use if not final completion
        logger: Logger instance
        memory: Optional memory system for context
        use_context: Whether to include memory context in the completion
        
    Returns:
        str: A polished final response
    """
    
    # Skip completion for very short responses or error messages
    if len(raw_response) < 25 or "error" in raw_response.lower():
        return raw_response
        
    logger.info("Using completion agent to polish response")
    
    # Get context from memory if available
    memory_context = ""
    if memory and use_context:
        # Get related interactions from memory using semantic search
        related_interactions = memory.get_related_interactions(query, [], limit=3, use_semantic=True)
        if related_interactions:
            memory_context = "Previous relevant interactions:\n"
            for interaction in related_interactions[:3]:
                if "query" in interaction and "response" in interaction:
                    memory_context += f"Q: {interaction['query']}\nA: {interaction['response']}\n\n"
    
    # Determine which model to use
    # For final completions, always use llama3:70b
    completion_model = "llama3:70b"
    logger.info(f"Using {completion_model} for final completion")
    
    # Create a prompt that instructs the model to polish the response
    prompt = f"""You are a helpful AI assistant specializing in data analysis and explanations.
Given the user query and a draft response, improve the response to be clear, direct, and helpful.
Maintain all factual information but improve clarity and readability.

User query: {query}
"""

    # Include memory context if available
    if memory_context:
        prompt += f"\nRelevant context from previous interactions:\n{memory_context}\n"

    prompt += f"""
Draft response: {raw_response}

Improved response:"""

    response = ollama.generate(
        model=completion_model,
        prompt=prompt,
    )
    
    improved_response = response.response.strip()
    
    # If the improved response is significantly shorter, stick with the original
    if len(improved_response) < len(raw_response) * 0.7 and len(raw_response) > 100:
        logger.warning("Completion agent produced significantly shorter response, using original")
        return raw_response
        
    return improved_response

def handle_basic_query(query, memory=None, report_structure=None):
    """
    Handle basic queries without needing the index
    
    Args:
        query: The user query
        memory: Optional memory system to check for available data
        report_structure: Optional report structure for data awareness
    """
    query_lower = query.lower()
    
    # For greeting patterns, provide more detailed information about available data
    is_greeting = query_lower in ["hi", "hello", "hey", "greetings"] or any(greeting in query_lower.split() for greeting in ["hi", "hello", "hey", "greetings"])
    
    if is_greeting:
        # Build a more informative response that includes what data is available
        response = "Hello! I'm your data analysis assistant. "
        
        # Include information about available datasets if we have report structure
        if report_structure:
            datasets = list(report_structure.keys())
            if datasets:
                if len(datasets) <= 5:
                    datasets_str = ", ".join(datasets)
                    response += f"I have access to the following datasets: {datasets_str}. "
                else:
                    datasets_sample = ", ".join(datasets[:5])
                    response += f"I have access to {len(datasets)} datasets including {datasets_sample}, and more. "
            
            # Count file types in the report structure
            file_counts = {"csv": 0, "image": 0, "markdown": 0, "other": 0}
            for group_data in report_structure.values():
                for file_name in group_data.get('files', {}):
                    if file_name.lower().endswith('.csv'):
                        file_counts["csv"] += 1
                    elif file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                        file_counts["image"] += 1
                    elif file_name.lower().endswith('.md'):
                        file_counts["markdown"] += 1
                    else:
                        file_counts["other"] += 1
            
            # Add file type info
            file_types = []
            if file_counts["csv"] > 0:
                file_types.append(f"{file_counts['csv']} CSV files")
            if file_counts["image"] > 0:
                file_types.append(f"{file_counts['image']} images")
            if file_counts["markdown"] > 0:
                file_types.append(f"{file_counts['markdown']} markdown documents")
            if file_counts["other"] > 0:
                file_types.append(f"{file_counts['other']} other files")
            
            if file_types:
                response += f"I can help analyze {', '.join(file_types)}. "
        
        # Add instructions for how to proceed
        response += "You can ask me to:\n"
        response += "1. List available files by type (try 'list images' or 'list csv files')\n"
        response += "2. Analyze specific files (try 'analyze [filename]')\n"
        response += "3. Ask questions about the data\n"
        response += "4. Get summaries of reports\n\n"
        response += "What would you like to know about?"
        
        return response
    
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


# ==== QUERY FUNCTION ====
def ask_query(engine, question, conversation_history, memory=None, DEFAULT_MODEL=None):
    # Get context from memory if available
    memory_context = ""
    if memory:
        # Get related interactions from memory using semantic search
        related_interactions = memory.get_related_interactions(question, [], limit=3, use_semantic=True)
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

class UserQueryAgent:
    """
    Agent that handles user input before it reaches the query understanding pipeline.
    Responsible for classifying queries, processing commands, and maintaining conversation context.
    """
    
    def __init__(self, memory, query_understanding, response_generator, report_structure, logger, DEFAULT_MODEL):
        """
        Initialize the UserQueryAgent with memory, query understanding, and response generator
        
        Args:
            memory: Memory system instance
            query_understanding: QueryUnderstanding instance
            response_generator: ResponseGenerator instance
            report_structure: Structure of available reports
            logger: Logger instance
            DEFAULT_MODEL: Default model to use for LLM operations
        """
        # Store components
        self.memory = memory
        
        # If query_understanding is not pre-initialized, create it with default model
        if query_understanding is None:
            from query_understanding import QueryUnderstanding
            self.query_understanding = QueryUnderstanding(default_model=DEFAULT_MODEL)
        else:
            # Use the provided instance but ensure default_model is set
            self.query_understanding = query_understanding
            if hasattr(self.query_understanding, 'default_model') and not self.query_understanding.default_model:
                self.query_understanding.default_model = DEFAULT_MODEL
        
        self.response_generator = response_generator
        self.report_structure = report_structure
        self.logger = logger
        self.DEFAULT_MODEL = DEFAULT_MODEL
        
        # Track last mentioned entities for context
        self.last_mentioned_files = []
        self.last_mentioned_dataset = None
        
        # Track conversation history
        self.conversation_history = []
        
        # Define command patterns
        self.command_patterns = {
            "list": r"(?i)^list\s+(.*?)(?:\s+files)?$",
            "analyze": r"(?i)^anal[yz]e\s+(.*?)$",
            "show": r"(?i)^show\s+(.*?)$",
            "what_is": r"(?i)^what(?:'s| is)\s+(.*?)$",
            "tell_about": r"(?i)^tell\s+(?:me\s+)?(?:about\s+)?(.*?)$",
            "help": r"(?i)^(?:help|assist)(?:\s+me)?(?:\s+with)?(?:\s+how\s+to)?(?:\s+)?(.*?)$"
        }
    

    # ==== FILE LISTING FUNCTION ====
    def list_files(self, file_type):
        """
        List files of a specific type or all files in the report structure.
        
        Args:
            file_type: Type of files to list ('csv', 'image', 'markdown', 'all')
            
        Returns:
            str: Formatted string with the list of files
        """
        self.logger.info(f"list_files function called with file_type: '{file_type}'")
        
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
            self.logger.warning(f"Unknown file type: {file_type}. Falling back to 'all'")
            file_type_message = f"I'm not familiar with '{file_type}' files. Showing all files instead."
            print(file_type_message)  # Still print for console users
            extensions = None
            file_type = 'all'
        else:
            file_type_message = ""
        
        # Collect files matching the criteria
        matching_files = []
        
        for group_name, group_data in self.report_structure.items():
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
            print(message)  # Still print for console users
            
            # Store in memory
            query_info = {
                "keywords": ["list", file_type, "files"],
                "intent": "search"
            }
            self.memory.store_interaction(f"list {file_type} files", message, query_info, [])
            return message
        
        # Group files by their group (folder)
        files_by_group = {}
        for file in matching_files:
            if file['group'] not in files_by_group:
                files_by_group[file['group']] = []
            files_by_group[file['group']].append(file)
        
        # Format output message
        output_lines = [f"Found {len(matching_files)} {file_type} files:"]
        if file_type_message:
            output_lines.insert(0, file_type_message)
            
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
            file_records = self.memory.get_related_files(os.path.basename(file['path']), [os.path.basename(file['path'])]) 
            if file_records:
                analyzed_files.append(file['name'])
        
        if analyzed_files:
            output_lines.append(f"\nI've already analyzed {len(analyzed_files)} of these files:")
            for i, name in enumerate(analyzed_files[:5]):
                output_lines.append(f"  • {name}")
            if len(analyzed_files) > 5:
                output_lines.append(f"  • ...and {len(analyzed_files) - 5} more")
            output_lines.append("\nYou can ask me about these files directly.")
        
        # Create the final output message
        output_message = "\n".join(output_lines)
        print(output_message)  # Still print for console users
        
        # Save to memory
        query_info = {
            "keywords": ["list", file_type, "files"],
            "intent": "search"
        }
        self.memory.store_interaction(
            f"list {file_type} files", 
            output_message,
            query_info,
            []
        )
        
        return output_message

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
        simple_response = handle_basic_query(user_input, self.memory, self.report_structure)
        if simple_response:
            # Apply completion agent to simple responses for consistency
            polished_response = completion_agent(
                simple_response, 
                user_input, 
                self.DEFAULT_MODEL, 
                self.logger,
                memory=self.memory,
                use_context=True
            )
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
        
        # Get related files using semantic search
        related_files = self.memory.get_related_files(user_input, query_info.get("keywords", []), use_semantic=True)
        related_file_paths = [file.get("original_path") for file in related_files if "original_path" in file]
        
        # Update last mentioned files if we found any
        if related_file_paths:
            self.last_mentioned_files = related_file_paths
            
        if query_engine:
            # Get related interactions using semantic search
            related_interactions = self.memory.get_related_interactions(user_input, query_info.get("keywords", []), use_semantic=True)
            
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
            
            # Polish with completion agent with full memory context
            polished_answer = completion_agent(
                answer, 
                user_input, 
                self.DEFAULT_MODEL, 
                self.logger,
                memory=self.memory,
                use_context=True
            )
            
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
            
            # Polish with completion agent with full memory context
            polished_answer = completion_agent(
                answer, 
                user_input, 
                self.DEFAULT_MODEL, 
                self.logger,
                memory=self.memory,
                use_context=True
            )
            
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
        
    def _extract_keywords(self, query):
        """Extract keywords from a query using the query understanding module"""
        return self.query_understanding._extract_keywords(query)
        
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
            response = self.list_files(file_type)
            
            # Store command in memory with appropriate intent
            query_info = {
                "keywords": ["list", file_type, "files"],
                "intent": "search"
            }
            self.memory.store_interaction(original_query, response, query_info, [])
            
            # Return the response from list_files
            return response
            
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
            for dataset_name in self.query_understanding.known_datasets:
                if isinstance(self.query_understanding.known_datasets, list):
                    # Handle the case where known_datasets is a list
                    dataset_terms = []
                    if dataset_name in lower_args:
                        query_info = {
                            "intent": "dataset_inquiry",
                            "target_dataset": dataset_name,
                            "keywords": lower_args.split()
                        }
                        enhanced_query = self.query_understanding.enhance_query(original_query, query_info, self.memory)
                        result = self._handle_dataset_inquiry(enhanced_query, original_query)
                        if result:
                            return result
                else:
                    # Handle the case where known_datasets is a dict
                    dataset_terms = self.query_understanding.known_datasets.get(dataset_name, [])[:3]
                    if dataset_name in lower_args or any(term in lower_args for term in dataset_terms):
                        query_info = {
                            "intent": "dataset_inquiry",
                            "target_dataset": dataset_name,
                            "keywords": lower_args.split()
                        }
                        enhanced_query = self.query_understanding.enhance_query(original_query, query_info, self.memory)
                        result = self._handle_dataset_inquiry(enhanced_query, original_query)
                        if result:
                            return result
            
            # Not a dataset, treat as general query with query engine
            query_engine = getattr(self, 'query_engine', None)
            try:
                # Process through regular query understanding pipeline
                query_info = self.query_understanding.analyze_query(original_query)
                enhanced_query = self.query_understanding.enhance_query(original_query, query_info, self.memory)
                
                # Get RAG response for general questions if query_engine is available
                if query_engine:
                    rag_response, history = ask_query(
                        query_engine, 
                        original_query, 
                        self.conversation_history, 
                        self.memory, 
                        self.DEFAULT_MODEL
                    )
                    self.conversation_history = history
                    
                    # Polish with completion agent
                    polished_answer = completion_agent(rag_response, original_query, self.DEFAULT_MODEL, self.logger)
                    
                    # Store in memory
                    self.memory.store_interaction(original_query, polished_answer, query_info, [])
                    
                    return polished_answer
            except Exception as e:
                self.logger.warning(f"Error using query engine: {str(e)}")
            
            # Fallback message if no query engine or it fails
            fallback_response = f"I don't have specific information about '{command_args}'. Try asking about available datasets or files, or use the 'list' command to see what's available."
            return completion_agent(fallback_response, original_query, self.DEFAULT_MODEL, self.logger, memory=self.memory, use_context=True)
        
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
            
            # Polish the response with completion agent for consistency
            polished_response = completion_agent(help_response, original_query, self.DEFAULT_MODEL, self.logger, memory=self.memory, use_context=True)
            return polished_response
        
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
        """
        Resolve references to previously mentioned entities
        
        Args:
            query: Raw query string
            query_info: Query analysis information
            
        Returns:
            Updated query_info with resolved references
        """
        self.logger.info("Resolving references in query")
        
        # Check for previously mentioned datasets
        if self.last_mentioned_dataset:
            if "dataset_references" not in query_info:
                query_info["dataset_references"] = []
            
            if self.last_mentioned_dataset not in query_info["dataset_references"]:
                query_info["dataset_references"].append(self.last_mentioned_dataset)
                self.logger.info(f"Resolved pronoun to dataset: {self.last_mentioned_dataset}")
        
        # Check for previously mentioned files
        if self.last_mentioned_files:
            if "file_references" not in query_info:
                query_info["file_references"] = []
            
            for file in self.last_mentioned_files:
                if file not in query_info["file_references"]:
                    query_info["file_references"].append(file)
                    self.logger.info(f"Resolved pronoun to file: {file}")
            
            # Also check if the query might be an "analyze" intent with a pronoun referring to a file
            if not query_info.get("intent") or query_info.get("intent") == "general_inquiry":
                # Check for analysis keywords
                analysis_terms = ["analyze", "show", "display", "explain", "describe", "plot", "graph"]
                if any(term in query.lower() for term in analysis_terms):
                    query_info["intent"] = "analyze"
                    self.logger.info("Updated intent to 'analyze' based on resolved file references")
        
        # For more complex reference resolution, look at conversation history
        conversation_history = self.memory.get_conversation_history(limit=3)
        
        # Use semantic search to find related past interactions
        related_interactions = self.memory.get_related_interactions(query, query_info.get("keywords", []), use_semantic=True)
        
        for interaction in related_interactions:
            # Check for dataset references in past interactions
            if "query_analysis" in interaction and "dataset_references" in interaction["query_analysis"]:
                past_datasets = interaction["query_analysis"]["dataset_references"]
                if past_datasets and "dataset_references" not in query_info:
                    query_info["dataset_references"] = past_datasets
                    self.logger.info(f"Resolved context to datasets: {past_datasets}")
                
            # Check for file references in past interactions
            if "query_analysis" in interaction and "file_references" in interaction["query_analysis"]:
                past_files = interaction["query_analysis"]["file_references"]
                if past_files and "file_references" not in query_info:
                    query_info["file_references"] = past_files
                    self.logger.info(f"Resolved context to files: {past_files}")
        
        return query_info

