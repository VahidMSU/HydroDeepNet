import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from llama_index.embeddings.ollama import OllamaEmbedding
from collections import defaultdict
import re
import time
import shutil
try:
    from .Logger import LoggerSetup
except ImportError:
    from Logger import LoggerSetup
import uuid
import copy

class MemorySystem:
    """
    System for storing, indexing, and retrieving files, datasets, and conversation history
    to provide context for the assistant.
    """
    
    def __init__(self, base_memory_path: str = None, logger=None, embedding_model_name: str = "nomic-embed-text"):
        """
        Initialize the memory system with directories for storing session data and files
        
        Args:
            base_memory_path: Base directory for storing memory files and session data
            logger: Optional logger instance
            embedding_model_name: Name of the embedding model to use
        """
        self.logger = LoggerSetup(rewrite=False, verbose=True)
        
        # Set default memory path if not provided
        if not base_memory_path:
            # Default to a directory in the current working directory
            base_memory_path = os.path.join(os.getcwd(), "assistant_memory")
        
        # Convert to Path object for easier path manipulation
        self.base_path = Path(base_memory_path)
        
        # Define subdirectories
        self.files_dir = self.base_path / "files"
        self.session_dir = self.base_path / "sessions"
        self.index_dir = self.base_path / "index"
        self.datasets_dir = self.base_path / "datasets"
        self.embeddings_dir = self.base_path / "embeddings"
        
        # Create required directories
        self._create_memory_directories()
        
        # Initialize memory components
        self.file_memory = {}  # Store file metadata
        self.search_index = {}  # Simple keyword search index (legacy)
        self.conversation_history = []  # Store conversation interactions
        
        # Initialize embedding model
        self.embedding_model_name = embedding_model_name
        self.embedding_model = OllamaEmbedding(model_name=embedding_model_name)
        
        # Vector store for embeddings
        self.document_embeddings = {}  # file_id -> embedding vector
        
        # Session data
        self.session_id = str(uuid.uuid4())
        self.session_data = {
            "id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "interactions": [],
            "user_preferences": {},
            "mentioned_files": set(),
            "mentioned_datasets": set(),
            "context": {}
        }
        
        # Load existing dataset information
        self.known_datasets = self._load_known_datasets()
        
        # Load existing embeddings if available
        self._load_embeddings()
        
        self.logger.info(f"Memory system initialized with base path: {self.base_path}")
    
    def _load_known_datasets(self) -> Dict[str, Dict]:
        """
        Load information about known datasets from disk
        
        Returns:
            dict: Dictionary of dataset information
        """
        datasets = {}
        
        # Create datasets directory if it doesn't exist
        os.makedirs(self.datasets_dir, exist_ok=True)
        
        # Check for dataset information files
        dataset_files = list(self.datasets_dir.glob("*.json"))
        
        for dataset_file in dataset_files:
            with open(dataset_file, 'r') as f:
                dataset_info = json.load(f)
                if 'name' in dataset_info:
                    datasets[dataset_info['name']] = dataset_info
                    self.logger.debug(f"Loaded dataset info: {dataset_info['name']}")
                else:
                    self.logger.warning(f"Dataset file {dataset_file} does not contain 'name' field")
        
        return datasets
    
    def _create_memory_directories(self):
        """Create necessary directories for the memory system"""
        # Create all required directories
        os.makedirs(self.files_dir, exist_ok=True)
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)
        os.makedirs(self.datasets_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        self.logger.debug("Memory directories created or verified")
    
    def _load_embeddings(self):
        """Load document embeddings from disk"""
        embedding_file = self.embeddings_dir / "document_embeddings.json"
        if embedding_file.exists():
            with open(embedding_file, 'r') as f:
                embedding_data = json.load(f)
                # Convert string representations of arrays back to numpy arrays
                for file_id, embedding_str in embedding_data.items():
                    self.document_embeddings[file_id] = np.array(json.loads(embedding_str))
            self.logger.info(f"Loaded {len(self.document_embeddings)} document embeddings")
        else:
            self.logger.info("No existing embeddings found")
    
    def _save_embeddings(self):
        """Save document embeddings to disk"""
        if not self.document_embeddings:
            return
            
        # Convert numpy arrays to serializable format (list)
        serializable_embeddings = {}
        for file_id, embedding in self.document_embeddings.items():
            # Store as JSON string to preserve numeric precision
            serializable_embeddings[file_id] = json.dumps(embedding.tolist())
            
        embedding_file = self.embeddings_dir / "document_embeddings.json"
        with open(embedding_file, 'w') as f:
            json.dump(serializable_embeddings, f)
            
        self.logger.debug(f"Saved {len(self.document_embeddings)} document embeddings")
    
    def add_file(self, 
                file_path: str, 
                content: Any = None, 
                file_type_or_metadata: Union[str, Dict] = None, 
                file_context: str = None) -> Dict:
        """
        Add a file to memory with its metadata and optional content
        
        Args:
            file_path: Path to the file
            content: File content (optional - will be indexed if provided)
            file_type_or_metadata: Either a string indicating the file type or a metadata dictionary
            file_context: Description or context for the file
            
        Returns:
            dict: File record with metadata
        """
        # Create a unique ID for this file
        file_id = str(uuid.uuid4())
        
        # Extract filename from path
        filename = os.path.basename(file_path)
        
        # Process the file_type_or_metadata parameter
        metadata = {}
        if isinstance(file_type_or_metadata, dict):
            metadata = file_type_or_metadata
        elif isinstance(file_type_or_metadata, str):
            metadata = {"file_type_tag": file_type_or_metadata}
        
        # Create basic file record
        file_record = {
            "id": file_id,
            "original_path": file_path,
            "filename": filename,
            "added_time": datetime.now().isoformat(),
            "metadata": metadata,
            "content": content,  # Store content directly in the record
            "context": file_context or "",
            "memory_path": str(self.files_dir / file_id)
        }
        
        # Add file extension information
        file_ext = os.path.splitext(filename)[1].lower()
        file_record["file_extension"] = file_ext
        
        # Determine file type based on extension
        if file_ext in ['.csv', '.xlsx', '.xls']:
            file_record["file_type"] = "tabular_data"
        elif file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.tif', '.tiff']:
            file_record["file_type"] = "image"
        elif file_ext in ['.txt', '.md', '.log']:
            file_record["file_type"] = "text"
        elif file_ext in ['.py', '.r', '.js', '.html', '.css', '.java', '.cpp']:
            file_record["file_type"] = "code"
        else:
            file_record["file_type"] = "other"
        
        # Store the file record
        self.file_memory[file_id] = file_record
        
        # Index the file content if provided
        if content is not None:
            # Generate semantic embedding for text content
            if isinstance(content, str) and len(content) > 0:
                self._generate_embedding(file_id, content)
            
            # Also perform legacy keyword indexing
            self._index_file(file_id, content, file_record)
        
        # Save the file record
        self._save_file_record(file_id, file_record)
        
        self.logger.info(f"Added file to memory: {filename} (ID: {file_id})")
        return file_record
    
    def _generate_embedding(self, file_id: str, content: str):
        """
        Generate and store embedding for text content
        
        Args:
            file_id: File ID to associate with the embedding
            content: Text content to embed
        """
        if not content or not isinstance(content, str):
            return
            
        self.logger.debug(f"Generating embedding for file {file_id}")
        
        # Create embedding
        try:
            # Truncate if needed (most embedding models have token limits)
            truncated_content = content[:10000]  # Reasonable limit
            embedding = self.embedding_model.get_text_embedding(truncated_content)
            
            # Store the embedding
            self.document_embeddings[file_id] = np.array(embedding)
            
            # Save embeddings to disk
            self._save_embeddings()
            
            self.logger.debug(f"Generated and saved embedding for file {file_id}")
        except Exception as e:
            self.logger.error(f"Error generating embedding for file {file_id}: {str(e)}")
        
    def _index_file(self, file_id: str, content: Any, file_record: Dict):
        """
        Index file content for searching (legacy keyword indexing)
        
        Args:
            file_id: ID of the file to index
            content: File content to index
            file_record: File record dictionary
        """
        # Get the content to index - either from the parameter or from the file_record
        content_to_index = content
        
        # Simple keyword extraction and indexing
        if isinstance(content_to_index, str):
            # For text content, extract keywords
            keywords = self._extract_keywords(content_to_index)
            
            # Add filename keywords
            filename_keywords = self._extract_keywords(file_record["filename"])
            keywords.extend(filename_keywords)
            
            # Add metadata keywords if available
            if file_record["metadata"]:
                for key, value in file_record["metadata"].items():
                    if isinstance(value, str):
                        metadata_keywords = self._extract_keywords(value)
                        keywords.extend(metadata_keywords)
            
            # Add context keywords if available
            if file_record["context"]:
                context_keywords = self._extract_keywords(file_record["context"])
                keywords.extend(context_keywords)
            
            # Update search index
            for keyword in set(keywords):  # Remove duplicates
                if keyword not in self.search_index:
                    self.search_index[keyword] = []
                
                if file_id not in self.search_index[keyword]:
                    self.search_index[keyword].append(file_id)
            
            # Save keywords in file record
            file_record["keywords"] = list(set(keywords))
            
            # Save the updated index
            self._save_search_index()
            
            self.logger.debug(f"Indexed file {file_id} with {len(keywords)} keywords")

    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text for indexing
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            list: List of extracted keywords
        """
        # Convert to string if not already
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove common stop words (simplified version)
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                     'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like', 'from'}
        
        # Tokenize and filter
        tokens = re.findall(r'\b\w+\b', text)
        keywords = [token for token in tokens if token not in stop_words and len(token) > 2]
        
        return keywords
    
    def search_files(self, query: str, limit: int = 5, use_semantic: bool = True) -> List[Dict]:
        """
        Search for files based on a query using semantic search by default
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            use_semantic: Whether to use semantic search (if False, falls back to keyword search)
            
        Returns:
            list: List of matching file records
        """
        if use_semantic and len(self.document_embeddings) > 0:
            return self.semantic_search(query, limit)
        else:
            return self.keyword_search(query, limit)
    
    def semantic_search(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search for files based on semantic similarity to the query
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            list: List of matching file records
        """
        if not self.document_embeddings:
            self.logger.warning("No document embeddings available, falling back to keyword search")
            return self.keyword_search(query, limit)
        
        self.logger.info(f"Performing semantic search for: '{query}'")
        
        # Generate query embedding
        query_embedding = np.array(self.embedding_model.get_text_embedding(query))
        
        # Calculate similarity scores
        similarity_scores = {}
        for file_id, doc_embedding in self.document_embeddings.items():
            # Compute cosine similarity
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarity_scores[file_id] = similarity
        
        # Sort by similarity score (descending)
        sorted_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get the file records
        results = []
        for file_id, score in sorted_scores[:limit]:
            if file_id in self.file_memory:
                file_record = copy.deepcopy(self.file_memory[file_id])
                file_record["match_score"] = float(score)  # Convert numpy float to Python float
                results.append(file_record)
        
        self.logger.info(f"Semantic search for '{query}' found {len(results)} results")
        return results
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            float: Cosine similarity score
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def keyword_search(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search for files based on keyword matching (legacy method)
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            list: List of matching file records
        """
        # Extract keywords from the query
        keywords = self._extract_keywords(query)
        
        # Find matching files
        matching_files = {}
        
        for keyword in keywords:
            if keyword in self.search_index:
                for file_id in self.search_index[keyword]:
                    if file_id not in matching_files:
                        matching_files[file_id] = 0
                    matching_files[file_id] += 1
        
        # Sort by match count (descending)
        sorted_matches = sorted(matching_files.items(), key=lambda x: x[1], reverse=True)
        
        # Get the file records
        results = []
        for file_id, score in sorted_matches[:limit]:
            if file_id in self.file_memory:
                file_record = copy.deepcopy(self.file_memory[file_id])
                file_record["match_score"] = score
                results.append(file_record)
        
        self.logger.info(f"Keyword search for '{query}' found {len(results)} results")
        return results
    
    def _save_file_record(self, file_id: str, file_record: Dict):
        """
        Save a file record to disk
        
        Args:
            file_id: ID of the file
            file_record: File record to save
        """
        # Create a copy to avoid modifying the original
        record_to_save = copy.deepcopy(file_record)
        
        # Convert sets to lists for JSON serialization
        for key, value in record_to_save.items():
            if isinstance(value, set):
                record_to_save[key] = list(value)
        
        # Save to disk
        record_path = self.index_dir / f"file_{file_id}.json"
        with open(record_path, 'w') as f:
            json.dump(record_to_save, f, indent=2)
            
        self.logger.debug(f"Saved file record for {file_id}")

    
    def _save_search_index(self):
        """Save the search index to disk"""
        # Save the index to disk
        index_path = self.index_dir / "search_index.json"
        with open(index_path, 'w') as f:
            json.dump(self.search_index, f, indent=2)
            
        self.logger.debug("Saved search index")
    
    def get_related_files(self, 
                         file_reference: str, 
                         keywords: List[str] = None, 
                         limit: int = 3,
                         use_semantic: bool = True) -> List[Dict]:
        """
        Get files related to a reference based on name or semantic similarity
        
        Args:
            file_reference: File name or reference
            keywords: Additional keywords to match
            limit: Maximum number of results to return
            use_semantic: Whether to use semantic search (if False, falls back to keyword matching)
            
        Returns:
            list: List of related file records
        """
        # If we have embeddings and semantic search is enabled, try semantic search first
        if use_semantic and self.document_embeddings:
            # Combine file reference and keywords into a query
            search_query = file_reference
            if keywords:
                search_query += " " + " ".join(keywords)
                
            semantic_results = self.semantic_search(search_query, limit=limit)
            
            # If we got good results semantically, return them
            if semantic_results:
                return semantic_results
        
        # Fall back to keyword matching
        matching_files = {}
        
        # First check for exact filename matches
        for file_id, record in self.file_memory.items():
            if file_reference.lower() in record["filename"].lower():
                matching_files[file_id] = 10  # High score for filename match
        
        # Then check for keyword matches
        if keywords:
            for keyword in keywords:
                if keyword in self.search_index:
                    for file_id in self.search_index[keyword]:
                        if file_id not in matching_files:
                            matching_files[file_id] = 0
                        matching_files[file_id] += 1
        
        # Sort by match score
        sorted_matches = sorted(matching_files.items(), key=lambda x: x[1], reverse=True)
        
        # Get the file records
        results = []
        for file_id, score in sorted_matches[:limit]:
            if file_id in self.file_memory:
                file_record = copy.deepcopy(self.file_memory[file_id])
                file_record["match_score"] = score
                results.append(file_record)
        
        return results

    
    def store_interaction(self, 
                         query: str, 
                         response: str, 
                         query_analysis: Dict, 
                         relevant_files: List[Dict] = None) -> str:
        """
        Store a conversation interaction with query, analysis, and response
        
        Args:
            query: User's query
            response: Assistant's response
            query_analysis: Analysis of the query
            relevant_files: Files relevant to this interaction
            
        Returns:
            str: ID of the stored interaction
        """
        # Create interaction record
        interaction_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        interaction = {
            "id": interaction_id,
            "timestamp": timestamp,
            "query": query,
            "query_analysis": query_analysis,
            "response": response,
            "relevant_files": relevant_files or []
        }
        
        # Add to conversation history
        self.conversation_history.append(interaction)
        
        # Add to session data
        self.session_data["interactions"].append(interaction)
        
        # Update mentioned files in session
        if relevant_files:
            for file in relevant_files:
                if "filename" in file:
                    self.session_data["mentioned_files"].add(file["filename"])
        
        # Update mentioned datasets
        if "dataset_references" in query_analysis:
            for dataset in query_analysis["dataset_references"]:
                self.session_data["mentioned_datasets"].add(dataset)
        
        # Save session data
        self._save_session_data()
        
        # Also generate embedding for the interaction
        interaction_text = f"Query: {query}\nResponse: {response}"
        self._generate_embedding(f"interaction_{interaction_id}", interaction_text)
        
        self.logger.info(f"Stored interaction {interaction_id}")
        return interaction_id
        

    
    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """
        Get recent conversation history
        
        Args:
            limit: Maximum number of interactions to return
            
        Returns:
            list: List of recent interactions
        """
        # Return the most recent interactions
        return self.conversation_history[-limit:] if self.conversation_history else []
    
    def get_related_interactions(self, 
                               query: str, 
                               keywords: List[str] = None, 
                               limit: int = 3,
                               use_semantic: bool = True) -> List[Dict]:
        """
        Get interactions related to a query based on semantic similarity
        
        Args:
            query: Current query
            keywords: Keywords to match (for fallback)
            limit: Maximum number of interactions to return
            use_semantic: Whether to use semantic search
            
        Returns:
            list: List of related interactions
        """
        if not self.conversation_history:
            return []
        
        # If semantic search is enabled and we have embeddings for interactions, use them
        if use_semantic and any(k.startswith("interaction_") for k in self.document_embeddings.keys()):
            # Generate embedding for the query
            query_embedding = np.array(self.embedding_model.get_text_embedding(query))
            
            # Calculate similarity with all interaction embeddings
            similarity_scores = {}
            for file_id, embedding in self.document_embeddings.items():
                if file_id.startswith("interaction_"):
                    interaction_id = file_id.replace("interaction_", "")
                    similarity = self._cosine_similarity(query_embedding, embedding)
                    similarity_scores[interaction_id] = similarity
            
            # Find matching interactions
            matching_interactions = {}
            for i, interaction in enumerate(self.conversation_history):
                if interaction["id"] in similarity_scores:
                    matching_interactions[i] = similarity_scores[interaction["id"]]
            
            # Sort by score (descending)
            sorted_matches = sorted(matching_interactions.items(), key=lambda x: x[1], reverse=True)
            
            # Get interactions
            results = []
            for idx, score in sorted_matches[:limit]:
                interaction = copy.deepcopy(self.conversation_history[idx])
                interaction["match_score"] = float(score)  # Convert numpy float to Python float
                results.append(interaction)
                
            if results:
                self.logger.info(f"Found {len(results)} semantically related interactions")
                return results
        
        # Fall back to keyword-based matching
        matching_interactions = {}
        
        # Extract query keywords if not provided
        if not keywords:
            keywords = self._extract_keywords(query)
        
        # Score interactions based on keyword matches
        for i, interaction in enumerate(self.conversation_history):
            score = 0
            
            # Check query text
            interaction_query = interaction.get("query", "")
            for keyword in keywords:
                if keyword in interaction_query.lower():
                    score += 1
            
            # Check query analysis keywords
            if "query_analysis" in interaction and "keywords" in interaction["query_analysis"]:
                analysis_keywords = interaction["query_analysis"]["keywords"]
                common_keywords = set(keywords).intersection(set(analysis_keywords))
                score += len(common_keywords) * 2  # Higher weight for analysis keywords
            
            # Add recency bonus (more recent = higher score)
            recency_bonus = (i + 1) / len(self.conversation_history)
            score += recency_bonus
            
            if score > 0:
                matching_interactions[i] = score
        
        # Sort by score (descending)
        sorted_matches = sorted(matching_interactions.items(), key=lambda x: x[1], reverse=True)
        
        # Get the interactions
        results = []
        for idx, score in sorted_matches[:limit]:
            interaction = copy.deepcopy(self.conversation_history[idx])
            interaction["match_score"] = score
            results.append(interaction)
        
        self.logger.info(f"Found {len(results)} keyword-matched related interactions")
        return results

    
    def _save_session_data(self):
        """Save current session data to disk"""
        # Create a copy to avoid modifying the original
        session_to_save = copy.deepcopy(self.session_data)
        
        # Convert sets to lists for JSON serialization
        for key, value in session_to_save.items():
            if isinstance(value, set):
                session_to_save[key] = list(value)
        
        # Save to disk
        os.makedirs(self.session_dir, exist_ok=True)
        session_path = self.session_dir / f"session_{self.session_id}.json"
        with open(session_path, 'w') as f:
            json.dump(session_to_save, f, indent=2)
            
        self.logger.debug(f"Saved session data for {self.session_id}")
        

    def update_dataset_info(self, dataset_name: str, dataset_info: Dict):
        """
        Update information about a dataset
        
        Args:
            dataset_name: Name of the dataset
            dataset_info: Information about the dataset
        """
        # Update in memory
        self.known_datasets[dataset_name] = dataset_info
        
        # Save to disk
        dataset_path = self.datasets_dir / f"{dataset_name}.json"
        with open(dataset_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
            
        self.logger.info(f"Updated dataset info for {dataset_name}")

    
    def get_dataset_info(self, dataset_name: str) -> Optional[Dict]:
        """
        Get information about a specific dataset
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            dict: Dataset information or None if not found
        """
        return self.known_datasets.get(dataset_name)
    
    def get_all_dataset_info(self) -> Dict[str, Dict]:
        """
        Get information about all known datasets
        
        Returns:
            dict: Dictionary of dataset information
        """
        return self.known_datasets
    
    def clear_memory(self, clear_files: bool = False, clear_embeddings: bool = False):
        """
        Clear memory components
        
        Args:
            clear_files: Whether to also remove file records
            clear_embeddings: Whether to clear document embeddings
        """
        # Clear conversation history
        self.conversation_history = []
        
        # Clear session data
        self.session_data["interactions"] = []
        self.session_data["mentioned_files"] = set()
        self.session_data["mentioned_datasets"] = set()
        
        # Save empty session
        self._save_session_data()
        
        if clear_embeddings:
            # Clear embeddings
            self.document_embeddings = {}
            # Remove embedding file
            embedding_file = self.embeddings_dir / "document_embeddings.json"
            if embedding_file.exists():
                os.remove(embedding_file)
            self.logger.info("Cleared document embeddings")
        
        if clear_files:
            # Clear file memory and search index
            self.file_memory = {}
            self.search_index = {}
            
            # Save empty search index
            self._save_search_index()
            
            # Remove file records from disk
            for file_record in list(self.index_dir.glob("file_*.json")):
                os.remove(file_record)
            
            self.logger.info("Cleared memory including file records")
        else:
            self.logger.info("Cleared conversation memory (kept file records)")

    
    def load_session(self, session_id: str) -> bool:
        """
        Load a previous session
        
        Args:
            session_id: ID of the session to load
            
        Returns:
            bool: Success or failure
        """
        session_path = self.session_dir / f"session_{session_id}.json"
        
        if not os.path.exists(session_path):
            self.logger.warning(f"Session {session_id} not found")
            return False
            
        with open(session_path, 'r') as f:
            session_data = json.load(f)
        
        # Update session data
        self.session_id = session_id
        self.session_data = session_data
        
        # Convert lists back to sets
        if "mentioned_files" in self.session_data:
            self.session_data["mentioned_files"] = set(self.session_data["mentioned_files"])
        
        if "mentioned_datasets" in self.session_data:
            self.session_data["mentioned_datasets"] = set(self.session_data["mentioned_datasets"])
        
        # Load conversation history
        self.conversation_history = self.session_data.get("interactions", [])
        
        self.logger.info(f"Loaded session {session_id}")
        return True

    
    def get_user_preferences(self) -> Dict:
        """
        Get user preferences from the current session
        
        Returns:
            dict: User preferences
        """
        return self.session_data.get("user_preferences", {})
    
    def update_user_preferences(self, preferences: Dict):
        """
        Update user preferences in the current session
        
        Args:
            preferences: Dictionary of user preferences
        """
        # Update preferences
        if "user_preferences" not in self.session_data:
            self.session_data["user_preferences"] = {}
            
        self.session_data["user_preferences"].update(preferences)
        
        # Save session data
        self._save_session_data()
        
        self.logger.info("Updated user preferences")

    
    def add_to_context(self, key: str, value: Any):
        """
        Add information to the session context
        
        Args:
            key: Context key
            value: Context value
        """
        # Update context
        self.session_data["context"][key] = value
        
        # Save session data
        self._save_session_data()
        
        self.logger.debug(f"Added to context: {key}")

    
    def get_context(self, key: str = None) -> Any:
        """
        Get information from the session context
        
        Args:
            key: Context key (if None, return all context)
            
        Returns:
            Context value or dictionary of all context values
        """
        context = self.session_data.get("context", {})
        
        if key is not None:
            return context.get(key)
        
        return context
    
    def get_mentioned_datasets(self) -> List[str]:
        """
        Get datasets mentioned in this session
        
        Returns:
            list: List of mentioned dataset names
        """
        return list(self.session_data.get("mentioned_datasets", set()))
    
    def get_mentioned_files(self) -> List[str]:
        """
        Get files mentioned in this session
        
        Returns:
            list: List of mentioned filenames
        """
        return list(self.session_data.get("mentioned_files", set())) 