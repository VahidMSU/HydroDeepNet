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


class MemorySystem:
    """
    Component responsible for storing and retrieving user interactions
    and managing file information in the system's memory.
    """
    
    def __init__(self, memory_dir: str = "/data/SWATGenXApp/codes/assistant/memory", logger=None):
        """
        Initialize the memory system
        
        Args:
            memory_dir (str): Path to directory for storing memory
        """
        self.logger = logger
        self.logger.info(f"Initializing Memory System with directory: {memory_dir}")
        
        self.memory_dir = Path(memory_dir)
        self.interactions_dir = self.memory_dir / "interactions"
        self.files_dir = self.memory_dir / "files"
        self.geographic_dir = self.memory_dir / "geographic"
        
        # Create memory directories if they don't exist
        self._create_directories()
        
        # Session ID based on timestamp
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.interaction_count = 0
        self.logger.info(f"Created new session with ID: {self.session_id}")
        
    def _create_directories(self):
        """Create necessary directories for memory storage"""
        try:
            for directory in [self.memory_dir, self.interactions_dir, 
                             self.files_dir, self.geographic_dir]:
                directory.mkdir(exist_ok=True, parents=True)
                
            self.logger.debug(f"Memory directories created/verified")
        except Exception as e:
            self.logger.error(f"Error creating memory directories: {str(e)}", exc_info=True)
            raise
            
    def store_interaction(self, query: str, response: str, 
                         query_info: Dict[str, Any], 
                         files_accessed: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Store a user interaction in memory
        
        Args:
            query (str): User's query
            response (str): System's response
            query_info (dict): Information about the query
            files_accessed (list, optional): List of files accessed
            
        Returns:
            dict: Stored interaction data
        """
        self.interaction_count += 1
        
        # Create interaction data
        interaction = {
            "session_id": self.session_id,
            "interaction_id": f"{self.session_id}_{self.interaction_count}",
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "query_info": query_info,
            "files_accessed": files_accessed or []
        }
        
        try:
            # Save to session file
            interaction_path = self.interactions_dir / f"{self.session_id}_{self.interaction_count}.json"
            with open(interaction_path, 'w', encoding='utf-8') as f:
                json.dump(interaction, f, indent=2)
                
            self.logger.debug(f"Stored interaction {self.interaction_count} to {interaction_path}")
            return interaction
            
        except Exception as e:
            self.logger.error(f"Error storing interaction: {str(e)}", exc_info=True)
            return interaction
    
    def add_file(self, file_path: Union[str, Path], content: str, 
                file_type: Optional[str] = None, metadata: Optional[Dict] = None) -> bool:
        """
        Add a file to memory for future reference
        
        Args:
            file_path (str or Path): Path to the file
            content (str): File content
            file_type (str, optional): Type of file
            metadata (dict, optional): Additional metadata
            
        Returns:
            bool: Success status
        """
        try:
            # Convert to Path object if string
            if isinstance(file_path, str):
                file_path = Path(file_path)
                
            # Get file type from extension if not provided
            if not file_type:
                file_type = file_path.suffix.lstrip('.')
                
            # Create file record
            file_record = {
                "original_path": str(file_path),
                "file_name": file_path.name,
                "file_type": file_type,
                "content": content,
                "added_timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            # Calculate unique ID for file (based on name and timestamp)
            file_id = f"{file_path.stem}_{int(time.time())}"
            
            # Save to files directory
            file_record_path = self.files_dir / f"{file_id}.json"
            with open(file_record_path, 'w', encoding='utf-8') as f:
                json.dump(file_record, f, indent=2)
                
            self.logger.info(f"Added file {file_path.name} to memory with ID {file_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding file to memory: {str(e)}", exc_info=True)
            return False
    
    def get_related_interactions(self, query: str, keywords: List[str], 
                                limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve interactions related to the current query
        
        Args:
            query (str): Current query
            keywords (list): Keywords from the query
            limit (int): Maximum number of interactions to return
            
        Returns:
            list: Related interactions
        """
        try:
            # Get all interaction files
            interaction_files = list(self.interactions_dir.glob("*.json"))
            
            if not interaction_files:
                return []
                
            # For each interaction, compute relevance score
            scored_interactions = []
            
            for file_path in interaction_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        interaction = json.load(f)
                        
                    # Score based on keyword overlap
                    score = 0
                    if "query_info" in interaction and "keywords" in interaction["query_info"]:
                        interaction_keywords = interaction["query_info"]["keywords"]
                        # Count overlapping keywords
                        for keyword in keywords:
                            if keyword in interaction_keywords:
                                score += 1
                                
                    # Add interaction with score
                    scored_interactions.append((score, interaction))
                    
                except Exception as e:
                    self.logger.warning(f"Error processing interaction file {file_path}: {str(e)}")
                    continue
            
            # Sort by score (descending) and return top results
            scored_interactions.sort(key=lambda x: x[0], reverse=True)
            return [interaction for score, interaction in scored_interactions[:limit]]
            
        except Exception as e:
            self.logger.error(f"Error retrieving related interactions: {str(e)}", exc_info=True)
            return []
    
    def get_related_files(self, query: str, keywords: List[str], 
                         file_type: Optional[str] = None,
                         limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve files related to the current query
        
        Args:
            query (str): Current query
            keywords (list): Keywords from the query
            file_type (str, optional): Filter by file type
            limit (int): Maximum number of files to return
            
        Returns:
            list: Related file records
        """
        try:
            # Get all file records
            file_records = []
            
            for file_path in self.files_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_record = json.load(f)
                        
                    # Filter by file type if specified
                    if file_type and file_record.get("file_type") != file_type:
                        continue
                        
                    file_records.append(file_record)
                except Exception as e:
                    self.logger.warning(f"Error reading file record {file_path}: {str(e)}")
                    continue
            
            if not file_records:
                return []
            
            # Extract topic keywords from the query
            query_lower = query.lower()
            
            # List of specific data products or datasets that might be queried
            known_datasets = {
                "cdl": ["cdl", "cropland", "crop", "agriculture", "land use", "land cover", "crop data layer", "usda", "crop type"],
                "climate_change": ["climate change", "climate", "global warming", "temperature change", "precipitation change", "future climate", "climate projections", "climate scenarios"],
                "gov_units": ["gov units", "governmental units", "administrative boundaries", "counties", "states", "boundaries", "administrative regions", "political boundaries"],
                "groundwater": ["groundwater", "aquifer", "well", "water table", "ground water", "subsurface water", "water level", "water depth", "hydrology"],
                "gssurgo": ["gssurgo", "soil", "sand", "clay", "silt", "organic matter", "bulk density", "soil properties", "soil survey", "usda soil"],
                "modis": ["modis", "evi", "ndvi", "lai", "vegetation", "mod13q1", "mod16a2", "mod15a2h", "satellite", "remote sensing", "vegetation index", "leaf area", "evapotranspiration", "et"],
                "nsrdb": ["nsrdb", "solar", "radiation", "solar power", "renewable energy", "irradiance", "pv", "photovoltaic", "solar potential", "insolation", "solar resource"],
                "prism": ["prism", "climate", "temperature", "precipitation", "rainfall", "weather", "drought", "historical climate", "climate data", "meteorology"],
                "snowdas": ["snow", "snowdas", "swe", "snow water equivalent", "snow cover", "snow depth", "snowpack", "snow accumulation", "snow melt"]
            }
            
            # Check if the query is asking about a specific dataset
            target_dataset = None
            for dataset, terms in known_datasets.items():
                if dataset in query_lower or any(term in query_lower for term in terms):
                    target_dataset = dataset
                    self.logger.info(f"Query appears to be about the '{dataset}' dataset")
                    break
            
            # If we identified a specific dataset, prioritize files from that dataset
            if target_dataset:
                dataset_matches = []
                
                for file_record in file_records:
                    file_name = file_record.get("file_name", "").lower()
                    file_path = file_record.get("original_path", "").lower()
                    file_content = file_record.get("content", "").lower()
                    
                    # First verify file is actually in the correct dataset directory
                    # This is the most important check - file must be in the target dataset directory
                    is_in_dataset_dir = False
                    dataset_dir_patterns = [
                        f"/{target_dataset}/", 
                        f"\\{target_dataset}\\",
                        f"/{target_dataset}_",
                        f"\\{target_dataset}_"
                    ]
                    
                    for pattern in dataset_dir_patterns:
                        if pattern in file_path:
                            is_in_dataset_dir = True
                            break
                    
                    # Calculate a dataset relevance score
                    score = 0
                    
                    # Give high score for being in the right directory
                    if is_in_dataset_dir:
                        score += 50  # This should outweigh all other factors
                    
                    # Check filename and path
                    if target_dataset in file_name:
                        score += 10
                    if target_dataset in file_path:
                        score += 8
                    
                    # Check related terms in the dataset's vocabulary
                    for term in known_datasets[target_dataset]:
                        if term in file_name:
                            score += 5
                        if term in file_path:
                            score += 3
                        if term in file_content:
                            score += 1
                    
                    if score > 0:
                        dataset_matches.append((score, file_record))
                        self.logger.debug(f"Dataset match for '{target_dataset}': {file_name} (score: {score})")
                
                # If we found dataset-specific matches, return those
                if dataset_matches:
                    dataset_matches.sort(key=lambda x: x[0], reverse=True)
                    self.logger.info(f"Found {len(dataset_matches)} matches for '{target_dataset}' dataset")
                    return [file for score, file in dataset_matches[:limit]]
            
            # Check for filename references in the query
            # First pass: Look for exact filename references with extensions
            extensions = ['.csv', '.png', '.jpg', '.jpeg', '.md', '.txt']
            exact_filename_matches = []
            
            # Look for filenames with extensions in the query
            for ext in extensions:
                pattern = rf'[\w\-_\s]+{ext}'
                matches = re.findall(pattern, query_lower)
                for match in matches:
                    match = match.strip()
                    # Look for matching files
                    for file_record in file_records:
                        file_name = file_record.get("file_name", "").lower()
                        if match in file_name:
                            exact_filename_matches.append(file_record)
                            break
            
            # Also check for exact and partial filename matches
            for file_record in file_records:
                file_name = file_record.get("file_name", "").lower()
                if file_name and file_name in query_lower:
                    # Add if not already included
                    if file_record not in exact_filename_matches:
                        exact_filename_matches.append(file_record)
                        
                # Check filename without extension
                if file_name:
                    filename_base = os.path.splitext(file_name)[0]
                    if filename_base and len(filename_base) > 3:
                        if filename_base in query_lower:
                            if file_record not in exact_filename_matches:
                                exact_filename_matches.append(file_record)
            
            # If we found exact filename matches, prioritize these
            if exact_filename_matches:
                self.logger.info(f"Found {len(exact_filename_matches)} filename matches for '{query}'")
                return exact_filename_matches[:limit]
                
            # If exact filename matching didn't work, try fuzzy matching for filenames
            fuzzy_matches = []
            query_words = query_lower.split()
            
            for file_record in file_records:
                file_name = file_record.get("file_name", "").lower()
                if not file_name:
                    continue
                    
                # Calculate fuzzy score based on word overlap
                filename_words = re.findall(r'\w+', file_name)
                overlap = set(query_words) & set(filename_words)
                
                # If at least one word overlaps, consider it a match
                if overlap:
                    score = len(overlap) * 5  # Prioritize filename matches
                    fuzzy_matches.append((score, file_record))
                    continue
                    
            # If we found fuzzy matches for filenames, return these first
            if fuzzy_matches:
                fuzzy_matches.sort(key=lambda x: x[0], reverse=True)
                return [file for score, file in fuzzy_matches[:limit]]
                
            # If filename matching didn't yield results, score based on content match
            scored_files = []
            
            for file_record in file_records:
                score = 0
                content = file_record.get("content", "").lower()
                file_name = file_record.get("file_name", "").lower()
                file_path = file_record.get("original_path", "").lower()
                
                # Check for keyword presence in content
                for keyword in keywords:
                    keyword_lower = keyword.lower()
                    # Give higher score for filename matches
                    if keyword_lower in file_name:
                        score += 5
                    if keyword_lower in file_path:
                        score += 3
                    if keyword_lower in content:
                        score += 1
                
                # Check for query terms in content
                for term in query_words:
                    if len(term) > 2 and term not in ["the", "and", "for", "in", "on", "of", "to", "a", "an"]:
                        if term in file_name:
                            score += 3
                        if term in file_path:
                            score += 2
                        if term in content:
                            score += 1
                        
                if score > 0:
                    scored_files.append((score, file_record))
                
            # Sort by score (descending) and return top results
            scored_files.sort(key=lambda x: x[0], reverse=True)
            return [file for score, file in scored_files[:limit]]
            
        except Exception as e:
            self.logger.error(f"Error retrieving related files: {str(e)}", exc_info=True)
            return []
    
    def get_geographic_info(self, entity: str) -> Optional[Dict[str, Any]]:
        """
        Get geographic information for an entity if available
        
        Args:
            entity (str): Geographic entity name
            
        Returns:
            dict or None: Geographic information if available
        """
        try:
            # Check if we have information about this entity
            entity_file = self.geographic_dir / f"{entity.lower().replace(' ', '_')}.json"
            
            if entity_file.exists():
                with open(entity_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving geographic info for {entity}: {str(e)}")
            return None
    
    def store_geographic_info(self, entity: str, info: Dict[str, Any]) -> bool:
        """
        Store geographic information about an entity
        
        Args:
            entity (str): Geographic entity name
            info (dict): Geographic information
            
        Returns:
            bool: Success status
        """
        try:
            # Prepare entity file name (lowercase with underscores)
            entity_filename = entity.lower().replace(' ', '_') + '.json'
            entity_path = self.geographic_dir / entity_filename
            
            # Add timestamp
            info['stored_timestamp'] = datetime.now().isoformat()
            
            # Save to file
            with open(entity_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=2)
                
            self.logger.info(f"Stored geographic information for entity: {entity}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing geographic info for {entity}: {str(e)}")
            return False
    
    def get_related_geographic_files(self, entities: List[str], 
                                    limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get files related to geographic entities
        
        Args:
            entities (list): Geographic entity names
            limit (int): Maximum number of files to return
            
        Returns:
            list: Related file records
        """
        try:
            # Search for files mentioning the geographic entities
            file_records = []
            entity_keywords = [entity.lower() for entity in entities]
            
            for file_path in self.files_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_record = json.load(f)
                        
                    # Check if file content mentions any of the entities
                    content = file_record.get("content", "").lower()
                    
                    for entity in entity_keywords:
                        if entity in content:
                            file_records.append(file_record)
                            break
                            
                except Exception as e:
                    self.logger.warning(f"Error reading file record {file_path}: {str(e)}")
                    continue
                    
            return file_records[:limit]
            
        except Exception as e:
            self.logger.error(f"Error retrieving geographic-related files: {str(e)}")
            return []
            
    def get_session_interactions(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all interactions for a specific session
        
        Args:
            session_id (str, optional): Session ID to retrieve (current session if None)
            
        Returns:
            list: Session interactions
        """
        try:
            session_id = session_id or self.session_id
            interactions = []
            
            # Get all interaction files for this session
            pattern = f"{session_id}_*.json"
            session_files = list(self.interactions_dir.glob(pattern))
            
            # Sort by interaction number
            session_files.sort()
            
            for file_path in session_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        interaction = json.load(f)
                        interactions.append(interaction)
                except Exception as e:
                    self.logger.warning(f"Error reading interaction file {file_path}: {str(e)}")
                    continue
                    
            return interactions
            
        except Exception as e:
            self.logger.error(f"Error retrieving session interactions: {str(e)}")
            return []
            
    def clear_memory(self, confirm: bool = False) -> bool:
        """
        Clear all memory (USE WITH CAUTION)
        
        Args:
            confirm (bool): Confirmation flag to prevent accidental clearing
            
        Returns:
            bool: Success status
        """
        if not confirm:
            self.logger.warning("Memory clear attempted without confirmation")
            return False
            
        try:
            # Remove and recreate all memory directories
            shutil.rmtree(self.memory_dir)
            self._create_directories()
            
            # Reset session counters
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.interaction_count = 0
            
            self.logger.info("Memory system cleared successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing memory: {str(e)}")
            return False
            
    def get_session_info(self) -> Dict[str, Any]:
        """
        Get information about the current session
        
        Returns:
            dict: Session information including ID, interactions, and active files
        """
        try:
            # Get all interactions for current session
            interactions = self.get_session_interactions()
            
            # Get list of unique files accessed in this session
            active_files = set()
            for interaction in interactions:
                if "files_accessed" in interaction:
                    active_files.update(interaction["files_accessed"])
            
            # Compile session information
            session_info = {
                "session_id": self.session_id,
                "interactions": interactions,
                "active_files": list(active_files),
                "interaction_count": self.interaction_count
            }
            
            return session_info
            
        except Exception as e:
            self.logger.error(f"Error getting session info: {str(e)}")
            return {
                "session_id": self.session_id,
                "error": str(e)
            } 