import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import os
from pathlib import Path
import logging

from RedisTools import RedisTools
from VectorStore import VectorStore
from Logger import Logger

class PersistenceManager:
    """Manages persistence across Redis and vector storage for the RAG system."""
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379/0",
                 vector_store_url: str = "postgresql://ai:ai@localhost:5432/ai",
                 session_id: Optional[str] = None,
                 log_dir: str = "logs",
                 log_level: int = logging.INFO):
        """
        Initialize the persistence manager.
        
        Args:
            redis_url: URL for Redis connection
            vector_store_url: URL for vector store connection
            session_id: Optional session identifier
            log_dir: Directory for log files
            log_level: Logging level
        """
        # Set up logging
        self.logger = Logger(
            log_dir=log_dir,
            app_name="persistence_manager",
            log_level=log_level
        )
        
        self.logger.info("Initializing PersistenceManager")
        self.session_id = session_id
        
        # Initialize storage systems
        self.redis_client = RedisTools(redis_url=redis_url, session_id=session_id)
        self.vector_store = VectorStore(
            connection_string=vector_store_url,
            log_dir=log_dir,
            log_level=log_level
        )
        
        # Track persistence status
        self.persistence_status = {
            "redis_connected": self.redis_client.has_redis,
            "vector_store_initialized": True
        }
        
        self.logger.info("PersistenceManager initialization completed")
    
    def save_conversation(self, 
                         session_id: str, 
                         messages: List[Dict[str, Any]], 
                         expire_seconds: int = 86400) -> bool:
        """
        Save conversation history to Redis.
        
        Args:
            session_id: Session identifier
            messages: List of conversation messages
            expire_seconds: Time until expiration in seconds
            
        Returns:
            bool: Success status
        """
        try:
            # Add timestamp if not present
            for message in messages:
                if 'timestamp' not in message:
                    message['timestamp'] = datetime.now().isoformat()
            
            # Save to Redis
            success = self.redis_client.save_to_redis(
                f"conversation:{session_id}",
                messages,
                expire_seconds=expire_seconds
            )
            
            if success:
                self.logger.info(f"Saved conversation for session {session_id}")
            else:
                self.logger.warning(f"Failed to save conversation for session {session_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error saving conversation: {str(e)}")
            return False
    
    def save_analysis(self, 
                     file_path: str, 
                     analysis: Union[str, Dict[str, Any]],
                     chunk_size: int = 1000) -> bool:
        """
        Save analysis results to both Redis and vector store.
        
        Args:
            file_path: Path of the analyzed file
            analysis: Analysis results (text or structured data)
            chunk_size: Size of text chunks for vector storage
            
        Returns:
            bool: Success status
        """
        # Prepare metadata
        metadata = {
            "source": os.path.basename(file_path),
            "full_path": file_path,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to Redis for quick retrieval
        redis_success = self.redis_client.save_to_redis(
            f"analysis:{file_path}",
            analysis
        )
        
        # Handle Document objects
        if hasattr(analysis, 'content'):
            # It's a Document object
            text = analysis.content
            if hasattr(analysis, 'metadata'):
                metadata.update(analysis.metadata)
        # For vector storage, we need text format
        elif isinstance(analysis, dict):
            text = json.dumps(analysis)
        else:
            text = str(analysis)
        
        # Split long text into chunks for vector storage
        chunks = self._chunk_text(text, chunk_size)
        
        # Store chunks in vector store
        vector_success = self.vector_store.batch_store_embeddings(
            texts=chunks,
            metadatas=[{**metadata, "chunk_index": i} for i in range(len(chunks))]
        )
            
        return redis_success and vector_success
    
    def get_conversation(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve conversation history from Redis.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of conversation messages or None if not found
        """
        try:
            result = self.redis_client.load_from_redis(f"conversation:{session_id}")
            if result:
                self.logger.debug(f"Retrieved conversation for session {session_id}")
            else:
                self.logger.debug(f"No conversation found for session {session_id}")
            return result
        except Exception as e:
            self.logger.error(f"Error retrieving conversation: {str(e)}")
            return None
    
    def get_analysis(self, 
                    file_path: str, 
                    use_vector_search: bool = False,
                    query: Optional[str] = None) -> Optional[Union[str, Dict[str, Any]]]:
        """
        Retrieve analysis results from storage.
        
        Args:
            file_path: Path of the analyzed file
            use_vector_search: Whether to use vector search
            query: Optional query for semantic search
            
        Returns:
            Analysis results or None if not found
        """
        try:
            self.logger.debug(f"Retrieving analysis for {file_path}, vector_search={use_vector_search}")
            if use_vector_search and query:
                # Search vector store
                results = self.vector_store.search_similar(
                    query=query,
                    limit=5,
                    threshold=0.7
                )
                
                if results:
                    self.logger.debug(f"Found {len(results)} vector results for query: {query}")
                    # Combine relevant chunks
                    return self._combine_chunks(results)
            
            # Fall back to Redis
            result = self.redis_client.get_cached_analysis(file_path)
            if result:
                self.logger.debug(f"Retrieved analysis from Redis for {file_path}")
            else:
                self.logger.debug(f"No analysis found for {file_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error retrieving analysis: {str(e)}")
            return None
    
    def search_knowledge_base(self, 
                            query: str,
                            k: int = 5,
                            threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search the knowledge base using vector similarity.
        
        Args:
            query: Search query
            k: Maximum number of results to return
            threshold: Similarity threshold
            
        Returns:
            List of relevant results with metadata
        """
        try:
            self.logger.debug(f"Searching knowledge base for: {query}")
            results = self.vector_store.search_similar(
                query=query,
                limit=k,  # Vector store still uses limit parameter
                threshold=threshold
            )
            
            self.logger.debug(f"Found {len(results)} results for query: {query}")
            return results
        except Exception as e:
            self.logger.error(f"Error searching knowledge base: {str(e)}")
            return []
    
    def clear_session_data(self, session_id: str) -> bool:
        """
        Clear all data for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            bool: Success status
        """
        try:
            self.logger.info(f"Clearing data for session: {session_id}")
            # Clear Redis data
            self.redis_client.clear_redis_session()
            
            # Clear vector store data
            deleted_count = self.vector_store.delete_by_metadata({"session_id": session_id})
            
            self.logger.info(f"Cleared session data for {session_id}: removed {deleted_count} vector entries")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing session data: {str(e)}")
            return False
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about the storage systems."""
        try:
            self.logger.debug("Gathering storage statistics")
            vector_stats = self.vector_store.get_stats()
            
            stats = {
                "redis_connected": self.persistence_status["redis_connected"],
                "vector_store": vector_stats
            }
            
            self.logger.debug(f"Storage stats gathered successfully")
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting storage stats: {str(e)}")
            return {}
    
    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks of specified size."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_size = len(word) + 1  # Add 1 for space
            if current_size + word_size > chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = word_size
            else:
                current_chunk.append(word)
                current_size += word_size
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _combine_chunks(self, results: List[Dict[str, Any]]) -> str:
        """Combine text chunks from search results."""
        # Sort chunks by their index
        sorted_chunks = sorted(
            results,
            key=lambda x: x['metadata'].get('chunk_index', 0)
        )
        
        # Combine text
        return " ".join(result['text'] for result in sorted_chunks)

