from typing import List, Dict, Any, Optional, Callable, Union, Type
import inspect
from Logger import Logger

class AgentKnowledge:
    """Base class for agent knowledge sources."""
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize the agent knowledge source.
        
        Args:
            log_dir: Directory for log files
        """
        self.logger = Logger(
            log_dir=log_dir,
            app_name="agent_knowledge"
        )
    
    def search(self, query: str, k: int = 5, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Search the knowledge base.
        
        Args:
            query: Search query
            k: Maximum number of results to return
            threshold: Similarity threshold
            
        Returns:
            List of search results with metadata
        """
        if hasattr(self, 'vector_db'):
            results = self.vector_db.search_similar(
                query=query,
                limit=k,  # Vector store uses limit parameter
                threshold=threshold
            )
            return self._format_results(results)
        return []
            
    def _format_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format search results with consistent structure."""
        formatted = []
        for result in results:
            formatted.append({
                "id": result.get("id", ""),
                "content": result.get("content", ""),
                "metadata": result.get("metadata", {}),
                "similarity": result.get("similarity", 0.0)
            })
        return formatted

class KnowledgeBaseAdapter:
    """
    Adapter for knowledge bases that handles different parameter formats.
    This class wraps any knowledge base and provides a consistent interface.
    """
    
    def __init__(self, knowledge_base: Any, log_dir: str = "logs"):
        """
        Initialize the adapter with a knowledge base.
        
        Args:
            knowledge_base: Any knowledge base implementation with a search method
            log_dir: Directory for log files
        """
        self.knowledge_base = knowledge_base
        self.logger = Logger(
            log_dir=log_dir,
            app_name="knowledge_base_adapter"
        )
    
    def search(self, query: str, k: int = 5, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Search the knowledge base with adaptive parameter handling.
        
        Args:
            query: Search query
            k: Maximum number of results to return
            threshold: Similarity threshold
            
        Returns:
            List of search results
        """
        # Use standard agno format with k parameter
        results = self.knowledge_base.search(query=query, k=k, threshold=threshold)
        return self._ensure_list(results)
    
    def _ensure_list(self, results: Any) -> List[Dict[str, Any]]:
        """Ensure the results are in a consistent list format."""
        if results is None:
            return []
        elif isinstance(results, list):
            return results
        elif isinstance(results, dict):
            return [results]
        else:
            # Convert to list if it's iterable
            return list(results) 