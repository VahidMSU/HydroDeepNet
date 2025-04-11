from typing import List, Dict, Any, Optional, Union, Tuple
from agno.vectordb.base import VectorDb
from VectorStore import VectorStore
from Logger import Logger

class VectorDbAdapter(VectorDb):
    """
    Adapter class that wraps our custom VectorStore to make it compatible 
    with agno's VectorDb interface.
    """
    
    def __init__(self, 
                 connection_string: str = "postgresql://ai:ai@localhost:5432/ai",
                 table_name: str = "vector_store",
                 log_dir: str = "logs",
                 log_level: int = 20):  # Default to INFO level (20)
        """
        Initialize the VectorDbAdapter.
        
        Args:
            connection_string: PostgreSQL connection string
            table_name: Name of the table to store vectors
            log_dir: Directory for log files
            log_level: Logging level (default: logging.INFO)
        """
        # Initialize logger
        self.logger = Logger(
            log_dir=log_dir,
            app_name="vector_db_adapter",
            log_level=log_level
        )
        self.logger.info(f"Initializing VectorDbAdapter with table: {table_name}")
        
        # Initialize the wrapped VectorStore
        self.vector_store = VectorStore(
            connection_string=connection_string,
            table_name=table_name,
            log_dir=log_dir,
            log_level=log_level
        )
        
        # Initialize required attributes for agno compatibility
        self.id_field = "id"
        self.content_field = "text"
        self.embedding_field = "embedding"
        self.metadata_field = "metadata"
        
        # Database name/collection info
        self.db_name = "postgres"
        self.collection_name = table_name
    
    def search(self, 
              query: str, 
              k: int = 5, 
              threshold: float = 0.5,
              limit: Optional[int] = None,
              **kwargs) -> List[Dict[str, Any]]:
        """
        Search for similar documents to the query.
        
        Args:
            query: The query text
            k: Maximum number of results to return (preferred parameter)
            threshold: Minimum similarity threshold
            limit: Alternative parameter for maximum results
            
        Returns:
            List of search results with metadata
        """
        # Use limit if k is not explicitly provided (i.e., k is default)
        # Or if limit is provided and k is the default value
        effective_k = k
        if limit is not None and k == 5: # Check if k is the default value
            effective_k = limit
        
        # Delegate to the underlying VectorStore using 'limit'
        results = self.vector_store.search_similar(
            query=query,
            limit=effective_k, # Use effective_k for the underlying call's limit
            threshold=threshold
        )
        
        # Format results for agno compatibility
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result.get("id", ""),
                "content": result.get("text", ""),
                "similarity": result.get("similarity", 0.0),
                "metadata": result.get("metadata", {})
            })
        
        return formatted_results
    
    # Required synchronous methods from VectorDb base class
    
    def add_documents(self, 
                     texts: List[str], 
                     metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            texts: List of document texts
            metadatas: Optional list of metadata for each document
            
        Returns:
            List of document IDs
        """
        # Handle missing metadatas
        if metadatas is None:
            metadatas = [{}] * len(texts)
            
        # Delegate to batch store in VectorStore
        success = self.vector_store.batch_store_embeddings(
            texts=texts,
            metadatas=metadatas
        )
        
        if success:
            # We don't have direct access to the IDs, so return placeholders
            return ["doc_" + str(i) for i in range(len(texts))]
        else:
            return []
    
    def add_document(self, 
                    text: str, 
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a single document to the vector store.
        
        Args:
            text: Document text
            metadata: Optional metadata for the document
            
        Returns:
            Document ID
        """
        # Delegate to store_embedding in VectorStore
        success = self.vector_store.store_embedding(
            text=text,
            metadata=metadata or {}
        )
        
        if success:
            # We don't have direct access to the ID, so return a placeholder
            return "doc_" + str(hash(text) % 10000)
        else:
            return ""
    
    def delete(self, document_id: str) -> bool:
        """
        Delete a document from the vector store.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            Success status
        """
        # Delegate to delete in VectorStore
        # Note: This is an approximation since our VectorStore doesn't support direct ID-based deletion
        deleted = self.vector_store.delete_by_metadata({"id": document_id})
        return deleted > 0
    
    # Additional required methods for VectorDb
    
    def create(self, overwrite: bool = False) -> bool:
        """Create the vector database."""
        # The database is auto-created in our VectorStore initialization
        return True
    
    def exists(self) -> bool:
        """Check if the vector database exists."""
        # Our VectorStore auto-creates the database, so we'll assume it exists
        return True
    
    def drop(self) -> bool:
        """Drop the vector database."""
        # Reset the table by recreating it
        self.vector_store._initialize_db()
        return True
    
    def doc_exists(self, document_id: str) -> bool:
        """Check if a document exists in the database."""
        # Not directly supported in our implementation
        return False
    
    def name_exists(self) -> bool:
        """Check if the database name exists."""
        # We assume the PostgreSQL database exists
        return True
    
    def insert(self, documents: List[Any], filters: Optional[Any] = None, **kwargs) -> List[str]:
        """Insert documents with explicit IDs."""
        # Handle Document objects instead of expecting dictionaries
        texts = []
        metadatas = []
        ids = []
        
        for i, doc in enumerate(documents):
            # Handle both Document objects and dictionaries
            if hasattr(doc, 'content') and hasattr(doc, 'page_content'):
                # It's an Agno Document object
                # Some Document implementations use 'content', others use 'page_content'
                text_content = doc.content if doc.content else doc.page_content
                texts.append(text_content)
                ids.append(getattr(doc, 'id', f"doc_{i}"))
                # Get metadata from the document object if available
                if hasattr(doc, 'metadata'):
                    metadatas.append(doc.metadata)
                else:
                    metadatas.append({})
            elif hasattr(doc, 'content'):
                # It's a Document object with direct content attribute
                texts.append(doc.content)
                ids.append(getattr(doc, 'id', f"doc_{i}"))
                # Get metadata from the document object if available
                if hasattr(doc, 'metadata'):
                    metadatas.append(doc.metadata)
                else:
                    metadatas.append({})
            elif hasattr(doc, 'page_content'):
                # It's a Document object with page_content attribute (LangChain style)
                texts.append(doc.page_content)
                ids.append(getattr(doc, 'id', f"doc_{i}"))
                # Get metadata from the document object if available
                if hasattr(doc, 'metadata'):
                    metadatas.append(doc.metadata)
                else:
                    metadatas.append({})
            elif isinstance(doc, dict):
                # It's a dictionary
                content = doc.get("content", doc.get("page_content", ""))
                texts.append(content)
                ids.append(doc.get("id", f"doc_{i}"))
                metadatas.append(doc.get("metadata", {}))
            else:
                # For any other type, try to convert to string
                texts.append(str(doc))
                ids.append(f"doc_{i}")
                metadatas.append({})
        
        # Log the extraction process
        self.logger.debug(f"Extracted {len(texts)} documents for insertion")
        
        success = self.vector_store.batch_store_embeddings(
            texts=texts,
            metadatas=metadatas
        )
        
        if success:
            return ids
        else:
            return []
    
    def upsert(self, documents: List[Dict[str, Any]], filters: Optional[Any] = None, **kwargs) -> List[str]:
        """Upsert documents with explicit IDs."""
        # Same as insert since our implementation doesn't distinguish
        return self.insert(documents)
    
    # Async methods (wrap the synchronous methods)
    
    async def async_search(self, 
                          query: str, 
                          k: int = 5, 
                          threshold: float = 0.5,
                          limit: Optional[int] = None,
                          **kwargs) -> List[Dict[str, Any]]:
        """Async version of search."""
        # Use limit if k is not explicitly provided or is default
        effective_k = k
        if limit is not None and k == 5:
            effective_k = limit
            
        return self.search(query, effective_k, threshold)
    
    async def async_add_documents(self, 
                                 texts: List[str], 
                                 metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Async version of add_documents."""
        return self.add_documents(texts, metadatas)
    
    async def async_add_document(self, 
                                text: str, 
                                metadata: Optional[Dict[str, Any]] = None) -> str:
        """Async version of add_document."""
        return self.add_document(text, metadata)
    
    async def async_delete(self, document_id: str) -> bool:
        """Async version of delete."""
        return self.delete(document_id)
    
    async def async_create(self, overwrite: bool = False) -> bool:
        """Async version of create."""
        return self.create(overwrite)
    
    async def async_exists(self) -> bool:
        """Async version of exists."""
        return self.exists()
    
    async def async_drop(self) -> bool:
        """Async version of drop."""
        return self.drop()
    
    async def async_doc_exists(self, document_id: str) -> bool:
        """Async version of doc_exists."""
        return self.doc_exists(document_id)
    
    async def async_name_exists(self) -> bool:
        """Async version of name_exists."""
        return self.name_exists()
    
    async def async_insert(self, documents: List[Dict[str, Any]], filters: Optional[Any] = None, **kwargs) -> List[str]:
        """Async version of insert."""
        return self.insert(documents)
    
    async def async_upsert(self, documents: List[Dict[str, Any]], filters: Optional[Any] = None, **kwargs) -> List[str]:
        """Async version of upsert."""
        return self.upsert(documents) 