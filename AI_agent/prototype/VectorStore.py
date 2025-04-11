import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import torch
from Logger import Logger
import json

class VectorStore:
    """Vector storage and retrieval using PostgreSQL with pgvector extension."""
    
    def __init__(self, 
                 connection_string: str = "postgresql://ai:ai@localhost:5432/ai",
                 table_name: str = "vector_store",
                 embedding_dim: int = 1536,  # Default OpenAI embedding dimension
                 model_name: str = "all-MiniLM-L6-v2",
                 log_dir: str = "logs",
                 log_level: int = 20,  # Default to INFO level (20)
                 prefer_local_embeddings: bool = True):  # New parameter to prefer local embeddings
        """
        Initialize VectorStore.
        
        Args:
            connection_string: PostgreSQL connection string
            table_name: Name of the table to store vectors
            embedding_dim: Dimension of the embedding vectors
            model_name: Name of the sentence transformer model for local embeddings
            log_dir: Directory for log files
            log_level: Logging level (default: logging.INFO)
            prefer_local_embeddings: Whether to prefer local embeddings over cloud embeddings
        """
        self.connection_string = connection_string
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.prefer_local_embeddings = prefer_local_embeddings
        
        self.logger = Logger(
            log_dir=log_dir,
            app_name="vector_store",
            log_level=log_level
        )
        
        # Initialize embedders - load both cloud and local embedders
        self.has_gemini_embedder = False
        self.has_openai_embedder = False
        
        # Always initialize local embedder first
        self.logger.info(f"Initializing local embedder with model: {model_name}")
        self.local_embedder = SentenceTransformer(model_name)
        self.logger.info(f"Local embedder initialized with model: {model_name}")
        
        # Only initialize cloud embedders if not preferring local
        if not self.prefer_local_embeddings:
            try:
                from agno.embedder.google import GeminiEmbedder
                self.gemini_embedder = GeminiEmbedder()
                self.has_gemini_embedder = True
                self.logger.info("GeminiEmbedder initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize GeminiEmbedder: {str(e)}")
                self.has_gemini_embedder = False
            
            try:
                from agno.embedder.openai import OpenAIEmbedder
                self.openai_embedder = OpenAIEmbedder()
                self.has_openai_embedder = True
                self.logger.info("OpenAIEmbedder initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenAIEmbedder: {str(e)}")
                self.has_openai_embedder = False
        
        # Initialize database
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize the database with required extensions and tables."""
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                # Enable pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # Create the vector store table
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id SERIAL PRIMARY KEY,
                        text TEXT NOT NULL,
                        embedding vector({self.embedding_dim}),
                        metadata JSONB,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Create an index for similarity search
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx 
                    ON {self.table_name} 
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """)
                
                conn.commit()
                self.logger.info(f"Initialized vector store table: {self.table_name}")
    
    def store_embedding(self, 
                       text: str, 
                       metadata: Optional[Dict[str, Any]] = None,
                       use_cloud_embedder: bool = True) -> bool:
        """
        Store text and its embedding in the database.
        
        Args:
            text: Text to embed and store
            metadata: Additional metadata to store with the embedding
            use_cloud_embedder: Whether to use cloud embedders (Gemini or OpenAI) or local model
            
        Returns:
            bool: Success status
        """
        # Generate embedding based on available embedders and preferences
        if self.prefer_local_embeddings or not use_cloud_embedder:
            # Use local embedder if preferred or explicitly requested
            embedding = self.local_embedder.encode(text)
            embedding = embedding.astype(np.float32)  # Convert to float32 for PostgreSQL
            self.logger.debug("Generated embedding using local embedder (by preference)")
        else:
            # Try cloud embedders if available and preferred
            if self.has_gemini_embedder:
                # Try Gemini embedder first
                embedding = self.gemini_embedder.get_embedding(text)
                embedding = np.array(embedding, dtype=np.float32)
                self.logger.debug("Generated embedding using GeminiEmbedder")
            elif self.has_openai_embedder:
                # Fall back to OpenAI if available
                embedding = self.openai_embedder.get_embedding(text)
                embedding = np.array(embedding, dtype=np.float32)
                self.logger.debug("Generated embedding using OpenAIEmbedder")
            else:
                # Fall back to local if no cloud embedders are available
                embedding = self.local_embedder.encode(text)
                embedding = embedding.astype(np.float32)
                self.logger.debug("Generated embedding using local embedder (no cloud embedders available)")
        
        # Store in database
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                # Convert metadata to JSON for PostgreSQL compatibility
                json_metadata = json.dumps(metadata) if metadata else None
                
                cur.execute(
                    f"""
                    INSERT INTO {self.table_name} (text, embedding, metadata)
                    VALUES (%s, %s, %s)
                    """,
                    (text, embedding.tolist(), json_metadata)
                )
                conn.commit()
                
        return True
    
    def search_similar(self, 
                      query: str, 
                      limit: int = 5,
                      threshold: float = 0.7,
                      use_cloud_embedder: bool = True) -> List[Dict[str, Any]]:
        """
        Search for similar texts using vector similarity.
        
        Args:
            query: Query text to search for
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold
            use_cloud_embedder: Whether to use cloud embedders (Gemini or OpenAI) or local model
            
        Returns:
            List of similar items with their metadata and similarity scores
        """
        # Generate query embedding based on available embedders and preferences
        if self.prefer_local_embeddings or not use_cloud_embedder:
            # Use local embedder if preferred or explicitly requested
            query_embedding = self.local_embedder.encode(query)
            query_embedding = query_embedding.astype(np.float32)
            self.logger.debug("Generated query embedding using local embedder (by preference)")
        else:
            # Try cloud embedders if available and preferred
            if self.has_gemini_embedder:
                # Try Gemini embedder first
                query_embedding = self.gemini_embedder.get_embedding(query)
                query_embedding = np.array(query_embedding, dtype=np.float32)
                self.logger.debug("Generated query embedding using GeminiEmbedder")
            elif self.has_openai_embedder:
                # Fall back to OpenAI if available
                query_embedding = self.openai_embedder.get_embedding(query)
                query_embedding = np.array(query_embedding, dtype=np.float32)
                self.logger.debug("Generated query embedding using OpenAIEmbedder")
            else:
                # Fall back to local if no cloud embedders are available
                query_embedding = self.local_embedder.encode(query)
                query_embedding = query_embedding.astype(np.float32)
                self.logger.debug("Generated query embedding using local embedder (no cloud embedders available)")
        
        # Search in database
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT text, metadata, 1 - (embedding <=> %s) as similarity
                    FROM {self.table_name}
                    WHERE 1 - (embedding <=> %s) > %s
                    ORDER BY similarity DESC
                    LIMIT %s
                    """,
                    (query_embedding.tolist(), query_embedding.tolist(), threshold, limit)
                )
                
                results = []
                for text, metadata, similarity in cur.fetchall():
                    results.append({
                        "text": text,
                        "metadata": metadata,
                        "similarity": float(similarity)
                    })
                
                return results
    
    def batch_store_embeddings(self, 
                             texts: List[str],
                             metadatas: Optional[List[Dict[str, Any]]] = None,
                             use_cloud_embedder: bool = True) -> bool:
        """
        Store multiple texts and their embeddings in batch.
        
        Args:
            texts: List of texts to embed and store
            metadatas: List of metadata dictionaries for each text
            use_cloud_embedder: Whether to use cloud embedders (Gemini or OpenAI) or local model
            
        Returns:
            bool: Success status
        """
        if metadatas is None:
            metadatas = [None] * len(texts)
        
        # Generate embeddings in batch based on available embedders and preferences
        if self.prefer_local_embeddings or not use_cloud_embedder:
            # Use local embedder if preferred or explicitly requested
            embeddings = self.local_embedder.encode(texts)
            embeddings = embeddings.astype(np.float32)
            self.logger.debug("Generated batch embeddings using local embedder (by preference)")
        else:
            # Try cloud embedders if available and preferred
            if self.has_gemini_embedder:
                # Process texts one by one since GeminiEmbedder might not have batch processing
                embeddings = []
                for text in texts:
                    embedding = self.gemini_embedder.get_embedding(text)
                    embeddings.append(np.array(embedding, dtype=np.float32))
                self.logger.debug("Generated batch embeddings using GeminiEmbedder")
            elif self.has_openai_embedder:
                # Process texts one by one since OpenAIEmbedder might not have batch processing
                embeddings = []
                for text in texts:
                    embedding = self.openai_embedder.get_embedding(text)
                    embeddings.append(np.array(embedding, dtype=np.float32))
                self.logger.debug("Generated batch embeddings using OpenAIEmbedder")
            else:
                # Fall back to local if no cloud embedders are available
                embeddings = self.local_embedder.encode(texts)
                embeddings = embeddings.astype(np.float32)
                self.logger.debug("Generated batch embeddings using local embedder (no cloud embedders available)")
            
        # Store in database
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                # Convert metadata dictionaries to JSON strings for PostgreSQL compatibility
                json_metadatas = []
                for meta in metadatas:
                    if meta is None:
                        json_metadatas.append(None)
                    else:
                        json_metadatas.append(json.dumps(meta))
                
                execute_values(
                    cur,
                    f"""
                    INSERT INTO {self.table_name} (text, embedding, metadata)
                    VALUES %s
                    """,
                    [(text, emb.tolist(), meta) 
                     for text, emb, meta in zip(texts, embeddings, json_metadatas)]
                )
                conn.commit()
                
        return True
    
    def delete_by_metadata(self, metadata_filter: Dict[str, Any]) -> int:
        """
        Delete embeddings based on metadata filter.
        
        Args:
            metadata_filter: Dictionary of metadata key-value pairs to match
            
        Returns:
            int: Number of deleted records
        """
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                # Convert metadata filter to JSON string for PostgreSQL
                json_filter = json.dumps(metadata_filter)
                
                cur.execute(
                    f"""
                    DELETE FROM {self.table_name}
                    WHERE metadata @> %s::jsonb
                    RETURNING id
                    """,
                    (json_filter,)
                )
                deleted_count = len(cur.fetchall())
                conn.commit()
                return deleted_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT 
                        COUNT(*) as total_count,
                        MIN(created_at) as oldest_entry,
                        MAX(created_at) as newest_entry
                    FROM {self.table_name}
                """)
                total_count, oldest_entry, newest_entry = cur.fetchone()
                
                return {
                    "total_embeddings": total_count,
                    "oldest_entry": oldest_entry,
                    "newest_entry": newest_entry,
                    "embedding_dimension": self.embedding_dim,
                    "table_name": self.table_name
                }

# --- Unit Tests ---
import unittest
from unittest.mock import patch, MagicMock, call

class TestVectorStore(unittest.TestCase):
    """Test suite for the VectorStore class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create patch objects but don't start them yet - will be used in individual tests
        self.psycopg2_connect_patch = patch('psycopg2.connect')
        self.sentence_transformer_patch = patch('sentence_transformers.SentenceTransformer')
        
        # Mock data for tests
        self.test_text = "This is a test document"
        self.test_query = "test query"
        self.test_metadata = {"source": "test", "author": "unit test"}
        self.test_embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.test_connection_string = "postgresql://test:test@localhost:5432/testdb"
        self.test_table = "test_vectors"

    def _setup_common_mocks(self):
        """Set up common mocks used by multiple tests."""
        # Start patches and get mocks
        self.mock_connect = self.psycopg2_connect_patch.start()
        self.mock_transformer = self.sentence_transformer_patch.start()
        
        # Set up connection and cursor mocks
        self.mock_conn = MagicMock()
        self.mock_cur = MagicMock()
        self.mock_connect.return_value.__enter__.return_value = self.mock_conn
        self.mock_conn.cursor.return_value.__enter__.return_value = self.mock_cur
        
        # Create mock for GeminiEmbedder
        self.mock_gemini_embedder = MagicMock()
        self.mock_gemini_embedder.get_embedding.return_value = self.test_embedding.tolist()
        
        # Create mock for OpenAIEmbedder
        self.mock_openai_embedder = MagicMock()
        self.mock_openai_embedder.get_embedding.return_value = self.test_embedding.tolist()
        
        self.mock_transformer_instance = MagicMock()
        self.mock_transformer.return_value = self.mock_transformer_instance
        self.mock_transformer_instance.encode.return_value = self.test_embedding
        
        # Create VectorStore instance with the mock embedders injected
        with patch.object(VectorStore, '_initialize_db'):  # Skip DB initialization
            self.vector_store = VectorStore(
                connection_string=self.test_connection_string,
                table_name=self.test_table
            )
            # Replace the actual embedders with our mocks
            self.vector_store.gemini_embedder = self.mock_gemini_embedder
            self.vector_store.has_gemini_embedder = True
            self.vector_store.openai_embedder = self.mock_openai_embedder
            self.vector_store.has_openai_embedder = True

    def tearDown(self):
        """Clean up after each test method."""
        # Stop any patches that might have been started
        patch.stopall()

    def test_initialization(self):
        """Test VectorStore initialization and database setup."""
        # This test doesn't need to mock OpenAIEmbedder since we patch _initialize_db
        self.mock_connect = self.psycopg2_connect_patch.start()
        self.mock_transformer = self.sentence_transformer_patch.start()
        
        # Set up DB initialization mocks
        self.mock_conn = MagicMock()
        self.mock_cur = MagicMock()
        self.mock_connect.return_value.__enter__.return_value = self.mock_conn
        self.mock_conn.cursor.return_value.__enter__.return_value = self.mock_cur
        
        # Create VectorStore instance without patching _initialize_db
        self.vector_store = VectorStore(
            connection_string=self.test_connection_string,
            table_name=self.test_table
        )
        
        # Verify DB initialization calls
        self.mock_cur.execute.assert_any_call("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Check table creation
        table_create_call = call(f"""
                        CREATE TABLE IF NOT EXISTS {self.test_table} (
                            id SERIAL PRIMARY KEY,
                            text TEXT NOT NULL,
                            embedding vector(1536),
                            metadata JSONB,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
        self.mock_cur.execute.assert_has_calls([table_create_call], any_order=True)
        
        # Verify commit was called
        self.mock_conn.commit.assert_called_once()

    def test_store_embedding_gemini(self):
        """Test storing a single embedding with Gemini."""
        self._setup_common_mocks()
        
        # Reset mock counters from initialization
        self.mock_cur.reset_mock()
        self.mock_conn.reset_mock()
        self.mock_gemini_embedder.reset_mock()
        
        # Call the method with use_cloud_embedder=True (should use Gemini first)
        result = self.vector_store.store_embedding(
            text=self.test_text,
            metadata=self.test_metadata,
            use_cloud_embedder=True
        )
        
        # Verify Gemini embedder was called
        self.mock_gemini_embedder.get_embedding.assert_called_once_with(self.test_text)
        
        # Verify OpenAI embedder was NOT called (since Gemini was available)
        self.mock_openai_embedder.get_embedding.assert_not_called()
        
        # Verify database insert
        self.mock_cur.execute.assert_called_once()
        # Check the SQL contains INSERT and the table name
        self.assertIn(f"INSERT INTO {self.test_table}", self.mock_cur.execute.call_args[0][0])
        # Verify metadata and text were passed to execute
        args = self.mock_cur.execute.call_args[0][1]
        self.assertEqual(args[0], self.test_text)
        self.assertEqual(args[2], self.test_metadata)
        
        # Verify commit was called
        self.mock_conn.commit.assert_called_once()
        
        # Verify result
        self.assertTrue(result)

    def test_store_embedding_openai(self):
        """Test storing a single embedding with OpenAI."""
        self._setup_common_mocks()
        
        # Reset mock counters from initialization
        self.mock_cur.reset_mock()
        self.mock_conn.reset_mock()
        self.mock_openai_embedder.reset_mock()
        
        # Disable Gemini embedder to test fallback to OpenAI
        self.vector_store.has_gemini_embedder = False
        
        # Call the method
        result = self.vector_store.store_embedding(
            text=self.test_text,
            metadata=self.test_metadata,
            use_cloud_embedder=True
        )
        
        # Verify OpenAI embedder was called
        self.mock_openai_embedder.get_embedding.assert_called_once_with(self.test_text)
        
        # Verify database insert
        self.mock_cur.execute.assert_called_once()
        # Check the SQL contains INSERT and the table name
        self.assertIn(f"INSERT INTO {self.test_table}", self.mock_cur.execute.call_args[0][0])
        # Verify metadata and text were passed to execute
        args = self.mock_cur.execute.call_args[0][1]
        self.assertEqual(args[0], self.test_text)
        self.assertEqual(args[2], self.test_metadata)
        
        # Verify commit was called
        self.mock_conn.commit.assert_called_once()
        
        # Verify result
        self.assertTrue(result)

    def test_store_embedding_local(self):
        """Test storing a single embedding with local model."""
        self._setup_common_mocks()
        
        # Reset mock counters from initialization
        self.mock_cur.reset_mock()
        self.mock_conn.reset_mock()
        self.mock_transformer_instance.reset_mock()  # Reset transformer mock
        
        # Call the method
        result = self.vector_store.store_embedding(
            text=self.test_text,
            metadata=self.test_metadata,
            use_cloud_embedder=False
        )
        
        # Verify local embedder was called
        self.mock_transformer_instance.encode.assert_called_once_with(self.test_text)
        
        # Verify database insert
        self.mock_cur.execute.assert_called_once()
        self.mock_conn.commit.assert_called_once()
        
        # Verify result
        self.assertTrue(result)

    def test_store_embedding_handles_error(self):
        """Test handling of errors during embedding storage."""
        self._setup_common_mocks()
        
        # Make the database connection raise an exception
        self.mock_connect.side_effect = Exception("Database connection error")
        
        # Call the method
        result = self.vector_store.store_embedding(
            text=self.test_text,
            metadata=self.test_metadata
        )
        
        # Verify result
        self.assertFalse(result)

    def test_search_similar_gemini(self):
        """Test searching for similar texts using Gemini embeddings."""
        self._setup_common_mocks()
        
        # Reset mock counters from initialization
        self.mock_cur.reset_mock()
        self.mock_gemini_embedder.reset_mock()
        
        # Set up cursor fetchall return value
        self.mock_cur.fetchall.return_value = [
            ("Document 1", {"source": "doc1"}, 0.8),
            ("Document 2", {"source": "doc2"}, 0.7)
        ]
        
        # Call the method
        results = self.vector_store.search_similar(
            query=self.test_query,
            limit=5,
            threshold=0.6,
            use_cloud_embedder=True
        )
        
        # Verify Gemini embedder was called with query
        self.mock_gemini_embedder.get_embedding.assert_called_once_with(self.test_query)
        
        # Verify OpenAI embedder was NOT called (since Gemini was available)
        self.mock_openai_embedder.get_embedding.assert_not_called()
        
        # Verify database query execution
        self.mock_cur.execute.assert_called_once()
        # Check the SQL contains SELECT and the table name
        sql = self.mock_cur.execute.call_args[0][0]
        self.assertIn(f"SELECT text, metadata", sql)
        self.assertIn(f"FROM {self.test_table}", sql)
        
        # Verify the returned results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["text"], "Document 1")
        self.assertEqual(results[0]["metadata"], {"source": "doc1"})
        self.assertEqual(results[0]["similarity"], 0.8)
        self.assertEqual(results[1]["text"], "Document 2")

    def test_search_similar_openai(self):
        """Test searching for similar texts using OpenAI embeddings."""
        self._setup_common_mocks()
        
        # Reset mock counters from initialization
        self.mock_cur.reset_mock()
        self.mock_openai_embedder.reset_mock()
        
        # Disable Gemini embedder to test fallback to OpenAI
        self.vector_store.has_gemini_embedder = False
        
        # Set up cursor fetchall return value
        self.mock_cur.fetchall.return_value = [
            ("Document 1", {"source": "doc1"}, 0.8),
            ("Document 2", {"source": "doc2"}, 0.7)
        ]
        
        # Call the method
        results = self.vector_store.search_similar(
            query=self.test_query,
            limit=5,
            threshold=0.6,
            use_cloud_embedder=True
        )
        
        # Verify OpenAI embedder was called with query
        self.mock_openai_embedder.get_embedding.assert_called_once_with(self.test_query)
        
        # Verify database query execution
        self.mock_cur.execute.assert_called_once()
        # Check the SQL contains SELECT and the table name
        sql = self.mock_cur.execute.call_args[0][0]
        self.assertIn(f"SELECT text, metadata", sql)
        self.assertIn(f"FROM {self.test_table}", sql)
        
        # Verify the returned results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["text"], "Document 1")
        self.assertEqual(results[0]["metadata"], {"source": "doc1"})
        self.assertEqual(results[0]["similarity"], 0.8)
        self.assertEqual(results[1]["text"], "Document 2")

    def test_search_similar_local(self):
        """Test searching for similar texts using local embeddings."""
        self._setup_common_mocks()
        
        # Reset mock counters from initialization
        self.mock_cur.reset_mock()
        self.mock_transformer_instance.reset_mock()  # Reset transformer mock
        
        # Set up cursor fetchall return value
        self.mock_cur.fetchall.return_value = [
            ("Document 1", {"source": "doc1"}, 0.8)
        ]
        
        # Call the method
        results = self.vector_store.search_similar(
            query=self.test_query,
            limit=5,
            threshold=0.6,
            use_cloud_embedder=False
        )
        
        # Verify local embedder was called
        self.mock_transformer_instance.encode.assert_called_once_with(self.test_query)
        
        # Verify database query execution
        self.mock_cur.execute.assert_called_once()
        
        # Verify the returned results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], "Document 1")

    def test_search_similar_handles_error(self):
        """Test handling of errors during search."""
        self._setup_common_mocks()
        
        # Make the database connection raise an exception
        self.mock_connect.side_effect = Exception("Database search error")
        
        # Call the method
        results = self.vector_store.search_similar(self.test_query)
        
        # Verify empty results returned
        self.assertEqual(results, [])

    def test_batch_store_embeddings_gemini(self):
        """Test batch storing embeddings with Gemini."""
        self._setup_common_mocks()
        
        # Reset mock counters from initialization
        self.mock_cur.reset_mock()
        self.mock_conn.reset_mock()
        self.mock_gemini_embedder.reset_mock()
        
        # Set up batch data
        texts = ["Document 1", "Document 2", "Document 3"]
        metadatas = [{"source": "doc1"}, {"source": "doc2"}, {"source": "doc3"}]
        
        # Mock execute_values globally rather than with patch - this is more reliable
        execute_values_mock = MagicMock()
        
        # Temporarily replace the real execute_values with our mock
        original_execute_values = psycopg2.extras.execute_values
        psycopg2.extras.execute_values = execute_values_mock
        
        try:
            # Call the method
            result = self.vector_store.batch_store_embeddings(
                texts=texts,
                metadatas=metadatas,
                use_cloud_embedder=True
            )
            
            # Verify get_embedding was called for each text
            self.assertEqual(self.mock_gemini_embedder.get_embedding.call_count, len(texts))
            for text in texts:
                self.mock_gemini_embedder.get_embedding.assert_any_call(text)
            
            # Verify execute_values was called
            execute_values_mock.assert_called_once()
            # Check the SQL contains INSERT and the table name
            sql = execute_values_mock.call_args[0][1]
            self.assertIn(f"INSERT INTO {self.test_table}", sql)
            
            # Verify commit was called
            self.mock_conn.commit.assert_called_once()
            
            # Verify result
            self.assertTrue(result)
        finally:
            # Restore the original execute_values function
            psycopg2.extras.execute_values = original_execute_values

    def test_batch_store_embeddings_no_metadata(self):
        """Test batch storing embeddings without metadata."""
        self._setup_common_mocks()
        
        # Reset mock counters from initialization
        self.mock_cur.reset_mock()
        self.mock_conn.reset_mock()
        self.mock_gemini_embedder.reset_mock()
        
        # Set up batch data with no metadata
        texts = ["Document 1", "Document 2"]
        
        # Mock execute_values globally rather than with patch - this is more reliable
        execute_values_mock = MagicMock()
        
        # Temporarily replace the real execute_values with our mock
        original_execute_values = psycopg2.extras.execute_values
        psycopg2.extras.execute_values = execute_values_mock
        
        try:
            # Call the method without metadata
            result = self.vector_store.batch_store_embeddings(
                texts=texts,
                metadatas=None,
                use_cloud_embedder=True
            )
            
            # Verify get_embedding was called for each text
            self.assertEqual(self.mock_gemini_embedder.get_embedding.call_count, len(texts))
            for text in texts:
                self.mock_gemini_embedder.get_embedding.assert_any_call(text)
            
            # Verify result
            self.assertTrue(result)
            
            # Verify execute_values was called
            execute_values_mock.assert_called_once()
            
            # Verify the default None metadata was used
            args = execute_values_mock.call_args[0][2]
            self.assertEqual(len(args), 2)  # Two text items
            for i, arg_tuple in enumerate(args):
                self.assertEqual(arg_tuple[0], texts[i])  # Text
                self.assertIsNone(arg_tuple[2])  # Metadata should be None
        finally:
            # Restore the original execute_values function
            psycopg2.extras.execute_values = original_execute_values

    def test_delete_by_metadata(self):
        """Test deleting embeddings by metadata."""
        self._setup_common_mocks()
        
        # Reset mock counters from initialization
        self.mock_cur.reset_mock()
        self.mock_conn.reset_mock()
        
        # Set up delete return value (IDs of deleted records)
        self.mock_cur.fetchall.return_value = [(1,), (2,)]
        
        # Call the method
        delete_count = self.vector_store.delete_by_metadata({"source": "test"})
        
        # Verify database delete execution
        self.mock_cur.execute.assert_called_once()
        # Check the SQL contains DELETE and the table name
        sql = self.mock_cur.execute.call_args[0][0]
        self.assertIn(f"DELETE FROM {self.test_table}", sql)
        
        # Verify commit was called
        self.mock_conn.commit.assert_called_once()
        
        # Verify the returned count
        self.assertEqual(delete_count, 2)

    def test_get_stats(self):
        """Test getting statistics about the vector store."""
        self._setup_common_mocks()
        
        # Reset mock counters from initialization
        self.mock_cur.reset_mock()
        
        # Set up mock return value for stats query
        self.mock_cur.fetchone.return_value = (100, "2023-01-01", "2023-04-01")
        
        # Call the method
        stats = self.vector_store.get_stats()
        
        # Verify database query execution
        self.mock_cur.execute.assert_called_once()
        sql = self.mock_cur.execute.call_args[0][0]
        self.assertIn(f"SELECT ", sql)
        self.assertIn(f"FROM {self.test_table}", sql)
        
        # Verify the returned stats
        self.assertEqual(stats["total_embeddings"], 100)
        self.assertEqual(stats["oldest_entry"], "2023-01-01")
        self.assertEqual(stats["newest_entry"], "2023-04-01")
        self.assertEqual(stats["table_name"], self.test_table)

    def test_get_stats_handles_error(self):
        """Test handling of errors when getting stats."""
        self._setup_common_mocks()
        
        # Make the database connection raise an exception
        self.mock_connect.side_effect = Exception("Stats error")
        
        # Call the method
        stats = self.vector_store.get_stats()
        
        # Verify empty dict returned
        self.assertEqual(stats, {})

    def test_gemini_fallback_to_openai(self):
        """Test fallback to OpenAI when Gemini is not available."""
        self._setup_common_mocks()
        
        # Reset mock counters from initialization
        self.mock_cur.reset_mock()
        self.mock_conn.reset_mock()
        self.mock_gemini_embedder.reset_mock()
        self.mock_openai_embedder.reset_mock()
        
        # Simulate Gemini embedder not being available
        self.vector_store.has_gemini_embedder = False
        
        # Call the method with use_cloud_embedder=True (should fall back to OpenAI)
        result = self.vector_store.store_embedding(
            text=self.test_text,
            metadata=self.test_metadata,
            use_cloud_embedder=True
        )
        
        # Verify Gemini embedder was NOT called
        self.mock_gemini_embedder.get_embedding.assert_not_called()
        
        # Verify OpenAI embedder WAS called
        self.mock_openai_embedder.get_embedding.assert_called_once_with(self.test_text)
        
        # Verify database insert
        self.mock_cur.execute.assert_called_once()
        
        # Verify result
        self.assertTrue(result)


# Run the tests if this file is executed directly
if __name__ == '__main__':
    unittest.main()
