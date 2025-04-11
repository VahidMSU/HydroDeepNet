import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import json
import re

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.knowledge.text import TextKnowledgeBase
from agno.knowledge.csv import CSVKnowledgeBase
from agno.knowledge.json import JSONKnowledgeBase
from agno.knowledge.docx import DocxKnowledgeBase
from agno.vectordb.pgvector import PgVector

@dataclass
class DocumentInfo:
    """Information about a document."""
    path: str
    type: str
    size: int
    last_modified: datetime
    summary: Optional[str] = None
    metadata: Dict[str, Any] = None
    vector_id: Optional[str] = None

class DocumentNavigatorAgent:
    """Agent specialized in document analysis and navigation."""
    
    def __init__(self, 
                 vector_store_url: str = "postgresql://ai:ai@localhost:5432/ai",
                 openai_api_key: Optional[str] = None,
                 log_dir: str = "logs",
                 log_level: int = logging.INFO):
        """
        Initialize the document navigator agent.
        
        Args:
            vector_store_url: URL for vector store connection
            openai_api_key: Optional OpenAI API key
            log_dir: Directory for log files
            log_level: Logging level
        """
        # Set up logging
        self._setup_logging(log_dir, log_level)
        
        self.logger.info("Initializing DocumentNavigatorAgent")
        self.logger.debug(f"Vector store URL: {vector_store_url}")
        
        # Initialize agent
        self.agent = Agent(
            model=OpenAIChat(id="gpt-4"),
            agent_id="document-navigator",
            name="Document Navigator",
            instructions=[
                "You are a document analysis expert.",
                "Navigate and summarize documents.",
                "Extract key information.",
                "Provide relevant context."
            ]
        )
        self.logger.info("Agent initialized successfully")
        
        # Initialize knowledge bases
        self.knowledge_bases = {}
        self.vector_store_url = vector_store_url
        
        # Track discovered documents
        self.discovered_docs: Dict[str, DocumentInfo] = {}
        
        # Document type handlers
        self.type_handlers = {
            ".pdf": self._handle_pdf,
            ".txt": self._handle_text,
            ".csv": self._handle_csv,
            ".json": self._handle_json,
            ".docx": self._handle_docx,
            ".md": self._handle_text
        }
        self.logger.info(f"Registered handlers for file types: {', '.join(self.type_handlers.keys())}")
    
    def _setup_logging(self, log_dir: str, log_level: int):
        """Set up logging configuration."""
        # Create logger
        self.logger = logging.getLogger("DocumentNavigatorAgent")
        self.logger.setLevel(log_level)
        
        # Create log directory
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # File handler for operations
        ops_handler = logging.FileHandler(
            log_path / "document_navigator_operations.log"
        )
        ops_handler.setLevel(log_level)
        ops_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(ops_handler)
        
        # File handler for errors
        error_handler = logging.FileHandler(
            log_path / "document_navigator_errors.log"
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
            'Exception: %(exc_info)s'
        ))
        self.logger.addHandler(error_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(console_handler)
        
        self.logger.info("Logging setup completed")
    
    def discover_documents(self, base_path: str) -> List[DocumentInfo]:
        """
        Discover and analyze documents in the given path.
        
        Args:
            base_path: Base path to search for documents
            
        Returns:
            List of discovered documents
        """
        self.logger.info(f"Starting document discovery in: {base_path}")
        try:
            discovered = []
            base_path = Path(base_path)
            
            # Walk through directory
            for file_path in base_path.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in self.type_handlers:
                    self.logger.debug(f"Processing file: {file_path}")
                    try:
                        # Get file info
                        doc_info = DocumentInfo(
                            path=str(file_path),
                            type=file_path.suffix.lower(),
                            size=file_path.stat().st_size,
                            last_modified=datetime.fromtimestamp(file_path.stat().st_mtime)
                        )
                        
                        # Process document
                        doc_info = self.type_handlers[doc_info.type](doc_info)
                        
                        if doc_info:
                            self.discovered_docs[str(file_path)] = doc_info
                            discovered.append(doc_info)
                            self.logger.info(f"Successfully processed: {file_path}")
                            
                    except Exception as e:
                        self.logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
            
            self.logger.info(f"Document discovery completed. Found {len(discovered)} documents")
            return discovered
            
        except Exception as e:
            self.logger.error(f"Error discovering documents: {str(e)}", exc_info=True)
            return []
    
    def _handle_pdf(self, doc_info: DocumentInfo) -> Optional[DocumentInfo]:
        """Handle PDF document."""
        try:
            # Initialize knowledge base
            vector_db = PgVector(
                table_name=f"pdf_{hash(doc_info.path)}",
                db_url=self.vector_store_url
            )
            
            kb = PDFKnowledgeBase(
                path=doc_info.path,
                vector_db=vector_db,
                reader=PDFReader(chunk=True)
            )
            
            # Load and process
            kb.load(recreate=False)
            
            # Generate summary
            summary = self.agent.print_response(
                f"Summarize the content of {os.path.basename(doc_info.path)}",
                context={"content": kb.get_content()},
                stream=False
            )
            
            doc_info.summary = summary
            doc_info.vector_id = kb.id
            self.knowledge_bases[doc_info.path] = kb
            
            return doc_info
            
        except Exception as e:
            self.logger.error(f"Error handling PDF {doc_info.path}: {str(e)}")
            return None
    
    def _handle_text(self, doc_info: DocumentInfo) -> Optional[DocumentInfo]:
        """Handle text document."""
        try:
            # Initialize knowledge base
            vector_db = PgVector(
                table_name=f"text_{hash(doc_info.path)}",
                db_url=self.vector_store_url
            )
            
            kb = TextKnowledgeBase(
                path=doc_info.path,
                vector_db=vector_db
            )
            
            # Load and process
            kb.load(recreate=False)
            
            # Generate summary
            summary = self.agent.print_response(
                f"Summarize the content of {os.path.basename(doc_info.path)}",
                context={"content": kb.get_content()},
                stream=False
            )
            
            doc_info.summary = summary
            doc_info.vector_id = kb.id
            self.knowledge_bases[doc_info.path] = kb
            
            return doc_info
            
        except Exception as e:
            self.logger.error(f"Error handling text {doc_info.path}: {str(e)}")
            return None
    
    def _handle_csv(self, doc_info: DocumentInfo) -> Optional[DocumentInfo]:
        """Handle CSV document."""
        try:
            # Initialize knowledge base
            vector_db = PgVector(
                table_name=f"csv_{hash(doc_info.path)}",
                db_url=self.vector_store_url
            )
            
            kb = CSVKnowledgeBase(
                path=doc_info.path,
                vector_db=vector_db
            )
            
            # Load and process
            kb.load(recreate=False)
            
            # Generate summary
            summary = self.agent.print_response(
                f"Analyze the CSV data in {os.path.basename(doc_info.path)}",
                context={"content": kb.get_content()},
                stream=False
            )
            
            doc_info.summary = summary
            doc_info.vector_id = kb.id
            self.knowledge_bases[doc_info.path] = kb
            
            return doc_info
            
        except Exception as e:
            self.logger.error(f"Error handling CSV {doc_info.path}: {str(e)}")
            return None
    
    def _handle_json(self, doc_info: DocumentInfo) -> Optional[DocumentInfo]:
        """Handle JSON document."""
        try:
            # Initialize knowledge base
            vector_db = PgVector(
                table_name=f"json_{hash(doc_info.path)}",
                db_url=self.vector_store_url
            )
            
            kb = JSONKnowledgeBase(
                path=doc_info.path,
                vector_db=vector_db
            )
            
            # Load and process
            kb.load(recreate=False)
            
            # Generate summary
            summary = self.agent.print_response(
                f"Analyze the JSON data in {os.path.basename(doc_info.path)}",
                context={"content": kb.get_content()},
                stream=False
            )
            
            doc_info.summary = summary
            doc_info.vector_id = kb.id
            self.knowledge_bases[doc_info.path] = kb
            
            return doc_info
            
        except Exception as e:
            self.logger.error(f"Error handling JSON {doc_info.path}: {str(e)}")
            return None
    
    def _handle_docx(self, doc_info: DocumentInfo) -> Optional[DocumentInfo]:
        """Handle DOCX document."""
        try:
            # Initialize knowledge base
            vector_db = PgVector(
                table_name=f"docx_{hash(doc_info.path)}",
                db_url=self.vector_store_url
            )
            
            kb = DocxKnowledgeBase(
                path=doc_info.path,
                vector_db=vector_db
            )
            
            # Load and process
            kb.load(recreate=False)
            
            # Generate summary
            summary = self.agent.print_response(
                f"Summarize the content of {os.path.basename(doc_info.path)}",
                context={"content": kb.get_content()},
                stream=False
            )
            
            doc_info.summary = summary
            doc_info.vector_id = kb.id
            self.knowledge_bases[doc_info.path] = kb
            
            return doc_info
            
        except Exception as e:
            self.logger.error(f"Error handling DOCX {doc_info.path}: {str(e)}")
            return None
    
    def handle_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Handle a document-related query.
        
        Args:
            query: User query
            context: Optional additional context
            
        Returns:
            Response text
        """
        self.logger.info(f"Processing query: {query}")
        try:
            # Extract document references from query
            doc_refs = self._extract_doc_references(query)
            self.logger.debug(f"Extracted document references: {doc_refs}")
            
            # Get relevant documents
            relevant_docs = []
            if doc_refs:
                # Use explicitly referenced documents
                self.logger.debug("Using explicitly referenced documents")
                for ref in doc_refs:
                    matching_docs = [
                        doc for doc in self.discovered_docs.values()
                        if ref.lower() in doc.path.lower()
                    ]
                    relevant_docs.extend(matching_docs)
            else:
                # Use semantic search across all documents
                self.logger.debug("Performing semantic search")
                relevant_docs = self._semantic_search(query)
            
            if not relevant_docs:
                self.logger.info("No relevant documents found for query")
                return "I couldn't find any relevant documents to answer your query."
            
            self.logger.info(f"Found {len(relevant_docs)} relevant documents")
            
            # Build context from relevant documents
            doc_context = {
                "documents": [
                    {
                        "path": doc.path,
                        "content": self.knowledge_bases[doc.path].get_content(),
                        "summary": doc.summary
                    }
                    for doc in relevant_docs[:3]  # Limit to top 3 most relevant
                ]
            }
            
            # Combine with provided context
            if context:
                doc_context.update(context)
            
            # Generate response
            self.logger.debug("Generating response using agent")
            response = self.agent.print_response(
                query,
                context=doc_context,
                stream=False
            )
            
            self.logger.info("Query processing completed successfully")
            return response
            
        except Exception as e:
            self.logger.error(f"Error handling query: {str(e)}", exc_info=True)
            return f"I encountered an error while processing your query: {str(e)}"
    
    def _extract_doc_references(self, query: str) -> List[str]:
        """Extract document references from query."""
        # Look for file extensions
        extensions = "|".join(self.type_handlers.keys())
        matches = re.findall(f"\\w+(?:{extensions})", query, re.IGNORECASE)
        
        # Look for quoted filenames
        quoted = re.findall(r'"([^"]+)"', query)
        
        return matches + quoted
    
    def _semantic_search(self, query: str, limit: int = 3) -> List[DocumentInfo]:
        """Perform semantic search across all documents."""
        results = []
        
        for doc_path, kb in self.knowledge_bases.items():
            try:
                similarity = kb.similarity(query)
                if similarity > 0.7:  # Threshold
                    results.append((self.discovered_docs[doc_path], similarity))
            except Exception as e:
                self.logger.error(f"Error searching {doc_path}: {str(e)}")
        
        # Sort by similarity and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in results[:limit]]
    
    def get_document_info(self, doc_path: str) -> Optional[DocumentInfo]:
        """Get information about a specific document."""
        return self.discovered_docs.get(doc_path)
    
    def get_all_documents(self) -> List[DocumentInfo]:
        """Get information about all discovered documents."""
        return list(self.discovered_docs.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about discovered documents."""
        self.logger.info("Gathering document statistics")
        try:
            stats = {
                "total_documents": len(self.discovered_docs),
                "by_type": {},
                "total_size": 0,
                "indexed_count": len(self.knowledge_bases)
            }
            
            # Collect stats by document type
            for doc in self.discovered_docs.values():
                if doc.type not in stats["by_type"]:
                    stats["by_type"][doc.type] = 0
                stats["by_type"][doc.type] += 1
                stats["total_size"] += doc.size
            
            self.logger.info(f"Statistics gathered successfully: {json.dumps(stats, indent=2)}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting stats: {str(e)}", exc_info=True)
            return {}
