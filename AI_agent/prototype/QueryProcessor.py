from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import re
import os
import shutil

from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.knowledge.text import TextKnowledgeBase
from agno.knowledge.csv import CSVKnowledgeBase
from agno.knowledge.json import JSONKnowledgeBase
from agno.knowledge.docx import DocxKnowledgeBase
from agno.knowledge.website import WebsiteKnowledgeBase
from agno.knowledge.combined import CombinedKnowledgeBase
from agno.vectordb.pgvector import PgVector

from CoordinatorAgent import CoordinatorAgent, AgentResponse
from ContextManager import ContextManager
from Logger import Logger
from AgentKnowledge import AgentKnowledge, KnowledgeBaseAdapter
from VectorStore import VectorStore
from VectorDbAdapter import VectorDbAdapter

@dataclass
class ProcessedQuery:
    """Represents a processed query with its components and metadata."""
    query: str
    enhanced_query: Optional[str]
    context: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class QueryResponse:
    """Represents the final response to a query."""
    content: str
    source_responses: List[Dict[str, Any]]
    context_used: Dict[str, Any]
    confidence: float
    processing_time: float
    metadata: Dict[str, Any]

class QueryProcessor:
    """Processes user queries and coordinates responses from various components."""
    
    def __init__(self,
                 coordinator: Optional[CoordinatorAgent] = None,
                 context_manager: Optional[ContextManager] = None,
                 knowledge_dir: str = "/data/SWATGenXApp/Users/admin/Reports/20250324_222749",
                 vector_store_url: str = "postgresql://ai:ai@localhost:5432/ai",
                 log_dir: str = "logs",
                 log_level: int = 20):  # Default to INFO level (20)
        """
        Initialize the query processor.
        
        Args:
            coordinator: Optional CoordinatorAgent instance
            context_manager: Optional ContextManager instance
            knowledge_dir: Directory containing knowledge files
            vector_store_url: URL for vector store connection
            log_dir: Directory for log files
            log_level: Logging level (default: logging.INFO)
        """
        # Set up logging
        self.logger = Logger(
            log_dir=log_dir,
            app_name="query_processor",
            log_level=log_level
        )
        
        self.logger.info("Initializing QueryProcessor")
        
        # Initialize components
        self.coordinator = coordinator
        self.context_manager = context_manager
        self.knowledge_dir = Path(knowledge_dir)
        
        # Initialize vector database with adapter for agno compatibility
        self.vector_store = VectorStore(
            connection_string=vector_store_url,
            table_name="agent_knowledge",
            log_dir=log_dir,
            log_level=log_level,
            prefer_local_embeddings=True  # Always prefer local embeddings to avoid API quota issues
        )
        self.vector_db = VectorDbAdapter(
            connection_string=vector_store_url,
            table_name="agent_knowledge",
            log_dir=log_dir,
            log_level=log_level
        )
        
        # Initialize knowledge bases
        self.knowledge_bases = self._initialize_knowledge_bases()
        
        # Create combined knowledge base with adapter
        kb = CombinedKnowledgeBase(
            sources=list(self.knowledge_bases.values()),
            vector_db=self.vector_db
        )
        # Wrap with adapter for flexible parameter handling
        self.knowledge_base = KnowledgeBaseAdapter(kb)
        
        # Query enhancement patterns
        self.enhancement_patterns = {
            "clarification": [
                r"what .* mean",
                r"explain .*",
                r"define .*"
            ],
            "comparison": [
                r"compare .*",
                r"difference between .*",
                r"vs .*"
            ],
            "analysis": [
                r"analyze .*",
                r"evaluate .*",
                r"assess .*"
            ]
        }
        
        self.logger.info("QueryProcessor initialization completed")

    def _initialize_knowledge_bases(self) -> Dict[str, Any]:
        """Initialize knowledge bases for different file types."""
        self.logger.info("Initializing knowledge bases")
        knowledge_bases = {}
        
        # Document discovery statistics
        discovery_stats = {
            "pdf": 0,
            "text": 0,
            "md": 0,
            "csv": 0,
            "json": 0,
            "docx": 0,
            "total": 0,
            "loaded": 0,
            "failed": 0
        }
        
        try:
            # Convert knowledge_dir to Path if it's a string
            if isinstance(self.knowledge_dir, str):
                self.knowledge_dir = Path(self.knowledge_dir)
            
            self.logger.info(f"Searching for documents in: {self.knowledge_dir}")
            
            # Check knowledge directory for files
            if self.knowledge_dir.exists():
                # Initialize PDF knowledge bases
                pdf_files = list(self.knowledge_dir.rglob("*.pdf"))
                discovery_stats["pdf"] = len(pdf_files)
                discovery_stats["total"] += len(pdf_files)
                self.logger.info(f"Discovered {len(pdf_files)} PDF files")
                for pdf_file in pdf_files:
                    kb_id = f"pdf_{pdf_file.name}"
                    self.logger.debug(f"Initializing PDF knowledge base for: {pdf_file}")
                    kb = PDFKnowledgeBase(
                        path=str(pdf_file),
                        vector_db=self.vector_db,
                        reader=PDFReader(chunk=True),
                        id=kb_id
                    )
                    knowledge_bases[kb_id] = kb
                
                # Initialize text knowledge bases (including markdown)
                txt_files = list(self.knowledge_dir.rglob("*.txt"))
                md_files = list(self.knowledge_dir.rglob("*.md"))
                discovery_stats["text"] = len(txt_files)
                discovery_stats["md"] = len(md_files)
                discovery_stats["total"] += len(txt_files) + len(md_files)
                self.logger.info(f"Discovered {len(txt_files)} text files and {len(md_files)} markdown files")
                
                text_files = txt_files + md_files
                for text_file in text_files:
                    kb_id = f"text_{text_file.name}"
                    self.logger.debug(f"Initializing text knowledge base for: {text_file}")
                    kb = TextKnowledgeBase(
                        path=str(text_file),
                        vector_db=self.vector_db,
                        id=kb_id
                    )
                    knowledge_bases[kb_id] = kb
                
                # Initialize CSV knowledge bases
                csv_files = list(self.knowledge_dir.rglob("*.csv"))
                discovery_stats["csv"] = len(csv_files)
                discovery_stats["total"] += len(csv_files)
                self.logger.info(f"Discovered {len(csv_files)} CSV files")
                for csv_file in csv_files:
                    kb_id = f"csv_{csv_file.name}"
                    self.logger.debug(f"Initializing CSV knowledge base for: {csv_file}")
                    kb = CSVKnowledgeBase(
                        path=str(csv_file),
                        vector_db=self.vector_db,
                        id=kb_id
                    )
                    knowledge_bases[kb_id] = kb
                
                # Initialize JSON knowledge bases
                json_files = list(self.knowledge_dir.rglob("*.json"))
                discovery_stats["json"] = len(json_files)
                discovery_stats["total"] += len(json_files)
                self.logger.info(f"Discovered {len(json_files)} JSON files")
                for json_file in json_files:
                    kb_id = f"json_{json_file.name}"
                    self.logger.debug(f"Initializing JSON knowledge base for: {json_file}")
                    kb = JSONKnowledgeBase(
                        path=str(json_file),
                        vector_db=self.vector_db,
                        id=kb_id
                    )
                    knowledge_bases[kb_id] = kb
                
                # Initialize DOCX knowledge bases
                docx_files = list(self.knowledge_dir.rglob("*.docx"))
                discovery_stats["docx"] = len(docx_files)
                discovery_stats["total"] += len(docx_files)
                self.logger.info(f"Discovered {len(docx_files)} DOCX files")
                for docx_file in docx_files:
                    kb_id = f"docx_{docx_file.name}"
                    self.logger.debug(f"Initializing DOCX knowledge base for: {docx_file}")
                    kb = DocxKnowledgeBase(
                        path=str(docx_file),
                        vector_db=self.vector_db,
                        id=kb_id
                    )
                    knowledge_bases[kb_id] = kb
                
                # Load all knowledge bases
                self.logger.info(f"Loading {len(knowledge_bases)} knowledge bases")
                for kb_id, kb in knowledge_bases.items():
                    try:
                        self.logger.debug(f"Loading knowledge base: {kb_id}")
                        kb.load(recreate=False)
                        discovery_stats["loaded"] += 1
                    except Exception as e:
                        self.logger.error(f"Error loading knowledge base {kb_id}: {str(e)}")
                        discovery_stats["failed"] += 1
                
                # Verify document discovery and loading
                self._verify_document_discovery(discovery_stats)
            else:
                self.logger.error(f"Knowledge directory not found: {self.knowledge_dir}")
            
            self.logger.info(f"Initialized {len(knowledge_bases)} knowledge bases")
            return knowledge_bases
            
        except Exception as e:
            self.logger.error(f"Error initializing knowledge bases: {str(e)}", exc_info=True)
            return {}
    
    def _verify_document_discovery(self, stats: Dict[str, int]) -> None:
        """
        Verify document discovery process and log detailed statistics.
        
        Args:
            stats: Dictionary containing document discovery statistics
        """
        self.logger.info("--- Document Discovery Verification ---")
        self.logger.info(f"Total documents discovered: {stats['total']}")
        self.logger.info(f"Documents by type: PDF={stats['pdf']}, Text={stats['text']}, Markdown={stats['md']}, CSV={stats['csv']}, JSON={stats['json']}, DOCX={stats['docx']}")
        self.logger.info(f"Knowledge bases loaded: {stats['loaded']}/{stats['total']}")
        
        if stats['failed'] > 0:
            self.logger.warning(f"Failed to load {stats['failed']} knowledge bases")
        
        if stats['total'] == 0:
            self.logger.error("No documents were discovered. Possible issues:")
            self.logger.error(f"1. Knowledge directory may be empty: {self.knowledge_dir}")
            self.logger.error("2. Permissions issues accessing the directory")
            self.logger.error("3. File types might not match the expected extensions")
            self.logger.error("Please check the knowledge directory and file permissions")
        
        if stats['loaded'] == 0 and stats['total'] > 0:
            self.logger.error("No knowledge bases were successfully loaded despite documents being discovered")
            self.logger.error("Please check the log for specific loading errors")
        
        # Log success rate
        success_rate = (stats['loaded'] / stats['total']) * 100 if stats['total'] > 0 else 0
        self.logger.info(f"Document loading success rate: {success_rate:.1f}%")
        self.logger.info("-----------------------------------")

    def load_knowledge_bases(self, recreate: bool = False) -> bool:
        """
        Load or reload all knowledge bases.
        
        Args:
            recreate: Whether to recreate the knowledge bases
            
        Returns:
            bool: Success status
        """
        self.logger.info(f"Loading knowledge bases (recreate={recreate})")
        success_count = 0
        total_count = len(self.knowledge_bases)
        
        try:
            for kb_id, kb in self.knowledge_bases.items():
                try:
                    self.logger.debug(f"Loading knowledge base: {kb_id}")
                    kb.load(recreate=recreate)
                    success_count += 1
                except Exception as e:
                    self.logger.error(f"Error loading knowledge base {kb_id}: {str(e)}", exc_info=True)
            
            # Recreate combined knowledge base with adapter
            combined_kb = CombinedKnowledgeBase(
                sources=list(self.knowledge_bases.values()),
                vector_db=self.vector_db
            )
            # Wrap with adapter for flexible parameter handling
            self.knowledge_base = KnowledgeBaseAdapter(combined_kb)
            
            self.logger.info(f"Successfully loaded {success_count}/{total_count} knowledge bases")
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"Error loading knowledge bases: {str(e)}", exc_info=True)
            return False

    def add_knowledge_source(self, path: str) -> bool:
        """
        Add a knowledge source from a file path.
        
        Args:
            path: Path to the file
            
        Returns:
            bool: Success status
        """
        try:
            file_path = Path(path)
            if not file_path.exists():
                self.logger.error(f"File not found: {path}")
                return False
                
            # Create knowledge directory if it doesn't exist
            if not self.knowledge_dir.exists():
                self.knowledge_dir.mkdir(parents=True, exist_ok=True)
                
            # Create a copy of the file in the knowledge directory
            destination = self.knowledge_dir / file_path.name
            
            # Check if source and destination are different before copying
            if file_path.resolve() != destination.resolve():
                shutil.copy2(file_path, destination)
            else:
                self.logger.debug(f"Source and destination are the same file: {file_path}. Skipping copy.")
            
            # Determine the type of file and create appropriate knowledge base
            if file_path.suffix.lower() == '.pdf':
                kb_id = f"pdf_{file_path.name}"
                kb = PDFKnowledgeBase(
                    path=str(destination),
                    vector_db=self.vector_db,
                    reader=PDFReader(chunk=True),
                    id=kb_id
                )
            elif file_path.suffix.lower() in ['.txt', '.md']:
                kb_id = f"text_{file_path.name}"
                kb = TextKnowledgeBase(
                    path=str(destination),
                    vector_db=self.vector_db,
                    id=kb_id
                )
            elif file_path.suffix.lower() == '.csv':
                kb_id = f"csv_{file_path.name}"
                kb = CSVKnowledgeBase(
                    path=str(destination),
                    vector_db=self.vector_db,
                    id=kb_id
                )
            elif file_path.suffix.lower() == '.json':
                kb_id = f"json_{file_path.name}"
                kb = JSONKnowledgeBase(
                    path=str(destination),
                    vector_db=self.vector_db,
                    id=kb_id
                )
            elif file_path.suffix.lower() == '.docx':
                kb_id = f"docx_{file_path.name}"
                kb = DocxKnowledgeBase(
                    path=str(destination),
                    vector_db=self.vector_db,
                    id=kb_id
                )
            else:
                self.logger.error(f"Unsupported file type: {file_path.suffix}")
                return False
            
            if kb and kb_id:
                # Load the knowledge base
                kb.load(recreate=True)
                
                # Add to knowledge bases
                self.knowledge_bases[kb_id] = kb
                
                # Recreate combined knowledge base with adapter
                combined_kb = CombinedKnowledgeBase(
                    sources=list(self.knowledge_bases.values()),
                    vector_db=self.vector_db
                )
                # Wrap with adapter for flexible parameter handling
                self.knowledge_base = KnowledgeBaseAdapter(combined_kb)
                
                self.logger.info(f"Successfully added knowledge source: {path}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error adding knowledge source: {str(e)}", exc_info=True)
            return False

    def process_query(self,
                     query: str,
                     context: Optional[Dict[str, Any]] = None) -> QueryResponse:
        """
        Process a user query and generate a comprehensive response.
        
        Args:
            query: The user's query
            context: Optional additional context
            
        Returns:
            QueryResponse containing the final response and metadata
        """
        start_time = datetime.now()
        self.logger.info(f"Processing query: {query}")
        
        try:
            # 1. Preprocess and enhance query
            processed_query = self._preprocess_query(query, context)
            self.logger.debug(f"Preprocessed query: {processed_query.enhanced_query}")
            
            # 2. Get relevant conversation history
            relevant_history = self._get_relevant_history(processed_query)
            self.logger.debug(f"Found {len(relevant_history)} relevant history entries")
            
            # 3. Search knowledge base
            kb_results = self._search_knowledge_base(processed_query)
            self.logger.debug(f"Found {len(kb_results)} knowledge base results")
            
            # 4. Route to appropriate agent(s)
            agent_response = self.coordinator.route_query(
                processed_query.enhanced_query or query,
                context=processed_query.context
            )
            self.logger.debug("Received agent response")
            
            # 5. Combine and enhance response
            final_response = self._enhance_response(
                processed_query,
                relevant_history,
                kb_results,
                agent_response.content
            )
            
            # 6. Update context
            self._update_context(processed_query, final_response)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Query processing completed in {processing_time:.2f} seconds")
            
            return QueryResponse(
                content=final_response,
                source_responses=[
                    {"type": "history", "items": relevant_history},
                    {"type": "knowledge_base", "items": kb_results},
                    {"type": "agent", "response": agent_response}
                ],
                context_used=processed_query.context,
                confidence=agent_response.confidence,
                processing_time=processing_time,
                metadata={
                    "query_type": agent_response.metadata.get("query_type"),
                    "enhanced": bool(processed_query.enhanced_query),
                    "sources_used": len(relevant_history) + len(kb_results) + 1
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return QueryResponse(
                content="I apologize, but I encountered an error processing your request.",
                source_responses=[],
                context_used={},
                confidence=0.0,
                processing_time=(datetime.now() - start_time).total_seconds(),
                metadata={"error": str(e)}
            )

    def _preprocess_query(self,
                         query: str,
                         context: Optional[Dict[str, Any]] = None) -> ProcessedQuery:
        """Preprocess and enhance the query."""
        self.logger.debug("Preprocessing query")
        try:
            # Clean query
            cleaned_query = query.strip()
            
            # Detect query patterns
            patterns_matched = []
            for pattern_type, patterns in self.enhancement_patterns.items():
                if any(re.search(p, cleaned_query, re.IGNORECASE) for p in patterns):
                    patterns_matched.append(pattern_type)
            
            # Enhance query if needed
            enhanced_query = None
            if patterns_matched:
                if "clarification" in patterns_matched:
                    enhanced_query = f"Provide a detailed explanation of: {cleaned_query}"
                elif "comparison" in patterns_matched:
                    enhanced_query = f"Compare and contrast in detail: {cleaned_query}"
                elif "analysis" in patterns_matched:
                    enhanced_query = f"Provide a comprehensive analysis of: {cleaned_query}"
            
            # Build context
            query_context = {
                "patterns_matched": patterns_matched,
                "timestamp": datetime.now().isoformat()
            }
            if context:
                query_context.update(context)
            
            processed = ProcessedQuery(
                query=cleaned_query,
                enhanced_query=enhanced_query,
                context=query_context,
                timestamp=datetime.now(),
                metadata={"patterns_matched": patterns_matched}
            )
            
            self.logger.debug(f"Query preprocessing completed: {json.dumps(processed.metadata)}")
            return processed
            
        except Exception as e:
            self.logger.error(f"Error preprocessing query: {str(e)}", exc_info=True)
            return ProcessedQuery(
                query=query,
                enhanced_query=None,
                context=context or {},
                timestamp=datetime.now(),
                metadata={"error": str(e)}
            )

    def _get_relevant_history(self, processed_query: ProcessedQuery) -> List[Dict[str, Any]]:
        """Get relevant conversation history."""
        self.logger.debug("Retrieving relevant conversation history")
        try:
            # Get recent history
            history = self.context_manager.get_conversation_history(limit=5)
            
            # Filter and sort by relevance
            relevant = []
            for turn in history:
                relevance = self._calculate_relevance(
                    processed_query.query,
                    turn.user_input + " " + turn.agent_response
                )
                if relevance > 0.3:  # Relevance threshold
                    relevant.append({
                        "turn": turn,
                        "relevance": relevance
                    })
            
            # Sort by relevance
            relevant.sort(key=lambda x: x["relevance"], reverse=True)
            
            self.logger.debug(f"Found {len(relevant)} relevant history entries")
            return relevant
            
        except Exception as e:
            self.logger.error(f"Error retrieving history: {str(e)}", exc_info=True)
            return []

    def _search_knowledge_base(self, processed_query: ProcessedQuery) -> List[Dict[str, Any]]:
        """Search the knowledge base for relevant information."""
        self.logger.debug("Searching knowledge base")
        try:
            # Search using both original and enhanced queries
            queries = [processed_query.query]
            if processed_query.enhanced_query:
                queries.append(processed_query.enhanced_query)
            
            results = []
            for query in queries:
                try:
                    # Check if the knowledge base has been loaded
                    if hasattr(self.knowledge_base, 'search'):
                        # Use the adapter's search method which handles different parameter formats
                        kb_results = self.knowledge_base.search(
                            query=query,
                            k=5,
                            threshold=0.3
                        )
                        if isinstance(kb_results, list):
                            results.extend(kb_results)
                    else:
                        self.logger.warning("Knowledge base search method not available")
                except Exception as e:
                    self.logger.error(f"Error in knowledge base search: {str(e)}", exc_info=True)
            
            # Remove duplicates and sort by relevance
            unique_results = {}
            for result in results:
                if isinstance(result, dict) and result.get("id") and result["id"] not in unique_results:
                    unique_results[result["id"]] = result
            
            # Sort by similarity
            sorted_results = sorted(
                unique_results.values(),
                key=lambda x: x.get("similarity", 0),
                reverse=True
            )
            
            self.logger.debug(f"Found {len(sorted_results)} knowledge base results")
            return sorted_results[:5]  # Return top 5 unique results
            
        except Exception as e:
            self.logger.error(f"Error searching knowledge base: {str(e)}", exc_info=True)
            return []

    def _enhance_response(self,
                         processed_query: ProcessedQuery,
                         relevant_history: List[Dict[str, Any]],
                         kb_results: List[Dict[str, Any]],
                         agent_response: str) -> str:
        """Enhance the response with context and knowledge base results."""
        try:
            content_parts = []
            
            # Add knowledge base results if available
            if kb_results:
                kb_content = [str(result.get('content', '')) for result in kb_results if result.get('content')]
                if kb_content:
                    content_parts.extend(kb_content)
            
            # Add agent response if available
            if agent_response:
                content_parts.append(str(agent_response))
            
            # If no content, return a default message
            if not content_parts:
                return "I don't have enough information to provide a helpful response."
            
            # Generate enhanced response
            enhanced_response = "\n\n".join(filter(None, content_parts))
            
            # Check if there's an agent response to use as a base
            if agent_response:
                # Use agent response as base and add context from KB
                if len(kb_results) > 0:
                    # Add citation or context
                    sources = [result.get('source', 'Unknown Source') for result in kb_results 
                              if result.get('source')]
                    if sources:
                        unique_sources = list(set(sources))
                        if len(unique_sources) <= 3:  # Only add sources if there are 3 or fewer
                            source_text = ", ".join(unique_sources)
                            enhanced_response += f"\n\nInformation sourced from: {source_text}"
            
            self.logger.debug(f"Enhanced response from {len(content_parts)} parts")
            return enhanced_response
            
        except Exception as e:
            self.logger.error(f"Error enhancing response: {str(e)}", exc_info=True)
            # Return original response if enhancement fails
            return agent_response if agent_response else "I encountered an error while processing your query."

    def _update_context(self,
                       processed_query: ProcessedQuery,
                       response: str):
        """Update conversation context with the latest interaction."""
        self.logger.debug("Updating conversation context")
        try:
            # Add conversation turn
            self.context_manager.add_conversation_turn(
                user_input=processed_query.query,
                agent_response=response,
                context_used=processed_query.context,
                metadata={
                    "enhanced_query": processed_query.enhanced_query,
                    "confidence": 0.9,  # Assuming a default confidence
                    "response_metadata": {}
                }
            )
            
            # Update state if needed
            state_updates = {}
            if "query_type" in processed_query.metadata:
                state_updates["last_query_type"] = processed_query.metadata["query_type"]
            if processed_query.metadata.get("patterns_matched"):
                state_updates["last_patterns_matched"] = processed_query.metadata["patterns_matched"]
            
            if state_updates:
                self.context_manager.update_state(state_updates)
            
            self.logger.debug("Context updated successfully")
            
        except Exception as e:
            self.logger.error(f"Error updating context: {str(e)}", exc_info=True)

    def _calculate_relevance(self, query: str, content: str) -> float:
        """Calculate relevance score between query and content."""
        try:
            # Convert to lowercase for comparison
            query = query.lower()
            content = content.lower()
            
            # Calculate word overlap
            query_words = set(query.split())
            content_words = set(content.split())
            
            overlap = len(query_words.intersection(content_words))
            total = len(query_words.union(content_words))
            
            return overlap / total if total > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error calculating relevance: {str(e)}", exc_info=True)
            return 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the query processor's state."""
        self.logger.info("Gathering query processor statistics")
        try:
            stats = {
                "coordinator_stats": self.coordinator.get_metrics(),
                "context_stats": self.context_manager.get_stats(),
                "knowledge_bases": {
                    "total": len(self.knowledge_bases),
                    "types": self._get_knowledge_base_types()
                }
            }
            
            self.logger.debug(f"Current stats: {json.dumps(stats, indent=2)}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting stats: {str(e)}", exc_info=True)
            return {}
    
    def _get_knowledge_base_types(self) -> Dict[str, int]:
        """Get counts of knowledge base types."""
        type_counts = {}
        for kb_id in self.knowledge_bases:
            kb_type = kb_id.split('_')[0]  # Extract type from ID (pdf_file.pdf -> pdf)
            if kb_type not in type_counts:
                type_counts[kb_type] = 0
            type_counts[kb_type] += 1
        return type_counts

    def verify_documents(self, directory: Optional[str] = None) -> Dict[str, Any]:
        """
        Verify documents in a specified directory or the default knowledge directory.
        
        Args:
            directory: Optional directory path to verify
            
        Returns:
            Dictionary with verification results
        """
        verify_dir = Path(directory) if directory else self.knowledge_dir
        self.logger.info(f"Verifying documents in: {verify_dir}")
        
        verification_results = {
            "directory": str(verify_dir),
            "exists": False,
            "readable": False,
            "document_count": 0,
            "documents_by_type": {},
            "sample_files": {},
            "issues": []
        }
        
        try:
            # Check directory existence
            if not verify_dir.exists():
                verification_results["issues"].append(f"Directory does not exist: {verify_dir}")
                return verification_results
            
            verification_results["exists"] = True
            
            # Check directory permissions
            try:
                next(verify_dir.iterdir(), None)
                verification_results["readable"] = True
            except PermissionError:
                verification_results["issues"].append(f"Permission denied: Cannot read directory {verify_dir}")
                return verification_results
            
            # Count documents by type
            extensions = {
                "pdf": "*.pdf",
                "txt": "*.txt",
                "md": "*.md",
                "csv": "*.csv",
                "json": "*.json",
                "docx": "*.docx"
            }
            
            for ext_name, ext_pattern in extensions.items():
                files = list(verify_dir.rglob(ext_pattern))
                file_count = len(files)
                verification_results["documents_by_type"][ext_name] = file_count
                verification_results["document_count"] += file_count
                
                # Add sample files (up to 3) for each type
                if files:
                    verification_results["sample_files"][ext_name] = [
                        str(f.relative_to(verify_dir)) for f in files[:3]
                    ]
            
            # Check for potential issues
            if verification_results["document_count"] == 0:
                verification_results["issues"].append(f"No supported documents found in {verify_dir}")
                
                # Check if there are any files at all
                any_files = list(verify_dir.rglob("*.*"))
                if any_files:
                    file_extensions = set(f.suffix.lower() for f in any_files if f.suffix)
                    verification_results["issues"].append(
                        f"Directory contains files with unsupported extensions: {', '.join(file_extensions)}"
                    )
            
            self.logger.info(f"Document verification complete: {verification_results['document_count']} documents found")
            return verification_results
            
        except Exception as e:
            self.logger.error(f"Error verifying documents: {str(e)}", exc_info=True)
            verification_results["issues"].append(f"Error during verification: {str(e)}")
            return verification_results