from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.knowledge.text import TextKnowledgeBase
from agno.knowledge.csv import CSVKnowledgeBase
from agno.knowledge.json import JSONKnowledgeBase
from agno.knowledge.docx import DocxKnowledgeBase
from agno.knowledge.website import WebsiteKnowledgeBase
from agno.vectordb.pgvector import PgVector
from agno.agent import Agent
from agno.embedder.openai import OpenAIEmbedder
from agno.media import Image
from agno.models.openai import OpenAIChat

import os
import json
import pandas as pd
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import uuid
import re
import traceback
from PIL import Image as PILImage
import logging

from RedisTools import RedisTools
from ContextManager import ContextManager
from KnowledgeGraph import KnowledgeGraph
from QueryProcessor import QueryProcessor
from PersistenceManager import PersistenceManager
from Logger import Logger
from CoordinatorAgent import CoordinatorAgent

class InteractiveRAGSystem:
    def __init__(self, 
                 log_dir: str = "logs",
                 log_level: int = logging.INFO):
        """
        Initialize the RAG system.
        
        Args:
            log_dir: Directory for log files
            log_level: Logging level
        """
        # Set up logging
        self.logger = Logger(
            log_dir=log_dir,
            app_name="rag_system",
            log_level=log_level
        )
        
        self.logger.info("Initializing InteractiveRAGSystem")
        
        # Initialize components with consistent logging
        self.context = ContextManager(
            log_dir=log_dir,
            log_level=log_level
        )
        
        self.knowledge_graph = KnowledgeGraph(
            log_dir=log_dir,
            log_level=log_level
        )
        
        self.persistence = PersistenceManager(
            log_dir=log_dir,
            log_level=log_level
        )
        
        # Initialize CoordinatorAgent
        self.coordinator = CoordinatorAgent(
            log_dir=log_dir,
            log_level=log_level
        )
        
        self.query_processor = QueryProcessor(
            coordinator=self.coordinator,  # Pass the initialized coordinator
            context_manager=self.context,
            log_dir=log_dir,
            log_level=log_level
        )
        
        self.logger.info("InteractiveRAGSystem initialization completed")
        
    def initialize(self, base_path: str):
        """
        Initialize the system with data from the specified path.
        
        Args:
            base_path: Path to the data directory
        """
        self.logger.info(f"Initializing system with data from: {base_path}")
        try:
            # Check if path exists
            if not os.path.exists(base_path):
                self.logger.error(f"Path does not exist: {base_path}")
                return False
                
            # Set knowledge directory in query processor to avoid copying files
            self.query_processor.knowledge_dir = Path(base_path)
            self.logger.info(f"Set knowledge directory to: {base_path}")
                
            # Initialize document discovery with verification
            self.logger.info("Discovering documents...")
            
            # Use the enhanced document verification system
            document_verification = self.query_processor.verify_documents(base_path)
            
            if document_verification["document_count"] > 0:
                self.logger.info(f"Found {document_verification['document_count']} documents in {base_path}")
                
                # Log document types found
                type_summary = ", ".join([
                    f"{doc_type}={count}" 
                    for doc_type, count in document_verification["documents_by_type"].items() 
                    if count > 0
                ])
                self.logger.info(f"Document types: {type_summary}")
                
                # Add documents to the knowledge base - using direct paths to avoid copying
                for doc_type, sample_files in document_verification["sample_files"].items():
                    for rel_path in sample_files:
                        full_path = os.path.join(base_path, rel_path)
                        self.logger.info(f"Adding knowledge source: {full_path}")
                        self.query_processor.add_knowledge_source(full_path)
            else:
                # If verification found issues, log them
                self.logger.warning("No documents were found during verification")
                for issue in document_verification["issues"]:
                    self.logger.warning(f"Document issue: {issue}")
            
            self.logger.info(f"Initialization complete. Discovered {document_verification['document_count']} documents.")
            return True
        except Exception as e:
            self.logger.error(f"Error during initialization: {str(e)}", exc_info=True)
            return False
        
    def chat(self, message: str):
        """
        Process a user message and generate a response.
        
        Args:
            message: User message
            
        Returns:
            Response text
        """
        self.logger.info(f"Processing user message: {message[:50]}...")
        try:
            # 1. Add to conversation history
            self.context.add_conversation_turn(
                user_input=message,
                agent_response="",  # Will be filled later
                context_used={}
            )
            
            # 2. Process query
            query_response = self.query_processor.process_query(message)
            response = query_response.content
            
            # 3. Update conversation with actual response
            # Normally we would update the conversation turn, but for now we'll just add a new one
            last_turn = list(self.context.conversation_history)[-1]
            self.context.add_conversation_turn(
                user_input=last_turn.user_input,
                agent_response=response,
                context_used=query_response.context_used,
                metadata=query_response.metadata
            )
            
            # 4. Update knowledge graph
            self.knowledge_graph.update_from_interaction(message, response)
            
            # 5. Save to persistence layer
            conversation_history = [
                {
                    "role": "user" if i % 2 == 0 else "assistant",
                    "content": turn.user_input if i % 2 == 0 else turn.agent_response,
                    "timestamp": turn.timestamp.isoformat()
                }
                for i, turn in enumerate(self.context.conversation_history)
            ]
            
            self.persistence.save_conversation(
                self.context.session_id or str(uuid.uuid4()),
                conversation_history
            )
            
            self.logger.info("Successfully processed user message")
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}", exc_info=True)
            return f"I'm sorry, but I encountered an error while processing your message: {str(e)}"

    def get_stats(self):
        """Get system statistics."""
        try:
            return {
                "context": self.context.get_stats(),
                "persistence": self.persistence.get_storage_stats(),
                "knowledge_graph": self.knowledge_graph.get_stats(),
                "query_processor": self.query_processor.get_stats()
            }
        except Exception as e:
            self.logger.error(f"Error getting stats: {str(e)}")
            return {"error": str(e)}

if __name__ == "__main__":
    # Set up logging configuration
    log_dir = "logs"
    log_level = logging.INFO
    ### clean up the logs
    if os.path.exists(log_dir) and len(os.listdir(log_dir)) > 0:
        for file in os.listdir(log_dir):
            if file.endswith(".log"):
                os.remove(os.path.join(log_dir, file))
    
    # Initialize RAG system
    rag_system = InteractiveRAGSystem(
        log_dir=log_dir,
        log_level=log_level
    )
    
    # Initialize with data
    username = "admin"
    report_num = "20250324_222749"
    generated_report_path = f"/data/SWATGenXApp/Users/{username}/Reports/{report_num}/"
    rag_system.initialize(base_path=generated_report_path)

    # Interactive chat loop
    print("RAG System initialized. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting RAG System.")
            break
            
        response = rag_system.chat(user_input)
        print(f"AI: {response}")