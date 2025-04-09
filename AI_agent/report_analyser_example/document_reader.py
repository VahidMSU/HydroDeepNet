from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.knowledge.text import TextKnowledgeBase
from agno.knowledge.csv import CSVKnowledgeBase
from agno.knowledge.json import JSONKnowledgeBase
from agno.knowledge.docx import DocxKnowledgeBase
from agno.knowledge.website import WebsiteKnowledgeBase
from agno.vectordb.pgvector import PgVector
from agno.agent import Agent
from agno.embedder.openai import OpenAIEmbedder
import logging
import os
from agno.knowledge.combined import CombinedKnowledgeBase


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentReader:
    """A unified reader for different types of documents."""
    
    def __init__(self, config):
        """
        Initialize the document reader.
        
        Args:
            config: Dictionary containing configuration for different document types and database
        """
        self.config = config
        self.db_url = config.get('db_url', "postgresql+psycopg://ai:ai@localhost:5432/ai")
        self.knowledge_bases = {}
        self.combined_knowledge_base = None
        self.agent = None
        self.embedder = OpenAIEmbedder()
        
    def initialize(self):
        """Initialize individual knowledge bases and create combined knowledge base."""
        try:
            # Initialize individual knowledge bases
            if 'pdf' in self.config and self.config['pdf'].get('path'):
                self._initialize_knowledge_base('pdf', self.config['pdf'])
                
            if 'text' in self.config and self.config['text'].get('path'):
                self._initialize_knowledge_base('text', self.config['text'])
                
            if 'csv' in self.config and self.config['csv'].get('path'):
                self._initialize_knowledge_base('csv', self.config['csv'])
                
            if 'json' in self.config and self.config['json'].get('path'):
                self._initialize_knowledge_base('json', self.config['json'])
                
            if 'docx' in self.config and self.config['docx'].get('path'):
                self._initialize_knowledge_base('docx', self.config['docx'])
                
            if 'website' in self.config and self.config['website'].get('url'):
                self._initialize_knowledge_base('website', self.config['website'])
            
            # Create combined knowledge base if multiple knowledge bases exist
            if len(self.knowledge_bases) > 1:
                self._create_combined_knowledge_base()
            elif len(self.knowledge_bases) == 1:
                # Use the single knowledge base if only one was initialized
                doc_type = list(self.knowledge_bases.keys())[0]
                self.agent = Agent(
                    knowledge=self.knowledge_bases[doc_type],
                    search_knowledge=True
                )
            else:
                logger.warning("No knowledge bases were initialized")
                
            return True
            
        except Exception as e:
            logger.error(f"Error initializing knowledge bases: {str(e)}")
            raise
    
    def _initialize_knowledge_base(self, doc_type, config):
        """Initialize a specific knowledge base based on document type."""
        try:
            table_name = config.get('table_name', f"{doc_type}_documents")
            vector_db = PgVector(
                table_name=table_name,
                db_url=self.db_url,
            )
            
            if doc_type == 'pdf':
                self.knowledge_bases[doc_type] = PDFKnowledgeBase(
                    path=config['path'],
                    vector_db=vector_db,
                    reader=PDFReader(chunk=True),
                    embedder=self.embedder
                )
            elif doc_type == 'text':
                self.knowledge_bases[doc_type] = TextKnowledgeBase(
                    path=config['path'],
                    vector_db=vector_db,
                    embedder=self.embedder
                )
            elif doc_type == 'csv':
                self.knowledge_bases[doc_type] = CSVKnowledgeBase(
                    path=config['path'],
                    vector_db=vector_db,
                    embedder=self.embedder
                )
            elif doc_type == 'json':
                self.knowledge_bases[doc_type] = JSONKnowledgeBase(
                    path=config['path'],
                    vector_db=vector_db,
                    embedder=self.embedder
                )
            elif doc_type == 'docx':
                self.knowledge_bases[doc_type] = DocxKnowledgeBase(
                    path=config['path'],
                    vector_db=vector_db,
                    embedder=self.embedder
                )
            elif doc_type == 'website':
                self.knowledge_bases[doc_type] = WebsiteKnowledgeBase(
                    urls=[config['url']],
                    max_links=config.get('max_links', 10),
                    vector_db=vector_db,
                    embedder=self.embedder
                )
                
            self.knowledge_bases[doc_type].load(recreate=False)
            logger.info(f"{doc_type.upper()} knowledge base initialized and loaded")
            
        except Exception as e:
            logger.error(f"Error initializing {doc_type} knowledge base: {str(e)}")
            raise

    def _create_combined_knowledge_base(self):
        """Create a combined knowledge base from all initialized knowledge bases."""
        try:
            combined_table = self.config.get('combined_table_name', 'combined_documents')
            vector_db = PgVector(
                table_name=combined_table,
                db_url=self.db_url,
            )
            
            self.combined_knowledge_base = CombinedKnowledgeBase(
                sources=list(self.knowledge_bases.values()),
                vector_db=vector_db,
                embedder=self.embedder
            )
            
            self.agent = Agent(
                knowledge=self.combined_knowledge_base,
                search_knowledge=True
            )
            
            self.combined_knowledge_base.load(recreate=False)
            logger.info("Combined knowledge base created and loaded")
            
        except Exception as e:
            logger.error(f"Error creating combined knowledge base: {str(e)}")
            raise
            
    def query(self, question):
        """Query the knowledge base."""
        if not self.agent:
            raise RuntimeError("Knowledge base not initialized. Call initialize() first.")
            
        try:
            return self.agent.print_response(question)
        except Exception as e:
            logger.error(f"Error querying knowledge base: {str(e)}")
            raise

def agno_reader(args):
    """Example usage of the DocumentReader."""
    # Prepare configuration for the document reader
    config = {
        'db_url': "postgresql+psycopg://ai:ai@localhost:5432/ai",
        'combined_table_name': 'all_documents',
    }
    
    # Add configurations for each document type
    if args.get('pdf_path'):
        config['pdf'] = {
            'path': args['pdf_path'],
            'table_name': args['pdf_table_name']
        }
    
    if args.get('csv_path'):
        config['csv'] = {
            'path': args['csv_path'],
            'table_name': args['csv_table_name']
        }
    
    if args.get('website_url'):
        config['website'] = {
            'url': args['website_url'],
            'table_name': args['website_table_name']
        }
    
    if args.get('docx_path'):
        config['docx'] = {
            'path': args['docx_path'],
            'table_name': args['docx_table_name']
        }
    
    if args.get('json_path'):
        config['json'] = {
            'path': args['json_path'],
            'table_name': args['json_table_name']
        }
    
    # Initialize document reader with the config
    reader = DocumentReader(config)
    reader.initialize()
    
    # Example query using the combined knowledge base
    reader.query("Tell me about the information from all documents")

if __name__ == "__main__":
    generated_report_path = f"/data/SWATGenXApp/codes/reports/"  # example path to the generated report
    args = {
    "pdf_path": "/data/SWATGenXApp/codes/PrivacyTermsOfUse.pdf",
    "pdf_table_name": "pdf_documents",
    "csv_path": f"{generated_report_path}/groundwater/groundwater_stats.csv",
    "csv_table_name": "csv_documents",
    "website_url": "https://docs.agno.com/introduction",
    "website_table_name": "website_documents",
    "docx_path": f"{generated_report_path}/groundwater/groundwater_stats.docx",
    "docx_table_name": "docx_documents",
    "json_path": f"{generated_report_path}/groundwater/groundwater_stats.json",
    "json_table_name": "json_documents"
    }

    agno_reader(args)
