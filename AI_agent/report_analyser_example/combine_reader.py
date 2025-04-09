from agno.knowledge.combined import CombinedKnowledgeBase
from agno.vectordb.pgvector import PgVector
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.knowledge.text import TextKnowledgeBase
from agno.embedder.openai import OpenAIEmbedder
from agno.agent import Agent
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set OpenAI API key
#os.environ["OPENAI_API_KEY"] = "your-api-key-here"

try:
    # Initialize the embedder
    embedder = OpenAIEmbedder()
    
    # Create knowledge bases for different file types with proper configuration
    pdf_knowledge_base = PDFKnowledgeBase(
        path="/data/SWATGenXApp/codes/reports/groundwater/groundwater_report.html",
        vector_db=PgVector(
            table_name="pdf_documents",
            db_url="postgresql+psycopg://ai:ai@localhost:5432/ai",
        ),
        reader=PDFReader(chunk=True),
        embedder=embedder
    )
    logger.info("PDF knowledge base initialized")

    text_knowledge_base = TextKnowledgeBase(
        path="/data/SWATGenXApp/codes/reports/groundwater/groundwater_stats.csv",
        vector_db=PgVector(
            table_name="text_documents",
            db_url="postgresql+psycopg://ai:ai@localhost:5432/ai",
        ),
        embedder=embedder
    )
    logger.info("Text knowledge base initialized")

    # Combine the knowledge bases with proper configuration
    knowledge_base = CombinedKnowledgeBase(
        sources=[pdf_knowledge_base, text_knowledge_base],
        vector_db=PgVector(
            table_name="combined_documents",
            db_url="postgresql+psycopg://ai:ai@localhost:5432/ai",
        ),
        embedder=embedder
    )
    logger.info("Combined knowledge base created")

    # Create agent with combined knowledge
    agent = Agent(
        knowledge=knowledge_base,
        search_knowledge=True
    )
    logger.info("Agent created with combined knowledge")

    # Load the knowledge base
    agent.knowledge.load(recreate=False)
    logger.info("Knowledge base loaded successfully")

    # Example query
    agent.print_response("Ask me about something from the knowledge base")

except Exception as e:
    logger.error(f"Error in knowledge base setup: {str(e)}")
    raise