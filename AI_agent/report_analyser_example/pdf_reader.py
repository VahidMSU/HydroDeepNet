from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.vectordb.pgvector import PgVector
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Create PDF knowledge base
    pdf_knowledge_base = PDFKnowledgeBase(
        path="/data/SWATGenXApp/codes/PrivacyTermsOfUse.pdf",
        # Table name: ai.pdf_documents
        vector_db=PgVector(
            table_name="pdf_documents",
            db_url="postgresql+psycopg://ai:ai@localhost:5432/ai",
        ),
        reader=PDFReader(chunk=True),
    )
    logger.info("PDF knowledge base initialized")

    # Create agent with PDF knowledge
    from agno.agent import Agent
    agent = Agent(
        knowledge=pdf_knowledge_base,
        search_knowledge=True,
    )
    logger.info("Agent created with PDF knowledge")

    # Load the knowledge base
    agent.knowledge.load(recreate=False)
    logger.info("Knowledge base loaded successfully")

    # Example query
    agent.print_response("Ask me about something from the knowledge base")

except Exception as e:
    logger.error(f"Error in PDF knowledge base setup: {str(e)}")
    raise