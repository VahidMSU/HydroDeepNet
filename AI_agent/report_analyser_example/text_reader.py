from agno.knowledge.text import TextKnowledgeBase
from agno.vectordb.pgvector import PgVector
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Create text knowledge base
    knowledge_base = TextKnowledgeBase(
        path="/data/SWATGenXApp/codes/reports/index.html",
        # Table name: ai.text_documents
        vector_db=PgVector(
            table_name="text_documents",
            db_url="postgresql+psycopg://ai:ai@localhost:5432/ai",
        ),
    )
    logger.info("Text knowledge base initialized")

    # Create agent with text knowledge
    from agno.agent import Agent
    agent = Agent(
        knowledge=knowledge_base,
        search_knowledge=True,
    )
    logger.info("Agent created with text knowledge")

    # Load the knowledge base
    agent.knowledge.load(recreate=False)
    logger.info("Knowledge base loaded successfully")

    # Example query
    agent.print_response("Ask me about something from the knowledge base")

except Exception as e:
    logger.error(f"Error in text knowledge base setup: {str(e)}")
    raise