from agno.knowledge.website import WebsiteKnowledgeBase
from agno.vectordb.pgvector import PgVector
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Create website knowledge base
    knowledge_base = WebsiteKnowledgeBase(
        urls=["https://docs.agno.com/introduction"],
        max_links=10,
        vector_db=PgVector(
            table_name="website_documents",
            db_url="postgresql+psycopg://ai:ai@localhost:5432/ai",
        )
    )
    logger.info("Website knowledge base initialized")

    # Create agent with website knowledge
    from agno.agent import Agent
    agent = Agent(
        knowledge=knowledge_base,
        search_knowledge=True
    )
    logger.info("Agent created with website knowledge")

    # Load the knowledge base
    agent.knowledge.load(recreate=False)
    logger.info("Knowledge base loaded successfully")

    # Example query
    agent.print_response("Ask me about something from the knowledge base")

except Exception as e:
    logger.error(f"Error in website knowledge base setup: {str(e)}")
    raise