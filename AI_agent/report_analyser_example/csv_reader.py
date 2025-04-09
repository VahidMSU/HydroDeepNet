from agno.knowledge.csv import CSVKnowledgeBase
from agno.vectordb.pgvector import PgVector
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Create CSV knowledge base
    knowledge_base = CSVKnowledgeBase(
        path="/data/SWATGenXApp/codes/reports/groundwater/groundwater_stats.csv",
        # Table name: ai.csv_documents
        vector_db=PgVector(
            table_name="csv_documents",
            db_url="postgresql+psycopg://ai:ai@localhost:5432/ai",
        )
    )
    logger.info("CSV knowledge base initialized")

    # Create agent with CSV knowledge
    from agno.agent import Agent
    agent = Agent(
        knowledge=knowledge_base,
        search_knowledge=True,
    )
    logger.info("Agent created with CSV knowledge")

    # Load the knowledge base
    agent.knowledge.load(recreate=False)
    logger.info("Knowledge base loaded successfully")

    # Example query
    agent.print_response("Ask me about something from the knowledge base")

except Exception as e:
    logger.error(f"Error in CSV knowledge base setup: {str(e)}")
    raise