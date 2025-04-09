from agno.knowledge.json import JSONKnowledgeBase
from agno.vectordb.pgvector import PgVector
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Create JSON knowledge base
    knowledge_base = JSONKnowledgeBase(
        path="/data/SWATGenXApp/codes/AI_agent/.vscode/settings.json",
        # Table name: ai.json_documents
        vector_db=PgVector(
            table_name="json_documents",
            db_url="postgresql+psycopg://ai:ai@localhost:5432/ai",
        ),
    )
    logger.info("JSON knowledge base initialized")

    # Create agent with JSON knowledge
    from agno.agent import Agent
    agent = Agent(
        knowledge=knowledge_base,
        search_knowledge=True,
    )
    logger.info("Agent created with JSON knowledge")

    # Load the knowledge base
    agent.knowledge.load(recreate=False)
    logger.info("Knowledge base loaded successfully")

    # Example query
    agent.print_response("Ask me about something from the knowledge base")

except Exception as e:
    logger.error(f"Error in JSON knowledge base setup: {str(e)}")
    raise