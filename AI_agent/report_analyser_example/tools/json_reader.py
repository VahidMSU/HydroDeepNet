from agno.knowledge.json import JSONKnowledgeBase
from agno.vectordb.pgvector import PgVector
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def json_reader(json_path="/data/SWATGenXApp/Users/admin/Reports/20250324_222749/config.json"):
    # Create JSON knowledge base
    knowledge_base = JSONKnowledgeBase(
        path=json_path,
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
    agent.knowledge.load(recreate=True)
    logger.info("Knowledge base loaded successfully")

    # Example query
    response = agent.print_response("Tell me about the settings in the json file")
    return response

if __name__ == "__main__":
    json_path = "/data/SWATGenXApp/Users/admin/Reports/20250324_222749/config.json"
    json_reader(json_path)
