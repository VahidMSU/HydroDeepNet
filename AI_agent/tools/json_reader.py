from agno.knowledge.json import JSONKnowledgeBase
from agno.vectordb.pgvector import PgVector
import logging
import os
from pathlib import Path
from typing import Optional
# Import config loader
from config_loader import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def json_reader(json_path: Optional[str] = None, recreate_db_str: str = "false", logger=None):
    """
    Analyze a JSON configuration file.

    Args:
        json_path: Path to the JSON file. If None, tries to find config.json in the latest report.
        recreate_db_str: If 'true', drops and recreates the database table. Defaults to 'false'.
        logger: Logger instance to use

    Returns:
        Analysis text or error message.
    """
    # Use provided logger or default
    log = logger or logging.getLogger(__name__)

    # Convert recreate_db_str to boolean
    recreate_db = (recreate_db_str.lower() == "true")

    # Get config
    config = get_config()
    db_url = config.get('database_url', 'postgresql+psycopg://ai:ai@localhost:5432/ai')
    table_name = config.get('db_tables', {}).get('json', 'json_documents')
    base_report_dir = config.get('base_report_dir')

    # Determine JSON path if not provided
    if json_path is None:
        if not base_report_dir or not os.path.isdir(base_report_dir):
            log.error("Base report directory not configured or found.")
            return "Error: Base report directory not configured or found."
        try:
            # Find the latest report directory
            report_dirs = sorted([d for d in os.listdir(base_report_dir) if os.path.isdir(os.path.join(base_report_dir, d))], reverse=True)
            if not report_dirs:
                log.error(f"No report directories found in {base_report_dir}")
                return f"Error: No reports found in {base_report_dir}."
            latest_report_path = os.path.join(base_report_dir, report_dirs[0])
            json_path = os.path.join(latest_report_path, "config.json")
            if not os.path.exists(json_path):
                log.error(f"config.json not found in latest report: {latest_report_path}")
                return f"Error: config.json not found in the latest report ({report_dirs[0]})."
        except Exception as e:
            log.error(f"Error determining default json_path: {e}")
            return "Error: Could not automatically determine the path to config.json."

    if not os.path.exists(json_path):
        log.error(f"JSON file not found: {json_path}")
        return f"Error: JSON file not found at {json_path}."

    log.info(f"Analyzing JSON file: {json_path}")

    # Create vector database connection
    try:
        vector_db = PgVector(
            table_name=table_name,
            db_url=db_url,
        )
    except Exception as e:
        log.error(f"Error connecting to vector database: {e}")
        return "Error: Could not connect to the vector database."

    # If recreate_db is True, drop the existing table
    if recreate_db:
        log.info(f"Recreating database table '{table_name}'...")
        try:
            vector_db.drop_table()
            log.info(f"Existing table '{table_name}' dropped successfully")
        except Exception as e:
            log.warning(f"Error dropping table '{table_name}' (this is normal if it doesn't exist): {e}")

    # Create JSON knowledge base
    knowledge_base = JSONKnowledgeBase(
        path=json_path,
        vector_db=vector_db,
    )
    log.info("JSON knowledge base initialized")

    # Create agent with JSON knowledge
    from agno.agent import Agent
    agent = Agent(
        knowledge=knowledge_base,
        search_knowledge=True,
        debug_mode=False,
        show_tool_calls=False,
        reasoning=False
    )
    log.info("Agent created with JSON knowledge")

    # Load the knowledge base
    try:
        agent.knowledge.load(recreate=recreate_db)
        log.info("Knowledge base loaded successfully")
    except Exception as e:
        log.error(f"Error loading knowledge base for {json_path}: {e}")
        return f"Error loading knowledge base for {json_path}."

    # Example query
    log.info("Generating JSON analysis")
    response = agent.print_response("Tell me about the settings in the json file")
    return response

if __name__ == "__main__":
    # Example: Analyze the config.json from the latest report
    response = json_reader()
    if response.startswith("Error:"):
        print(response)
    else:
        print("\nAnalysis:")
        print(response)
