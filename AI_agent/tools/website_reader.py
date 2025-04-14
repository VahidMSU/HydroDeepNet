from agno.knowledge.website import WebsiteKnowledgeBase
from agno.vectordb.pgvector import PgVector
import logging
from urllib.parse import urlparse, urljoin
from typing import Optional
# Import config loader
from config_loader import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def website_reader(url: Optional[str] = None, max_links: Optional[int] = None, max_depth: Optional[int] = None, recreate_db_str: str = "false", logger=None):
    """
    Analyze content from a website URL.

    Args:
        url: Website URL to analyze. Defaults to config value.
        max_links: Maximum number of links to follow. Defaults to config value.
        max_depth: Maximum crawl depth. Defaults to config value.
        recreate_db_str: If 'true', drops and recreates the database table.
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
    table_name = config.get('db_tables', {}).get('website', 'website_documents')

    # Use provided args or get from config
    target_url = url or config.get('default_website_url', 'https://docs.agno.com/introduction')
    target_max_links = max_links if max_links is not None else config.get('max_website_links', 5)
    target_max_depth = max_depth if max_depth is not None else config.get('max_website_depth', 1)

    # Validate URL
    try:
        result = urlparse(target_url)
        if not all([result.scheme, result.netloc]):
            raise ValueError("Invalid URL format")
    except Exception as e:
        log.error(f"Invalid URL: {target_url}")
        return f"Error: Invalid URL provided - {str(e)}"

    # Get base domain for limiting crawl scope
    base_domain = result.netloc

    log.info(f"Analyzing website: {target_url} (max_links={target_max_links}, max_depth={target_max_depth})")

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

    # Create website knowledge base with limited scope
    knowledge_base = WebsiteKnowledgeBase(
        urls=[target_url],
        max_links=target_max_links,
        allowed_domains=[base_domain],
        max_depth=target_max_depth,
        vector_db=vector_db
    )
    log.info(f"Website knowledge base initialized for {target_url}")

    # Create agent with website knowledge
    from agno.agent import Agent
    agent = Agent(
        knowledge=knowledge_base,
        search_knowledge=True,
        debug_mode=False,
        show_tool_calls=False,
        reasoning=False
    )
    log.info("Agent created with website knowledge")

    # Load the knowledge base
    try:
        agent.knowledge.load(recreate=recreate_db)
        log.info("Knowledge base loaded successfully")
    except Exception as e:
        log.error(f"Error loading knowledge base for {target_url}: {e}")
        return f"Error loading knowledge base for {target_url}."

    # Create a focused analysis prompt
    analysis_prompt = f"""Please analyze the main content of {target_url} and provide:
    1. Main topic and purpose of this specific page
    2. Key information presented on this page
    3. Important points or highlights

    Focus only on the main content of this specific URL, not the entire website."""

    log.info("Generating website analysis")
    response = agent.print_response(analysis_prompt)
    if not response or "I can't browse the internet" in response:
        log.error("Failed to analyze website content")
        return f"Error: Unable to analyze the website at {target_url}. Please check if the URL is accessible and try again."

    log.info("Website analysis completed")
    return response

if __name__ == "__main__":
    # Example usage with a specific page, falling back to config for limits
    test_url = "https://www.python.org/about/"
    print(f"Analyzing: {test_url}")
    response = website_reader(url=test_url, recreate_db_str="true")
    print(response)

