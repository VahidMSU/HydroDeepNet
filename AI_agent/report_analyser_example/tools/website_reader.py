from agno.knowledge.website import WebsiteKnowledgeBase
from agno.vectordb.pgvector import PgVector
import logging
from urllib.parse import urlparse, urljoin

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def website_reader(url="https://docs.agno.com/introduction", max_links=5, recreate_db=False):
    # Validate URL
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            raise ValueError("Invalid URL format")
    except Exception as e:
        logger.error(f"Invalid URL: {url}")
        return f"Error: Invalid URL provided - {str(e)}"

    # Get base domain for limiting crawl scope
    base_domain = result.netloc

    # Create website knowledge base with limited scope
    knowledge_base = WebsiteKnowledgeBase(
        urls=[url],
        max_links=max_links,  # Limit to fewer links
        allowed_domains=[base_domain],  # Only crawl within same domain
        max_depth=1,  # Only crawl one level deep
        vector_db=PgVector(
            table_name="website_documents",
            db_url="postgresql+psycopg://ai:ai@localhost:5432/ai",
        )
    )
    logger.info(f"Website knowledge base initialized for {url}")

    # Create agent with website knowledge
    from agno.agent import Agent
    agent = Agent(
        knowledge=knowledge_base,
        search_knowledge=True
    )
    logger.info("Agent created with website knowledge")

    # Load the knowledge base
    agent.knowledge.load(recreate=recreate_db)
    logger.info("Knowledge base loaded successfully")

    # Create a focused analysis prompt
    analysis_prompt = f"""Please analyze the main content of {url} and provide:
    1. Main topic and purpose of this specific page
    2. Key information presented on this page
    3. Important points or highlights
    
    Focus only on the main content of this specific URL, not the entire website."""

    response = agent.print_response(analysis_prompt)
    if not response or "I can't browse the internet" in response:
        logger.error("Failed to analyze website content")
        return f"Error: Unable to analyze the website at {url}. Please check if the URL is accessible and try again."
    
    return response

if __name__ == "__main__":
    # Example usage with a specific page
    test_url = "https://www.python.org/about/"  # More focused URL
    response = website_reader(test_url, max_links=5, recreate_db=True)
    print(response)

