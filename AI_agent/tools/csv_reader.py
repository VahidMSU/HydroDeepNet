from agno.knowledge.csv import CSVKnowledgeBase
from agno.vectordb.pgvector import PgVector
import logging
import pandas as pd
from dir_discover import discover_reports

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def csv_reader(csv_path, recreate_db=False, logger=None):
    """
    Analyzes CSV files using vector database and AI.
    
    Args:
        csv_path: Path to the CSV file
        recreate_db: If True, drops and recreates the database table. Use this when getting duplicate key errors.
        logger: Logger instance to use
    """
    # Use provided logger or default to module logger
    log = logger or logging.getLogger(__name__)
    
    # First read the CSV directly to get basic info
    df = pd.read_csv(csv_path)
    log.info(f"CSV file contains {len(df)} rows and {len(df.columns)} columns")

    # Create vector database connection with table recreation option
    vector_db = PgVector(
        table_name="csv_documents",
        db_url="postgresql+psycopg://ai:ai@localhost:5432/ai",
    )
    
    # If recreate_db is True, drop the existing table
    if recreate_db:
        log.info("Recreating database table...")
        try:
            vector_db.drop_table()
            log.info("Existing table dropped successfully")
        except Exception as e:
            log.warning(f"Error dropping table (this is normal if it doesn't exist): {e}")

    # Create CSV knowledge base
    knowledge_base = CSVKnowledgeBase(
        path=csv_path,
        vector_db=vector_db
    )
    log.info("CSV knowledge base initialized")

    # Create agent with CSV knowledge
    from agno.agent import Agent
    agent = Agent(
        knowledge=knowledge_base,
        search_knowledge=True,
    )
    log.info("Agent created with CSV knowledge")

    # Load the knowledge base with recreation option
    try:
        agent.knowledge.load(recreate=recreate_db)
        log.info("Knowledge base loaded successfully")
    except Exception as e:
        log.error(f"Error loading knowledge base: {e}")
        log.info("Trying to proceed with analysis anyway...")

    # Create a more specific prompt with CSV context
    analysis_prompt = f"""The CSV file at {csv_path} contains {len(df)} rows and {len(df.columns)} columns.
    The columns are: {', '.join(df.columns.tolist())}

    Please analyze this groundwater statistics data and provide:
    1. A detailed description of what each column represents and the type of data it contains
    2. Statistical summary of the numerical columns (min, max, mean if applicable)
    3. Any temporal patterns or trends in the data
    4. Notable relationships between different variables
    5. Key insights about groundwater conditions based on this data

    Present this information in a clear, structured format with specific numbers and examples from the data."""

    response = agent.print_response(analysis_prompt)
    if not response:
        # If no response, provide basic statistical analysis
        log.info("Agent did not provide analysis, showing basic statistics:")
        return str(df.describe())
    return response

if __name__ == "__main__":
    reports_dict = discover_reports()
    report_timestamp = sorted(reports_dict.keys())[-1]
    print(report_timestamp)
    
    # Find the cdl_data.csv file in the nsrdb group
    csv_path = reports_dict[report_timestamp]["groups"]["cdl"]["files"]['.csv']['cdl_data.csv']['path']

    print(csv_path)
    # Note: Set recreate_db=True if getting duplicate key errors
    response = csv_reader(csv_path, recreate_db=True)
    print(response)