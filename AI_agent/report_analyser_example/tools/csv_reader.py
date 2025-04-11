from agno.knowledge.csv import CSVKnowledgeBase
from agno.vectordb.pgvector import PgVector
import logging
import pandas as pd
from dir_discover import discover_reports
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def csv_reader(csv_path, recreate_db=False):
    # First read the CSV directly to get basic info
    df = pd.read_csv(csv_path)
    logger.info(f"CSV file contains {len(df)} rows and {len(df.columns)} columns")

    # Create CSV knowledge base
    knowledge_base = CSVKnowledgeBase(
        path=csv_path,
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

    # Load the knowledge base with recreation option
    agent.knowledge.load(recreate=recreate_db)
    logger.info("Knowledge base loaded successfully")

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
        logger.info("Agent did not provide analysis, showing basic statistics:")
        return str(df.describe())
    return response


if __name__ == "__main__":
    reports_dict = discover_reports()
    report_timestamp = sorted(reports_dict.keys())[-1]
    print(report_timestamp)
    
    # Find the cdl_data.csv file in the nsrdb group
    nsrdb_files = reports_dict[report_timestamp]["groups"]["cdl"]["files"]
    csv_path = None
    #
    # Files are organized by extension
    if ".csv" in nsrdb_files:
        for file_info in nsrdb_files[".csv"]:
            if file_info["name"] == "cdl_data.csv":
                csv_path = file_info["path"]
                break
    
    if csv_path is None:
        raise FileNotFoundError("Could not find cdl_data.csv in the report")
        
    print(csv_path)
    response = csv_reader(csv_path, recreate_db=True)
    print(response)