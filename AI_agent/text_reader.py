from agno.agent import Agent
from agno.document.base import Document
from agno.knowledge.document import DocumentKnowledgeBase
from agno.vectordb.pgvector import PgVector
from agno.models.openai import OpenAIChat
import os
import logging

def analyze_report(doc_path, logger=None):
    """
    Analyze a text report document.
    
    Args:
        doc_path: Path to the document
        logger: Logger instance to use
        
    Returns:
        Analysis text
    """
    # Use provided logger or default
    log = logger or logging.getLogger(__name__)
    
    assert os.path.exists(doc_path), f"File does not exist: {doc_path}"
    log.info(f"Analyzing text report: {doc_path}")
    
    # Read the document content
    with open(doc_path, "r") as f:
        doc_content = f.read()

    # Create document object
    document = Document(content=doc_content)
    
    log.info("Creating document knowledge base")
    # Create knowledge base with the document
    knowledge_base = DocumentKnowledgeBase(
        documents=[document],
        vector_db=PgVector(
            table_name="documents",
            db_url="postgresql+psycopg://ai:ai@localhost:5432/ai",
        ),
    )

    # Load the knowledge base with recreation to ensure fresh data
    knowledge_base.load(recreate=True)
    log.info("Knowledge base loaded successfully")

    # Create an agent with model and knowledge
    agent = Agent(
        model=OpenAIChat(id="gpt-4"),
        knowledge=knowledge_base,
        search_knowledge=True,
        markdown=True
    )
    log.info("Agent created with document knowledge")

    # Create analysis prompt with document content
    analysis_prompt = f"""Here is the content of the groundwater report:

{doc_content}

Based on this report, please provide:
1. Overview of the study area and available data
2. Key findings about the report (statistics, trends, patterns, etc.)
3. Important patterns and relationships between the data
4. Main implications of the report
5. Key recommendations

Focus on the most significant findings and their practical implications."""

    # Get the analysis
    log.info("Generating report analysis")
    return agent.print_response(analysis_prompt, stream=True)

if __name__ == "__main__":
    from dir_discover import discover_reports
    reports_dict = discover_reports()
    report_timestamp = sorted(reports_dict.keys())[-1]
    doc_path = reports_dict[report_timestamp]["groups"]["climate_change"]["files"]['.md']['climate_change_report.md']['path']
    print(doc_path)
    analyze_report(doc_path)