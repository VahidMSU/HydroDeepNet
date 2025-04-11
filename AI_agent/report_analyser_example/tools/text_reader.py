from agno.agent import Agent
from agno.document.base import Document
from agno.knowledge.document import DocumentKnowledgeBase
from agno.vectordb.pgvector import PgVector
from agno.models.openai import OpenAIChat
import os
def analyze_report(doc_path):
    assert os.path.exists(doc_path), f"File does not exist: {doc_path}"
    # Read the document content
    with open(doc_path, "r") as f:
        doc_content = f.read()

    # Create document object
    document = Document(content=doc_content)

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

    # Create an agent with model and knowledge
    agent = Agent(
        model=OpenAIChat(id="gpt-4"),
        knowledge=knowledge_base,
        search_knowledge=True,
        markdown=True
    )

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
    return agent.print_response(analysis_prompt, stream=True)

if __name__ == "__main__":
    from dir_discover import discover_reports
    reports_dict = discover_reports()
    report_timestamp = sorted(reports_dict.keys())[-1]
    print(report_timestamp)
    
    # Find the cdl_data.csv file in the nsrdb group
    climate_change_files = reports_dict[report_timestamp]["groups"]["climate_change"]["files"]
    doc_path = None
    #
    # Files are organized by extension
    if ".md" in climate_change_files:
        for file_info in climate_change_files[".md"]:
            if file_info["name"] == "climate_change_report.md":
                doc_path = file_info["path"]
                break



    analyze_report(doc_path)