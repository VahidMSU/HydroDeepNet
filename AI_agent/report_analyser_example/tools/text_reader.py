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
2. Key findings about aquifer properties (thickness, conductivity, transmissivity)
3. Important patterns and relationships between properties
4. Main hydrogeologic implications
5. Key recommendations

Focus on the most significant findings and their practical implications."""

    # Get the analysis
    return agent.print_response(analysis_prompt, stream=True)

if __name__ == "__main__":
    doc_path = "/data/SWATGenXApp/Users/admin/Reports/20250324_222749/groundwater/groundwater_report.md"
    analyze_report(doc_path)