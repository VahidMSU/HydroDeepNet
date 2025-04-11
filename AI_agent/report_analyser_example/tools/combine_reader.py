from agno.knowledge.combined import CombinedKnowledgeBase
from agno.vectordb.pgvector import PgVector
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.knowledge.text import TextKnowledgeBase
from agno.knowledge.csv import CSVKnowledgeBase
from agno.document.base import Document
from agno.embedder.openai import OpenAIEmbedder
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.media import Image
import os
from pathlib import Path

def combined_reader(report_dir="/data/SWATGenXApp/Users/admin/Reports/20250324_222749", recreate_db=False):
    """
    Analyzes a complete report by combining multiple types of data.
    
    Args:
        report_dir: Path to the report directory
        recreate_db: Whether to recreate the database
        
    Returns:
        The analysis response
    """
    # Check if directory exists
    if not os.path.exists(report_dir):
        print(f"Report directory not found: {report_dir}")
        return

    # Define paths based on report directory
    groundwater_dir = os.path.join(report_dir, "groundwater")
    config_path = os.path.join(report_dir, "config.json")
    report_path = os.path.join(groundwater_dir, "groundwater_report.md")
    stats_path = os.path.join(groundwater_dir, "groundwater_stats.csv")
    image_path = os.path.join(groundwater_dir, "groundwater_correlation.png")
    
    print(f"Processing report in: {report_dir}")
    
    # Initialize the embedder
    embedder = OpenAIEmbedder()
    
    # Create knowledge bases for different file types
    knowledge_bases = []
    
    # Add markdown report if it exists
    if os.path.exists(report_path):
        print(f"Adding report: {report_path}")
        with open(report_path, "r") as f:
            report_content = f.read()
        
        document = Document(content=report_content)
        report_kb = TextKnowledgeBase(
            path=report_path,
            text=report_content,
            vector_db=PgVector(
                table_name="report_documents",
                db_url="postgresql+psycopg://ai:ai@localhost:5432/ai",
            ),
            embedder=embedder
        )
        knowledge_bases.append(report_kb)
    
    # Add CSV stats if it exists
    if os.path.exists(stats_path):
        print(f"Adding stats: {stats_path}")
        csv_kb = CSVKnowledgeBase(
            path=stats_path,
            vector_db=PgVector(
                table_name="csv_documents",
                db_url="postgresql+psycopg://ai:ai@localhost:5432/ai",
            ),
            embedder=embedder
        )
        knowledge_bases.append(csv_kb)
    
    # Add config file if it exists
    if os.path.exists(config_path):
        print(f"Adding config: {config_path}")
        config_kb = TextKnowledgeBase(
            path=config_path,
            vector_db=PgVector(
                table_name="config_documents",
                db_url="postgresql+psycopg://ai:ai@localhost:5432/ai",
            ),
            embedder=embedder
        )
        knowledge_bases.append(config_kb)
    
    # Combine the knowledge bases
    combined_kb = CombinedKnowledgeBase(
        sources=knowledge_bases,
        vector_db=PgVector(
            table_name="combined_documents",
            db_url="postgresql+psycopg://ai:ai@localhost:5432/ai",
        ),
        embedder=embedder
    )
    print("Combined knowledge base created")
    
    # Create agent with combined knowledge
    agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        knowledge=combined_kb,
        search_knowledge=True,
        markdown=True
    )
    print("Agent created with combined knowledge")
    
    # Load the knowledge base
    combined_kb.load(recreate=recreate_db)
    print("Knowledge base loaded successfully")
    
    # Process image if it exists
    image_description = ""
    if os.path.exists(image_path):
        print(f"Processing image: {image_path}")
        image_agent = Agent(
            model=OpenAIChat(id="gpt-4o"),
            markdown=True,
            instructions=[
                "You are analyzing groundwater data visualizations.",
                "Provide detailed analysis of the charts and plots.",
                "Focus on trends, patterns, and relationships in the data."
            ]
        )
        image_description = image_agent.print_response(
            "What does this visualization show? Provide a detailed analysis.",
            images=[Image(filepath=image_path)]
        )
    
    # Create comprehensive analysis prompt
    analysis_prompt = f"""
    Please provide a comprehensive analysis of the groundwater report by synthesizing all available information:
    
    1. Overview of the study area and objectives
    2. Data sources and methodology used
    3. Key findings about groundwater properties and conditions
    4. Statistical analysis of the groundwater parameters
    5. Interpretation of visualizations and their significance
    6. Main conclusions and recommendations
    
    {image_description if image_description else ""}
    
    Focus on providing an integrated analysis that connects all pieces of information.
    """
    
    # Get the comprehensive analysis
    return agent.print_response(analysis_prompt, stream=True)

if __name__ == "__main__":
    response = combined_reader(recreate_db=True)
    print(response)