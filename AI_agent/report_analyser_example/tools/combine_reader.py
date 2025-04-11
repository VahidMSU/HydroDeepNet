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
from dir_discover import discover_reports
import os
from pathlib import Path

def combined_reader(report_timestamp=None, group_name=None, recreate_db=False):
    """
    Analyzes a complete report group by combining multiple types of data.
    
    Args:
        report_timestamp: Specific report timestamp to analyze (e.g., "20250324_222749")
        group_name: Specific group to analyze (e.g., "groundwater", "climate")
        recreate_db: Whether to recreate the database
        
    Returns:
        The analysis response
    """
    # Get reports structure
    reports_dict = discover_reports()
    
    # If no timestamp provided, use the latest
    if report_timestamp is None:
        report_timestamp = sorted(reports_dict.keys())[-1]
    
    if report_timestamp not in reports_dict:
        print(f"Report {report_timestamp} not found")
        return
    
    report_data = reports_dict[report_timestamp]
    print(f"Processing report: {report_timestamp}")
    
    # If no group specified, show available groups and return
    if group_name is None:
        print("\nAvailable groups:")
        for group in report_data["groups"].keys():
            print(f"- {group}")
        return
    
    if group_name not in report_data["groups"]:
        print(f"Group {group_name} not found in report {report_timestamp}")
        print("Available groups:", list(report_data["groups"].keys()))
        return
    
    # Get group data
    group_data = report_data["groups"][group_name]
    print(f"\nProcessing group: {group_name}")
    
    # Initialize the embedder
    embedder = OpenAIEmbedder()
    
    # Create knowledge bases for different file types
    knowledge_bases = []
    
    # Process config file if it exists
    if report_data["config"]:
        print(f"Adding config: {report_data['config']}")
        config_kb = TextKnowledgeBase(
            path=report_data["config"],
            vector_db=PgVector(
                table_name="config_documents",
                db_url="postgresql+psycopg://ai:ai@localhost:5432/ai",
            ),
            embedder=embedder
        )
        knowledge_bases.append(config_kb)
    
    # Process files by type
    files = group_data["files"]
    
    # Add markdown/text files
    for ext in [".md", ".txt"]:
        if ext in files:
            for file_data in files[ext]:
                print(f"Adding text: {file_data['name']}")
                with open(file_data["path"], "r") as f:
                    content = f.read()
                
                text_kb = TextKnowledgeBase(
                    path=file_data["path"],
                    text=content,
                    vector_db=PgVector(
                        table_name="text_documents",
                        db_url="postgresql+psycopg://ai:ai@localhost:5432/ai",
                    ),
                    embedder=embedder
                )
                knowledge_bases.append(text_kb)
    
    # Add CSV files
    if ".csv" in files:
        for file_data in files[".csv"]:
            print(f"Adding CSV: {file_data['name']}")
            csv_kb = CSVKnowledgeBase(
                path=file_data["path"],
                vector_db=PgVector(
                    table_name="csv_documents",
                    db_url="postgresql+psycopg://ai:ai@localhost:5432/ai",
                ),
                embedder=embedder
            )
            knowledge_bases.append(csv_kb)
    
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
    
    # Process images if they exist
    image_descriptions = []
    if ".png" in files:
        image_agent = Agent(
            model=OpenAIChat(id="gpt-4o"),
            markdown=True,
            instructions=[
                f"You are analyzing {group_name} visualizations.",
                "Provide detailed analysis of the charts and plots.",
                "Focus on trends, patterns, and relationships in the data."
            ]
        )
        
        for file_data in files[".png"]:
            print(f"Processing image: {file_data['name']}")
            description = image_agent.print_response(
                "What does this visualization show? Provide a detailed analysis.",
                images=[Image(filepath=file_data["path"])]
            )
            image_descriptions.append(f"Analysis of {file_data['name']}:\n{description}")
    
    # Create comprehensive analysis prompt
    image_section = "\n\nImage Analyses:\n" + "\n".join(image_descriptions) if image_descriptions else ""
    
    analysis_prompt = f"""Please provide a comprehensive analysis of the {group_name} report by synthesizing all available information:

1. Overview of the study area and objectives
2. Data sources and methodology used
3. Key findings and observations
4. Statistical analysis of the parameters
5. Interpretation of visualizations and their significance
6. Main conclusions and recommendations{image_section}

Focus on providing an integrated analysis that connects all pieces of information."""
    
    # Get the comprehensive analysis
    return agent.print_response(analysis_prompt, stream=True)

if __name__ == "__main__":
    # Example: Analyze the latest report's groundwater group
    response = combined_reader(group_name="groundwater", recreate_db=True)
    print("\nAnalysis:")
    print(response)
    
    # To see available groups in a specific report:
    # combined_reader("20250324_222749")