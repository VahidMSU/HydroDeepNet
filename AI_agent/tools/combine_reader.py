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
from typing import Dict, Optional
import os
import logging
from pathlib import Path

def combined_reader(reports_dict: Dict, report_timestamp: Optional[str] = None, group_name: Optional[str] = None, recreate_db: bool = False, logger=None):
    """
    Analyzes a complete report group by combining multiple types of data.
    
    Args:
        reports_dict: Dictionary containing the report structure
        report_timestamp: Specific report timestamp to analyze (e.g., "20250324_222749")
        group_name: Specific group to analyze (e.g., "groundwater", "climate")
        recreate_db: Whether to recreate the database
        logger: Logger instance to use
        
    Returns:
        The analysis response
    """
    # Use provided logger or default
    log = logger or logging.getLogger(__name__)
    
    # If no timestamp provided, use the latest
    if report_timestamp is None:
        report_timestamp = sorted(reports_dict.keys())[-1]
    
    if report_timestamp not in reports_dict:
        log.error(f"Report {report_timestamp} not found")
        print(f"Report {report_timestamp} not found")
        return
    
    report_data = reports_dict[report_timestamp]
    log.info(f"Processing report: {report_timestamp}")
    print(f"Processing report: {report_timestamp}")
    
    # If no group specified, show available groups and return
    if group_name is None:
        log.info("No group specified, listing available groups")
        print("\nAvailable groups:")
        for group in report_data["groups"].keys():
            print(f"- {group}")
        return
    
    if group_name not in report_data["groups"]:
        log.error(f"Group {group_name} not found in report {report_timestamp}")
        print(f"Group {group_name} not found in report {report_timestamp}")
        print("Available groups:", list(report_data["groups"].keys()))
        return
    
    # Get group data
    group_data = report_data["groups"][group_name]
    log.info(f"Processing group: {group_name}")
    print(f"\nProcessing group: {group_name}")
    
    # Initialize the embedder
    embedder = OpenAIEmbedder()
    
    # Create knowledge bases for different file types
    knowledge_bases = []
    
    # Process config file if it exists
    if report_data["config"]:
        log.info(f"Adding config: {report_data['config']}")
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
            for filename, file_info in files[ext].items():
                log.info(f"Adding text: {filename}")
                print(f"Adding text: {filename}")
                with open(file_info["path"], "r") as f:
                    content = f.read()
                
                text_kb = TextKnowledgeBase(
                    path=file_info["path"],
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
        for filename, file_info in files[".csv"].items():
            log.info(f"Adding CSV: {filename}")
            print(f"Adding CSV: {filename}")
            csv_kb = CSVKnowledgeBase(
                path=file_info["path"],
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
    log.info("Combined knowledge base created")
    print("Combined knowledge base created")
    
    # Create agent with combined knowledge
    agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        knowledge=combined_kb,
        search_knowledge=True,
        markdown=True
    )
    log.info("Agent created with combined knowledge")
    print("Agent created with combined knowledge")
    
    # Load the knowledge base
    combined_kb.load(recreate=recreate_db)
    log.info("Knowledge base loaded successfully")
    print("Knowledge base loaded successfully")
    
    # Process images if they exist
    image_descriptions = []
    if ".png" in files:
        log.info("Processing image files")
        image_agent = Agent(
            model=OpenAIChat(id="gpt-4o"),
            markdown=True,
            instructions=[
                f"You are analyzing {group_name} visualizations.",
                "Provide detailed analysis of the charts and plots.",
                "Focus on trends, patterns, and relationships in the data."
            ]
        )
        
        for filename, file_info in files[".png"].items():
            log.info(f"Processing image: {filename}")
            print(f"Processing image: {filename}")
            description = image_agent.print_response(
                "What does this visualization show? Provide a detailed analysis.",
                images=[Image(filepath=file_info["path"])]
            )
            image_descriptions.append(f"Analysis of {filename}:\n{description}")
    
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
    
    log.info("Generating comprehensive analysis")
    # Get the comprehensive analysis
    return agent.print_response(analysis_prompt, stream=True)

if __name__ == "__main__":
    # Import discover_reports only when running as main
    from dir_discover import discover_reports
    
    # Get the reports structure
    reports = discover_reports()
    
    # Example: Analyze the latest report's groundwater group
    response = combined_reader(reports, group_name="groundwater", recreate_db=True)
    print("\nAnalysis:")
    print(response)
    
    # To see available groups in a specific report:
    # combined_reader(reports, "20250324_222749")