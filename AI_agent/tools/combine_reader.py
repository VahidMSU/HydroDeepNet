from agno.knowledge.combined import CombinedKnowledgeBase
from agno.vectordb.pgvector import PgVector
from agno.knowledge.pdf import PDFKnowledgeBase
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
# Import config loader
from config_loader import get_config
# Import discover_reports relatively
from dir_discover import discover_reports

def combined_reader(reports_dict: Dict, report_timestamp: Optional[str] = None, group_name: Optional[str] = None, recreate_db_str: str = "false", logger=None):
    """
    Analyzes a complete report group by combining multiple types of data.

    Args:
        reports_dict: Dictionary containing the report structure
        report_timestamp: Specific report timestamp to analyze (e.g., "20250324_222749")
        group_name: Specific group to analyze (e.g., "groundwater", "climate")
        recreate_db_str: If 'true', drops and recreates the database table.
        logger: Logger instance to use

    Returns:
        The analysis response or error message.
    """
    # Use provided logger or default
    log = logger or logging.getLogger(__name__)

    # Convert recreate_db_str to boolean
    recreate_db = (recreate_db_str.lower() == "true")

    # Get config
    config = get_config()
    db_url = config.get('database_url', 'postgresql+psycopg://ai:ai@localhost:5432/ai')
    db_tables = config.get('db_tables', {})
    combined_table = db_tables.get('combined', 'combined_documents')
    config_table = db_tables.get('config', 'config_documents')
    text_table = db_tables.get('text', 'text_documents')
    csv_table = db_tables.get('csv', 'csv_documents')
    agent_model_id = config.get('default_model', 'gpt-4o')
    image_model_id = config.get('image_model', 'gpt-4o')

    # If no timestamp provided, use the latest
    if report_timestamp is None:
        if not reports_dict:
             log.warning("Reports dictionary is empty.")
             return "Error: No reports found to determine the latest timestamp."
        report_timestamp = sorted(reports_dict.keys())[-1]

    if report_timestamp not in reports_dict:
        log.error(f"Report {report_timestamp} not found")
        print(f"Report {report_timestamp} not found")
        return f"Error: Report {report_timestamp} not found."

    report_data = reports_dict[report_timestamp]
    log.info(f"Processing report: {report_timestamp}")
    print(f"Processing report: {report_timestamp}")

    # If no group specified, show available groups and return
    if group_name is None:
        log.info("No group specified, listing available groups")
        available_groups = list(report_data.get("groups", {}).keys())
        if not available_groups:
            return f"No groups found in report {report_timestamp}."
        print("\nAvailable groups:")
        for group in available_groups:
            print(f"- {group}")
        return "Please specify a group to analyze using --group [group_name] or select one interactively."

    if group_name not in report_data.get("groups", {}):
        log.error(f"Group {group_name} not found in report {report_timestamp}")
        print(f"Group {group_name} not found in report {report_timestamp}")
        print("Available groups:", list(report_data.get("groups", {}).keys()))
        return f"Error: Group {group_name} not found in report {report_timestamp}."

    # Get group data
    group_data = report_data["groups"][group_name]
    log.info(f"Processing group: {group_name}")
    print(f"\nProcessing group: {group_name}")

    # Initialize the embedder (Consider making configurable if needed)
    embedder = OpenAIEmbedder()

    # --- Vector DB Setup ---
    try:
        base_vector_db_args = {"db_url": db_url}
        combined_vector_db = PgVector(table_name=combined_table, **base_vector_db_args)
        config_vector_db = PgVector(table_name=config_table, **base_vector_db_args)
        text_vector_db = PgVector(table_name=text_table, **base_vector_db_args)
        csv_vector_db = PgVector(table_name=csv_table, **base_vector_db_args)

        # Handle recreation for the combined table (individual tables handle their own)
        if recreate_db:
            log.info(f"Recreating database table '{combined_table}'...")
            try:
                combined_vector_db.drop_table()
                log.info(f"Existing table '{combined_table}' dropped successfully")
            except Exception as e:
                 log.warning(f"Error dropping table '{combined_table}' (this is normal if it doesn't exist): {e}")

    except Exception as e:
        log.error(f"Error setting up vector databases: {e}")
        return "Error: Could not set up vector databases."

    # --- Knowledge Base Creation ---
    knowledge_bases = []

    # Process config file if it exists
    if report_data.get("config") and os.path.exists(report_data["config"]):
        log.info(f"Adding config: {report_data['config']}")
        print(f"Adding config: {report_data['config']}")
        try:
            config_kb = TextKnowledgeBase(
                path=report_data["config"],
                vector_db=config_vector_db,
                embedder=embedder
            )
            knowledge_bases.append(config_kb)
        except Exception as e:
            log.error(f"Failed to initialize TextKnowledgeBase for config file: {e}")

    # Process files by type
    files = group_data.get("files", {})

    # Add markdown/text files
    for ext in [".md", ".txt"]:
        if ext in files:
            for filename, file_info in files[ext].items():
                file_path = file_info["path"]
                if os.path.exists(file_path):
                    log.info(f"Adding text: {filename}")
                    print(f"Adding text: {filename}")
                    try:
                        # Pass path, let TextKnowledgeBase read it
                        text_kb = TextKnowledgeBase(
                            path=file_path,
                            vector_db=text_vector_db,
                            embedder=embedder
                        )
                        knowledge_bases.append(text_kb)
                    except Exception as e:
                        log.error(f"Failed to initialize TextKnowledgeBase for {filename}: {e}")
                else:
                     log.warning(f"Text file path not found: {file_path}")

    # Add CSV files
    if ".csv" in files:
        for filename, file_info in files[".csv"].items():
            file_path = file_info["path"]
            if os.path.exists(file_path):
                log.info(f"Adding CSV: {filename}")
                print(f"Adding CSV: {filename}")
                try:
                    csv_kb = CSVKnowledgeBase(
                        path=file_path,
                        vector_db=csv_vector_db,
                        embedder=embedder
                    )
                    knowledge_bases.append(csv_kb)
                except Exception as e:
                     log.error(f"Failed to initialize CSVKnowledgeBase for {filename}: {e}")
            else:
                log.warning(f"CSV file path not found: {file_path}")

    if not knowledge_bases:
        log.warning(f"No valid knowledge sources found for group {group_name} in report {report_timestamp}.")
        return f"Error: No readable files found to analyze in group {group_name}."

    # --- Combined Knowledge and Agent --- 
    combined_kb = CombinedKnowledgeBase(
        sources=knowledge_bases,
        vector_db=combined_vector_db,
        embedder=embedder
    )
    log.info("Combined knowledge base created")
    print("Combined knowledge base created")

    agent = Agent(
        model=OpenAIChat(id=agent_model_id),
        knowledge=combined_kb,
        search_knowledge=True,
        markdown=True
    )
    log.info("Agent created with combined knowledge")
    print("Agent created with combined knowledge")

    # Load the knowledge base (recreate applies mainly to the combined table here)
    try:
        combined_kb.load(recreate=recreate_db)
        log.info("Knowledge base loaded successfully")
        print("Knowledge base loaded successfully")
    except Exception as e:
        log.error(f"Error loading combined knowledge base for group {group_name}: {e}")
        return f"Error loading combined knowledge base for group {group_name}."

    # --- Image Processing --- 
    image_descriptions = []
    if ".png" in files:
        log.info("Processing image files")
        try:
            image_agent = Agent(
                model=OpenAIChat(id=image_model_id),
                markdown=True,
                instructions=[
                    f"You are analyzing {group_name} visualizations.",
                    "Provide detailed analysis of the charts and plots.",
                    "Focus on trends, patterns, and relationships in the data."
                ]
            )
        except Exception as e:
            log.error(f"Failed to initialize image agent: {e}")
            image_agent = None # Continue without image analysis

        if image_agent:
            for filename, file_info in files[".png"].items():
                file_path = file_info["path"]
                if os.path.exists(file_path):
                    log.info(f"Processing image: {filename}")
                    print(f"Processing image: {filename}")
                    try:
                        description = image_agent.print_response(
                            "What does this visualization show? Provide a detailed analysis.",
                            images=[Image(filepath=file_path)]
                        )
                        image_descriptions.append(f"Analysis of {filename}:\n{description}")
                    except Exception as e:
                        log.error(f"Error processing image {filename}: {e}")
                else:
                     log.warning(f"Image file path not found: {file_path}")

    # --- Final Analysis --- 
    image_section = "\n\nImage Analyses:\n" + "\n".join(image_descriptions) if image_descriptions else ""

    analysis_prompt = f"""Please provide a comprehensive analysis of the {group_name} report by synthesizing all available information:

1. Overview of the study area and objectives (if available)
2. Data sources and methodology used (if described)
3. Key findings and observations from text and data files
4. Statistical analysis of the parameters (if CSV data exists)
5. Interpretation of visualizations and their significance (if images exist)
6. Main conclusions and recommendations{image_section}

Focus on providing an integrated analysis that connects all pieces of information."""

    log.info("Generating comprehensive analysis")
    # Get the comprehensive analysis
    return agent.print_response(analysis_prompt, stream=True)

if __name__ == "__main__":
    cfg = get_config()
    reports = discover_reports(base_dir=cfg.get('base_report_dir'))

    if not reports:
        print("No reports found in the configured directory.")
    else:
        # Example: Analyze the latest report's groundwater group
        try:
            response = combined_reader(reports, group_name="groundwater", recreate_db_str="true")
            print("\nAnalysis:")
            # Response might be streamed, ensure it's handled appropriately if not printed directly
            # print(response) # This might not work directly if response is a generator
        except Exception as e:
            print(f"\nError during combined analysis: {e}")

        # To see available groups in a specific report:
        # print("\nListing groups for latest report:")
        # combined_reader(reports)