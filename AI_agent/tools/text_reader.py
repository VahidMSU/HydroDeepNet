from pathlib import Path
from dir_discover import discover_reports
from ai_logger import LoggerSetup
import logging
from agno.agent import Agent
from agno.models.openai import OpenAIChat

def text_reader(text_path, logger=None):
    """
    Analyzes text files using an AI agent.

    Args:
        text_path: Path to the text file (e.g., .txt, .md)
        logger: Logger instance to use

    Returns:
        Analysis text from the agent
    """
    # Use provided logger or default
    log = logger or logging.getLogger(__name__)
    log.info(f"Reading and analyzing text file: {text_path}")

    # Read the text content
    try:
        with open(text_path, 'r', encoding='utf-8') as file:
            text_content = file.read()
        log.info(f"Successfully read {len(text_content)} characters from the file.")
    except Exception as e:
        log.error(f"Error reading text file {text_path}: {e}")
        return f"Error: Could not read file {text_path}."

    # Initialize the agent
    agent = Agent(
        model=OpenAIChat(id="gpt-4o"), # Or another suitable model
        agent_id="text-analyzer",
        name="Text Analysis Agent",
        markdown=True,
        debug_mode=True,
        show_tool_calls=True,
        instructions=[
            "Your Name is HydroLinguist",
            "You are an AI agent that analyzes text content, particularly reports.",
            "Assume the text might contain technical information, numbers, and tables related to environmental and water resources.",
            "Provide a structured summary of the key information, findings, and any data presented in the text.",
            "Identify the main topics and conclusions.",
        ],
        reasoning=True
    )

    log.info("Sending text content for analysis")

    # Create a prompt for the agent
    analysis_prompt = f"""Please analyze the following text content from the file {Path(text_path).name}:

    --- TEXT START ---
    {text_content[:5000]}
    --- TEXT END ---

    Provide a detailed summary including:
    1. The main subject or purpose of the text.
    2. Key findings, data points, or conclusions mentioned.
    3. Any structure or sections identified (e.g., introduction, methodology, results).
    4. Potential insights or implications based on the content.
    """
    if len(text_content) > 5000:
        analysis_prompt += "\n\nNote: The provided text was truncated to the first 5000 characters for analysis."


    result = agent.print_response(
        analysis_prompt,
        stream=True # Set to False if streaming is not desired or causes issues
    )

    log.info("Text analysis completed")
    return result

if __name__ == "__main__":
    reports_dict = discover_reports()
    report_timestamp = sorted(reports_dict.keys())[-1]
    print(report_timestamp)
    
    # Find the .md file in the climate_change group (adjust if needed)
    # This assumes the structure hasn't changed drastically
    try:
        md_path = reports_dict[report_timestamp]["groups"]["climate_change"]["files"]['.md']['climate_change_report.md']['path']
        print(f"Found text file: {md_path}")
        # Call the reader function (logging will be handled by the default logger inside)
        response = text_reader(md_path) 
        # The response is already printed by agent.print_response(stream=True)
        # print(response) # Uncomment if stream=False
    except KeyError as e:
        print(f"Error: Could not find the expected markdown file path. Key not found: {e}")
        print("Please check the discover_reports() output and update the path accordingly.")