from pathlib import Path
from dir_discover import discover_reports
import random



def image_reader(image_path):
    from agno.agent import Agent
    from agno.media import Image
    from agno.models.openai import OpenAIChat

    agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        agent_id="image-to-text",
        name="Image to Text Agent",
        markdown=True,
        debug_mode=True,
        show_tool_calls=True,
        instructions=[
            "Your Name is HydroDeepNet",
            "You are an AI agent that can generate text descriptions based on an image.",
            "You analyse the image and provide a detailed description of the image and your key findings.",
            "The context of the images are related to various environmental and water resources parameters",
        ],
        reasoning=True
    )
    
    return agent.print_response(
        "what do u see?",
        images=[Image(filepath=image_path)],
        stream=True
    )

if __name__ == "__main__":
    # Get a random image from the reports
    """Get a random PNG image from the reports structure."""
    
    reports_dict = discover_reports()
    
    report_timestamp = sorted(reports_dict.keys())[-1]

    image_path = reports_dict[report_timestamp]["groups"]["cdl"]["files"]['.png']['cdl_rotation.png']['path']
    
    print(image_path)

    response = image_reader(image_path)
    print(response)