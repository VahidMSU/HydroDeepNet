from pathlib import Path

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
        "You are an AI agent that can generate text descriptions based on an image.",
        "You have to return a text response describing the image.",
    ],
)
image_path = "/data/SWATGenXApp/Users/admin/Reports/20250324_222749/groundwater/groundwater_correlation.png"
agent.print_response(
    "what do u see?",
    images=[Image(filepath=image_path)],
    stream=True,
)