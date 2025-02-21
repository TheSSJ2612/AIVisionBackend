from agno.agent import Agent, RunResponse
from agno.models.openrouter import OpenRouter
from agno.media import Image
from agno.tools.tavily import TavilyTools
import base64
import mimetypes


def image_file_to_data_url(image_path):
    """
    Convert a local image file to a Data URL.

    Args:
        image_path (str): The file path to the image.

    Returns:
        str: The Data URL containing the Base64-encoded image.
    """
    # Determine the MIME type of the image
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        raise ValueError(f"Unable to determine MIME type for {image_path}")

    # Read the image file in binary mode and encode it to Base64
    with open(image_path, "rb") as image_file:
        base64_encoded = base64.b64encode(image_file.read()).decode("utf-8")

    # Create the Data URL
    data_url = f"data:{mime_type};base64,{base64_encoded}"
    return data_url


agent = Agent(
    model=OpenRouter(
        id="google/gemini-2.0-flash-exp:free",
        api_key="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",
    ),
    tools=[TavilyTools(api_key="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")],
    show_tool_calls=True,
    markdown=True,
)

# Print the response in the terminal
agent.print_response(
    "Describe the image and get me the latest price of products in the image online",
    images=[Image(url=image_file_to_data_url("hq720.jpg"))],
    stream=True,
)
