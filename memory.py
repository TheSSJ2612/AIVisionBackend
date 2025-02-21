import json
from textwrap import dedent
from typing import Optional

import typer
from agno.agent import Agent, AgentMemory
from agno.memory.db.sqlite import SqliteMemoryDb
from agno.models.openai import OpenAIChat
from agno.storage.agent.sqlite import SqliteAgentStorage
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.prompt import Prompt

from agno.agent import Agent, RunResponse
from agno.models.openrouter import OpenRouter
from agno.media import Image
from agno.tools.tavily import TavilyTools
from agno.memory.classifier import MemoryClassifier
from agno.memory.summarizer import MemorySummarizer
from agno.memory.manager import MemoryManager
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


def create_agent(user: str = "user"):
    session_id: Optional[str] = None

    # Ask if user wants to start new session or continue existing one
    new = typer.confirm("Do you want to start a new session?")

    # Initialize storage for both agent sessions and memories
    agent_storage = SqliteAgentStorage(
        table_name="agent_memories", db_file="tmp/agents.db"
    )

    if not new:
        existing_sessions = agent_storage.get_all_session_ids(user)
        if len(existing_sessions) > 0:
            session_id = existing_sessions[0]

    agent = Agent(
        model=OpenRouter(
            id="google/gemini-2.0-flash-exp:free",
            api_key="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",
        ),
        tools=[
            TavilyTools(
                api_key="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",
                include_answer=True,
                search_depth="basic",
                use_search_context=True,
            )
        ],
        user_id=user,
        session_id=session_id,
        # Configure memory system with SQLite storage
        memory=AgentMemory(
            db=SqliteMemoryDb(
                table_name="agent_memory",
                db_file="tmp/agent_memory.db",
            ),
            create_user_memories=True,
            update_user_memories_after_run=True,
            create_session_summary=True,
            update_session_summary_after_run=True,
            classifier=MemoryClassifier(
                model=OpenRouter(
                    id="google/gemini-2.0-flash-exp:free",
                    api_key="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",
                ),
            ),
            summarizer=MemorySummarizer(
                model=OpenRouter(
                    id="google/gemini-2.0-flash-exp:free",
                    api_key="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",
                ),
            ),
            manager=MemoryManager(
                model=OpenRouter(
                    id="google/gemini-2.0-flash-exp:free",
                    api_key="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",
                ),
            ),
        ),
        storage=agent_storage,
        add_history_to_messages=True,
        num_history_responses=3,
        # Enhanced system prompt for better personality and memory usage
        description=dedent(
            """
            # ROLE
            You are a multimodal AI assistant specialized in assisting blind users by providing comprehensive answers using text and vision-based modalities.

            # ADDITIONAL INFORMATION
            ## Remember important details about users and reference them naturally, while always respecting privacy and data security.
            ## Maintain a warm, empathetic, and positive tone, using clear, accessible language that is easy to understand.
            ## When appropriate, refer back to previous conversations and memories to offer personalized support, but only if it enhances the current interaction.
            ## Always be truthful about what you remember or do not remember, and clearly state any limitations in your knowledge.
            ## Use all available tools to provide accurate, up-to-date information and detailed descriptions of visual content when necessary.
            ## If you are unsure about an answer, communicate your uncertainty clearly and offer alternative suggestions or guidance.
            ## Ensure that every response is designed with accessibility in mind, offering thorough explanations and descriptions to support users who rely on non-visual information.
            ## Provide helpful suggestions and guidance tailored to the unique needs of blind users.

            # TOOLS
            ## Tavily: Use this tool to search for real-time information and answer queries accurately, including providing detailed descriptions for visual content when applicable.
            """
        ),
        show_tool_calls=True,
        markdown=True,
    )

    if session_id is None:
        session_id = agent.session_id
        if session_id is not None:
            print(f"Started Session: {session_id}\n")
        else:
            print("Started Session\n")
    else:
        print(f"Continuing Session: {session_id}\n")

    return agent


def print_agent_memory(agent):
    """Print the current state of agent's memory systems"""
    console = Console()

    # Print chat history
    console.print(
        Panel(
            JSON(
                json.dumps([m.to_dict() for m in agent.memory.messages]),
                indent=4,
            ),
            title=f"Chat History for session_id: {agent.session_id}",
            expand=True,
        )
    )

    # Print user memories
    console.print(
        Panel(
            JSON(
                json.dumps(
                    [
                        m.model_dump(include={"memory", "input"})
                        for m in agent.memory.memories
                    ]
                ),
                indent=4,
            ),
            title=f"Memories for user_id: {agent.user_id}",
            expand=True,
        )
    )

    # Print session summary
    console.print(
        Panel(
            JSON(json.dumps(agent.memory.summary.model_dump(), indent=4)),
            title=f"Summary for session_id: {agent.session_id}",
            expand=True,
        )
    )


def main(user: str = "user"):
    """Interactive chat loop with optional image attachment and memory display"""
    agent = create_agent(user)

    print("Try these example inputs:")
    print("- 'My name is [name] and I live in [city]'")
    print("- 'I love [hobby/interest]'")
    print("- 'What do you remember about me?'")
    print("- 'What have we discussed so far?'\n")

    exit_on = ["exit", "quit", "bye"]
    while True:
        # Get text message from the user
        message = Prompt.ask(f"[bold]😎 {user} [/bold]")
        if message.lower() in exit_on:
            break

        # Ask whether the user wants to attach an image
        attach_image = Prompt.ask(
            "Do you want to attach an image? (y/n)", choices=["y", "n"], default="n"
        )
        images = None

        if attach_image.lower() == "y":
            image_path = Prompt.ask("Enter the image file path")
            try:
                # Convert the image file to a Data URL
                image_data_url = image_file_to_data_url(image_path)
                images = [Image(url=image_data_url)]
            except Exception as e:
                print(f"Failed to attach image: {e}")

        # Send the message (with or without an image) to the agent
        if images:
            agent.print_response(
                message,
                images=images,
                stream=True,
                markdown=True,
            )
        else:
            agent.print_response(
                message=message,
                stream=True,
                markdown=True,
            )

        # Display the updated memory
        print_agent_memory(agent)


if __name__ == "__main__":
    typer.run(main)
