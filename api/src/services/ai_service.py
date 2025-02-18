from PIL import Image
import logging
import io

from google.cloud import speech
from google.cloud import texttospeech
from google.cloud import translate_v2 as translate

import base64
import os

from typing import List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate

from api.src.services.memory_service import MultimodalConversationMemory

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AIService:
    def __init__(self):
        self.agent = None  # This will be our LangChain agent instance.
        self.agent_executor = None
        self.memory = MultimodalConversationMemory(memory_key="chat_history")
        self.users_db = []

    def initalize(self):
        try:

            if self.agent:
                return "already initialized"
            logger.info(
                "Initializing LangChain agent with OpenRouter API model and Tavily search tool..."
            )

            llm = ChatOpenAI(
                openai_api_key=os.getenv("OPENROUTER_API_KEY"),
                openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
                model_name=os.getenv("OPENROUTER_MODEL_NAME"),
                temperature=os.getenv("OPENROUTER_MODEL_TEMPERATURE"),
            )

            # Initialize the Tavily search tool for web search.
            tavily_tool = TavilySearchResults(
                name="tavily_search_engine",
                description="A search engine optimized for retrieving information from web based on user query.",
                max_results=5,
                search_depth="basic",
                include_answer=True,
                include_raw_content=True,
                include_images=False,
                verbose=False,
            )
            tools = [tavily_tool]

            # Create multimodal prompt template
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """You are a multimodal assistant for blind user. 
                        Analyze both text and images to answer questions.
                        You have access to Chat history between assistant and user.
                        You have access to tavily_tool which can be used to search the web.
                        Follow these rules:
                        1. Focus FIRST on [CURRENT REQUEST] and [CURRENT IMAGES]
                        2. Use tavily_tool to search for information whenever latest information or news or real-time data is needed
                        3. If you cant give a good answer, say so and ask for more information
                        4. Only use [CONVERSATION HISTORY CONTEXT] if relevant and necessary. Give preference to latest user query.
                        5. Never describe old images unless explicitly asked
                        6. State clearly when using historical context""",
                    ),
                    ("placeholder", "{chat_history}"),
                    ("human", "{input}"),
                    ("placeholder", "{agent_scratchpad}"),
                ]
            )

            # Create a LangChain agent that can call the Tavily tool when needed.
            self.agent = create_tool_calling_agent(llm, tools, prompt)
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=tools,
                memory=self.memory,
                verbose=True,
                handle_parsing_errors=True,
            )
            return "initialized successfully"
        except Exception as e:
            logger.error(f"Error in initalize: {e}")
            raise RuntimeError("Error occurred while initializing the AI model.") from e

    def consume(self, image: Image.Image, query_text: str) -> dict:
        """
        Process a new user query along with an optional image.
        The user sends only the latest query; conversation history is maintained on the backend.
        """
        if not self.agent:
            self.initalize()
        try:
            user_text = query_text.strip()
            image_data = None
            if image:
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # Update memory with the new user message.
            self.memory.add_message("user", user_text, image_data)

            # Build the prompt by relying on the agent's internal memory.
            # For this implementation, we simply use the latest query.
            combined_input = user_text
            if image_data:
                combined_input += " [Image data received]"
            print("combined_input", combined_input)
            # Run the agent. The agent will consider the full conversation history stored in memory.
            response = self.agent.run(combined_input)
            logger.info(f"Assistant response: {response}")
            print("Assistant response: ", response)
            # Update memory with the assistant's response.
            self.memory.add_message("assistant", response)
            return {"generated_text": response}
        except Exception as e:
            logger.error(f"Pipeline Error: {e}")
            raise RuntimeError("Error occurred while processing the AI model.") from e

    def process_multimodal_input(
        self, text: str = None, images: Union[str, bytes] = None
    ):
        """Handle text, images, or both"""
        content = []
        user_text = ""
        image_data = ""

        if text:
            content.append({"type": "text", "text": text})
            user_text = text.strip()

        if images:
            img = images  # Handle different image formats
            b64_img = ""
            if isinstance(img, str):
                if img.startswith("http"):
                    # URL case
                    b64_img = img
                    content.append({"type": "image_url", "image_url": {"url": img}})
                elif os.path.exists(img):  # Actual file path
                    with open(img, "rb") as f:
                        img_bytes = f.read()
                    b64_img = base64.b64encode(img_bytes).decode("utf-8")
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                        }
                    )
                else:  # Base64 string case
                    b64_img = img
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                        }
                    )
            elif isinstance(img, bytes):
                b64_img = base64.b64encode(img).decode("utf-8")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                    }
                )
            image_data = b64_img

        # Update memory with the new user message.
        self.memory.add_message("user", user_text, image_data)

        response = self.agent_executor.invoke({"input": content})

        output = response["output"]

        self.memory.add_message("assistant", output)

        return {"generated_text": output}

    def load_image_as_base64(self, file_path: str) -> str:
        """Return base64 string only (original version)"""
        with open(file_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # def stt(self, audio_bytes: bytes) -> str:
    #     """Convert Hindi audio bytes to text using Google Cloud Speech-to-Text."""
    #     client = speech.SpeechClient()
    #     config = speech.RecognitionConfig(
    #         encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    #         sample_rate_hertz=44100,  # Adjust to match your audio file sample rate
    #         language_code="hi-IN",
    #     )
    #     audio = speech.RecognitionAudio(content=audio_bytes)
    #     response = client.recognize(config=config, audio=audio)
    #     transcript = " ".join(
    #         result.alternatives[0].transcript for result in response.results
    #     )
    #     logger.info(f"STT transcript: {transcript}")
    #     return transcript

    # def translate_text(self, text: str, target_language: str) -> str:
    #     """Translate text using Google Cloud Translation API."""
    #     client = translate.Client()
    #     result = client.translate(text, target_language=target_language)
    #     translated_text = result["translatedText"]
    #     logger.info(f"Translated '{text}' to '{translated_text}' ({target_language})")
    #     return translated_text

    # def synthesize_speech(self, text: str) -> bytes:
    #     """Convert Hindi text to speech using Google Cloud Text-to-Speech."""
    #     client = texttospeech.TextToSpeechClient()
    #     synthesis_input = texttospeech.SynthesisInput(text=text)
    #     voice = texttospeech.VoiceSelectionParams(
    #         language_code="hi-IN",
    #         ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
    #     )
    #     audio_config = texttospeech.AudioConfig(
    #         audio_encoding=texttospeech.AudioEncoding.MP3,
    #     )
    #     response = client.synthesize_speech(
    #         input=synthesis_input, voice=voice, audio_config=audio_config
    #     )
    #     logger.info("Synthesized speech from text.")
    #     return response.audio_content

    # def consume_voice(self, audio_file, image_file) -> bytes:
    #     """
    #     Process a voice query:
    #     1. Convert Hindi audio to text (STT)
    #     2. Translate Hindi text to English
    #     3. Build conversation prompt and run the Visual QA model
    #     4. Translate the English answer to Hindi
    #     5. Convert the Hindi answer to audio (TTS)
    #     Returns:
    #         Audio bytes (MP3) of the Hindi answer.
    #     """
    #     import google.auth

    #     credentials, project = google.auth.default()
    #     print("Active project:", project)

    #     try:
    #         # 1. Read audio bytes and run STT
    #         audio_bytes = audio_file.file.read()
    #         hindi_question = self.stt(audio_bytes)

    #         # 2. Translate Hindi question to English
    #         english_question = self.translate_text(hindi_question, target_language="en")

    #         # 3. Read image bytes and open as PIL Image
    #         image_data = image_file.file.read()
    #         image = Image.open(io.BytesIO(image_data))

    #         # 4. Build conversation prompt with the English question
    #         conversation = [
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {"type": "text", "text": english_question},
    #                     {"type": "image"},
    #                 ],
    #             },
    #         ]

    #         if not self.processor:
    #             self.initalize()

    #         prompt = self.processor.apply_chat_template(
    #             conversation, add_generation_prompt=True
    #         )

    #         # 5. Get the model's response (assumes pipeline returns a list with a 'generated_text' key)
    #         outputs = self.pipe(
    #             images=image, prompt=prompt, generate_kwargs={"max_new_tokens": 200}
    #         )
    #         english_answer = outputs[0].get("generated_text", "No answer generated")
    #         logger.info(f"Model English answer: {english_answer}")

    #         assistant_text = self.extract_assistant_response(english_answer)
    #         logger.info(f"Extracted assistant response: {assistant_text}")

    #         # 6. Translate the English answer back to Hindi
    #         hindi_answer = self.translate_text(assistant_text, target_language="hi")

    #         # 7. Convert the Hindi answer to speech (TTS)
    #         tts_audio = self.synthesize_speech(hindi_answer)

    #         return tts_audio

    #     except Exception as e:
    #         logger.error(f"Error in consume_voice: {e}", exc_info=True)
    #         raise e

    def extract_assistant_response(self, full_text: str) -> str:
        marker = "assistant"
        lower_text = full_text.lower()
        index = lower_text.find(marker)
        if index != -1:
            extracted = full_text[index + len(marker) :].strip()
            return extracted
        return full_text
