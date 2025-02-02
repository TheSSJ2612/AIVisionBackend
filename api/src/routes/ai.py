from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from ..services.ai_service import AIService
from PIL import Image
import io
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AIRouter:
    def __init__(self):
        self.router = APIRouter()
        self.ai_service = AIService()

        self.router.get("/ai/initialize", response_model=str)(self.initialize)
        self.router.get("/ai/consume", response_model=list)(self.consume)
        self.router.post("/ai/consume/image", response_model=list)(self.consume_image)
        self.router.post("/ai/consume/voice")(self.consume_voice)

    def initialize(self):
        start_time = time.time()
        result = self.ai_service.initalize()
        execution_time = time.time() - start_time
        print(f"Execution time for /ai/initialize: {execution_time:.2f} seconds")
        return result

    def consume(self):
        start_time = time.time()
        image = Image.open("api/assets/sample_img.jpg")
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "describe the image to a blind person in 50 words",
                    },
                    {"type": "image"},
                ],
            },
        ]
        result = self.ai_service.consume(image, conversation)
        execution_time = time.time() - start_time
        print(f"Execution time for /ai/consume: {execution_time:.2f} seconds")
        return result
    
    def consume_image(self, textInput: str, file: UploadFile = File(...)):
        start_time = time.time()

        try:
            # Read and validate the uploaded image file
            image_data = file.file.read()
            image = Image.open(io.BytesIO(image_data))

            # Verify image type
            if not isinstance(image, Image.Image):
                raise ValueError("Uploaded file is not a valid image.")

            # Prepare the conversation
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": textInput,
                        },
                        {"type": "image"},
                    ],
                },
            ]

            # Pass the image and conversation to the AI service
            result = self.ai_service.consume(image, conversation)

        except Exception as e:
            logger.error(f"Error in /ai/consume/image: {e}")
            raise RuntimeError("Error occurred while processing AI service.") from e

        # Measure execution time
        execution_time = time.time() - start_time
        print(f"Execution time for /ai/consume/image: {execution_time:.2f} seconds")

        return result
    
    
    def consume_voice(self, audio: UploadFile = File(...), image: UploadFile = File(...)):
        """
        Endpoint to process a voice query:
        - 'audio': Hindi voice input (question)
        - 'image': Associated image file for Visual QA
        Returns synthesized Hindi audio as the answer.
        """
        start_time = time.time()
        try:
            # Process the voice input through the AI service
            tts_audio = self.ai_service.consume_voice(audio, image)
        except Exception as e:
            logger.error(f"Error in /ai/consume/voice: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error occurred while processing voice input.") from e

        execution_time = time.time() - start_time
        print(f"Execution time for /ai/consume/voice: {execution_time:.2f} seconds")

        # Return the audio as a streaming response (MIME type 'audio/mpeg' for MP3)
        headers = {"Content-Disposition": "attachment; filename=output.mp3"}
        return StreamingResponse(io.BytesIO(tts_audio), media_type="audio/mpeg", headers=headers)


# Create an instance of UserRouter to use in main.py
ai_router = AIRouter().router
