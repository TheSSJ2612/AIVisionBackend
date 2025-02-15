from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from api.src.services.ai_service import AIService
from PIL import Image
import io
import time
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AIRouter:
    def __init__(self):
        print("AIRouter START")
        self.router = APIRouter()
        self.ai_service = AIService()
        self.router.get("/ai/initialize", response_model=str)(self.initialize)
        self.router.get("/ai/consume", response_model=dict)(self.consume)
        self.router.post("/ai/consume/image", response_model=dict)(self.consume_image)
        # self.router.post("/ai/consume/voice")(self.consume_voice)
        print("AIRouter END")

    def initialize(self):
        print("Initializing AI service...")
        try:
            start_time = time.time()
            print("Initializing AI service...")
            result = self.ai_service.initalize()
            execution_time = time.time() - start_time
            print(f"Execution time for /ai/initialize: {execution_time:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"Error in /ai/initialize: {e}")
            print(f"Error in /ai/initialize: {e}")
            raise HTTPException(
                status_code=500, detail="Error occurred while initializing AI service."
            )

    def consume(self):
        start_time = time.time()
        print("Consuming AI service...")
        # For testing: load a local sample image.
        image = Image.open("api/assets/sample_img.jpg")
        query_text = "describe the image to a blind person in 50 words"
        result = self.ai_service.consume(image, query_text)
        execution_time = time.time() - start_time
        print(f"Execution time for /ai/consume: {execution_time:.2f} seconds")
        return result

    def consume_image(self, textInput: str = Form(""), file: UploadFile = File(...)):
        start_time = time.time()
        print("Consuming AI service with image...")
        try:
            image_bytes = file.file.read()
            image = Image.open(io.BytesIO(image_bytes))
            if not isinstance(image, Image.Image):
                raise ValueError("Uploaded file is not a valid image.")
            result = self.ai_service.consume(image, textInput)
        except Exception as e:
            logger.error(f"Error in /ai/consume/image: {e}")

            raise HTTPException(
                status_code=500, detail="Error occurred while processing AI service."
            ) from e
        execution_time = time.time() - start_time
        print(f"Execution time for /ai/consume/image: {execution_time:.2f} seconds")
        return result

    # def consume_voice(
    #     self, audio: UploadFile = File(...), image: UploadFile = File(...)
    # ):
    #     """
    #     Endpoint to process a voice query:
    #     - 'audio': Hindi voice input (question)
    #     - 'image': Associated image file for Visual QA
    #     Returns synthesized Hindi audio as the answer.
    #     """
    #     start_time = time.time()
    #     try:
    #         # Process the voice input through the AI service
    #         tts_audio = self.ai_service.consume_voice(audio, image)
    #     except Exception as e:
    #         logger.error(f"Error in /ai/consume/voice: {e}", exc_info=True)
    #         raise HTTPException(
    #             status_code=500, detail="Error occurred while processing voice input."
    #         ) from e

    #     execution_time = time.time() - start_time
    #     print(f"Execution time for /ai/consume/voice: {execution_time:.2f} seconds")

    #     # Return the audio as a streaming response (MIME type 'audio/mpeg' for MP3)
    #     headers = {"Content-Disposition": "attachment; filename=output.mp3"}
    #     return StreamingResponse(
    #         io.BytesIO(tts_audio), media_type="audio/mpeg", headers=headers
    #     )


# Create an instance of AIRouter to be included in main.py
ai_router = AIRouter().router
