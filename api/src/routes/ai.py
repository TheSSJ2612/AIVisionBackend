from fastapi import APIRouter, File, UploadFile
from ..services.ai_service import AIService
from PIL import Image
import io
import time


class AIRouter:
    def __init__(self):
        self.router = APIRouter()
        self.ai_service = AIService()

        self.router.get("/ai/initialize", response_model=str)(self.initialize)
        self.router.get("/ai/consume", response_model=list)(self.consume)
        self.router.post("/ai/consume/image", response_model=list)(self.consume_image)

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

    def consume_image(self, textInput, file: UploadFile = File(...)):
        start_time = time.time()
        # Read the uploaded image file
        image_data = file.file.read()
        image = Image.open(io.BytesIO(image_data))

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
        result = self.ai_service.consume(image, conversation)
        execution_time = time.time() - start_time
        print(f"Execution time for /ai/consume/image: {execution_time:.2f} seconds")
        return result


# Create an instance of UserRouter to use in main.py
ai_router = AIRouter().router
