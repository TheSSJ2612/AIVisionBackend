from fastapi import APIRouter, File, UploadFile
from ..services.ai_service import AIService
from PIL import Image
import io


class AIRouter:
    def __init__(self):
        self.router = APIRouter()
        self.ai_service = AIService()

        self.router.get("/ai/initialize", response_model=str)(self.initialize)
        self.router.get("/ai/consume", response_model=list)(self.consume)
        self.router.post("/ai/consume/image", response_model=list)(self.consume_image)

    def initialize(self):
        return self.ai_service.initalize()

    def consume(self):
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
        return self.ai_service.consume(image, conversation)

    def consume_image(self, file: UploadFile = File(...)):
        # Read the uploaded image file
        image_data = file.file.read()
        image = Image.open(io.BytesIO(image_data))

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
        return self.ai_service.consume(image, conversation)


# Create an instance of UserRouter to use in main.py
ai_router = AIRouter().router
