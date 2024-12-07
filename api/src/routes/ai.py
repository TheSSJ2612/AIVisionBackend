from fastapi import APIRouter
from ..services.ai_service import AIService
from PIL import Image


class AIRouter:
    def __init__(self):
        self.router = APIRouter()
        self.ai_service = AIService()

        self.router.get("/ai/initialize", response_model=str)(self.initialize)
        self.router.get("/ai/consume", response_model=list)(self.consume)

    def initialize(self):
        return self.ai_service.initalize()

    def consume(self):

        image = Image.open(
            "C:\\Users\\madan\\Projects\\BlindGlasses\\LocalLLMs\\TestData\\testimage2.jpg"
        )

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
        return self.ai_service.consume("some input text or media")


# Create an instance of UserRouter to use in main.py
ai_router = AIRouter().router
