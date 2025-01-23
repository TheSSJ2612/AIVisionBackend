from transformers import pipeline, AutoProcessor
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AIService:
    def __init__(self):
        self.processor = None
        self.users_db = []

    def initalize(self):
        if self.processor:
            return "already initialized"
        print("initializing....")
        model_id = "llava-hf/llava-interleave-qwen-0.5b-hf"
        self.pipe = pipeline("image-to-text", model=model_id, device_map="cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        return "initialized successfully"

    def consume(self, image: Image.Image, conversation: str):
        if not self.processor:
            self.initalize()

        try:
            # Generate the prompt using the processor
            prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )

            # Ensure the image is a valid PIL Image
            if not isinstance(image, Image.Image):
                raise ValueError("The 'image' input must be a PIL Image object.")

            # Call the pipeline
            outputs = self.pipe(
                images=image,
                prompt=prompt,
                generate_kwargs={"max_new_tokens": 200}
            )
            return outputs

        except Exception as e:
            logger.error(f"Pipeline Error: {e}")
            raise RuntimeError("Error occurred while processing the AI model.") from e



