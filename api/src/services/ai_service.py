from transformers import pipeline, AutoProcessor


class AIService:
    def __init__(self):
        self.processor = None
        self.users_db = []

    def initalize(self):
        if self.processor:
            return "already initialized"
        print("initializing....")
        model_id = "llava-hf/llava-interleave-qwen-0.5b-hf"
        self.pipe = pipeline("image-to-text", model=model_id, device=0)
        self.processor = AutoProcessor.from_pretrained(model_id)
        return "initialized successfully"

    def consume(self, image, conversation: str):
        if not self.processor:
            self.initalize()
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        outputs = self.pipe(
            image, prompt=prompt, generate_kwargs={"max_new_tokens": 200}
        )
        print("payload >>> ", image, conversation)
        return outputs
