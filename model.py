from transformers import pipeline, AutoProcessor
from PIL import Image
import requests

model_id = "llava-hf/llava-interleave-qwen-0.5b-hf"
pipe = pipeline("image-to-text", model=model_id, device=0)
processor = AutoProcessor.from_pretrained(model_id)
image = Image.open("test2.jpeg")

# Define a chat histiry and use apply_chat_template to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image")
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
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
print(outputs)
