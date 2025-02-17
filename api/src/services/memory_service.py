from typing import List, Dict
from pydantic import BaseModel, Field
from langchain.memory.chat_memory import BaseChatMemory


class MultimodalConversationMemory(BaseChatMemory, BaseModel):
    memory_key: str = Field(default="chat_history")
    messages: List[Dict] = Field(default_factory=list)

    @property
    def memory_variables(self) -> list:
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict) -> dict:
        memory_text = ""
        for msg in self.messages:
            if "image" in msg:
                memory_text += f"{msg['role']}: {msg['text']} [Image data: {msg['image'][:50]}...]\n"
            else:
                memory_text += f"{msg['role']}: {msg['text']}\n"
        return {self.memory_key: memory_text}

    def save_context(self, inputs: dict, outputs: dict) -> None:
        user_input = inputs.get("input", "")
        assistant_output = outputs.get("output", "")
        self.messages.append({"role": "user", "text": user_input})
        self.messages.append({"role": "assistant", "text": assistant_output})

    def add_message(self, role: str, text: str, image_data: str = None):
        entry = {"role": role, "text": text}
        if image_data:
            entry["image"] = image_data
        self.messages.append(entry)
