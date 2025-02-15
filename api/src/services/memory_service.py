class MultimodalConversationMemory:
    """A simple in-memory conversation memory that preserves multimodal inputs."""

    def __init__(self, memory_key: str = "chat_history"):
        self.memory_key = memory_key
        self.messages = (
            []
        )  # Each message is a dict: {"role": str, "text": str, "image": Optional[str]}

    def add_message(self, role: str, text: str, image_data: str = None):
        entry = {"role": role, "text": text}
        if image_data:
            entry["image"] = image_data
        self.messages.append(entry)

    def load_memory_variables(self) -> dict:
        """Combine the conversation history into a single string."""
        memory_text = ""
        for msg in self.messages:
            if "image" in msg:
                # Display a shortened version of the image data for readability.
                memory_text += f"{msg['role']}: {msg['text']} [Image data: {msg['image'][:50]}...]\n"
            else:
                memory_text += f"{msg['role']}: {msg['text']}\n"
        return {self.memory_key: memory_text}

    def clear(self):
        self.messages = []
