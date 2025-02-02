from transformers import pipeline, AutoProcessor
from PIL import Image
import logging
import io

from google.cloud import speech
from google.cloud import texttospeech
from google.cloud import translate_v2 as translate

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
        
    # ----- New Helper Methods for STT, Translation, and TTS -----

    def stt(self, audio_bytes: bytes) -> str:
        """Convert Hindi audio bytes to text using Google Cloud Speech-to-Text."""
        client = speech.SpeechClient()
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,  # Adjust to match your audio file sample rate
            language_code="hi-IN",
        )
        audio = speech.RecognitionAudio(content=audio_bytes)
        response = client.recognize(config=config, audio=audio)
        transcript = " ".join(result.alternatives[0].transcript for result in response.results)
        logger.info(f"STT transcript: {transcript}")
        return transcript

    def translate_text(self, text: str, target_language: str) -> str:
        """Translate text using Google Cloud Translation API."""
        client = translate.Client()
        result = client.translate(text, target_language=target_language)
        translated_text = result["translatedText"]
        logger.info(f"Translated '{text}' to '{translated_text}' ({target_language})")
        return translated_text

    def synthesize_speech(self, text: str) -> bytes:
        """Convert Hindi text to speech using Google Cloud Text-to-Speech."""
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="hi-IN",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
        )
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        logger.info("Synthesized speech from text.")
        return response.audio_content

    def consume_voice(self, audio_file, image_file) -> bytes:
        """
        Process a voice query:
        1. Convert Hindi audio to text (STT)
        2. Translate Hindi text to English
        3. Build conversation prompt and run the Visual QA model
        4. Translate the English answer to Hindi
        5. Convert the Hindi answer to audio (TTS)
        Returns:
            Audio bytes (MP3) of the Hindi answer.
        """
        try:
            # 1. Read audio bytes and run STT
            audio_bytes = audio_file.file.read()
            hindi_question = self.stt(audio_bytes)

            # 2. Translate Hindi question to English
            english_question = self.translate_text(hindi_question, target_language="en")

            # 3. Read image bytes and open as PIL Image
            image_data = image_file.file.read()
            image = Image.open(io.BytesIO(image_data))

            # 4. Build conversation prompt with the English question
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": english_question},
                        {"type": "image"},
                    ],
                },
            ]

            if not self.processor:
                self.initalize()

            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

            # 5. Get the model's response (assumes pipeline returns a list with a 'generated_text' key)
            outputs = self.pipe(
                images=image,
                prompt=prompt,
                generate_kwargs={"max_new_tokens": 200}
            )
            english_answer = outputs[0].get("generated_text", "No answer generated")
            logger.info(f"Model English answer: {english_answer}")

            # 6. Translate the English answer back to Hindi
            hindi_answer = self.translate_text(english_answer, target_language="hi")

            # 7. Convert the Hindi answer to speech (TTS)
            tts_audio = self.synthesize_speech(hindi_answer)

            return tts_audio

        except Exception as e:
            logger.error(f"Error in consume_voice: {e}", exc_info=True)
            raise e
