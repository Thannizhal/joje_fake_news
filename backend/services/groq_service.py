import json
import base64
from groq import Groq
from config import settings
from prompts.templates import (
    CLASSIFICATION_SYSTEM_PROMPT,
    CLASSIFICATION_USER_PROMPT,
    IMAGE_CLASSIFICATION_USER_PROMPT,
    AUDIO_CLASSIFICATION_USER_PROMPT,
)

client = Groq(api_key=settings.groq_api_key)

TEXT_MODEL = "llama-3.3-70b-versatile"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
WHISPER_MODEL = "whisper-large-v3"


def _parse_groq_json(raw: str) -> dict:
    """Strip markdown code fences if present, then parse JSON."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # remove first and last fence lines
        text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    return json.loads(text)


def classify_text(content: str) -> dict:
    """Classify plain text news using Groq LLM."""
    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": CLASSIFICATION_USER_PROMPT.format(content=content)},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    return _parse_groq_json(response.choices[0].message.content)


def classify_image(image_bytes: bytes, media_type: str) -> dict:
    """Classify a meme/infographic image using Groq Vision."""
    b64_image = base64.standard_b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{media_type};base64,{b64_image}"

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": IMAGE_CLASSIFICATION_USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    return _parse_groq_json(response.choices[0].message.content)


def transcribe_audio(audio_bytes: bytes, filename: str) -> str:
    """Transcribe audio using Groq Whisper. Returns transcribed text."""
    transcription = client.audio.transcriptions.create(
        model=WHISPER_MODEL,
        file=(filename, audio_bytes),
        response_format="text",
    )
    return transcription


def classify_audio(audio_bytes: bytes, filename: str) -> tuple[dict, str]:
    """Transcribe audio then classify the transcribed text. Returns (result, transcribed_text)."""
    transcribed_text = transcribe_audio(audio_bytes, filename)

    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": AUDIO_CLASSIFICATION_USER_PROMPT.format(
                    transcribed_text=transcribed_text
                ),
            },
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    result = _parse_groq_json(response.choices[0].message.content)
    return result, transcribed_text
