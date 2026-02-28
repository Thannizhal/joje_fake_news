from pydantic import BaseModel
from typing import Optional


class TextClassifyRequest(BaseModel):
    content: str


class ClassificationResponse(BaseModel):
    classification: str  # "FAKE" or "REAL"
    fake_percentage: int  # 0-100
    reasons: list[str]
    summary: str
    extracted_content: Optional[str] = None  # text from image/audio, null for text input
