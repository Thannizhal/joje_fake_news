from fastapi import APIRouter, UploadFile, File, HTTPException
from schemas.models import TextClassifyRequest, ClassificationResponse
from services import groq_service

router = APIRouter(prefix="/api/classify", tags=["classify"])

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
ALLOWED_AUDIO_TYPES = {"audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/mp4", "audio/ogg", "audio/webm"}
MAX_FILE_SIZE_MB = 25


def _check_file_size(data: bytes, filename: str):
    size_mb = len(data) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File '{filename}' exceeds the {MAX_FILE_SIZE_MB}MB limit.",
        )


@router.post("/text", response_model=ClassificationResponse)
async def classify_text(request: TextClassifyRequest):
    if not request.content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty.")

    result = groq_service.classify_text(request.content)
    return ClassificationResponse(
        classification=result["classification"],
        fake_percentage=result["fake_percentage"],
        reasons=result["reasons"],
        summary=result["summary"],
        extracted_content=None,
    )


@router.post("/image", response_model=ClassificationResponse)
async def classify_image(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image type '{file.content_type}'. Allowed: JPEG, PNG, WEBP, GIF.",
        )

    image_bytes = await file.read()
    _check_file_size(image_bytes, file.filename)

    result = groq_service.classify_image(image_bytes, file.content_type)
    return ClassificationResponse(
        classification=result["classification"],
        fake_percentage=result["fake_percentage"],
        reasons=result["reasons"],
        summary=result["summary"],
        extracted_content=result.get("extracted_content"),
    )


@router.post("/audio", response_model=ClassificationResponse)
async def classify_audio(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio type '{file.content_type}'. Allowed: MP3, WAV, MP4, OGG, WEBM.",
        )

    audio_bytes = await file.read()
    _check_file_size(audio_bytes, file.filename)

    result, transcribed_text = groq_service.classify_audio(audio_bytes, file.filename)
    return ClassificationResponse(
        classification=result["classification"],
        fake_percentage=result["fake_percentage"],
        reasons=result["reasons"],
        summary=result["summary"],
        extracted_content=transcribed_text,
    )
