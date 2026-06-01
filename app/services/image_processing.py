import io
import logging
from PIL import Image

from app.config import Settings

logger = logging.getLogger(__name__)

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "application/octet-stream",
}


def validate_upload(content_type: str | None, data: bytes, settings: Settings) -> None:
    from fastapi import HTTPException

    if content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported type: {content_type}")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="File too large")
    if len(data) < settings.min_upload_bytes:
        raise HTTPException(status_code=400, detail="File too small")


def preprocess_image(img_bytes: bytes, settings: Settings) -> bytes:
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        width, height = image.size
        logger.info("Original image: %sx%s, %.1fKB", width, height, len(img_bytes) / 1024)

        if width < settings.target_min_dimension or height < settings.target_min_dimension:
            scale = max(settings.target_min_dimension / width, settings.target_min_dimension / height)
            image = image.resize((int(width * scale), int(height * scale)), Image.LANCZOS)

        if image.width > settings.max_image_dimension or image.height > settings.max_image_dimension:
            image.thumbnail((settings.max_image_dimension, settings.max_image_dimension), Image.LANCZOS)

        output = io.BytesIO()
        image.save(output, format="JPEG", quality=settings.jpeg_quality, optimize=True)
        return output.getvalue()
    except Exception as exc:
        logger.warning("Image preprocessing failed, using original bytes: %s", exc)
        return img_bytes

