import json
import logging
from functools import lru_cache

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def get_vision_module():
    try:
        from google.cloud import vision
    except ImportError as exc:
        raise RuntimeError(
            "Google Vision dependency is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return vision


@lru_cache
def get_vision_client():
    vision = get_vision_module()
    credentials_json = get_settings().google_service_account_json

    if credentials_json:
        try:
            from google.oauth2 import service_account

            credentials_info = json.loads(credentials_json)
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
            return vision.ImageAnnotatorClient(credentials=credentials)
        except (ImportError, json.JSONDecodeError, ValueError, TypeError) as exc:
            raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON is invalid") from exc

    try:
        return vision.ImageAnnotatorClient()
    except Exception as exc:
        raise RuntimeError(f"Google Vision credentials are unavailable: {exc}") from exc


def read_barcode(img_bytes: bytes) -> str | None:
    try:
        import io

        from PIL import Image
        from pyzbar.pyzbar import decode

        image = Image.open(io.BytesIO(img_bytes))
        codes = decode(image)
        return codes[0].data.decode("utf-8") if codes else None
    except ImportError as exc:
        logger.warning("Barcode dependencies are not installed: %s", exc)
        return None
    except Exception as exc:
        logger.warning("Barcode scan failed: %s", exc)
        return None


def extract_product_raw_data(img_bytes: bytes) -> dict:
    vision = get_vision_module()
    client = get_vision_client()
    image = vision.Image(content=img_bytes)

    try:
        text_response = client.text_detection(image=image)
        label_response = client.label_detection(image=image)
    except Exception as exc:
        raise RuntimeError(f"Google Vision OCR failed: {exc}") from exc

    if text_response.error.message:
        raise RuntimeError(text_response.error.message)
    if label_response.error.message:
        raise RuntimeError(label_response.error.message)

    text = text_response.text_annotations[0].description if text_response.text_annotations else ""
    labels = [label.description for label in label_response.label_annotations]
    barcode = read_barcode(img_bytes) or "Not found"
    return {"text": text, "labels": labels, "barcode": barcode}


def extract_document_text(img_bytes: bytes) -> str:
    vision = get_vision_module()
    client = get_vision_client()
    image = vision.Image(content=img_bytes)
    try:
        response = client.document_text_detection(image=image)
    except Exception as exc:
        raise RuntimeError(f"Google Vision OCR failed: {exc}") from exc
    if response.error.message:
        raise RuntimeError(f"Google Vision OCR failed: {response.error.message}")
    return response.full_text_annotation.text or ""
