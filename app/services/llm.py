import base64
import json
import logging
import re
from typing import Any

import httpx
from google import genai
from google.genai import types

from app.config import Settings

logger = logging.getLogger(__name__)


def call_ollama(prompt: str, settings: Settings, img_bytes: bytes | None = None) -> str:
    payload: dict[str, Any] = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 2048,
            "stop": ["Note:", "Explanation:", "---"],
        },
    }
    if img_bytes is not None:
        payload["images"] = [base64.b64encode(img_bytes).decode("utf-8")]

    try:
        with httpx.Client(timeout=settings.ollama_timeout_seconds) as client:
            response = client.post(f"{settings.ollama_base_url}/api/generate", json=payload)
            response.raise_for_status()
            return response.json().get("response", "")
    except httpx.ConnectError as exc:
        raise RuntimeError(f"Cannot connect to Ollama at {settings.ollama_base_url}") from exc
    except httpx.TimeoutException as exc:
        raise RuntimeError(f"Ollama timed out after {settings.ollama_timeout_seconds}s") from exc


def call_gemini(
    prompt: str,
    settings: Settings,
    system_prompt: str | None = None,
    img_bytes: bytes | None = None,
) -> str:
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    contents: list[Any] = [prompt]
    if img_bytes is not None:
        contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))

    try:
        with genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(timeout=int(settings.gemini_timeout_seconds * 1000)),
        ) as client:
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0,
                    max_output_tokens=4000,
                ),
            )
    except Exception as exc:
        raise RuntimeError(f"Gemini request failed: {exc}") from exc

    if not response.text:
        raise RuntimeError("Gemini returned empty content")
    return response.text


def call_text_llm(prompt: str, settings: Settings, system_prompt: str | None = None) -> str:
    if settings.gemini_api_key:
        return call_gemini(prompt, settings=settings, system_prompt=system_prompt)

    ollama_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
    return call_ollama(ollama_prompt, settings=settings)


def parse_json_response(text: str, fallback: dict[str, Any]) -> dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?", "", text.strip()).strip().rstrip("`")
    candidates = [cleaned]

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        candidates.append(match.group())

    start = cleaned.find("{")
    if start != -1:
        depth = 0
        for index, char in enumerate(cleaned[start:], start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(cleaned[start : index + 1])
                    break

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    logger.warning("Model JSON parse failed, using OCR fallback")
    return fallback


def product_fallback(raw_data: dict[str, Any]) -> dict[str, Any]:
    text = raw_data.get("text", "")
    first_line = text.split("\n")[0].strip() if text else "Unknown Product"
    brand = first_line.split()[0] if first_line else "unknown"
    return {
        "name": first_line or "Unknown Product",
        "brand": brand,
        "category": raw_data.get("labels", ["Unknown"])[0] if raw_data.get("labels") else "Unknown",
        "hsn_code": "unknown",
        "barcode": raw_data.get("barcode", "unknown"),
        "sku": f"{brand.upper()}-PRODUCT",
        "slug": re.sub(r"[^a-z0-9]+", "-", first_line.lower()).strip("-")[:50] or "unknown-product",
        "description": text[:200] or "Product details not available.",
        "base_unit": "unknown",
        "price": 0.0,
        "mrp": 0.0,
        "quantity": "unknown",
        "ingredients": "unknown",
        "country_of_origin": "India",
        "manufacturer": brand,
    }


def sanitize_product(product: dict[str, Any], raw_data: dict[str, Any]) -> dict[str, Any]:
    defaults = product_fallback(raw_data)
    for key, default in defaults.items():
        if product.get(key) in (None, "", "null"):
            product[key] = default

    if product["brand"] == "unknown" and product["name"] not in ("Unknown Product", "unknown"):
        product["brand"] = str(product["name"]).split()[0]

    for field in ("price", "mrp"):
        try:
            product[field] = float(product[field])
        except (TypeError, ValueError):
            product[field] = 0.0
    return product


def generate_product_json(raw_data: dict[str, Any], img_bytes: bytes, settings: Settings) -> dict[str, Any]:
    ocr_snippet = raw_data.get("text", "")[:1200]
    ocr_section = (
        f"OCR Text from packaging:\n{ocr_snippet}"
        if len(raw_data.get("text", "").strip()) > 10
        else "OCR: no text detected. Use image details only."
    )
    prompt = f"""You are a product data extraction assistant analyzing a product image.

{ocr_section}
Detected Labels: {raw_data.get("labels", [])}
Barcode: {raw_data.get("barcode", "Not found")}

Return only one valid JSON object with these exact fields:
{{"name":"Dove Cream Beauty Bar","brand":"Dove","category":"Personal Care","hsn_code":"34011110","barcode":"8901234567890","sku":"DOVE-CREAM-BAR-100G","slug":"dove-cream-beauty-bar","description":"Dove Cream Beauty Bar is a gentle moisturising soap.","base_unit":"100g","price":45.00,"mrp":45.00,"quantity":"100g","ingredients":"Sodium Lauroyl Isethionate, Stearic Acid, Water","country_of_origin":"India","manufacturer":"Hindustan Unilever Ltd"}}

Rules: never return null, never include markdown, and use real values from OCR/image when available."""
    if settings.gemini_api_key:
        response = call_gemini(prompt, settings=settings, img_bytes=img_bytes)
    else:
        response = call_ollama(prompt, settings=settings, img_bytes=img_bytes)
    if not response.strip():
        response = call_text_llm(f"Return product JSON from this OCR text:\n{ocr_snippet}", settings=settings)
    product = parse_json_response(response, product_fallback(raw_data))
    return sanitize_product(product, raw_data)
