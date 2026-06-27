import asyncio
import logging
import traceback
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.schemas import HealthResponse, OCRResponse, ScanResponse
from app.services.image_processing import preprocess_image, validate_upload
from app.services.invoice import parse_invoice_text
from app.services.llm import generate_product_json
from app.services.search import download_image, search_product_image_url
from app.services.vision import extract_document_text, extract_product_raw_data

settings = get_settings()
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("epose-ai")
scan_semaphore = asyncio.Semaphore(settings.max_concurrent_scans)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.install_google_credentials_env()
    logger.info("%s %s starting", settings.app_name, settings.app_version)
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


async def _read_validated_image(file: UploadFile) -> bytes:
    data = await file.read()
    validate_upload(file.content_type, data, settings)
    logger.info(
        "Received file=%s type=%s size=%.1fKB",
        file.filename,
        file.content_type,
        len(data) / 1024,
    )
    return data


async def _scan_image_bytes(img_bytes: bytes, include_web_image: bool = True) -> dict:
    async with scan_semaphore:
        processed = await run_in_threadpool(preprocess_image, img_bytes, settings)
        raw = await run_in_threadpool(extract_product_raw_data, processed)
        product = await run_in_threadpool(generate_product_json, raw, processed, settings)

        product_name = product.get("name", "")
        if include_web_image and product_name and product_name != "Unknown Product":
            try:
                image_url = await run_in_threadpool(search_product_image_url, product_name, settings)
                if image_url:
                    product["web_image_url"] = image_url
                    product["image_url"] = image_url
            except Exception as exc:
                logger.warning("Product image lookup unavailable: %s", exc)

        logger.info("Scan complete: %s | MRP: %s", product.get("name"), product.get("mrp"))
        return product


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    if settings.gemini_api_key:
        return HealthResponse(
            status="ok",
            version=settings.app_version,
            provider="gemini",
            model=settings.gemini_model,
        )

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{settings.ollama_base_url}/api/tags")
            response.raise_for_status()
            models = [model["name"] for model in response.json().get("models", [])]
            model_ok = any(settings.ollama_model in model for model in models)
        return HealthResponse(
            status="ok" if model_ok else "degraded",
            version=settings.app_version,
            provider="ollama",
            model=settings.ollama_model,
        )
    except Exception as exc:
        return HealthResponse(
            status="degraded",
            version=settings.app_version,
            provider=f"ollama (unreachable: {exc})",
            model=settings.ollama_model,
        )


@app.post("/scan", response_model=ScanResponse)
async def scan_product(file: UploadFile = File(...)) -> ScanResponse:
    try:
        img_bytes = await _read_validated_image(file)
        product = await _scan_image_bytes(img_bytes)
        return ScanResponse(data=product)
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Product scan failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@app.get("/scan_by_name", response_model=ScanResponse)
async def scan_product_by_name(
    product_name: str = Query(..., min_length=1, description="Product name to search")
) -> ScanResponse:
    try:
        image_url = await run_in_threadpool(search_product_image_url, product_name.strip(), settings)
        if not image_url:
            raise HTTPException(status_code=404, detail=f"No image found for product '{product_name}'")
        img_bytes = await run_in_threadpool(download_image, image_url, settings)
        product = await _scan_image_bytes(img_bytes, include_web_image=False)
        product["web_image_url"] = image_url
        product["image_url"] = image_url
        return ScanResponse(data=product)
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Name scan failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@app.post("/ocr", response_model=OCRResponse)
async def ocr_file(file: UploadFile = File(...)) -> OCRResponse:
    try:
        img_bytes = await _read_validated_image(file)
        processed = await run_in_threadpool(preprocess_image, img_bytes, settings)
        raw = await run_in_threadpool(extract_product_raw_data, processed)
        return OCRResponse(
            text=raw.get("text", ""),
            labels=raw.get("labels", []),
            barcode=raw.get("barcode"),
        )
    except HTTPException:
        raise
    except RuntimeError as exc:
        logger.error("OCR service unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("OCR failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@app.post("/invoice-scan", response_model=ScanResponse)
async def scan_invoice(file: UploadFile = File(...)) -> ScanResponse:
    try:
        img_bytes = await _read_validated_image(file)
        async with scan_semaphore:
            processed = await run_in_threadpool(preprocess_image, img_bytes, settings)
            logger.info("Running Google Vision document OCR")
            raw_ocr = await run_in_threadpool(extract_document_text, processed)
            logger.info("Invoice OCR extracted %s characters", len(raw_ocr))
            logger.info("Parsing invoice OCR text")
            data = await run_in_threadpool(parse_invoice_text, raw_ocr, settings)
        logger.info("Invoice scan complete: %s items", len(data.get("items", [])))
        return ScanResponse(data=data)
    except HTTPException:
        raise
    except RuntimeError as exc:
        logger.error("Invoice scan service unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Invoice scan failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error") from exc
