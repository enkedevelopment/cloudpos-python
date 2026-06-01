import logging
import time

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)


def _is_reachable_image(url: str, settings: Settings) -> bool:
    try:
        with httpx.Client(timeout=settings.web_search_timeout_seconds, follow_redirects=True) as client:
            response = client.get(url)
        if response.status_code >= 400:
            return False
        return response.headers.get("content-type", "").lower().startswith("image/")
    except Exception as exc:
        logger.info("Skipping image result that could not be fetched: %s", exc)
        return False


def search_product_image_url(product_name: str, settings: Settings) -> str | None:
    from ddgs import DDGS

    query = f"{product_name} product"
    for attempt in range(settings.web_search_retries):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.images(query, max_results=5))
            for result in results:
                image_url = result.get("image")
                if image_url and _is_reachable_image(image_url, settings):
                    return image_url
            return None
        except Exception as exc:
            error = str(exc).lower()
            if "ratelimit" in error or "403" in error:
                time.sleep(2**attempt)
                continue
            logger.warning("Image search failed: %s", exc)
            return None
    return None


def download_image(url: str, settings: Settings) -> bytes:
    with httpx.Client(timeout=settings.web_search_timeout_seconds, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if not content_type.startswith("image/"):
            raise RuntimeError(f"Search result was not an image: {content_type}")
        return response.content
