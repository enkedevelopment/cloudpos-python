import json
import logging
import re
from typing import Any

from openai import OpenAI

from app.config import Settings
from app.services.llm import call_ollama, parse_json_response

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a ledger OCR parser. Extract product rows from OCR invoice text and return structured JSON.

Instructions:
1. Extract only product rows with SI number, item name, quantity, rate, and amount.
2. Ignore headers, voucher details, totals, footers, and party details.
3. Merge multiline item names when adjacent lines continue a product name.
4. Keep original SI numbers.
5. Output only valid JSON, no markdown, no explanation.

Expected JSON:
{"items":[{"SI":"1","Item":"PRODUCT NAME","Qty":"10","Rate":"100.00","Amount":"1000.00"}]}"""


def clean_invoice_ocr(text: str) -> str:
    garbage_words = {
        "voucher",
        "party",
        "gross",
        "round",
        "bill amount",
        "net total",
        "sales",
        "return",
        "date",
        "step",
    }
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not any(word in stripped.lower() for word in garbage_words):
            lines.append(stripped)
    return "\n".join(lines)[:6000]


def _is_number(value: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:,\d{2,3})*(?:\.\d+)?|\d+(?:\.\d+)?", value))


def _clean_number(value: str) -> str:
    return value.replace(",", "")


def _clean_item_name(value: str) -> str:
    value = re.sub(r"\s+", " ", value)
    return value.strip(" -:\t")


def _parse_invoice_fallback(ocr_text: str) -> dict[str, Any]:
    items: list[dict[str, str]] = []
    current: dict[str, str] | None = None

    for raw_line in clean_invoice_ocr(ocr_text).splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue

        row_match = re.match(r"^(\d{1,4})\s+(.+)$", line)
        if row_match:
            if current:
                items.append(current)

            serial = row_match.group(1)
            content = row_match.group(2)
            tokens = content.split()
            numeric_positions = [
                (index, _clean_number(token))
                for index, token in enumerate(tokens)
                if _is_number(token)
            ]

            qty = rate = amount = ""
            item_end = len(tokens)
            if len(numeric_positions) >= 3:
                qty = numeric_positions[-3][1]
                rate = numeric_positions[-2][1]
                amount = numeric_positions[-1][1]
                item_end = numeric_positions[-3][0]
            elif len(numeric_positions) >= 2:
                qty = numeric_positions[-2][1]
                rate = numeric_positions[-1][1]
                item_end = numeric_positions[-2][0]

            current = {
                "SI": serial,
                "Item": _clean_item_name(" ".join(tokens[:item_end])),
                "Qty": qty,
                "Rate": rate,
                "Amount": amount,
            }
            continue

        if not current:
            continue

        numbers = [_clean_number(value) for value in re.findall(r"\d+(?:,\d{2,3})*(?:\.\d+)?|\d+(?:\.\d+)?", line)]
        if numbers:
            if not current["Amount"]:
                current["Amount"] = numbers[-1]
            if not current["Rate"] and len(numbers) >= 2:
                current["Rate"] = numbers[-2]
            if not current["Qty"] and len(numbers) >= 3:
                current["Qty"] = numbers[-3]
        else:
            current["Item"] = _clean_item_name(f"{current['Item']} {line}")

    if current:
        items.append(current)

    return {"items": [item for item in items if item["Item"]]}


def _normalize_invoice(data: dict[str, Any]) -> dict[str, Any]:
    items = data.get("items")
    if not isinstance(items, list):
        data["items"] = []
    return data


def parse_invoice_text(ocr_text: str, settings: Settings) -> dict[str, Any]:
    cleaned = clean_invoice_ocr(ocr_text)
    user_prompt = f"Parse this OCR invoice text and extract product rows:\n\n{cleaned}"

    try:
        if settings.openrouter_api_key:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=settings.openrouter_api_key,
                timeout=settings.openrouter_timeout_seconds,
            )
            completion = client.chat.completions.create(
                model=settings.openrouter_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=4000,
                extra_body={"reasoning": {"effort": "none", "exclude": True}},
            )
            content = completion.choices[0].message.content if completion.choices else None
            if not content:
                raise RuntimeError("Invoice parser returned empty content")
            data = parse_json_response(content, {"items": []})
            normalized = _normalize_invoice(data)
        else:
            prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
            response = call_ollama(prompt, settings=settings)
            data = parse_json_response(response, {"items": []})
            normalized = _normalize_invoice(data)

        if normalized["items"]:
            return normalized
        logger.warning("Invoice parser returned no items, using OCR fallback parser")
    except Exception as exc:
        logger.warning("Invoice parser unavailable, using OCR fallback parser: %s", exc)

    return _parse_invoice_fallback(ocr_text)


def extract_json_text(response: str) -> dict[str, Any]:
    cleaned = response.strip().replace("```json", "").replace("```", "")
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise RuntimeError("No JSON object found in response")
    json_text = re.sub(r",\s*([}\]])", r"\1", cleaned[start : end + 1])
    return json.loads(json_text)
