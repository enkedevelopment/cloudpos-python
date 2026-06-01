from typing import Any
from pydantic import BaseModel, Field


class ScanResponse(BaseModel):
    success: bool = True
    data: dict[str, Any] | None = None
    error: str | None = None


class OCRResponse(BaseModel):
    success: bool = True
    text: str
    labels: list[str] = Field(default_factory=list)
    barcode: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    ollama: str
    model: str

