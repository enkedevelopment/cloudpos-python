# Epose AI Scanner API

Production-ready FastAPI service for product OCR, barcode scanning, product JSON extraction, product image lookup, raw OCR, and invoice item extraction.

## Endpoints

- `GET /health`
- `POST /scan` with multipart field `file`
- `GET /scan_by_name?product_name=...`
- `POST /ocr` with multipart field `file`
- `POST /invoice-scan` with multipart field `file`

## Run Locally

```bash
cd epose-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 5000
```

## Production Run

```bash
cd epose-ai
gunicorn app.main:app \
  --worker-class uvicorn.workers.UvicornWorker \
  --workers 4 \
  --bind 0.0.0.0:5000 \
  --timeout 240
```

Set `GOOGLE_APPLICATION_CREDENTIALS` to a Google Vision service-account JSON path locally. On Vercel, set `GOOGLE_SERVICE_ACCOUNT_JSON` to the complete JSON content instead. Set `GEMINI_API_KEY` to use Gemini for product and invoice JSON extraction; otherwise the service falls back to Ollama. Keep all credentials out of source control.
# python-ai-epos
